use std::io;

use proxima_tensor::spec::{
    RopePairing, causal_mask, elementwise, fused_rope_pair, input_leaf, reduce, rmsnorm,
    scalar_constant,
};
use proxima_tensor::{DType, Extent, NodeId, Op, ReduceInit, ScalarOp};

pub struct Graph {
    pub program: Vec<Op>,
    pub root: NodeId,
    pub taps: Vec<(&'static str, NodeId)>,
}

impl Graph {
    pub fn norm_sites(&self) -> usize {
        self.program
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    Op::Elementwise {
                        body: ScalarOp::SquareRoot,
                        ..
                    }
                )
            })
            .count()
    }
}

#[derive(Clone, Copy)]
pub struct Shape {
    pub hidden: u32,
    pub intermediate: u32,
    pub heads: u32,
    pub head_width: u32,
    pub vocab: u32,
}

fn apply(program: &mut Vec<Op>, body: ScalarOp, operands: &[(NodeId, &str)]) -> io::Result<NodeId> {
    elementwise(program, DType::Float32, body, operands).map_err(io::Error::other)
}

fn sum(program: &mut Vec<Op>, operand: NodeId, input: &str, output: &str) -> io::Result<NodeId> {
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        operand,
        input,
        output,
    )
    .map_err(io::Error::other)
}

fn leaf(program: &mut Vec<Op>, name: &str, dimensions: &[u32]) -> NodeId {
    input_leaf(
        program,
        DType::Float32,
        dimensions.iter().copied().map(Extent::Static).collect(),
        name,
    )
}

fn state(program: &mut Vec<Op>, hidden: u32, name: &str) -> NodeId {
    input_leaf(
        program,
        DType::Float32,
        vec![Extent::Symbolic(0), Extent::Static(hidden)],
        name,
    )
}

fn norm(
    program: &mut Vec<Op>,
    input: NodeId,
    scale: NodeId,
    inverse: NodeId,
    epsilon: NodeId,
) -> io::Result<NodeId> {
    rmsnorm(program, input, scale, inverse, epsilon).map_err(io::Error::other)
}

pub fn projection(program: &mut Vec<Op>, input: NodeId, weight: NodeId) -> io::Result<NodeId> {
    let product = apply(
        program,
        ScalarOp::Multiply,
        &[(input, "si->sio"), (weight, "io->sio")],
    )?;
    sum(program, product, "sio->sio", "so->sio")
}

pub fn layer(shape: Shape) -> io::Result<Graph> {
    let mut program = Vec::new();
    let hidden = state(&mut program, shape.hidden, "hidden");
    let epsilon = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Symbolic(0)],
        "epsilon",
    );
    let inverse = scalar_constant(&mut program, 1.0 / shape.hidden as f32);
    let attention_scale = scalar_constant(&mut program, 1.0 / (shape.head_width as f32).sqrt());
    let input_norm = leaf(&mut program, "input_norm", &[shape.hidden]);
    let attention_norm = leaf(&mut program, "attention_norm", &[shape.hidden]);
    let ffn_norm = leaf(&mut program, "ffn_norm", &[shape.hidden]);
    let output_norm = leaf(&mut program, "output_norm", &[shape.hidden]);
    let query_weight = leaf(
        &mut program,
        "query",
        &[shape.hidden, shape.heads, shape.head_width],
    );
    let key_weight = leaf(
        &mut program,
        "key",
        &[shape.hidden, shape.heads, shape.head_width],
    );
    let value_weight = leaf(
        &mut program,
        "value",
        &[shape.hidden, shape.heads, shape.head_width],
    );
    let attention_weight = leaf(
        &mut program,
        "attention",
        &[shape.heads, shape.head_width, shape.hidden],
    );
    let gate_weight = leaf(&mut program, "gate", &[shape.hidden, shape.intermediate]);
    let up_weight = leaf(&mut program, "up", &[shape.hidden, shape.intermediate]);
    let down_weight = leaf(&mut program, "down", &[shape.intermediate, shape.hidden]);
    let cosine = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Symbolic(0), Extent::Static(shape.head_width / 2)],
        "cosine",
    );
    let sine = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Symbolic(0), Extent::Static(shape.head_width / 2)],
        "sine",
    );
    let normalized_input = norm(&mut program, hidden, input_norm, inverse, epsilon)?;
    let mut projected = Vec::new();
    for weight in [query_weight, key_weight, value_weight] {
        let product = apply(
            &mut program,
            ScalarOp::Multiply,
            &[(normalized_input, "si->shdi"), (weight, "ihd->shdi")],
        )?;
        projected.push(sum(&mut program, product, "shdi->shdi", "shd->shdi")?);
    }
    let pairing = RopePairing::SplitHalf {
        pairs: shape.head_width / 2,
    };
    let query_pair = fused_rope_pair(&mut program, projected[0], 'h', cosine, sine, pairing)
        .map_err(io::Error::other)?;
    let key_pair = fused_rope_pair(&mut program, projected[1], 'h', cosine, sine, pairing)
        .map_err(io::Error::other)?;
    let first_product = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(query_pair.0, "shi->sthi"), (key_pair.0, "thi->sthi")],
    )?;
    let first_score = sum(&mut program, first_product, "sthi->sthi", "sth->sthi")?;
    let second_product = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(query_pair.1, "shi->sthi"), (key_pair.1, "thi->sthi")],
    )?;
    let second_score = sum(&mut program, second_product, "sthi->sthi", "sth->sthi")?;
    let scores = apply(
        &mut program,
        ScalarOp::Add,
        &[(first_score, "sth->sth"), (second_score, "sth->sth")],
    )?;
    let scaled = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(scores, "sth->sth"), (attention_scale, "->sth")],
    )?;
    let (future, negative_infinity) = causal_mask(&mut program).map_err(io::Error::other)?;
    let masked = apply(
        &mut program,
        ScalarOp::Select,
        &[
            (future, "st->sth"),
            (negative_infinity, "->sth"),
            (scaled, "sth->sth"),
        ],
    )?;
    let maximum = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        masked,
        "sth->sth",
        "sh->sth",
    )
    .map_err(io::Error::other)?;
    let shifted = apply(
        &mut program,
        ScalarOp::Subtract,
        &[(masked, "sth->sth"), (maximum, "sh->sth")],
    )?;
    let exponentials = apply(
        &mut program,
        ScalarOp::Exponential,
        &[(shifted, "sth->sth")],
    )?;
    let denominator = sum(&mut program, exponentials, "sth->sth", "sh->sth")?;
    let reciprocal = apply(
        &mut program,
        ScalarOp::Reciprocal,
        &[(denominator, "sh->sh")],
    )?;
    let probabilities = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(exponentials, "sth->sth"), (reciprocal, "sh->sth")],
    )?;
    let attended_product = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(probabilities, "sth->sthd"), (projected[2], "thd->sthd")],
    )?;
    let attended = sum(&mut program, attended_product, "sthd->sthd", "shd->sthd")?;
    let output_product = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(attended, "shd->shdo"), (attention_weight, "hdo->shdo")],
    )?;
    let attention_output = sum(&mut program, output_product, "shdo->shdo", "so->shdo")?;
    let attention_branch = norm(
        &mut program,
        attention_output,
        attention_norm,
        inverse,
        epsilon,
    )?;
    let residual = apply(
        &mut program,
        ScalarOp::Add,
        &[(hidden, "sd->sd"), (attention_branch, "sd->sd")],
    )?;
    let normalized_ffn = norm(&mut program, residual, ffn_norm, inverse, epsilon)?;
    let gate = projection(&mut program, normalized_ffn, gate_weight)?;
    let up = projection(&mut program, normalized_ffn, up_weight)?;
    let negative = scalar_constant(&mut program, -1.0);
    let one = scalar_constant(&mut program, 1.0);
    let negated = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(gate, "sg->sg"), (negative, "->sg")],
    )?;
    let exponential = apply(&mut program, ScalarOp::Exponential, &[(negated, "sg->sg")])?;
    let added = apply(
        &mut program,
        ScalarOp::Add,
        &[(exponential, "sg->sg"), (one, "->sg")],
    )?;
    let sigmoid = apply(&mut program, ScalarOp::Reciprocal, &[(added, "sg->sg")])?;
    let silu = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(gate, "sg->sg"), (sigmoid, "sg->sg")],
    )?;
    let gated = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(silu, "sg->sg"), (up, "sg->sg")],
    )?;
    let ffn_output = projection(&mut program, gated, down_weight)?;
    let ffn_branch = norm(&mut program, ffn_output, output_norm, inverse, epsilon)?;
    let root = apply(
        &mut program,
        ScalarOp::Add,
        &[(residual, "sd->sd"), (ffn_branch, "sd->sd")],
    )?;
    Ok(Graph {
        program,
        root,
        taps: vec![
            ("pre_attention_norm", normalized_input),
            ("attention_output", attention_output),
            ("post_attention_norm", attention_branch),
            ("attention_residual", residual),
            ("pre_ffn_norm", normalized_ffn),
            ("ffn_output", ffn_output),
            ("post_ffn_norm", ffn_branch),
        ],
    })
}

pub fn normalization(hidden_size: u32) -> io::Result<Graph> {
    let mut program = Vec::new();
    let hidden = state(&mut program, hidden_size, "hidden");
    let scale = leaf(&mut program, "scale", &[hidden_size]);
    let epsilon = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Symbolic(0)],
        "epsilon",
    );
    let inverse = scalar_constant(&mut program, 1.0 / hidden_size as f32);
    let root = norm(&mut program, hidden, scale, inverse, epsilon)?;
    Ok(Graph {
        program,
        root,
        taps: Vec::new(),
    })
}

pub fn readout(shape: Shape) -> io::Result<Graph> {
    let mut program = Vec::new();
    let hidden = leaf(&mut program, "hidden", &[1, shape.hidden]);
    let weight = leaf(&mut program, "head", &[shape.hidden, shape.vocab]);
    let root = projection(&mut program, hidden, weight)?;
    Ok(Graph {
        program,
        root,
        taps: Vec::new(),
    })
}

pub fn contrast(vocab: u32) -> io::Result<Graph> {
    let mut program = Vec::new();
    let reference = leaf(&mut program, "reference", &[vocab]);
    let terminal = leaf(&mut program, "terminal", &[vocab]);
    let strength = leaf(&mut program, "strength", &[vocab]);
    let difference = apply(
        &mut program,
        ScalarOp::Subtract,
        &[(terminal, "d->d"), (reference, "d->d")],
    )?;
    let scaled = apply(
        &mut program,
        ScalarOp::Multiply,
        &[(difference, "d->d"), (strength, "d->d")],
    )?;
    let root = apply(
        &mut program,
        ScalarOp::Add,
        &[(terminal, "d->d"), (scaled, "d->d")],
    )?;
    Ok(Graph {
        program,
        root,
        taps: Vec::new(),
    })
}
