//! The activation side of a Hadamard-folded weight (PrismML `prism.hadamard.*`): a folded weight
//! `W_f = W * D * H` reads `H * (D * x)`, with `D` a fixed +1/-1 sign vector over the input width
//! and `H` the normalized Sylvester-Walsh Hadamard matrix applied to every consecutive `block`
//! elements. `W_f` is the quantized tensor; this module builds the `H * (D * x)` that feeds it.
//!
//! Teaching pointer: everything here is [`elementwise`] and [`reduce`] over the same read-side
//! index decompositions `gated_attention.rs` uses to split a flat axis into heads. The Sylvester
//! matrix factors by bits, `H_block = H_n1 (x) H_n2` with `block = n1 * n2`, so one block rotates
//! as two small contractions instead of one `block x block` one: `n1 + n2` multiplies per element
//! instead of `block`, and no `[.., block, block]` product to hold. The result is rank four
//! `[lead, blocks, n1, n2]` in flat order; a consumer contracts the three trailing axes against
//! the weight's flat input axis read as `block * b + n2 * e + f`. `lead` is `"s"` for a sequence
//! axis and `""` for the squeezed single-position programs.

use super::*;

pub const HADAMARD_ROTATION_LEAF_PREFIX: &str = "prism.hadamard.rotation.";

#[must_use]
pub fn hadamard_rotation_leaf(size: u32) -> String {
    alloc::format!("{HADAMARD_ROTATION_LEAF_PREFIX}{size}")
}

#[must_use]
pub fn hadamard_signs_leaf(width: u32) -> String {
    alloc::format!("prism.hadamard.signs.{width}")
}

#[derive(Debug, Clone)]
pub struct HadamardFold {
    gdn_v_grouped: bool,
    embedding: u32,
    feed_forward: u32,
    kv_heads: u32,
    block: u32,
    outer: u32,
    inner: u32,
    outer_rotation: NodeId,
    inner_rotation: NodeId,
    signs: Vec<(u32, NodeId)>,
}

impl HadamardFold {
    /// Declares the rotation leaves and one sign leaf per distinct projection input width: the
    /// embedding width, the feed-forward width, and the `extra_widths` of the recurrence and
    /// attention outputs.
    pub fn declare(
        program: &mut Vec<Op>,
        block: u32,
        embedding: u32,
        feed_forward: u32,
        kv_heads: u32,
        gdn_v_grouped: bool,
        extra_widths: &[u32],
    ) -> Self {
        let outer = 1u32 << (block.trailing_zeros() / 2);
        let inner = block / outer;
        let widths = [&[embedding, feed_forward][..], extra_widths].concat();
        let rotation_leaf = |program: &mut Vec<Op>, size: u32| {
            input_leaf(
                program,
                DType::FLOAT32,
                alloc::vec![Extent::Static(size), Extent::Static(size)],
                &hadamard_rotation_leaf(size),
            )
        };
        let outer_rotation = rotation_leaf(program, outer);
        let inner_rotation = match inner == outer {
            true => outer_rotation,
            false => rotation_leaf(program, inner),
        };
        let mut signs: Vec<(u32, NodeId)> = Vec::new();
        for &width in &widths {
            if signs.iter().all(|(existing, _)| *existing != width) {
                let leaf = input_leaf(
                    program,
                    DType::FLOAT32,
                    alloc::vec![Extent::Static(width / block), Extent::Static(block)],
                    &hadamard_signs_leaf(width),
                );
                signs.push((width, leaf));
            }
        }
        Self {
            gdn_v_grouped,
            embedding,
            feed_forward,
            kv_heads,
            block,
            outer,
            inner,
            outer_rotation,
            inner_rotation,
            signs,
        }
    }

    #[must_use]
    pub const fn embedding(&self) -> u32 {
        self.embedding
    }

    #[must_use]
    pub const fn feed_forward(&self) -> u32 {
        self.feed_forward
    }

    #[must_use]
    pub const fn gdn_v_grouped(&self) -> bool {
        self.gdn_v_grouped
    }

    #[must_use]
    pub const fn kv_heads(&self) -> u32 {
        self.kv_heads
    }

    #[must_use]
    pub const fn block(&self) -> u32 {
        self.block
    }

    fn signs_for(&self, width: u32) -> Result<NodeId, TensorError> {
        self.signs
            .iter()
            .find(|(candidate, _)| *candidate == width)
            .map(|(_, node)| *node)
            .ok_or(TensorError::NotLowerable {
                node: NodeId(0),
                reason: "no hadamard sign leaf was declared for this input width",
            })
    }

    /// `H * (D * x)` for a flat `[lead, width]` activation, as `[lead, width / block, n1, n2]`.
    pub fn rotate_flat(
        &self,
        program: &mut Vec<Op>,
        x: NodeId,
        width: u32,
        lead: &str,
    ) -> Result<NodeId, TensorError> {
        let signs = self.signs_for(width)?;
        let donor = ones(program, &[width / self.block, self.outer, self.inner]);
        let split = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (
                    x,
                    leading(
                        lead,
                        &alloc::format!("{}*b+{}*a+c", self.block, self.inner),
                        "bac",
                    )
                    .as_str(),
                ),
                (donor, alloc::format!("bac->{lead}bac").as_str()),
            ],
        )?;
        let signed = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (split, alloc::format!("{lead}bac->{lead}bac").as_str()),
                (
                    signs,
                    alloc::format!("b,{}*a+c->{lead}bac", self.inner).as_str(),
                ),
            ],
        )?;
        let first = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (signed, alloc::format!("{lead}bac->{lead}beac").as_str()),
                (self.outer_rotation, alloc::format!("ae->{lead}beac").as_str()),
            ],
        )?;
        let first = reduce(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            first,
            alloc::format!("{lead}beac->{lead}beac").as_str(),
            alloc::format!("{lead}bec->{lead}beac").as_str(),
        )?;
        self.contract_inner(program, first, lead)
    }

    /// `H * (D * x)` for an activation already split into `[lead, heads, head_dim]` heads whose
    /// flat order is `head * head_dim + dim`; a block spans `block / head_dim` consecutive heads
    /// and `head_dim` must be a multiple of the inner factor.
    pub fn rotate_heads(
        &self,
        program: &mut Vec<Op>,
        x: NodeId,
        heads: u32,
        head_dim: u32,
        lead: &str,
    ) -> Result<NodeId, TensorError> {
        if head_dim % self.inner != 0 || self.block % head_dim != 0 {
            return Err(TensorError::NotLowerable {
                node: NodeId(0),
                reason: "a hadamard block must hold whole heads, each a multiple of the inner factor",
            });
        }
        let heads_per_block = self.block / head_dim;
        let rows_per_head = head_dim / self.inner;
        let signs = self.signs_for(heads * head_dim)?;
        let donor = ones(
            program,
            &[heads / heads_per_block, heads_per_block, rows_per_head, self.inner],
        );
        let split = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (
                    x,
                    leading(
                        lead,
                        &alloc::format!("{heads_per_block}*b+q,{}*r+c", self.inner),
                        "bqrc",
                    )
                    .as_str(),
                ),
                (donor, alloc::format!("bqrc->{lead}bqrc").as_str()),
            ],
        )?;
        let signed = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (split, alloc::format!("{lead}bqrc->{lead}bqrc").as_str()),
                (
                    signs,
                    alloc::format!("b,{head_dim}*q+{}*r+c->{lead}bqrc", self.inner).as_str(),
                ),
            ],
        )?;
        let first = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (signed, alloc::format!("{lead}bqrc->{lead}beqrc").as_str()),
                (
                    self.outer_rotation,
                    alloc::format!("{rows_per_head}*q+r,e->{lead}beqrc").as_str(),
                ),
            ],
        )?;
        let first = reduce(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            first,
            alloc::format!("{lead}beqrc->{lead}beqrc").as_str(),
            alloc::format!("{lead}bec->{lead}beqrc").as_str(),
        )?;
        self.contract_inner(program, first, lead)
    }

    /// The second Kronecker factor: contracts the trailing `c` axis of `[lead, b, e, c]` with
    /// the inner rotation into `[lead, b, e, f]`.
    fn contract_inner(
        &self,
        program: &mut Vec<Op>,
        first: NodeId,
        lead: &str,
    ) -> Result<NodeId, TensorError> {
        let second = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (first, alloc::format!("{lead}bec->{lead}befc").as_str()),
                (self.inner_rotation, alloc::format!("cf->{lead}befc").as_str()),
            ],
        )?;
        reduce(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            second,
            alloc::format!("{lead}befc->{lead}befc").as_str(),
            alloc::format!("{lead}bef->{lead}befc").as_str(),
        )
    }

    /// A latent embedding row `z` of `[s, width]` back in the primal basis, `D * (H * z)`: the
    /// rotation comes first and the signs after, the reverse of a folded weight's input.
    /// A flat result needs the rotated axes merged again, which has no affine read map, so a
    /// one-hot of the flat index (`Equal` over `Iota`s) merges them.
    pub fn restore_flat(
        &self,
        program: &mut Vec<Op>,
        z: NodeId,
        width: u32,
    ) -> Result<NodeId, TensorError> {
        let blocks = width / self.block;
        let signs = self.signs_for(width)?;
        let donor = ones(program, &[blocks, self.outer, self.inner]);
        let split = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (
                    z,
                    alloc::format!("s,{}*b+{}*a+c->sbac", self.block, self.inner).as_str(),
                ),
                (donor, "bac->sbac"),
            ],
        )?;
        let first = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[(split, "sbac->sbeac"), (self.outer_rotation, "ae->sbeac")],
        )?;
        let first = reduce(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            first,
            "sbeac->sbeac",
            "sbec->sbeac",
        )?;
        let rotated = self.contract_inner(program, first, "s")?;
        let signed = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (rotated, "sbef->sbef"),
                (signs, alloc::format!("b,{}*e+f->sbef", self.inner).as_str()),
            ],
        )?;
        let iota = |program: &mut Vec<Op>, extent: u32| {
            op::append(
                program,
                Op::Iota {
                    dtype: DType::FLOAT32,
                    extent: Extent::Static(extent),
                },
            )
        };
        let flat_index = iota(program, width);
        let block_index = iota(program, blocks);
        let outer_index = iota(program, self.outer);
        let inner_index = iota(program, self.inner);
        let block_stride = scalar_constant(program, self.block as f32);
        let inner_stride = scalar_constant(program, self.inner as f32);
        let block_start = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[(block_index, "b->b"), (block_stride, "->b")],
        )?;
        let outer_start = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[(outer_index, "e->e"), (inner_stride, "->e")],
        )?;
        let in_block = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            &[(outer_start, "e->ef"), (inner_index, "f->ef")],
        )?;
        let position = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            &[(block_start, "b->bef"), (in_block, "ef->bef")],
        )?;
        let selector = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Equal,
            &[(flat_index, "k->kbef"), (position, "bef->kbef")],
        )?;
        let product = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[(signed, "sbef->skbef"), (selector, "kbef->skbef")],
        )?;
        reduce(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            product,
            "skbef->skbef",
            "sk->skbef",
        )
    }

    /// Contracts a rotated activation `[lead, blocks, n1, n2]` against a weight whose flat input
    /// axis is first and whose remaining axes are `free` letters (`"hc"` for `[in, h, c]`).
    pub fn project_folded(
        &self,
        program: &mut Vec<Op>,
        rotated: NodeId,
        weight: NodeId,
        lead: &str,
        free: &str,
    ) -> Result<NodeId, TensorError> {
        let free_axes = free
            .chars()
            .map(String::from)
            .collect::<Vec<String>>()
            .join(",");
        let product = elementwise(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            &[
                (rotated, alloc::format!("{lead}bef->{lead}{free}bef").as_str()),
                (
                    weight,
                    alloc::format!("{}*b+{}*e+f,{free_axes}->{lead}{free}bef", self.block, self.inner)
                        .as_str(),
                ),
            ],
        )?;
        reduce(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            product,
            alloc::format!("{lead}{free}bef->{lead}{free}bef").as_str(),
            alloc::format!("{lead}{free}->{lead}{free}bef").as_str(),
        )
    }
}

fn leading(lead: &str, read: &str, iteration: &str) -> String {
    match lead.is_empty() {
        true => alloc::format!("{read}->{iteration}"),
        false => alloc::format!("{lead},{read}->{lead}{iteration}"),
    }
}

/// Regroups an activation whose operand axes are named by `x_axes` (letters `u`, `g`, `d`, and
/// `s` when `lead` is `"s"`) into `[lead, kv_heads * group, d]`. Head `group * u + g` is the
/// grouped order a folded weight may have been built over, `kv_heads * g + u` the tiled order
/// llama.cpp computes in. A merge of two axes has no affine read map, so a one-hot selector built
/// from `Iota`s does it.
pub fn merge_head_axes(
    program: &mut Vec<Op>,
    x: NodeId,
    x_axes: &str,
    lead: &str,
    kv_heads: u32,
    group: u32,
    grouped: bool,
) -> Result<NodeId, TensorError> {
    let iota = |program: &mut Vec<Op>, extent: u32| {
        op::append(
            program,
            Op::Iota {
                dtype: DType::FLOAT32,
                extent: Extent::Static(extent),
            },
        )
    };
    let head_index = iota(program, kv_heads * group);
    let kv_index = iota(program, kv_heads);
    let group_index = iota(program, group);
    let flat_head = match grouped {
        true => {
            let stride = scalar_constant(program, group as f32);
            let start = elementwise(
                program,
                DType::FLOAT32,
                ScalarOp::Multiply,
                &[(kv_index, "u->u"), (stride, "->u")],
            )?;
            elementwise(
                program,
                DType::FLOAT32,
                ScalarOp::Add,
                &[(start, "u->ug"), (group_index, "g->ug")],
            )?
        }
        false => {
            let stride = scalar_constant(program, kv_heads as f32);
            let start = elementwise(
                program,
                DType::FLOAT32,
                ScalarOp::Multiply,
                &[(group_index, "g->g"), (stride, "->g")],
            )?;
            elementwise(
                program,
                DType::FLOAT32,
                ScalarOp::Add,
                &[(start, "g->ug"), (kv_index, "u->ug")],
            )?
        }
    };
    let selector = elementwise(
        program,
        DType::FLOAT32,
        ScalarOp::Equal,
        &[(head_index, "h->hug"), (flat_head, "ug->hug")],
    )?;
    let product = elementwise(
        program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[
            (x, alloc::format!("{x_axes}->{lead}hugd").as_str()),
            (selector, alloc::format!("hug->{lead}hugd").as_str()),
        ],
    )?;
    reduce(
        program,
        DType::FLOAT32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        alloc::format!("{lead}hugd->{lead}hugd").as_str(),
        alloc::format!("{lead}hd->{lead}hugd").as_str(),
    )
}

/// A flat `[lead, outer * inner]` projection output read as `[lead, outer, inner]`: the
/// read-side split every head layout uses, with a ones donor to bind the two extents.
pub fn split_projection(
    program: &mut Vec<Op>,
    flat: NodeId,
    lead: &str,
    outer: u32,
    inner: u32,
) -> Result<NodeId, TensorError> {
    let donor = ones(program, &[outer, inner]);
    elementwise(
        program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[
            (
                flat,
                leading(lead, &alloc::format!("{inner}*h+c"), "hc").as_str(),
            ),
            (donor, alloc::format!("hc->{lead}hc").as_str()),
        ],
    )
}

fn ones(program: &mut Vec<Op>, shape: &[u32]) -> NodeId {
    op::append(
        program,
        Op::Constant {
            dtype: DType::FLOAT32,
            shape: shape.iter().map(|extent| Extent::Static(*extent)).collect(),
            value: 1.0,
        },
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const BLOCK: u32 = 8;
    const OUTER: usize = 2;
    const INNER: usize = 4;
    const SEQUENCE: u32 = 2;

    fn sylvester(size: usize, row: usize, column: usize) -> f64 {
        let sign = if (row & column).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        sign / (size as f64).sqrt()
    }

    fn rotation_data(size: usize) -> Vec<f32> {
        (0..size)
            .flat_map(|row| (0..size).map(move |column| sylvester(size, row, column) as f32))
            .collect()
    }

    fn signs_data(width: usize) -> Vec<f32> {
        (0..width)
            .map(|index| if (index * 7 + 3) % 5 < 2 { -1.0 } else { 1.0 })
            .collect()
    }

    fn activation(count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 13 + 5) % 17) as f32 * 0.25 - 2.0)
            .collect()
    }

    fn reference(x: &[f32], signs: &[f32], width: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for row in x.chunks_exact(width) {
            for block in row
                .chunks_exact(BLOCK as usize)
                .zip(signs.chunks_exact(BLOCK as usize))
            {
                for column in 0..BLOCK as usize {
                    out.push(
                        (0..BLOCK as usize)
                            .map(|index| {
                                sylvester(BLOCK as usize, index, column)
                                    * f64::from(block.0[index])
                                    * f64::from(block.1[index])
                            })
                            .sum(),
                    );
                }
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (f64::from(*got) - want).abs() < 1e-5,
                "element {index}: got {got}, want {want}"
            );
        }
    }

    fn evaluate(program: &[Op], leading: &[&[f32]], width: usize, root: NodeId) -> Vec<f32> {
        let outer = rotation_data(OUTER);
        let inner = rotation_data(INNER);
        let signs = signs_data(width);
        let mut blocks: Vec<&[f32]> = leading.to_vec();
        blocks.extend([outer.as_slice(), inner.as_slice(), signs.as_slice()]);
        crate::cpu::evaluate(program, &[], &blocks, &[root])
            .expect("evaluates")
            .root()
            .to_vec()
    }

    #[test]
    fn the_block_factors_into_outer_and_inner_kronecker_sizes() {
        let mut program = Vec::new();

        let fold = HadamardFold::declare(&mut program, 1024, 2048, 2048, 1, true, &[]);

        assert_eq!((fold.outer, fold.inner), (32, 32));
        let odd = HadamardFold::declare(&mut program, BLOCK, 16, 16, 1, true, &[]);
        assert_eq!((odd.outer as usize, odd.inner as usize), (OUTER, INNER));
    }

    #[test]
    fn flat_rotation_matches_the_block_sylvester_reference() {
        let width = 24u32;
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![Extent::Static(SEQUENCE), Extent::Static(width)],
            "x",
        );
        let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
        let root = fold.rotate_flat(&mut program, x, width, "s").expect("lowers");
        let x_data = activation((SEQUENCE * width) as usize);

        let rotated = evaluate(&program, &[&x_data], width as usize, root);

        assert_close(
            &rotated,
            &reference(&x_data, &signs_data(width as usize), width as usize),
        );
    }

    #[test]
    fn head_rotation_matches_the_flat_reference_when_blocks_span_heads() {
        let (heads, head_dim) = (6u32, 4u32);
        let width = heads * head_dim;
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![
                Extent::Static(SEQUENCE),
                Extent::Static(heads),
                Extent::Static(head_dim)
            ],
            "x",
        );
        let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
        let root = fold
            .rotate_heads(&mut program, x, heads, head_dim, "s")
            .expect("lowers");
        let x_data = activation((SEQUENCE * width) as usize);

        let rotated = evaluate(&program, &[&x_data], width as usize, root);

        assert_close(
            &rotated,
            &reference(&x_data, &signs_data(width as usize), width as usize),
        );
    }

    #[test]
    fn restore_flat_applies_the_rotation_first_and_the_signs_after() {
        let width = 24u32;
        let mut program = Vec::new();
        let z = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![Extent::Static(SEQUENCE), Extent::Static(width)],
            "z",
        );
        let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
        let restored = fold.restore_flat(&mut program, z, width).expect("lowers");
        let z_data = activation((SEQUENCE * width) as usize);
        let signs = signs_data(width as usize);
        let ones_signs = vec![1.0f32; width as usize];
        let expected: Vec<f64> = reference(&z_data, &ones_signs, width as usize)
            .iter()
            .enumerate()
            .map(|(index, value)| value * f64::from(signs[index % width as usize]))
            .collect();

        let restored = evaluate(&program, &[&z_data], width as usize, restored);

        assert_close(&restored, &expected);
    }

    #[test]
    fn project_folded_contracts_the_rotated_blocks_against_the_flat_weight_axis() {
        let (width, outputs) = (24u32, 3u32);
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![Extent::Static(SEQUENCE), Extent::Static(width)],
            "x",
        );
        let weight = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![Extent::Static(width), Extent::Static(outputs)],
            "weight",
        );
        let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
        let rotated = fold.rotate_flat(&mut program, x, width, "s").expect("lowers");
        let root = fold
            .project_folded(&mut program, rotated, weight, "s", "o")
            .expect("lowers");
        let x_data = activation((SEQUENCE * width) as usize);
        let weight_data: Vec<f32> = (0..(width * outputs) as usize)
            .map(|index| ((index * 3 + 1) % 7) as f32 * 0.5 - 1.5)
            .collect();
        let expected: Vec<f64> = reference(&x_data, &signs_data(width as usize), width as usize)
            .chunks_exact(width as usize)
            .flat_map(|row| {
                (0..outputs as usize)
                    .map(|output| {
                        row.iter()
                            .enumerate()
                            .map(|(index, value)| {
                                value * f64::from(weight_data[index * outputs as usize + output])
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();

        let outer = rotation_data(OUTER);
        let inner = rotation_data(INNER);
        let signs = signs_data(width as usize);
        let blocks: [&[f32]; 5] = [&x_data, &weight_data, &outer, &inner, &signs];
        let evaluated = crate::cpu::evaluate(&program, &[], &blocks, &[root]).expect("evaluates");

        assert_close(evaluated.root(), &expected);
    }

    #[test]
    fn merge_head_axes_orders_heads_as_group_times_kv_plus_group_index() {
        let (kv_heads, group, dim) = (3u32, 2u32, 4u32);
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![
                Extent::Static(SEQUENCE),
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(dim)
            ],
            "x",
        );
        let merged =
            merge_head_axes(&mut program, x, "sugd", "s", kv_heads, group, true).expect("lowers");
        let x_data = activation((SEQUENCE * kv_heads * group * dim) as usize);

        let blocks: [&[f32]; 1] = [&x_data];
        let evaluated = crate::cpu::evaluate(&program, &[], &blocks, &[merged]).expect("evaluates");

        assert_eq!(evaluated.root(), x_data.as_slice());
    }

    #[test]
    fn merge_head_axes_tiled_order_puts_the_group_index_outermost() {
        let (kv_heads, group, dim) = (3u32, 2u32, 4u32);
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(dim)
            ],
            "x",
        );
        let merged =
            merge_head_axes(&mut program, x, "ugd", "", kv_heads, group, false).expect("lowers");
        let x_data = activation((kv_heads * group * dim) as usize);
        let expected: Vec<f32> = (0..group as usize)
            .flat_map(|inner| (0..kv_heads as usize).map(move |kv| (kv, inner)))
            .flat_map(|(kv, inner)| {
                let start = (kv * group as usize + inner) * dim as usize;
                x_data[start..start + dim as usize].to_vec()
            })
            .collect();

        let blocks: [&[f32]; 1] = [&x_data];
        let evaluated = crate::cpu::evaluate(&program, &[], &blocks, &[merged]).expect("evaluates");

        assert_eq!(evaluated.root(), expected.as_slice());
    }

    #[test]
    fn merge_head_axes_transposed_input_lands_in_the_same_head_order() {
        let (kv_heads, group, dim) = (3u32, 2u32, 4u32);
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![
                Extent::Static(group),
                Extent::Static(kv_heads),
                Extent::Static(dim)
            ],
            "x",
        );
        let merged =
            merge_head_axes(&mut program, x, "gud", "", kv_heads, group, true).expect("lowers");
        let x_data = activation((group * kv_heads * dim) as usize);
        let expected: Vec<f32> = (0..kv_heads as usize)
            .flat_map(|kv| (0..group as usize).map(move |inner| (kv, inner)))
            .flat_map(|(kv, inner)| {
                let start = (inner * kv_heads as usize + kv) * dim as usize;
                x_data[start..start + dim as usize].to_vec()
            })
            .collect();

        let blocks: [&[f32]; 1] = [&x_data];
        let evaluated = crate::cpu::evaluate(&program, &[], &blocks, &[merged]).expect("evaluates");

        assert_eq!(evaluated.root(), expected.as_slice());
    }
}
