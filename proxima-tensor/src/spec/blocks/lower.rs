use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::{DType, NumericFormat};

use super::{Attention, Block, DenseFfn, Input, Network, Norm, RoutedFfn};
use crate::error::TensorError;
use crate::op::{
    ConstantLiteral, Extent, NodeId, Op, ReduceInit, SLIDING_CACHED_LEN_INPUT, SLIDING_KV_SYMBOL,
    ScalarOp,
};
use crate::spec::scheduled_cached::causal_mask_cached_windowed;
use crate::spec::{
    TOKEN_IDS, CachedEmitContext, CachedLayerRoots, ForwardProgram, LayerCacheRoots, MoeSite, MoeSites,
    append_cached_dense_ffn, append_cached_routed_ffn, append_embedding_scale,
    append_gqa_cached_attention, append_gqa_cached_routed_attention, causal_mask,
    causal_mask_windowed, count_constant, elementwise, embedding_lookup_as, gather_last_row,
    input_leaf, merge_masks, reduce, rmsnorm, scalar_constant, symbolic_leaf, with_pad_key_mask,
};

const BUILDER: &str = "Network::lower";

enum Ffn<'network> {
    Dense(&'network DenseFfn),
    Routed(&'network RoutedFfn),
}

struct Layer<'network> {
    attn_norm: &'network Norm,
    attention: &'network Attention,
    ffn_norm: &'network Norm,
    ffn: Ffn<'network>,
}

enum FfnLeaves {
    Dense {
        w_gate: NodeId,
        w_up: NodeId,
        w_down: NodeId,
    },
    Routed {
        gate_inp: NodeId,
        expert_w_gate: NodeId,
        expert_w_up: NodeId,
        expert_w_down: NodeId,
    },
}

fn shape_error(feature: &'static str) -> TensorError {
    TensorError::UnsupportedInBuilder {
        builder: BUILDER,
        feature,
    }
}

fn leaf_name<'map>(
    leaves: &'map BTreeMap<String, String>,
    key: &str,
) -> Result<&'map str, TensorError> {
    leaves
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| shape_error("a block that omits a leaf name this engine declares"))
}

fn norm_weight(norm: &Norm) -> Result<&str, TensorError> {
    norm.weight
        .as_deref()
        .ok_or_else(|| shape_error("a norm with no weight leaf"))
}

fn residual_parts(block: &Block) -> Result<(Option<ConstantLiteral>, [&Block; 2]), TensorError> {
    match block {
        Block::Residual { scale, body } => match body.blocks.as_slice() {
            [norm, mixer] => Ok((*scale, [norm, mixer])),
            _ => Err(shape_error("a residual that is not one norm and one mixer")),
        },
        _ => Err(shape_error("a body that is not residual sublayers")),
    }
}

fn layers(network: &Network) -> Result<(Vec<Layer<'_>>, Option<ConstantLiteral>), TensorError> {
    let mut found = Vec::new();
    let mut scale = None;
    for pair in network.body.blocks.chunks(2) {
        let [attention_block, ffn_block] = pair else {
            return Err(shape_error("an odd number of residual sublayers"));
        };
        let (attention_scale, [first, second]) = residual_parts(attention_block)?;
        let (_, [ffn_first, ffn_second]) = residual_parts(ffn_block)?;
        scale = attention_scale;
        let (Block::Norm(attn_norm), Block::Attention(attention)) = (first, second) else {
            return Err(shape_error("an attention sublayer that is not norm then attention"));
        };
        let Block::Norm(ffn_norm) = ffn_first else {
            return Err(shape_error("a feed-forward sublayer that does not start with a norm"));
        };
        let ffn = match ffn_second {
            Block::DenseFfn(dense) => Ffn::Dense(dense),
            Block::RoutedFfn(routed) => Ffn::Routed(routed),
            _ => return Err(shape_error("a feed-forward sublayer that is not a feed-forward")),
        };
        found.push(Layer {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
        });
    }
    Ok((found, scale))
}

impl Network {
    /// Lowers a network [`Network::from_bounded`] produced to the program the
    /// Bounded cached engine builds: the same nodes in the same order, with
    /// each layer's leaves declared in three passes (residual norms, body
    /// blocks in order, QK-norm weights) before its sublayers are emitted
    /// through the cached attention and feed-forward bodies.
    ///
    /// # Errors
    ///
    /// [`TensorError::UnsupportedInBuilder`] for a body that is not that
    /// shape, a leaf name a block omits, or a host element with no dtype;
    /// anything a node builder refuses.
    pub fn lower(&self, key_pad_id: Option<u32>) -> Result<ForwardProgram, TensorError> {
        let Input::Tokens {
            vocab,
            width: embedding,
            table,
            scale: embedding_scale,
            activation,
        } = &self.input
        else {
            return Err(shape_error("an input that is not token ids"));
        };
        let (vocab, embedding) = (*vocab, *embedding);
        let host = NumericFormat::host_element()
            .to_dtype()
            .ok_or_else(|| shape_error("a host element with no dtype"))?;
        let activation = activation.to_dtype().unwrap_or(host);
        let (layers, residual_scale) = layers(self)?;
        let Some(first_layer) = layers.first() else {
            return Err(shape_error("a network with no layers"));
        };
        let shape = first_layer.attention;
        let (query_heads, kv_heads, head_dim) = (shape.query_heads, shape.kv_heads, shape.head_dim);
        let layer_windows: Vec<Option<u32>> =
            layers.iter().map(|layer| layer.attention.window).collect();
        let windows: Vec<u32> = layer_windows
            .iter()
            .flatten()
            .copied()
            .filter(|width| *width > 0)
            .collect();
        let sliding_kv_ring = self.decode.sliding_kv_ring;
        if sliding_kv_ring && windows.iter().any(|width| *width != windows[0]) {
            return Err(shape_error(
                "sliding_kv_ring with differing window widths (one sliding slot)",
            ));
        }
        let group = query_heads / kv_heads;
        let pairs = head_dim / 2;
        let fused_qkv_reduce = shape.fused_qkv;
        let qk_norm = shape.qk_norm;
        let qkv_biases = shape.biases;

        let mut program: Vec<Op> = Vec::new();
        let ids = input_leaf(
            &mut program,
            TOKEN_IDS,
            alloc::vec![Extent::Symbolic(0)],
            "ids",
        );
        let table = input_leaf(
            &mut program,
            host,
            alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
            table,
        );
        let mut x = embedding_lookup_as(&mut program, table, ids, activation);
        if let Some(scale) = embedding_scale {
            x = append_embedding_scale(&mut program, x, embedding, *scale, activation)?;
        }

        let inv_dim = count_constant(&mut program, embedding);
        let eps = symbolic_leaf(&mut program, host, "eps");
        let ones = scalar_constant(&mut program, 1.0);
        let inv_sqrt_head_dim = scalar_constant(&mut program, shape.score_scale.multiplier());
        let inv_head_dim = qk_norm.then(|| count_constant(&mut program, head_dim));
        let residual_scale = residual_scale.map(|scale| scalar_constant(&mut program, scale));
        let cos_new = input_leaf(
            &mut program,
            host,
            alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
            "rope_cos",
        );
        let sin_new = input_leaf(
            &mut program,
            host,
            alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
            "rope_sin",
        );
        let group_ones = crate::op::append(
            &mut program,
            Op::Constant {
                dtype: host,
                shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
                value: 1.0,
            },
        );
        let (head_shape_ones, kv_head_shape_ones) = if fused_qkv_reduce {
            let head_shape_ones = crate::op::append(
                &mut program,
                Op::Constant {
                    dtype: host,
                    shape: alloc::vec![Extent::Static(query_heads), Extent::Static(head_dim)],
                    value: 1.0,
                },
            );
            let kv_head_shape_ones = crate::op::append(
                &mut program,
                Op::Constant {
                    dtype: host,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                    value: 1.0,
                },
            );
            (head_shape_ones, kv_head_shape_ones)
        } else {
            (ones, ones)
        };
        let (is_future, _neg_infinity) = causal_mask(&mut program)?;
        let is_future = with_pad_key_mask(&mut program, is_future, ids, key_pad_id)?;
        let cached_len = input_leaf(&mut program, host, Vec::new(), "cached_len");
        let cached_padding_mask = key_pad_id
            .map(|_| {
                causal_mask_cached_windowed(&mut program, cached_len, Extent::Symbolic(1), None)
            })
            .transpose()?;
        let ring_len = (sliding_kv_ring && !windows.is_empty())
            .then(|| input_leaf(&mut program, host, Vec::new(), SLIDING_CACHED_LEN_INPUT));
        let mut window_masks: Vec<(u32, NodeId, (NodeId, NodeId))> = Vec::new();
        for &width in &windows {
            if window_masks.iter().any(|(built, ..)| *built == width) {
                continue;
            }
            let (new_block, _) = causal_mask_windowed(&mut program, Some(width))?;
            let new_block = with_pad_key_mask(&mut program, new_block, ids, key_pad_id)?;
            let cached_block = match ring_len {
                Some(ring_len) => causal_mask_cached_windowed(
                    &mut program,
                    ring_len,
                    Extent::Symbolic(SLIDING_KV_SYMBOL),
                    Some(width),
                )?,
                None => causal_mask_cached_windowed(
                    &mut program,
                    cached_len,
                    Extent::Symbolic(1),
                    Some(width),
                )?,
            };
            window_masks.push((width, new_block, cached_block));
        }

        let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(layers.len());
        let mut layer_residuals: Vec<NodeId> = Vec::with_capacity(layers.len());
        let mut moe_sites: Vec<MoeSite> = Vec::new();

        for (index, layer) in layers.iter().enumerate() {
            let layer_index =
                u32::try_from(index).map_err(|_| shape_error("a layer index past u32"))?;
            let window = layer_windows[index].filter(|width| *width > 0);
            let masks =
                window.and_then(|width| window_masks.iter().find(|(built, ..)| *built == width));
            let layer_is_future = masks.map_or(is_future, |(_, new_block, _)| *new_block);
            let cached_window_mask = match (
                masks.map(|(_, _, cached_block)| *cached_block),
                cached_padding_mask,
            ) {
                (None, None) => None,
                (Some(pair), None) | (None, Some(pair)) => Some(pair),
                (Some((window, neg_infinity)), Some((padding, _))) => {
                    let both = merge_masks(&mut program, window, padding, "st->st")?;
                    Some((both, neg_infinity))
                }
            };
            let cache_bound = if ring_len.is_some() && window.is_some() {
                Extent::Symbolic(SLIDING_KV_SYMBOL)
            } else {
                Extent::Symbolic(1)
            };

            // pass A: the residual norms, attention's then the feed-forward's
            let attn_norm_weight = input_leaf(
                &mut program,
                host,
                alloc::vec![Extent::Static(embedding)],
                norm_weight(layer.attn_norm)?,
            );
            let ffn_norm_weight = input_leaf(
                &mut program,
                host,
                alloc::vec![Extent::Static(embedding)],
                norm_weight(layer.ffn_norm)?,
            );

            // pass B: the body blocks in order, attention then feed-forward
            let leaves = &layer.attention.leaves;
            let (wq, wk, wv) = if fused_qkv_reduce {
                let rows = (query_heads + 2 * kv_heads) * head_dim;
                let w_qkv = input_leaf(
                    &mut program,
                    host,
                    alloc::vec![Extent::Static(rows), Extent::Static(embedding)],
                    leaf_name(leaves, "attn_qkv")?,
                );
                (w_qkv, w_qkv, w_qkv)
            } else {
                let wq = input_leaf(
                    &mut program,
                    host,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(query_heads),
                        Extent::Static(head_dim)
                    ],
                    leaf_name(leaves, "attn_q")?,
                );
                let kv_shape = alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads),
                    Extent::Static(head_dim)
                ];
                let wk = input_leaf(
                    &mut program,
                    host,
                    kv_shape.clone(),
                    leaf_name(leaves, "attn_k")?,
                );
                let wv = input_leaf(&mut program, host, kv_shape, leaf_name(leaves, "attn_v")?);
                (wq, wk, wv)
            };
            let mut bias_leaf = |heads: u32, key: &str| -> Result<Option<NodeId>, TensorError> {
                if !qkv_biases {
                    return Ok(None);
                }
                Ok(Some(input_leaf(
                    &mut program,
                    host,
                    alloc::vec![Extent::Static(heads), Extent::Static(head_dim)],
                    leaf_name(leaves, key)?,
                )))
            };
            let q_bias = bias_leaf(query_heads, "attn_q_bias")?;
            let k_bias = bias_leaf(kv_heads, "attn_k_bias")?;
            let v_bias = bias_leaf(kv_heads, "attn_v_bias")?;
            let wo = input_leaf(
                &mut program,
                host,
                alloc::vec![
                    Extent::Static(kv_heads),
                    Extent::Static(group),
                    Extent::Static(head_dim),
                    Extent::Static(embedding)
                ],
                leaf_name(leaves, "attn_output")?,
            );
            let k_even_cache = input_leaf(
                &mut program,
                host,
                alloc::vec![cache_bound, Extent::Static(kv_heads), Extent::Static(pairs)],
                &alloc::format!("kv_cache.{layer_index}.k_even"),
            );
            let k_odd_cache = input_leaf(
                &mut program,
                host,
                alloc::vec![cache_bound, Extent::Static(kv_heads), Extent::Static(pairs)],
                &alloc::format!("kv_cache.{layer_index}.k_odd"),
            );
            let v_cache = input_leaf(
                &mut program,
                host,
                alloc::vec![
                    cache_bound,
                    Extent::Static(kv_heads),
                    Extent::Static(head_dim)
                ],
                &alloc::format!("kv_cache.{layer_index}.v"),
            );
            let ffn_leaves = declare_ffn_leaves(&mut program, host, embedding, &layer.ffn)?;

            // pass C: the per-head q and k norm weights
            let qk_norm_weights = match inv_head_dim {
                Some(inv_head_dim) => {
                    let q_norm_weight = input_leaf(
                        &mut program,
                        host,
                        alloc::vec![Extent::Static(head_dim)],
                        leaf_name(leaves, "attn_q_norm")?,
                    );
                    let k_norm_weight = input_leaf(
                        &mut program,
                        host,
                        alloc::vec![Extent::Static(head_dim)],
                        leaf_name(leaves, "attn_k_norm")?,
                    );
                    Some((q_norm_weight, k_norm_weight, inv_head_dim))
                }
                None => None,
            };

            let context = CachedEmitContext {
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos_new,
                sin_new,
                group_ones,
                is_future: layer_is_future,
                cached_window_mask,
                residual_scale,
                group,
                head_dim,
                fused_qkv_shape: (fused_qkv_reduce && matches!(layer.ffn, Ffn::Dense(_)))
                    .then_some((head_shape_ones, kv_head_shape_ones, query_heads)),
            };
            let caches = (k_even_cache, k_odd_cache, v_cache);
            let (next_x, layer_roots) = match (&layer.ffn, ffn_leaves) {
                (
                    Ffn::Dense(dense),
                    FfnLeaves::Dense {
                        w_gate,
                        w_up,
                        w_down,
                    },
                ) => {
                    let (residual1, roots) = append_gqa_cached_attention(
                        &mut program,
                        &context,
                        x,
                        attn_norm_weight,
                        (wq, wk, wv),
                        wo,
                        caches,
                        qk_norm_weights,
                        (q_bias, k_bias, v_bias),
                        shape.rope_pairing,
                    )?;
                    let next_x = append_cached_dense_ffn(
                        &mut program,
                        &context,
                        residual1,
                        ffn_norm_weight,
                        w_gate,
                        w_up,
                        w_down,
                        dense.paired_gate_up,
                    )?;
                    (next_x, roots)
                }
                (
                    Ffn::Routed(routed),
                    FfnLeaves::Routed {
                        gate_inp,
                        expert_w_gate,
                        expert_w_up,
                        expert_w_down,
                    },
                ) => {
                    let (residual1, roots) = append_gqa_cached_routed_attention(
                        &mut program,
                        &context,
                        x,
                        attn_norm_weight,
                        (wq, wk, wv),
                        wo,
                        caches,
                        qk_norm_weights,
                    )?;
                    let (next_x, site) = append_cached_routed_ffn(
                        &mut program,
                        &context,
                        layer_index,
                        residual1,
                        ffn_norm_weight,
                        gate_inp,
                        (expert_w_gate, expert_w_up, expert_w_down),
                        (routed.expert_count, routed.expert_used_count),
                    )?;
                    moe_sites.push(site);
                    (next_x, roots)
                }
                _ => return Err(shape_error("a feed-forward whose leaves do not match its kind")),
            };
            x = next_x;
            cache_roots.push(layer_roots);
            layer_residuals.push(next_x);
        }

        let output_norm_weight = input_leaf(
            &mut program,
            host,
            alloc::vec![Extent::Static(embedding)],
            norm_weight(&self.head.norm)?,
        );
        let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;
        let normed_last = gather_last_row(&mut program, normed_final, self.decode.last_row_only);
        let lm_head = input_leaf(
            &mut program,
            host,
            alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
            &self.head.output,
        );
        let logits_product = elementwise(
            &mut program,
            host,
            ScalarOp::Multiply,
            &[(normed_last, "sd->sdv"), (lm_head, "dv->sdv")],
        )?;
        let logits = reduce(
            &mut program,
            host,
            ScalarOp::Add,
            ReduceInit::Zero,
            logits_product,
            "sdv->sdv",
            "sv->sdv",
        )?;
        let logits = match self.head.logit_scale {
            Some(scale) => {
                let inverse = scalar_constant(&mut program, 1.0 / scale);
                elementwise(
                    &mut program,
                    host,
                    ScalarOp::Multiply,
                    &[(logits, "sv->sv"), (inverse, "->sv")],
                )?
            }
            None => logits,
        };

        Ok(ForwardProgram {
            program,
            logits,
            layer_roots: cache_roots
                .into_iter()
                .map(LayerCacheRoots::Attention)
                .collect(),
            moe_sites: MoeSites(moe_sites),
            layer_residuals,
            hidden: Some(normed_last),
            duplicate_head_roots: Vec::new(),
            layer_diagnostics: Vec::new(),
            readouts: None,
        })
    }
}

fn declare_ffn_leaves(
    program: &mut Vec<Op>,
    host: DType,
    embedding: u32,
    ffn: &Ffn<'_>,
) -> Result<FfnLeaves, TensorError> {
    match ffn {
        Ffn::Dense(dense) => {
            let feed_forward = dense.feed_forward;
            let leaves = &dense.leaves;
            let (w_gate, w_up) = if dense.paired_gate_up {
                let w_gate_up = input_leaf(
                    program,
                    host,
                    alloc::vec![
                        Extent::Static(2),
                        Extent::Static(feed_forward),
                        Extent::Static(embedding)
                    ],
                    leaf_name(leaves, "ffn_gate_up")?,
                );
                (w_gate_up, w_gate_up)
            } else {
                let matrix =
                    || alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)];
                let w_gate = input_leaf(program, host, matrix(), leaf_name(leaves, "ffn_gate")?);
                let w_up = input_leaf(program, host, matrix(), leaf_name(leaves, "ffn_up")?);
                (w_gate, w_up)
            };
            let w_down = input_leaf(
                program,
                host,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                leaf_name(leaves, "ffn_down")?,
            );
            Ok(FfnLeaves::Dense {
                w_gate,
                w_up,
                w_down,
            })
        }
        Ffn::Routed(routed) => {
            let (feed_forward, expert_count) = (routed.feed_forward, routed.expert_count);
            let leaves = &routed.leaves;
            let gate_inp = input_leaf(
                program,
                host,
                alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
                leaf_name(leaves, "ffn_gate_inp")?,
            );
            let slab = || {
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ]
            };
            let expert_w_gate =
                input_leaf(program, host, slab(), leaf_name(leaves, "ffn_gate_exps")?);
            let expert_w_up = input_leaf(program, host, slab(), leaf_name(leaves, "ffn_up_exps")?);
            let expert_w_down = input_leaf(
                program,
                host,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(feed_forward),
                    Extent::Static(embedding)
                ],
                leaf_name(leaves, "ffn_down_exps")?,
            );
            Ok(FfnLeaves::Routed {
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
            })
        }
    }
}
