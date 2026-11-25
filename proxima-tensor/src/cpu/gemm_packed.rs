use super::*;

/// Offset of every reduction coordinate from an operand's own origin, in
/// row-major flat order over `reduction_dims`: the table [`PackedDotLayout`]
/// gathers through, built once per bound op (`reduction_len` entries).
pub(super) fn reduction_offsets(
    resolved: &BoundOp,
    reduction_dims: &[u16],
    view: &bind::Layout,
) -> Vec<i64> {
    reduction_dims.iter().fold(vec![0i64], |offsets, &dim| {
        let extent = resolved.extents[dim as usize] as i64;
        let stride = view.stride(dim);
        offsets
            .iter()
            .flat_map(|&origin| (0..extent).map(move |step| origin + step * stride))
            .collect()
    })
}

/// What [`PackedDotLayout::new`] reads: the tile plan from
/// [`packed_dot_tile_plan`] plus the reduce's axis split and output layout.
/// Bundled for the argument-count lint, like [`ConvGemmContext`].
pub(super) struct PackedDotShape<'a> {
    pub(super) resolved: &'a BoundOp,
    pub(super) plan: &'a NeonTilePlan,
    pub(super) reduction_strides: &'a [i64],
    pub(super) reduction_dims: &'a [u16],
    pub(super) leading_axis: u16,
    pub(super) last_output_dim: Option<u16>,
    pub(super) out_layout: &'a bind::Layout,
    pub(super) width: usize,
}

/// Strides, origins and panel sizes for running the dot tile over a reduce
/// whose operand reduction does not coalesce to stride 1: the non-flat
/// operand's reduction slice is gathered (BLIS-style packing) into a
/// contiguous panel that the unchanged [`gemm_tile_neon`] reads at stride
/// `reduction_len`. A packed column operand is packed once per column panel
/// ([`neon_column_panel_cols`] columns, inside
/// [`NEON_COLUMN_PANEL_BUDGET_BYTES`] of L2) and reused by every row strip; a
/// packed row operand once per [`TILE_ROWS`]-row strip. Scratch is sized from
/// those block constants by [`Self::scratch_a_len`]/[`Self::scratch_b_len`]
/// and allocated once per call by the caller.
///
/// Element-agnostic by construction: this type holds only offsets and
/// extents, and the pack/write helpers are generic over `Element: Copy`. The
/// kernel call, where the evaluator's element type is concrete, stays at the
/// caller (`run_reduce_planned`), beside the tile call it mirrors.
pub(super) struct PackedDotLayout {
    pub(super) reduction_len: usize,
    pub(super) pack_a: bool,
    pub(super) pack_b: bool,
    pub(super) offsets_a: Vec<i64>,
    pub(super) offsets_b: Vec<i64>,
    pub(super) base_a: i64,
    pub(super) base_b: i64,
    pub(super) row_stride: i64,
    pub(super) column_stride: i64,
    pub(super) out_base: i64,
    pub(super) out_row_stride: i64,
    pub(super) out_column_stride: i64,
    pub(super) rows_total: usize,
    pub(super) tiled_cols: usize,
    pub(super) panel_cols: usize,
}

impl PackedDotLayout {
    pub(super) fn new(shape: &PackedDotShape) -> Self {
        let resolved = shape.resolved;
        let plan = shape.plan;
        let view_a = &resolved.operands()[plan.index_a].1;
        let view_b = &resolved.operands()[plan.index_b].1;
        let origin = vec![0u64; resolved.extents.len()];
        let reduction_len: usize = shape
            .reduction_dims
            .iter()
            .map(|&dim| resolved.extents[dim as usize] as usize)
            .product();
        let tiled_cols = shape.width - shape.width % TILE_COLS;
        Self {
            reduction_len,
            pack_a: shape.reduction_strides[plan.index_a] != 1,
            pack_b: shape.reduction_strides[plan.index_b] != 1,
            offsets_a: reduction_offsets(resolved, shape.reduction_dims, view_a),
            offsets_b: reduction_offsets(resolved, shape.reduction_dims, view_b),
            base_a: view_a.offset_of(&origin),
            base_b: view_b.offset_of(&origin),
            row_stride: plan.row_stride_a as i64,
            column_stride: plan.col_stride_b as i64,
            out_base: shape.out_layout.offset_of(&origin),
            out_row_stride: shape.out_layout.stride(shape.leading_axis),
            out_column_stride: shape
                .last_output_dim
                .map_or(0, |dim| shape.out_layout.stride(dim)),
            rows_total: resolved.extents[shape.leading_axis as usize] as usize,
            tiled_cols,
            panel_cols: neon_column_panel_cols(reduction_len as u64, tiled_cols),
        }
    }

    pub(super) fn scratch_a_len(&self) -> usize {
        if self.pack_a {
            TILE_ROWS * self.reduction_len
        } else {
            0
        }
    }

    pub(super) fn scratch_b_len(&self) -> usize {
        if self.pack_b && self.tiled_cols > 0 {
            self.panel_cols * self.reduction_len
        } else {
            0
        }
    }

    /// Packs columns `panel_start..panel_end` of the column operand into
    /// `scratch`, one `reduction_len`-element slot per column.
    pub(super) fn pack_columns<Element: Copy>(
        &self,
        source: &[Element],
        scratch: &mut [Element],
        panel_start: usize,
        panel_end: usize,
    ) {
        let origins =
            (panel_start..panel_end).map(|column| self.base_b + column as i64 * self.column_stride);
        gather(source, &self.offsets_b, origins, scratch);
    }

    /// Packs `rows` rows starting at `row_start` of the row operand into
    /// `scratch`, one `reduction_len`-element slot per row.
    pub(super) fn pack_rows<Element: Copy>(
        &self,
        source: &[Element],
        scratch: &mut [Element],
        row_start: usize,
        rows: usize,
    ) {
        let origins =
            (row_start..row_start + rows).map(|row| self.base_a + row as i64 * self.row_stride);
        gather(source, &self.offsets_a, origins, scratch);
    }

    /// Writes one kernel tile into `output` at rows `row_start..` and
    /// columns `column..`.
    pub(super) fn write_tile<const ROWS: usize, Element: Copy>(
        &self,
        tile: &[[Element; TILE_COLS]; ROWS],
        output: &mut [Element],
        row_start: usize,
        column: usize,
    ) {
        for (row, tile_row) in tile.iter().enumerate() {
            let row_base = self.out_base + (row_start + row) as i64 * self.out_row_stride;
            for (lane, &value) in tile_row.iter().enumerate() {
                let position = row_base + (column + lane) as i64 * self.out_column_stride;
                output[position as usize] = value;
            }
        }
    }

    /// Output position of `(row, column)`, for the per-element column tail.
    pub(super) fn output_position(&self, row: usize, column: usize) -> usize {
        (self.out_base + row as i64 * self.out_row_stride + column as i64 * self.out_column_stride)
            as usize
    }
}

fn gather<Element: Copy>(
    source: &[Element],
    offsets: &[i64],
    origins: impl Iterator<Item = i64>,
    destination: &mut [Element],
) {
    if offsets.is_empty() {
        return;
    }
    for (slot, origin) in destination.chunks_exact_mut(offsets.len()).zip(origins) {
        for (value, &offset) in slot.iter_mut().zip(offsets) {
            *value = source[(origin + offset) as usize];
        }
    }
}
