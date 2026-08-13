# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_compressor (placeholder)."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    BLOCK_SIZE,
    CSA_CANDIDATES_PER_LEAF,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    CSA_PAIR_WIDTH,
    CSA_TOPK,
    CSA_MAX_QUERIES,
    CSA_TOPK_INVALID_TASK_SLOT,
    MAX_CONTEXT_TOKENS,
)
from decode_indexer_topk import active_score_topk_forest
from decode_metadata import (
    PHASE_D_LEAF_BEGIN,
    PHASE_D_LEAF_FIELDS,
    PHASE_D_LEAF_QUERY,
    PHASE_D_LEAF_VALID,
    PHASE_D_PAIR_FIELDS,
    PHASE_D_ROOT_FIELDS,
    PHASE_D_SINGLETON_FIELDS,
    PHASE_D_UPPER_FIELDS,
)

# Dynamic shape variables. S stays static: the score/topk scopes divide by it.
B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")  # T = B * S
PAGE_DYN = pl.dynamic("PAGE_DYN")
REQUEST_OFFSET_DYN = pl.dynamic("REQUEST_OFFSET_DYN")
LEAF_DYN = pl.dynamic("LEAF_DYN")
PAIR_GROUP_DYN = pl.dynamic("PAIR_GROUP_DYN")
SINGLETON_DYN = pl.dynamic("SINGLETON_DYN")
UPPER_MERGE_DYN = pl.dynamic("UPPER_MERGE_DYN")
ARENA_DYN = pl.dynamic("ARENA_DYN")
IDX_ROW_DYN = pl.dynamic("IDX_ROW_DYN")

# model config
B = DECODE_LOCAL_REQUESTS
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale

# kernel-local
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")

# tiling
Q_TILE = 256
# Q_OUT_TILE is the per-task N granularity (sets idx_qr_proj task count); MM_N_TILE
# is the Mat-safe cube N-tile. Q_OUT_TILE fans Q_OUT_TILE // MM_N_TILE cube ops per
# task so task count halves without growing the [Q_TILE, MM_N_TILE] L1 wq load.
Q_OUT_TILE = 1024
T_PAD = ((T + 16 - 1) // 16) * 16  # static upper bound on the token axis
# Matmul M at the 16-row cube floor: a tile taller than the dynamic source is not expressible.
MM_ROW_TILE = 16
# INT32 Acc is MM_ROW_TILE * MM_N_TILE * 4B and must stay under the 128KiB L0C wall.
MM_N_TILE = min(512, (128 * 1024) // (MM_ROW_TILE * 4))
QR_OT_COUNT = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE  # qr_proj N-tasks per row block
assert Q_OUT_TILE % MM_N_TILE == 0
# Dequant token tile: a whole-T [T, Q_OUT_TILE] FP32 tile does not fit UB.
DEQUANT_T_TILE = min(T, 8)
assert T % DEQUANT_T_TILE == 0
HEAD_DIM_TILE = 32
D_TILE = 512
# weights_proj splits K, not N: a [D_TILE, IDX_N_HEADS] row block reads contiguous GM,
# while an N slice would take 32B out of every 128B row. Each task writes its own
# partial row block, summed by a separate reduce scope. Partials are laid out
# [K slice][T_PAD rows] so the reduce adds whole T_PAD-row blocks.
# WEIGHTS_K_SLICE // D_TILE == 2, so the inner loop is a pl.range: a degenerate
# 2-iteration pl.pipeline(stage=2) miscompiles over matmul.
WEIGHTS_OK = 4
WEIGHTS_K_SLICE = D // WEIGHTS_OK
assert WEIGHTS_K_SLICE % D_TILE == 0
QH_QUANT_TILE = 64
# cube tile for q @ hadamard; L0C caps it at QH_MM_TILE * IDX_HEAD_DIM * 4B <= 64KiB.
QH_MM_TILE = 64
QH_HEAD_DIM_TILE = 64
ROPE_ROW_BLOCK = S * IDX_N_HEADS
# qr_rope SPMD tile == row block: one ROPE_ROW_TILE-row block per SPMD tile.
ROPE_ROW_TILE = 32
@pl.jit.inline
def indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos_il: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    query_vectors: pl.Tensor[
        [T_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8
    ],
    query_scales: pl.Tensor[[T_DYN, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[T_DYN, IDX_N_HEADS], pl.FP32],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_descriptors: pl.Tensor[[PAIR_GROUP_DYN, PHASE_D_PAIR_FIELDS], pl.INT32],
    singleton_descriptors: pl.Tensor[
        [SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32
    ],
    upper_descriptors: pl.Tensor[
        [UPPER_MERGE_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    root_descriptors: pl.Tensor[[T_DYN, PHASE_D_ROOT_FIELDS], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, CSA_TOPK], pl.INT32],
    index_commit_dep: pl.Scalar[pl.TASK_ID],
    late_dep: pl.Scalar[pl.TASK_ID],
    completion: pl.Array[1, pl.TASK_ID],
):
    """Project token-local index queries and run the exact active CSA forest."""
    query_count = pl.tensor.dim(x, 0)
    query_heads = query_count * IDX_N_HEADS
    row_blocks = (query_count + MM_ROW_TILE - 1) // MM_ROW_TILE

    qr_acc_pad = pl.create_tensor(
        [T_PAD, IDX_N_HEADS * IDX_HEAD_DIM],
        dtype=pl.INT32,
    )
    for qr_unit in pl.spmd(
        QR_OT_COUNT * row_blocks,
        name_hint="phase_d_idx_qr_proj_matmul",
        allow_early_resolve=True,
    ):
        qr_rb = qr_unit // QR_OT_COUNT
        ot = qr_unit - qr_rb * QR_OT_COUNT
        qr_r0 = qr_rb * MM_ROW_TILE
        qr_rows = pl.min(MM_ROW_TILE, query_count - qr_r0)
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(
                    qr,
                    [MM_ROW_TILE, Q_TILE],
                    [qr_r0, q0],
                    valid_shape=[qr_rows, Q_TILE],
                )
                wq_tile = wq_b[
                    q0 : q0 + Q_TILE,
                    o_base + ns : o_base + ns + MM_N_TILE,
                ]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[
                qr_r0 : qr_r0 + MM_ROW_TILE,
                o_base + ns : o_base + ns + MM_N_TILE,
            ] = qr_acc

    qr_proj = pl.create_tensor(
        [query_count, IDX_N_HEADS * IDX_HEAD_DIM],
        dtype=pl.FP32,
    )
    with pl.spmd(
        (query_count // DEQUANT_T_TILE) * QR_OT_COUNT,
        name_hint="phase_d_idx_qr_proj_dequant",
        allow_early_resolve=True,
    ):
        unit = pl.tile.get_block_idx()
        query_block = unit // QR_OT_COUNT
        ot = unit - query_block * QR_OT_COUNT
        query = query_block * DEQUANT_T_TILE
        o_base = ot * Q_OUT_TILE
        acc_fp32 = pl.cast(
            qr_acc_pad[
                query : query + DEQUANT_T_TILE,
                o_base : o_base + Q_OUT_TILE,
            ],
            target_type=pl.FP32,
            mode="none",
        )
        wq_scale = pl.reshape(
            wq_b_scale[o_base : o_base + Q_OUT_TILE],
            [1, Q_OUT_TILE],
        )
        qr_dequant = pl.col_expand_mul(
            pl.row_expand_mul(
                acc_fp32,
                qr_scale[query : query + DEQUANT_T_TILE, :],
            ),
            wq_scale,
        )
        qr_proj[
            query : query + DEQUANT_T_TILE,
            o_base : o_base + Q_OUT_TILE,
        ] = qr_dequant

    qr_proj_flat = pl.reshape(qr_proj, [query_heads, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([query_heads, IDX_HEAD_DIM], dtype=pl.BF16)
    rope_swap_idx_t = pl.create_tensor(
        [ROPE_ROW_TILE, ROPE_HEAD_DIM],
        dtype=pl.INT32,
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="phase_d_idx_rope_swap_idx",
        allow_early_resolve=True,
    ):
        sw_col = pl.col_expand_mul(
            pl.full(
                [ROPE_ROW_TILE, ROPE_HEAD_DIM],
                dtype=pl.FP32,
                value=1.0,
            ),
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        sw_dup_f = pl.cast(
            pl.cast(
                pl.mul(sw_col, 0.5),
                target_type=pl.INT32,
                mode="trunc",
            ),
            target_type=pl.FP32,
        )
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx_t[:, :] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)),
            target_type=pl.INT32,
        )

    for rope_unit in pl.spmd(
        query_heads // ROPE_ROW_TILE,
        name_hint="phase_d_idx_query_rope",
        allow_early_resolve=True,
    ):
        row0 = rope_unit * ROPE_ROW_TILE
        query = row0 // IDX_N_HEADS
        qr_nope = qr_proj_flat[
            row0 : row0 + ROPE_ROW_TILE,
            0:IDX_NOPE_HEAD_DIM,
        ]
        qr_rope = qr_proj_flat[
            row0 : row0 + ROPE_ROW_TILE,
            IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM,
        ]
        qr_swapped = pl.gather(
            qr_rope,
            dim=-1,
            index=rope_swap_idx_t,
        )
        rope_rot = pl.add(
            pl.col_expand_mul(
                qr_rope,
                cos_il[query : query + 1, :],
            ),
            pl.col_expand_mul(
                qr_swapped,
                sin_signed[query : query + 1, :],
            ),
        )
        qr_bf16[row0 : row0 + ROPE_ROW_TILE, :] = pl.concat(
            pl.cast(qr_nope, target_type=pl.BF16, mode="rint"),
            pl.cast(rope_rot, target_type=pl.BF16, mode="rint"),
        )

    qh_acc_gm = pl.create_tensor([query_heads, IDX_HEAD_DIM], dtype=pl.FP32)
    for query in pl.spmd(
        query_count,
        name_hint="phase_d_idx_hadamard_matmul",
        allow_early_resolve=True,
    ):
        row0 = query * IDX_N_HEADS
        qh_acc_gm[row0 : row0 + IDX_N_HEADS, :] = pl.matmul(
            qr_bf16[row0 : row0 + IDX_N_HEADS, :],
            hadamard,
            out_dtype=pl.FP32,
        )

    query_vectors_flat = pl.reshape(
        query_vectors, [query_heads, IDX_HEAD_DIM]
    )
    query_scales_flat = pl.reshape(query_scales, [query_heads, 1])
    for query in pl.spmd(
        query_count,
        name_hint="phase_d_idx_query_quant",
        allow_early_resolve=True,
    ):
        row0 = query * IDX_N_HEADS
        qh_amax = pl.full(
            [1, IDX_N_HEADS],
            dtype=pl.FP32,
            value=INT8_AMAX_EPS,
        )
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[
                row0 : row0 + IDX_N_HEADS,
                h0 : h0 + QH_HEAD_DIM_TILE,
            ]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_amax = pl.maximum(
                qh_amax,
                pl.reshape(pl.row_max(qh_a_abs), [1, IDX_N_HEADS]),
            )
        qh_scale_quant_row = pl.div(
            pl.full(
                [1, IDX_N_HEADS],
                dtype=pl.FP32,
                value=INT8_SCALE_MAX,
            ),
            qh_amax,
        )
        query_scales_flat[row0 : row0 + IDX_N_HEADS, :] = pl.reshape(
            pl.recip(qh_scale_quant_row),
            [IDX_N_HEADS, 1],
        )
        qh_scale_quant = pl.reshape(
            qh_scale_quant_row,
            [IDX_N_HEADS, 1],
        )
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[
                row0 : row0 + IDX_N_HEADS,
                h1 : h1 + QH_HEAD_DIM_TILE,
            ]
            qh_q_i32 = pl.cast(
                pl.row_expand_mul(qh_q_f32, qh_scale_quant),
                target_type=pl.INT32,
                mode="rint",
            )
            query_vectors_flat[
                row0 : row0 + IDX_N_HEADS,
                h1 : h1 + QH_HEAD_DIM_TILE,
            ] = pl.cast(
                pl.cast(qh_q_i32, target_type=pl.FP16, mode="round"),
                target_type=pl.INT8,
                mode="trunc",
            )

    weights_partial = pl.create_tensor(
        [WEIGHTS_OK * T_PAD, IDX_N_HEADS],
        dtype=pl.FP32,
    )
    with pl.spmd(
        WEIGHTS_OK * row_blocks,
        name_hint="phase_d_idx_weights_proj",
        deps=[late_dep],
    ) as _weights_proj_tid:
        unit = pl.tile.get_block_idx()
        row_block = unit // WEIGHTS_OK
        k_block = unit - row_block * WEIGHTS_OK
        row0 = row_block * MM_ROW_TILE
        valid_rows = pl.min(MM_ROW_TILE, query_count - row0)
        k_base = k_block * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor(
            [MM_ROW_TILE, IDX_N_HEADS],
            dtype=pl.FP32,
        )
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(
                x,
                [MM_ROW_TILE, D_TILE],
                [row0, d0],
                valid_shape=[valid_rows, D_TILE],
            )
            weight_tile = weights_proj[d0 : d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(
                    x_tile,
                    weight_tile,
                    out_dtype=pl.FP32,
                )
            else:
                weights_acc = pl.matmul_acc(
                    weights_acc,
                    x_tile,
                    weight_tile,
                )
        weights_partial[
            k_block * T_PAD + row0 : k_block * T_PAD + row0 + MM_ROW_TILE,
            :,
        ] = weights_acc

    with pl.spmd(
        query_count,
        name_hint="phase_d_idx_weights_reduce",
        allow_early_resolve=True,
    ):
        query = pl.tile.get_block_idx()
        weights_sum = weights_partial[query : query + 1, :]
        for k_block in pl.unroll(1, WEIGHTS_OK):
            weights_sum = pl.add(
                weights_sum,
                weights_partial[
                    k_block * T_PAD + query : k_block * T_PAD + query + 1,
                    :,
                ],
            )
        query_weights[query : query + 1, :] = pl.mul(
            weights_sum,
            WEIGHTS_SCALE,
        )

    active_score_topk_forest(
        query_vectors,
        query_scales,
        query_weights,
        idx_kv_cache_flat,
        idx_kv_scale_flat,
        query_request_ids,
        idx_pages,
        idx_page_offsets,
        idx_windows,
        request_epochs,
        leaf_descriptors,
        pair_descriptors,
        singleton_descriptors,
        upper_descriptors,
        root_descriptors,
        pair_arena,
        topk_scores,
        topk_indices,
        index_commit_dep,
        completion,
    )
    return topk_scores, topk_indices


@pl.jit
def phase_d_indexer_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos_il: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    query_vectors: pl.Tensor[
        [T_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8
    ],
    query_scales: pl.Tensor[[T_DYN, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[T_DYN, IDX_N_HEADS], pl.FP32],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_descriptors: pl.Tensor[[PAIR_GROUP_DYN, PHASE_D_PAIR_FIELDS], pl.INT32],
    singleton_descriptors: pl.Tensor[
        [SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32
    ],
    upper_descriptors: pl.Tensor[
        [UPPER_MERGE_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    root_descriptors: pl.Tensor[[T_DYN, PHASE_D_ROOT_FIELDS], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, CSA_TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, CSA_TOPK], pl.INT32]],
):
    """Standalone gate for projection plus the compact Phase D Top-K forest."""
    x.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    cos_il.bind_dynamic(0, T_DYN)
    sin_signed.bind_dynamic(0, T_DYN)
    query_vectors.bind_dynamic(0, T_DYN)
    query_scales.bind_dynamic(0, T_DYN)
    query_weights.bind_dynamic(0, T_DYN)
    idx_kv_cache_flat.bind_dynamic(0, IDX_ROW_DYN)
    idx_kv_scale_flat.bind_dynamic(0, IDX_ROW_DYN)
    query_request_ids.bind_dynamic(0, T_DYN)
    idx_pages.bind_dynamic(0, PAGE_DYN)
    idx_page_offsets.bind_dynamic(0, REQUEST_OFFSET_DYN)
    idx_windows.bind_dynamic(0, B_DYN)
    request_epochs.bind_dynamic(0, B_DYN)
    leaf_descriptors.bind_dynamic(0, LEAF_DYN)
    pair_descriptors.bind_dynamic(0, PAIR_GROUP_DYN)
    singleton_descriptors.bind_dynamic(0, SINGLETON_DYN)
    upper_descriptors.bind_dynamic(0, UPPER_MERGE_DYN)
    root_descriptors.bind_dynamic(0, T_DYN)
    pair_arena.bind_dynamic(0, ARENA_DYN)
    topk_scores.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    index_commit_dep = pl.system.task_dummy(deps=[])
    late_dep = pl.system.task_dummy(deps=[])
    completion = pl.array.create(1, pl.TASK_ID)
    indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos_il,
        sin_signed,
        hadamard,
        query_vectors,
        query_scales,
        query_weights,
        idx_kv_cache_flat,
        idx_kv_scale_flat,
        query_request_ids,
        idx_pages,
        idx_page_offsets,
        idx_windows,
        request_epochs,
        leaf_descriptors,
        pair_descriptors,
        singleton_descriptors,
        upper_descriptors,
        root_descriptors,
        pair_arena,
        topk_scores,
        topk_indices,
        index_commit_dep,
        late_dep,
        completion,
    )
    return topk_scores, topk_indices


def _phase_d_forest_descriptors(candidate_counts, logical_begins):
    """Pack a heterogeneous per-query forest using the production Phase-D ABI."""
    import torch

    if len(candidate_counts) != len(logical_begins):
        raise ValueError("candidate_counts and logical_begins must have equal length")

    ready_frontier = 8
    invalid_slot = CSA_TOPK_INVALID_TASK_SLOT
    node_kinds = []
    node_leaf_ids = []
    node_left_slots = []
    node_right_slots = []
    node_credit_slots = []
    leaf_rows = []
    root_slots = []

    def append_leaf(query, local_leaf, valid, credit_slot):
        leaf_id = len(leaf_rows)
        slot = len(node_kinds)
        begin = logical_begins[query] + local_leaf * CSA_CANDIDATES_PER_LEAF
        leaf_rows.append([query, begin, valid, slot, credit_slot])
        node_kinds.append(0)
        node_leaf_ids.append(leaf_id)
        node_left_slots.append(-1)
        node_right_slots.append(-1)
        node_credit_slots.append(credit_slot)
        return slot

    def append_merge(left_slot, right_slot):
        slot = len(node_kinds)
        node_kinds.append(1)
        node_leaf_ids.append(-1)
        node_left_slots.append(left_slot)
        node_right_slots.append(right_slot)
        node_credit_slots.append(invalid_slot)
        return slot

    for query, (candidates, begin) in enumerate(zip(candidate_counts, logical_begins)):
        leaves = (candidates + CSA_CANDIDATES_PER_LEAF - 1) // CSA_CANDIDATES_PER_LEAF
        if leaves == 0:
            root_slots.append(-1)
            continue

        level1_slots = []
        for group in range(leaves // 2):
            credit = invalid_slot
            if group >= ready_frontier:
                credit = level1_slots[group - ready_frontier]
            left_leaf = group * 2
            left_valid = min(
                CSA_CANDIDATES_PER_LEAF,
                candidates - left_leaf * CSA_CANDIDATES_PER_LEAF,
            )
            left_slot = append_leaf(query, left_leaf, left_valid, credit)
            right_leaf = left_leaf + 1
            right_valid = min(
                CSA_CANDIDATES_PER_LEAF,
                candidates - right_leaf * CSA_CANDIDATES_PER_LEAF,
            )
            right_slot = append_leaf(query, right_leaf, right_valid, credit)
            level1_slots.append(append_merge(left_slot, right_slot))

        if leaves % 2:
            local_leaf = leaves - 1
            credit = invalid_slot
            if len(level1_slots) >= ready_frontier:
                credit = level1_slots[-ready_frontier]
            valid = min(
                CSA_CANDIDATES_PER_LEAF,
                candidates - local_leaf * CSA_CANDIDATES_PER_LEAF,
            )
            level1_slots.append(append_leaf(query, local_leaf, valid, credit))

        current = level1_slots
        while len(current) > 1:
            next_level = []
            for pair in range(len(current) // 2):
                next_level.append(append_merge(current[2 * pair], current[2 * pair + 1]))
            if len(current) % 2:
                next_level.append(current[-1])
            current = next_level
        root_slots.append(current[0])

    pair_rows = []
    upper_rows = []
    paired_leaf_slots = set()
    for output_slot, kind in enumerate(node_kinds):
        if kind != 1:
            continue
        left_slot = node_left_slots[output_slot]
        right_slot = node_right_slots[output_slot]
        if node_kinds[left_slot] == 0 and node_kinds[right_slot] == 0:
            pair_rows.append([
                node_leaf_ids[left_slot],
                node_leaf_ids[right_slot],
                left_slot,
                right_slot,
                output_slot,
                node_credit_slots[left_slot],
            ])
            paired_leaf_slots.update((left_slot, right_slot))
        else:
            upper_rows.append([left_slot, right_slot, output_slot])

    singleton_slots = [
        slot
        for slot, kind in enumerate(node_kinds)
        if kind == 0 and slot not in paired_leaf_slots
    ]
    singleton_rows = [
        [node_leaf_ids[slot], slot, node_credit_slots[slot]]
        for slot in singleton_slots
    ]
    root_deps = [root if root >= 0 else invalid_slot for root in root_slots]

    def tensor(rows, width):
        if rows:
            return torch.tensor(rows, dtype=torch.int32)
        return torch.empty(0, width, dtype=torch.int32)

    return {
        "leaf_descriptors": tensor(leaf_rows, PHASE_D_LEAF_FIELDS),
        "pair_descriptors": tensor(pair_rows, PHASE_D_PAIR_FIELDS),
        "singleton_descriptors": tensor(singleton_rows, PHASE_D_SINGLETON_FIELDS),
        "upper_descriptors": tensor(upper_rows, PHASE_D_UPPER_FIELDS),
        "root_descriptors": tensor(
            [[root, dep] for root, dep in zip(root_slots, root_deps)],
            PHASE_D_ROOT_FIELDS,
        ),
        "candidate_counts": list(candidate_counts),
        "logical_begins": list(logical_begins),
        "root_slots": root_slots,
        "node_count": len(node_kinds),
    }


PHASE_D_TRACE_CASE_LENGTHS = {
    "trace_b4_s8_16k": (16 * 1024, 16 * 1024),
    "trace_b4_s8_1m": (MAX_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS),
    # One 16K request and one 496K request per 16-row CSA chunk. Two
    # identical chunks therefore exercise B=4/S=8 with exactly 1M total
    # resident history per rank while keeping both chunks equally loaded.
    "trace_b4_s8_ragged": (16 * 1024, 496 * 1024),
}


def build_phase_d_indexer_specs(
    query_count=8,
    case="phase_d_one_leaf",
    request_lengths=None,
    request_active=None,
):
    """Build compact Phase-D projection/forest fixtures."""
    import torch

    from golden import TensorSpec

    if query_count <= 0 or query_count > CSA_MAX_QUERIES or query_count % 8 != 0:
        raise ValueError(
            "Phase D indexer fixture query_count must be a multiple of 8 "
            f"in [8, {CSA_MAX_QUERIES}]"
        )
    trace_lengths = PHASE_D_TRACE_CASE_LENGTHS.get(case)
    if request_lengths is not None:
        trace_lengths = tuple(int(length) for length in request_lengths)
        if len(trace_lengths) != query_count // S:
            raise ValueError(
                "custom Phase-D lengths must provide one length per S-row request"
            )
        if any(length < S or length > MAX_CONTEXT_TOKENS for length in trace_lengths):
            raise ValueError("custom Phase-D lengths must stay inside [S, 1M]")
    if request_active is None:
        trace_request_active = None
    else:
        trace_request_active = tuple(bool(active) for active in request_active)
        if trace_lengths is None or len(trace_request_active) != len(trace_lengths):
            raise ValueError("request_active must match the custom/trace request count")
    if case == "phase_d_mixed_forest" or trace_lengths is not None:
        # One eight-row group spans the important empty, partial-leaf, exact-
        # leaf and odd-forest boundaries.  Repeat that complete group at the
        # 16-row production ceiling so a compile-time CSA shard may hold two
        # complete S=8 requests without weakening the boundary coverage.
        if trace_lengths is None:
            group_candidate_counts = [0, 1, 511, 512, 513, 2048, 4096, 4097]
            group_logical_begins = [0, 128, 256, 384, 512, 4096, 8192, 12288]
            groups = query_count // len(group_candidate_counts)
            candidate_counts = group_candidate_counts * groups
            logical_begins = group_logical_begins * groups
            query_request_ids = list(range(query_count))
            request_candidate_caps = candidate_counts
        else:
            if query_count != 2 * S:
                raise ValueError(
                    "B4/S8 CSA trace fixtures require one complete 16-row chunk"
                )
            candidate_counts = []
            query_request_ids = []
            request_candidate_caps = []
            for request, final_length in enumerate(trace_lengths):
                if trace_request_active is not None and not trace_request_active[request]:
                    candidate_counts.extend([0] * S)
                    query_request_ids.extend([-1] * S)
                    request_candidate_caps.append(0)
                    continue
                if final_length > MAX_CONTEXT_TOKENS:
                    raise ValueError("trace request exceeds the 1M context ceiling")
                first_position = final_length - S
                candidate_counts.extend(
                    (first_position + row + 1) // COMPRESS_RATIO
                    for row in range(S)
                )
                query_request_ids.extend([request] * S)
                request_candidate_caps.append(final_length // COMPRESS_RATIO)
            logical_begins = [0] * query_count
        forest = _phase_d_forest_descriptors(candidate_counts, logical_begins)
        page_counts = []
        request_logical_begins = (
            logical_begins
            if trace_lengths is None
            else [0] * len(trace_lengths)
        )
        for begin, candidates in zip(
            request_logical_begins, request_candidate_caps
        ):
            if candidates <= 0:
                page_counts.append(0)
            else:
                page_counts.append(
                    (begin + candidates + BLOCK_SIZE - 1) // BLOCK_SIZE
                    - begin // BLOCK_SIZE
                )
        page_offsets = [0]
        for count in page_counts:
            page_offsets.append(page_offsets[-1] + count)
        total_pages = page_offsets[-1]
        pages = []
        for request, count in enumerate(page_counts):
            base = page_offsets[request]
            # Compact physical pages are deliberately permuted per request;
            # ``head`` in idx_windows rotates the logical page walk as well.
            pages.extend([base + local for local in reversed(range(count))])
        # Preserve the public ``[P, 2]`` page-descriptor ABI when an inactive
        # request set contributes no pages.  ``torch.tensor([])`` would be
        # one-dimensional and makes the rank/chunk packer infer inconsistent
        # descriptor ranks at EP>2.
        pages = torch.tensor(
            [[page, 11] for page in pages], dtype=torch.int32
        ).reshape(-1, 2)
        windows = torch.tensor(
            [
                [begin, begin + candidates, request % max(page_counts[request], 1)]
                for request, (begin, candidates) in enumerate(
                    zip(request_logical_begins, request_candidate_caps)
                )
            ],
            dtype=torch.int32,
        )
        page_offsets_t = torch.tensor(page_offsets, dtype=torch.int32)
        zero = lambda shape, dtype: torch.zeros(shape, dtype=dtype)
        identity = torch.eye(IDX_HEAD_DIM, dtype=torch.bfloat16)
        return [
            TensorSpec("x", [query_count, D], torch.bfloat16, init_value=lambda: zero((query_count, D), torch.bfloat16)),
            TensorSpec("qr", [query_count, Q_LORA], torch.int8, init_value=lambda: zero((query_count, Q_LORA), torch.int8)),
            TensorSpec("qr_scale", [query_count, 1], torch.float32, init_value=lambda: torch.ones(query_count, 1)),
            TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: zero((Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM), torch.int8)),
            TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: torch.ones(IDX_N_HEADS * IDX_HEAD_DIM)),
            TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=lambda: zero((D, IDX_N_HEADS), torch.bfloat16)),
            TensorSpec("cos_il", [query_count, ROPE_HEAD_DIM], torch.float32, init_value=lambda: torch.ones(query_count, ROPE_HEAD_DIM)),
            TensorSpec("sin_signed", [query_count, ROPE_HEAD_DIM], torch.float32, init_value=lambda: zero((query_count, ROPE_HEAD_DIM), torch.float32)),
            TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=lambda: identity),
            TensorSpec("query_vectors", [query_count, IDX_N_HEADS, IDX_HEAD_DIM], torch.int8, init_value=lambda: zero((query_count, IDX_N_HEADS, IDX_HEAD_DIM), torch.int8)),
            TensorSpec("query_scales", [query_count, IDX_N_HEADS], torch.float32, init_value=lambda: torch.ones(query_count, IDX_N_HEADS)),
            TensorSpec("query_weights", [query_count, IDX_N_HEADS], torch.float32, init_value=lambda: torch.ones(query_count, IDX_N_HEADS)),
            TensorSpec("idx_kv_cache_flat", [total_pages * BLOCK_SIZE, IDX_HEAD_DIM], torch.int8, init_value=lambda: zero((total_pages * BLOCK_SIZE, IDX_HEAD_DIM), torch.int8)),
            TensorSpec("idx_kv_scale_flat", [total_pages * BLOCK_SIZE, 1], torch.float32, init_value=lambda: torch.ones(total_pages * BLOCK_SIZE, 1)),
            TensorSpec("query_request_ids", [query_count], torch.int32, init_value=lambda: torch.tensor(query_request_ids, dtype=torch.int32)),
            TensorSpec("idx_pages", list(pages.shape), torch.int32, init_value=lambda: pages),
            TensorSpec("idx_page_offsets", [len(page_offsets)], torch.int32, init_value=lambda: page_offsets_t),
            TensorSpec("idx_windows", list(windows.shape), torch.int32, init_value=lambda: windows),
            TensorSpec("request_epochs", [len(page_counts)], torch.int32, init_value=lambda: torch.full((len(page_counts),), 11, dtype=torch.int32)),
            TensorSpec("leaf_descriptors", list(forest["leaf_descriptors"].shape), torch.int32, init_value=lambda: forest["leaf_descriptors"]),
            TensorSpec("pair_descriptors", list(forest["pair_descriptors"].shape), torch.int32, init_value=lambda: forest["pair_descriptors"]),
            TensorSpec("singleton_descriptors", list(forest["singleton_descriptors"].shape), torch.int32, init_value=lambda: forest["singleton_descriptors"]),
            TensorSpec("upper_descriptors", list(forest["upper_descriptors"].shape), torch.int32, init_value=lambda: forest["upper_descriptors"]),
            TensorSpec("root_descriptors", list(forest["root_descriptors"].shape), torch.int32, init_value=lambda: forest["root_descriptors"]),
            TensorSpec("pair_arena", [max(forest["node_count"], 1), CSA_PAIR_WIDTH], torch.float32, init_value=lambda: zero((max(forest["node_count"], 1), CSA_PAIR_WIDTH), torch.float32)),
            TensorSpec("topk_scores", [query_count, CSA_TOPK], torch.float32, is_output=True),
            TensorSpec("topk_indices", [query_count, CSA_TOPK], torch.int32, is_output=True),
        ]
    logical_begin = 1
    valid_candidates = 2048
    page_count = 17
    physical_pages = list(reversed(range(page_count)))
    pages = torch.tensor(
        [[page, 11] for page in physical_pages], dtype=torch.int32
    )
    leaf_descriptors = torch.tensor(
        [
            [
                query,
                logical_begin,
                valid_candidates,
                query,
                CSA_TOPK_INVALID_TASK_SLOT,
            ]
            for query in range(query_count)
        ],
        dtype=torch.int32,
    )
    singleton_descriptors = torch.tensor(
        [
            [query, query, CSA_TOPK_INVALID_TASK_SLOT]
            for query in range(query_count)
        ],
        dtype=torch.int32,
    )
    root_descriptors = torch.tensor(
        [[query, query] for query in range(query_count)], dtype=torch.int32
    )
    return [
        TensorSpec("x", [query_count, D], torch.bfloat16),
        TensorSpec("qr", [query_count, Q_LORA], torch.int8),
        TensorSpec(
            "qr_scale",
            [query_count, 1],
            torch.float32,
            init_value=lambda: torch.ones(query_count, 1),
        ),
        TensorSpec(
            "wq_b",
            [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM],
            torch.int8,
        ),
        TensorSpec(
            "wq_b_scale",
            [IDX_N_HEADS * IDX_HEAD_DIM],
            torch.float32,
            init_value=lambda: torch.ones(IDX_N_HEADS * IDX_HEAD_DIM),
        ),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16),
        TensorSpec(
            "cos_il",
            [query_count, ROPE_HEAD_DIM],
            torch.float32,
            init_value=lambda: torch.ones(query_count, ROPE_HEAD_DIM),
        ),
        TensorSpec(
            "sin_signed", [query_count, ROPE_HEAD_DIM], torch.float32
        ),
        TensorSpec(
            "hadamard",
            [IDX_HEAD_DIM, IDX_HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: torch.eye(IDX_HEAD_DIM),
        ),
        TensorSpec(
            "query_vectors",
            [query_count, IDX_N_HEADS, IDX_HEAD_DIM],
            torch.int8,
        ),
        TensorSpec(
            "query_scales",
            [query_count, IDX_N_HEADS],
            torch.float32,
        ),
        TensorSpec(
            "query_weights",
            [query_count, IDX_N_HEADS],
            torch.float32,
        ),
        TensorSpec(
            "idx_kv_cache_flat",
            [page_count * BLOCK_SIZE, IDX_HEAD_DIM],
            torch.int8,
        ),
        TensorSpec(
            "idx_kv_scale_flat",
            [page_count * BLOCK_SIZE, 1],
            torch.float32,
            init_value=lambda: torch.ones(page_count * BLOCK_SIZE, 1),
        ),
        TensorSpec(
            "query_request_ids",
            [query_count],
            torch.int32,
            init_value=lambda: torch.zeros(query_count, dtype=torch.int32),
        ),
        TensorSpec("idx_pages", [page_count, 2], torch.int32, init_value=lambda: pages),
        TensorSpec(
            "idx_page_offsets",
            [2],
            torch.int32,
            init_value=lambda: torch.tensor([0, page_count], dtype=torch.int32),
        ),
        TensorSpec(
            "idx_windows",
            [1, 3],
            torch.int32,
            init_value=lambda: torch.tensor(
                [[logical_begin, logical_begin + valid_candidates, 0]],
                dtype=torch.int32,
            ),
        ),
        TensorSpec(
            "request_epochs",
            [1],
            torch.int32,
            init_value=lambda: torch.tensor([11], dtype=torch.int32),
        ),
        TensorSpec(
            "leaf_descriptors",
            [query_count, PHASE_D_LEAF_FIELDS],
            torch.int32,
            init_value=lambda: leaf_descriptors,
        ),
        TensorSpec(
            "pair_descriptors",
            [0, PHASE_D_PAIR_FIELDS],
            torch.int32,
            init_value=lambda: torch.empty(
                0, PHASE_D_PAIR_FIELDS, dtype=torch.int32
            ),
        ),
        TensorSpec(
            "singleton_descriptors",
            [query_count, PHASE_D_SINGLETON_FIELDS],
            torch.int32,
            init_value=lambda: singleton_descriptors,
        ),
        TensorSpec(
            "upper_descriptors",
            [0, PHASE_D_UPPER_FIELDS],
            torch.int32,
            init_value=lambda: torch.empty(
                0, PHASE_D_UPPER_FIELDS, dtype=torch.int32
            ),
        ),
        TensorSpec(
            "root_descriptors",
            [query_count, PHASE_D_ROOT_FIELDS],
            torch.int32,
            init_value=lambda: root_descriptors,
        ),
        TensorSpec(
            "pair_arena",
            [query_count, CSA_PAIR_WIDTH],
            torch.float32,
        ),
        TensorSpec(
            "topk_scores",
            [query_count, CSA_TOPK],
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "topk_indices",
            [query_count, CSA_TOPK],
            torch.int32,
            is_output=True,
        ),
    ]


def golden_phase_d_indexer(tensors):
    """Exact zero-score reference for every packed Phase-D forest shape."""
    import torch

    query_count = tensors["topk_indices"].shape[0]
    scores = torch.full(
        (query_count, CSA_TOPK), FP32_NEG_INF, dtype=torch.float32
    )
    indices = torch.full((query_count, CSA_TOPK), -1, dtype=torch.int32)
    leaf_descriptors = tensors["leaf_descriptors"]
    for query in range(query_count):
        candidates = []
        for row in leaf_descriptors:
            if int(row[PHASE_D_LEAF_QUERY].item()) != query:
                continue
            begin = int(row[PHASE_D_LEAF_BEGIN].item())
            valid = int(row[PHASE_D_LEAF_VALID].item())
            candidates.extend(range(begin, begin + max(valid, 0)))
        candidates.sort()
        kept = min(CSA_TOPK, len(candidates))
        if kept:
            scores[query, :kept] = 0.0
            indices[query, :kept] = torch.tensor(
                candidates[:kept], dtype=torch.int32
            )
    tensors["topk_scores"][:] = scores
    tensors["topk_indices"][:] = indices


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--query-count",
        type=int,
        default=8,
        help=f"Packed query rows, in [1, {CSA_MAX_QUERIES}].",
    )
    parser.add_argument(
        "--case",
        choices=(
            "phase_d_one_leaf",
            "phase_d_mixed_forest",
            *PHASE_D_TRACE_CASE_LENGTHS,
        ),
        default="phase_d_one_leaf",
    )
    parser.add_argument("--enable-l2-swimlane", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.query_count <= CSA_MAX_QUERIES:
        parser.error(
            f"--query-count must be in [1, {CSA_MAX_QUERIES}], got {args.query_count}"
        )

    result = run_jit(
        fn=phase_d_indexer_test,
        specs=build_phase_d_indexer_specs(
            query_count=args.query_count,
            case=args.case,
        ),
        golden_fn=golden_phase_d_indexer,
        runtime_dir=args.runtime_dir,
        compile_cfg={"dump_passes": args.dump_passes},
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_l2_swimlane": args.enable_l2_swimlane,
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
