# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed ratio-128 HCA attention for elastic 1M-context decode."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    FLASH as M,
    HCA_ROWS_PER_SHARD,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    SWA_SOURCE_INVALID,
    SWA_SOURCE_OVERLAY_BASE,
)


B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
HCA_WORK_DYN = pl.dynamic("HCA_WORK_DYN")
HCA_PAGES_DYN = pl.dynamic("HCA_PAGES_DYN")
HCA_REQUEST_OFFSETS_DYN = pl.dynamic("HCA_REQUEST_OFFSETS_DYN")
HCA_QUERY_OFFSETS_DYN = pl.dynamic("HCA_QUERY_OFFSETS_DYN")

B = DECODE_LOCAL_REQUESTS
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

NEG_INF = -1.0e20
H_TILE = 16
QK_M_TILE = 32
ATTN_K_TILE = HCA_ROWS_PER_SHARD
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256
PROJ_A_MM_N_TILE = 128
T_PAD = ((T + 16 - 1) // 16) * 16
MM_T_TILE = T_PAD
HCA_QUERY_CHUNK_T = S
HCA_MAX_QUERY_CHUNKS = (T + HCA_QUERY_CHUNK_T - 1) // HCA_QUERY_CHUNK_T
ROPE_CS_T_TILE = 8
PROJ_A_ROW_TILE = 16
PA_N_FRAGS = O_LORA // PROJ_A_MM_N_TILE
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512
PROJ_B_ACT_N_REGS = D // PROJ_B_ACT_N_TILE
QUANT_TOKEN_TILE = 8
PROJ_B_D_TILE = 512
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TASK_T_TILE = 8

assert WIN == ATTN_K_TILE == 128
assert HCA_QUERY_CHUNK_T % S == 0


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca_chunk_heads(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    hca_pages: pl.Tensor[[HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    query_base: pl.Scalar[pl.INDEX],
    query_count: pl.Scalar[pl.INDEX],
    work_base: pl.Scalar[pl.INDEX],
    work_count: pl.Scalar[pl.INDEX],
    cmp_dep: pl.Scalar[pl.TASK_ID],
):
    """Merge one bounded query chunk into the forward-wide packed heads."""
    total_t = pl.tensor.dim(q, 0)
    t_dim = query_count
    request_count = pl.tensor.dim(request_epochs, 0)
    item_count = t_dim + work_count
    total_t_heads = total_t * H
    t_hblocks = t_dim * (H // H_TILE)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    page_count = pl.tensor.dim(hca_pages, 0)
    ori_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])

    q_flat = pl.reshape(q, [total_t_heads, HEAD_DIM])
    partial_mi = pl.create_tensor([item_count * H, 1], dtype=pl.FP32)
    partial_li = pl.create_tensor([item_count * H, 1], dtype=pl.FP32)
    partial_oi = pl.create_tensor([item_count * H, HEAD_DIM], dtype=pl.FP32)
    with pl.scope():
        packed_kv = pl.create_tensor(
            [item_count * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16
        )
        packed_valid = pl.create_tensor(
            [item_count, ATTN_K_TILE], dtype=pl.FP32
        )
        with pl.spmd(
            item_count, name_hint="hca_packed_gather", deps=[cmp_dep]
        ) as gather_tid:
            item = pl.tile.get_block_idx()
            query = query_base + item
            work = work_base + item - t_dim
            is_raw = item < t_dim
            if not is_raw:
                query = pl.cast(
                    pl.read(hca_work_query_ids, [work]), pl.INDEX
                )
            request = pl.cast(pl.read(query_request_ids, [query]), pl.INDEX)
            row_begin = pl.cast(0, pl.INT32)
            valid_rows = pl.cast(ATTN_K_TILE, pl.INT32)
            if not is_raw:
                row_begin = pl.read(hca_work_row_begin, [work])
                valid_rows = pl.read(hca_work_valid_rows, [work])
            page_begin = pl.cast(0, pl.INT32)
            page_total = pl.cast(0, pl.INT32)
            valid_begin = pl.cast(0, pl.INT32)
            head = pl.cast(0, pl.INT32)
            request_epoch = pl.cast(-1, pl.INT32)
            if request >= 0:
                if request < request_count:
                    page_begin = pl.read(hca_page_offsets, [request])
                    page_end = pl.read(hca_page_offsets, [request + 1])
                    page_total = page_end - page_begin
                    valid_begin = pl.read(hca_windows, [request, 0])
                    head = pl.read(hca_windows, [request, 2])
                    request_epoch = pl.read(request_epochs, [request])
            for lane in pl.range(ATTN_K_TILE):
                dst = item * ATTN_K_TILE + lane
                source_valid = pl.cast(0, pl.INT32)
                source_row = pl.cast(0, pl.INDEX)
                source_overlay = pl.cast(-1, pl.INDEX)
                if is_raw:
                    source = pl.read(swa_sources, [query, lane])
                    if source >= 0:
                        if source < ori_block_num * BLOCK_SIZE:
                            source_valid = pl.cast(1, pl.INT32)
                            source_row = pl.cast(source, pl.INDEX)
                    else:
                        if source <= SWA_SOURCE_OVERLAY_BASE:
                            source_overlay = pl.cast(
                                SWA_SOURCE_OVERLAY_BASE - source, pl.INDEX
                            )
                            if source_overlay >= 0:
                                if source_overlay < total_t:
                                    source_valid = pl.cast(2, pl.INT32)
                else:
                    if lane < valid_rows:
                        logical_row = row_begin + lane
                        logical_page_base = valid_begin // BLOCK_SIZE
                        relative_page = logical_row // BLOCK_SIZE - logical_page_base
                        if page_total > 0:
                            if relative_page >= 0:
                                if relative_page < page_total:
                                    page_index = (head + relative_page) % page_total
                                    page_entry = page_begin + page_index
                                    if page_entry >= 0:
                                        if page_entry < page_count:
                                            physical_page = pl.read(
                                                hca_pages, [page_entry, 0]
                                            )
                                            page_epoch = pl.read(
                                                hca_pages, [page_entry, 1]
                                            )
                                            if physical_page >= 0:
                                                if page_epoch == request_epoch:
                                                    source_valid = pl.cast(1, pl.INT32)
                                                    source_row = pl.cast(
                                                        physical_page * BLOCK_SIZE
                                                        + logical_row % BLOCK_SIZE,
                                                        pl.INDEX,
                                                    )
                if source_valid == 1:
                    if is_raw:
                        packed_kv[dst : dst + 1, 0:HEAD_DIM] = ori_flat[
                            source_row : source_row + 1, 0:HEAD_DIM
                        ]
                    else:
                        packed_kv[dst : dst + 1, 0:HEAD_DIM] = cmp_flat[
                            source_row : source_row + 1, 0:HEAD_DIM
                        ]
                    pl.write(packed_valid, [item, lane], 1.0)
                elif source_valid == 2:
                    packed_kv[dst : dst + 1, 0:HEAD_DIM] = current_kv[
                        source_overlay : source_overlay + 1, 0:HEAD_DIM
                    ]
                    pl.write(packed_valid, [item, lane], 1.0)
                else:
                    packed_kv[dst : dst + 1, 0:HEAD_DIM] = pl.full(
                        [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                    )
                    pl.write(packed_valid, [item, lane], 0.0)

        with pl.spmd(
            item_count,
            name_hint="hca_packed_qk_pv",
            deps=[gather_tid],
            allow_early_resolve=True,
        ) as _qk_tid:
            item = pl.tile.get_block_idx()
            query = query_base + item
            if item >= t_dim:
                work = work_base + item - t_dim
                query = pl.cast(
                    pl.read(hca_work_query_ids, [work]), pl.INDEX
                )
            kv_tile = packed_kv[
                item * ATTN_K_TILE : (item + 1) * ATTN_K_TILE, 0:HEAD_DIM
            ]
            valid_row = packed_valid[item : item + 1, 0:ATTN_K_TILE]
            bias = pl.mul(pl.sub(valid_row, 1.0), -NEG_INF)
            for head_block in pl.pipeline(H // QK_M_TILE, stage=2):
                head0 = head_block * QK_M_TILE
                q_tile = q_flat[
                    query * H + head0 : query * H + head0 + QK_M_TILE,
                    0:HEAD_DIM,
                ]
                scores = pl.mul(
                    pl.matmul(q_tile, kv_tile, b_trans=True, out_dtype=pl.FP32),
                    SOFTMAX_SCALE,
                )
                scores = pl.add(
                    scores,
                    pl.col_expand(
                        pl.full(
                            [QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0
                        ),
                        bias,
                    ),
                )
                mi = pl.row_max(scores)
                exp_scores = pl.mul(
                    pl.exp(pl.row_expand_sub(scores, mi)),
                    pl.col_expand(
                        pl.full(
                            [QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0
                        ),
                        valid_row,
                    ),
                )
                li = pl.row_sum(exp_scores)
                oi = pl.matmul(
                    pl.cast(exp_scores, target_type=pl.BF16, mode="rint"),
                    kv_tile,
                    out_dtype=pl.FP32,
                )
                partial_mi[
                    item * H + head0 : item * H + head0 + QK_M_TILE, 0:1
                ] = mi
                partial_li[
                    item * H + head0 : item * H + head0 + QK_M_TILE, 0:1
                ] = li
                partial_oi[
                    item * H + head0 : item * H + head0 + QK_M_TILE, 0:HEAD_DIM
                ] = oi

    rope_cos_il = pl.create_tensor([HCA_QUERY_CHUNK_T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor(
        [HCA_QUERY_CHUNK_T, ROPE_DIM], dtype=pl.FP32
    )
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_swap"):
        sw_col = pl.col_expand_mul(
            pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(
                pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        sw_dup_f = pl.cast(
            pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)),
            target_type=pl.INT32,
        )

    for cp in pl.spmd(HALF_ROPE // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full(
                [ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE],
                dtype=pl.FP32,
                value=1.0,
            ),
            pl.cast(
                pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        cs_dup_f = pl.cast(
            pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_rb in pl.range(rope_cs_blocks):
            cs_t0 = cs_rb * ROPE_CS_T_TILE
            cs_global_t0 = query_base + cs_t0
            cs_cos = pl.cast(
                freqs_cos[
                    cs_global_t0 : cs_global_t0 + ROPE_CS_T_TILE,
                    cp_r0 : cp_r0 + ROPE_TILE,
                ],
                target_type=pl.FP32,
            )
            cs_sin = pl.cast(
                freqs_sin[
                    cs_global_t0 : cs_global_t0 + ROPE_CS_T_TILE,
                    cp_r0 : cp_r0 + ROPE_TILE,
                ],
                target_type=pl.FP32,
            )
            rope_cos_il[
                cs_t0 : cs_t0 + ROPE_CS_T_TILE,
                cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE,
            ] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            rope_sin_signed[
                cs_t0 : cs_t0 + ROPE_CS_T_TILE,
                cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE,
            ] = pl.mul(pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    with pl.spmd(t_hblocks, name_hint="hca_packed_merge_norm") as merge_tid:
        merge_item = pl.tile.get_block_idx()
        local_query = merge_item // (H // H_TILE)
        query = query_base + local_query
        head_index = merge_item - local_query * (H // H_TILE)
        head0 = head_index * H_TILE
        raw_row = local_query * H + head0
        merge_mi = partial_mi[raw_row : raw_row + H_TILE, 0:1]
        merge_li = partial_li[raw_row : raw_row + H_TILE, 0:1]
        merge_oi = partial_oi[raw_row : raw_row + H_TILE, 0:HEAD_DIM]
        work_begin = pl.cast(
            pl.read(hca_query_work_offsets, [query]), pl.INDEX
        )
        work_end = pl.cast(
            pl.read(hca_query_work_offsets, [query + 1]), pl.INDEX
        )
        for work in pl.pipeline(work_begin, work_end, stage=2):
            row = (t_dim + work - work_base) * H + head0
            cur_mi = partial_mi[row : row + H_TILE, 0:1]
            cur_li = partial_li[row : row + H_TILE, 0:1]
            cur_oi = partial_oi[row : row + H_TILE, 0:HEAD_DIM]
            merge_mi_new = pl.maximum(merge_mi, cur_mi)
            alpha = pl.exp(pl.sub(merge_mi, merge_mi_new))
            beta = pl.exp(pl.sub(cur_mi, merge_mi_new))
            merge_li = pl.add(
                pl.mul(alpha, merge_li), pl.mul(beta, cur_li)
            )
            merge_oi = pl.add(
                pl.row_expand_mul(merge_oi, alpha),
                pl.row_expand_mul(cur_oi, beta),
            )
            merge_mi = merge_mi_new
        sink = pl.reshape(attn_sink[head0 : head0 + H_TILE], [H_TILE, 1])
        denom = pl.add(merge_li, pl.exp(pl.sub(sink, merge_mi)))
        full = pl.row_expand_div(merge_oi, denom)
        full_bf16 = pl.cast(full, target_type=pl.BF16, mode="rint")
        rope = full[0:H_TILE, NOPE_DIM:HEAD_DIM]
        swapped = pl.gather(
            rope, dim=-1, index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM]
        )
        rotated = pl.add(
            pl.col_expand_mul(
                rope,
                rope_cos_il[local_query : local_query + 1, 0:ROPE_DIM],
            ),
            pl.col_expand_mul(
                swapped,
                rope_sin_signed[
                    local_query : local_query + 1, 0:ROPE_DIM
                ],
            ),
        )
        rope_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")
        merged_bf16 = pl.concat(full_bf16[:, :NOPE_DIM], rope_bf16)
        for lane in pl.unroll(H_TILE):
            packed_row = (
                ((head0 + lane) // HEADS_PER_GROUP) * T_PAD + query
            )
            packed_col = ((head0 + lane) % HEADS_PER_GROUP) * HEAD_DIM
            o_packed[
                packed_row : packed_row + 1,
                packed_col : packed_col + HEAD_DIM,
            ] = merged_bf16[lane : lane + 1, :]
    return o_packed


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca_local_o_proj(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    """Project all merged HCA heads after the bounded chunk frontier."""
    t_dim = pl.tensor.dim(attn_out, 0)
    act_t_blks = t_dim // PROJ_B_ACT_TASK_T_TILE
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE

    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    with pl.manual_scope():
        for group in pl.parallel(O_GROUPS):
            row_base = group * T_PAD
            out_col = group * O_LORA
            with pl.spmd(
                proj_a_rows * PA_N_FRAGS,
                name_hint="proj_a_mm",
                deps=[heads_dep],
                allow_early_resolve=True,
            ) as proj_a_tid:
                unit = pl.tile.get_block_idx()
                row_block = unit // PA_N_FRAGS
                n_frag = unit - row_block * PA_N_FRAGS
                row0 = row_block * PROJ_A_ROW_TILE
                rows = pl.min(PROJ_A_ROW_TILE, t_dim - row0)
                src0 = row_base + row0
                n0 = n_frag * PROJ_A_MM_N_TILE
                x_chunk = pl.slice(
                    o_packed,
                    [PROJ_A_ROW_TILE, A_K_TILE],
                    [src0, 0],
                    valid_shape=[rows, A_K_TILE],
                )
                w_chunk = wo_a[
                    group : group + 1,
                    n0 : n0 + PROJ_A_MM_N_TILE,
                    0:A_K_TILE,
                ]
                acc_a = pl.matmul(
                    x_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32
                )
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    acc_a = pl.matmul_acc(
                        acc_a,
                        pl.slice(
                            o_packed,
                            [PROJ_A_ROW_TILE, A_K_TILE],
                            [src0, k0],
                            valid_shape=[rows, A_K_TILE],
                        ),
                        wo_a[
                            group : group + 1,
                            n0 : n0 + PROJ_A_MM_N_TILE,
                            k0 : k0 + A_K_TILE,
                        ],
                        b_trans=True,
                    )
                o_r_pad = pl.assemble(o_r_pad, acc_a, [row0, out_col + n0])

            col = group * O_LORA
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="quant",
                deps=[proj_a_tid],
                allow_early_resolve=True,
            ) as quant_tid:
                for qt in pl.pipeline(0, t_dim, QUANT_TOKEN_TILE, stage=2):
                    values = o_r_pad[
                        qt : qt + QUANT_TOKEN_TILE, col : col + O_LORA
                    ]
                    row_max = pl.reshape(
                        pl.row_max(pl.abs(values)), [1, QUANT_TOKEN_TILE]
                    )
                    amax = pl.maximum(
                        pl.full(
                            [1, QUANT_TOKEN_TILE],
                            dtype=pl.FP32,
                            value=INT8_AMAX_EPS,
                        ),
                        row_max,
                    )
                    scale = pl.div(
                        pl.full(
                            [1, QUANT_TOKEN_TILE],
                            dtype=pl.FP32,
                            value=INT8_SCALE_MAX,
                        ),
                        amax,
                    )
                    act_scale_dq[
                        group : group + 1, qt : qt + QUANT_TOKEN_TILE
                    ] = pl.recip(scale)
                    scaled = pl.row_expand_mul(
                        values, pl.reshape(scale, [QUANT_TOKEN_TILE, 1])
                    )
                    as_i32 = pl.cast(
                        scaled, target_type=pl.INT32, mode="rint"
                    )
                    as_fp16 = pl.cast(
                        as_i32, target_type=pl.FP16, mode="round"
                    )
                    o_r_i8_pad[
                        qt : qt + QUANT_TOKEN_TILE, col : col + O_LORA
                    ] = pl.cast(as_fp16, target_type=pl.INT8, mode="trunc")
                for zero_t in pl.range(t_dim, T_PAD, QUANT_TOKEN_TILE):
                    o_r_i8_pad[
                        zero_t : zero_t + QUANT_TOKEN_TILE, col : col + O_LORA
                    ] = pl.cast(
                        pl.full(
                            [QUANT_TOKEN_TILE, O_LORA],
                            dtype=pl.FP16,
                            value=0.0,
                        ),
                        target_type=pl.INT8,
                        mode="trunc",
                    )

            with pl.spmd(
                D // PROJ_B_D_TILE,
                name_hint="proj_b_mm",
                deps=[quant_tid],
                allow_early_resolve=True,
            ) as proj_b_tid:
                d_chunk = pl.tile.get_block_idx()
                d0 = d_chunk * PROJ_B_D_TILE
                for n_frag in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + n_frag * PROJ_B_MM_N_TILE
                    acc_b = pl.matmul(
                        o_r_i8_pad[:, col : col + B_K_TILE],
                        wo_b[
                            n0 : n0 + PROJ_B_MM_N_TILE,
                            col : col + B_K_TILE,
                        ],
                        b_trans=True,
                        out_dtype=pl.INT32,
                    )
                    for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                        k0 = col + kb * B_K_TILE
                        acc_b = pl.matmul_acc(
                            acc_b,
                            o_r_i8_pad[:, k0 : k0 + B_K_TILE],
                            wo_b[
                                n0 : n0 + PROJ_B_MM_N_TILE,
                                k0 : k0 + B_K_TILE,
                            ],
                            b_trans=True,
                        )
                    partials[
                        0:MM_T_TILE,
                        group * D + n0 : group * D + n0 + PROJ_B_MM_N_TILE,
                    ] = acc_b
            proj_b_tids[group] = proj_b_tid

    with pl.spmd(
        act_t_blks * PROJ_B_ACT_N_REGS,
        name_hint="proj_b_act",
        deps=[proj_b_tids[index] for index in range(O_GROUPS)],
        allow_early_resolve=True,
    ) as act_tid:
        item = pl.tile.get_block_idx()
        token_block = item // PROJ_B_ACT_N_REGS
        n_register = item - token_block * PROJ_B_ACT_N_REGS
        n0 = n_register * PROJ_B_ACT_N_TILE
        token0 = token_block * PROJ_B_ACT_TASK_T_TILE
        weight_scale = pl.reshape(
            wo_b_scale[n0 : n0 + PROJ_B_ACT_N_TILE],
            [1, PROJ_B_ACT_N_TILE],
        )
        for token in pl.range(
            token0,
            token0 + PROJ_B_ACT_TASK_T_TILE,
            PROJ_B_ACT_T_TILE,
        ):
            acc = pl.full(
                [PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE],
                dtype=pl.FP32,
                value=0.0,
            )
            for group in pl.pipeline(O_GROUPS, stage=2):
                partial = partials[
                    token : token + PROJ_B_ACT_T_TILE,
                    group * D + n0 : group * D + n0 + PROJ_B_ACT_N_TILE,
                ]
                dequant_scale = pl.reshape(
                    act_scale_dq[
                        group : group + 1,
                        token : token + PROJ_B_ACT_T_TILE,
                    ],
                    [PROJ_B_ACT_T_TILE, 1],
                )
                acc = pl.add(
                    acc,
                    pl.row_expand_mul(
                        pl.cast(partial, target_type=pl.FP32, mode="none"),
                        dequant_scale,
                    ),
                )
            attn_out[
                token : token + PROJ_B_ACT_T_TILE,
                n0 : n0 + PROJ_B_ACT_N_TILE,
            ] = pl.cast(
                pl.col_expand_mul(acc, weight_scale),
                target_type=pl.BF16,
                mode="rint",
            )
    return act_tid


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    hca_pages: pl.Tensor[[HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    cmp_dep: pl.Scalar[pl.TASK_ID],
):
    """Stream exact HCA work through bounded query-local scratch scopes."""
    t_dim = pl.tensor.dim(q, 0)
    chunk_count = (
        t_dim + HCA_QUERY_CHUNK_T - 1
    ) // HCA_QUERY_CHUNK_T
    o_packed = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)

    for chunk in pl.range(chunk_count):
        query_base = pl.cast(chunk * HCA_QUERY_CHUNK_T, pl.INDEX)
        query_end = pl.min(query_base + HCA_QUERY_CHUNK_T, t_dim)
        query_count = query_end - query_base
        work_begin_i32 = pl.read(hca_query_work_offsets, [query_base])
        work_end = pl.read(hca_query_work_offsets, [query_end])
        work_base = pl.cast(work_begin_i32, pl.INDEX)
        work_count = pl.cast(work_end - work_begin_i32, pl.INDEX)
        with pl.scope():
            o_packed = sparse_attn_hca_chunk_heads(
                q,
                ori_kv,
                current_kv,
                swa_sources,
                cmp_kv,
                query_request_ids,
                hca_pages,
                hca_page_offsets,
                hca_windows,
                request_epochs,
                hca_query_work_offsets,
                hca_work_query_ids,
                hca_work_row_begin,
                hca_work_valid_rows,
                attn_sink,
                freqs_cos,
                freqs_sin,
                o_packed,
                query_base,
                query_count,
                work_base,
                work_count,
                cmp_dep,
            )

    heads_anchor = pl.create_tensor([1, 16], dtype=pl.BF16)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="hca_heads_join",
    ) as heads_done:
        heads_anchor[0:1, 0:16] = o_packed[0:1, 0:16]

    projection_done = sparse_attn_hca_local_o_proj(
        o_packed,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        heads_done,
    )
    return projection_done


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    hca_pages: pl.Tensor[[HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    q.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    current_kv.bind_dynamic(0, T_DYN)
    swa_sources.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    query_request_ids.bind_dynamic(0, T_DYN)
    hca_pages.bind_dynamic(0, HCA_PAGES_DYN)
    hca_page_offsets.bind_dynamic(0, HCA_REQUEST_OFFSETS_DYN)
    hca_windows.bind_dynamic(0, B_DYN)
    request_epochs.bind_dynamic(0, B_DYN)
    hca_query_work_offsets.bind_dynamic(0, HCA_QUERY_OFFSETS_DYN)
    hca_work_query_ids.bind_dynamic(0, HCA_WORK_DYN)
    hca_work_row_begin.bind_dynamic(0, HCA_WORK_DYN)
    hca_work_valid_rows.bind_dynamic(0, HCA_WORK_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    attn_out.bind_dynamic(0, T_DYN)
    dep = pl.system.task_dummy(deps=[])
    sparse_attn_hca(
        q,
        ori_kv,
        current_kv,
        swa_sources,
        cmp_kv,
        query_request_ids,
        hca_pages,
        hca_page_offsets,
        hca_windows,
        request_epochs,
        hca_query_work_offsets,
        hca_work_query_ids,
        hca_work_row_begin,
        hca_work_valid_rows,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        dep,
    )
    return attn_out


def golden_sparse_attn(tensors):
    """Torch reference using the same raw-plus-packed online merge order."""
    import torch

    q = tensors["q"].float()
    ori = tensors["ori_kv"].float().reshape(-1, HEAD_DIM)
    current = tensors["current_kv"].float()
    cmp_kv = tensors["cmp_kv"].float().reshape(-1, HEAD_DIM)
    sources = tensors["swa_sources"]
    request_ids = tensors["query_request_ids"]
    pages = tensors["hca_pages"]
    page_offsets = tensors["hca_page_offsets"]
    windows = tensors["hca_windows"]
    epochs = tensors["request_epochs"]
    work_offsets = tensors["hca_query_work_offsets"]
    work_rows = tensors["hca_work_row_begin"]
    work_valid = tensors["hca_work_valid_rows"]
    sink = tensors["attn_sink"].float()
    tokens = q.shape[0]
    o = torch.zeros(tokens, H, HEAD_DIM)

    def partial(query, kv_rows, valid):
        kv_tile = torch.stack(kv_rows).float()
        valid_t = torch.tensor(valid, dtype=torch.bool)
        scores = (q[query] @ kv_tile.T) * SOFTMAX_SCALE
        scores = scores.masked_fill(~valid_t.unsqueeze(0), NEG_INF)
        mi = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - mi).masked_fill(
            ~valid_t.unsqueeze(0), 0.0
        )
        li = exp_scores.sum(dim=-1, keepdim=True)
        oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
        return mi, li, oi

    for query in range(tokens):
        raw_rows = []
        raw_valid = []
        for source in sources[query].tolist():
            if source >= 0:
                if 0 <= int(source) < int(ori.shape[0]):
                    raw_rows.append(ori[source])
                    raw_valid.append(True)
                else:
                    raw_rows.append(torch.zeros(HEAD_DIM))
                    raw_valid.append(False)
            elif source <= SWA_SOURCE_OVERLAY_BASE:
                overlay = SWA_SOURCE_OVERLAY_BASE - int(source)
                same_request = (
                    0 <= overlay < tokens
                    and int(request_ids[overlay]) == int(request_ids[query])
                )
                if same_request and overlay <= query:
                    raw_rows.append(current[overlay])
                    raw_valid.append(True)
                else:
                    # The metadata contract is causal: a malformed/future
                    # overlay is treated as an invalid row rather than being
                    # allowed to leak a later query into this one.
                    raw_rows.append(torch.zeros(HEAD_DIM))
                    raw_valid.append(False)
            else:
                raw_rows.append(torch.zeros(HEAD_DIM))
                raw_valid.append(False)
        mi, li, oi = partial(query, raw_rows, raw_valid)
        request = int(request_ids[query])
        for work in range(int(work_offsets[query]), int(work_offsets[query + 1])):
            row_begin = int(work_rows[work])
            valid_rows = int(work_valid[work])
            kv_rows = []
            valid = []
            for lane in range(ATTN_K_TILE):
                if lane >= valid_rows:
                    kv_rows.append(torch.zeros(HEAD_DIM))
                    valid.append(False)
                    continue
                row = row_begin + lane
                begin = int(page_offsets[request])
                end = int(page_offsets[request + 1])
                total = end - begin
                row_valid = False
                row_value = torch.zeros(HEAD_DIM)
                if total > 0 and 0 <= begin <= end <= int(pages.shape[0]):
                    valid_begin = int(windows[request, 0])
                    head = int(windows[request, 2])
                    rel = row // BLOCK_SIZE - valid_begin // BLOCK_SIZE
                    if rel >= 0 and rel < total:
                        entry = begin + (head + rel) % total
                        if 0 <= entry < int(pages.shape[0]):
                            page_epoch = int(pages[entry, 1])
                            physical = int(pages[entry, 0])
                            source_row = physical * BLOCK_SIZE + row % BLOCK_SIZE
                            if (
                                physical >= 0
                                and page_epoch == int(epochs[request])
                                and 0 <= source_row < int(cmp_kv.shape[0])
                            ):
                                row_value = cmp_kv[source_row]
                                row_valid = True
                kv_rows.append(row_value)
                valid.append(row_valid)
            cur_mi, cur_li, cur_oi = partial(query, kv_rows, valid)
            mi_new = torch.maximum(mi, cur_mi)
            alpha = torch.exp(mi - mi_new)
            beta = torch.exp(cur_mi - mi_new)
            li = alpha * li + beta * cur_li
            oi = alpha * oi + beta * cur_oi
            mi = mi_new
        o[query] = oi / (li + torch.exp(sink.unsqueeze(-1) - mi))

    cos = tensors["freqs_cos"].float()[:, :HALF_ROPE].unsqueeze(1)
    sin = tensors["freqs_sin"].float()[:, :HALF_ROPE].unsqueeze(1)
    pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    even, odd = pair[..., 0], pair[..., 1]
    inv_even = (even * cos + odd * sin).to(torch.bfloat16).float()
    inv_odd = (odd * cos - even * sin).to(torch.bfloat16).float()
    o = torch.cat(
        [
            o[..., :NOPE_DIM],
            torch.stack([inv_even, inv_odd], dim=-1).flatten(-2),
        ],
        dim=-1,
    ).to(torch.bfloat16)
    wo_a = tensors["wo_a"].float()
    projected = torch.einsum(
        "tgd,grd->tgr",
        o.float().reshape(tokens, O_GROUPS, O_GROUP_IN),
        wo_a,
    )
    projected = projected.reshape(tokens, O_GROUPS, O_LORA)
    amax = projected.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = INT8_SCALE_MAX / amax
    quant = (
        torch.round(projected * scale)
        .to(torch.int32)
        .to(torch.float16)
        .to(torch.int8)
    )
    weight = tensors["wo_b"].reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(tokens, D)
    for group in range(O_GROUPS):
        out += (
            quant[:, group].to(torch.int32)
            @ weight[:, group].to(torch.int32).T
        ).float() / scale[:, group]
    out *= tensors["wo_b_scale"].float().unsqueeze(0)
    tensors["attn_out"][:] = out.to(torch.bfloat16)


_CASES = {
    "zero_compressed": (0, 0),
    "one_shard_tail": (1, 0),
    "one_full_shard": (128, 0),
    "two_shards": (256, 0),
    "cross_page_shard": (128, 1),
    "heterogeneous_shards": (-1, 0),
    "sixty_four_shards": (8192, 0),
    "page_permutation": (256, 0),
    "causal_overlay": (0, 0),
    "stale_page": (128, 0),
}


def build_tensor_specs(case="one_full_shard", batch=4):
    """Build exact-work fixtures for a runtime batch of four-request groups."""
    import torch
    from golden import TensorSpec
    from utils import quant_w_per_channel

    if case not in _CASES:
        raise ValueError(f"unknown HCA packed case: {case!r}")
    if batch < 4 or batch > B or batch % 4:
        raise ValueError(f"batch must be a multiple of 4 in [4, {B}], got {batch}")
    tokens = batch * S
    if case == "heterogeneous_shards":
        pattern = [0, 1, 256, 8192]
        rows_by_request = [pattern[index % len(pattern)] for index in range(batch)]
    elif case == "sixty_four_shards":
        pattern = [8192, 0, 0, 0]
        rows_by_request = [pattern[index % len(pattern)] for index in range(batch)]
    else:
        rows, _ = _CASES[case]
        rows_by_request = [rows] * batch
    begin_offset = _CASES[case][1]
    page_ids = []
    page_offsets = [0]
    windows = []
    page_heads = []
    page_epochs = []
    physical_page = 0
    for request, rows in enumerate(rows_by_request):
        pages = max(1, (begin_offset + rows + BLOCK_SIZE - 1) // BLOCK_SIZE)
        ids = list(range(physical_page, physical_page + pages))
        physical_page += pages
        head = 0
        if case == "page_permutation" and pages >= 2:
            # Rotate the logical page table while selecting the inverse head;
            # logical row 0 still resolves to the original physical page 0.
            ids = ids[1:] + ids[:1]
            head = pages - 1
        epoch = 6 if case == "stale_page" else 7
        page_ids.extend((page, epoch) for page in ids)
        page_offsets.append(len(page_ids))
        windows.append((begin_offset, begin_offset + rows, head))
        page_heads.append(head)
        page_epochs.append(epoch)
    work_query_ids = []
    work_rows = []
    work_valid_rows = []
    work_offsets = [0]
    for query in range(tokens):
        request = query // S
        rows = rows_by_request[request]
        row = begin_offset
        while row < begin_offset + rows:
            work_query_ids.append(query)
            work_rows.append(row)
            work_valid_rows.append(min(ATTN_K_TILE, begin_offset + rows - row))
            row += ATTN_K_TILE
        work_offsets.append(len(work_query_ids))

    q = torch.rand(tokens, H, HEAD_DIM) - 0.5
    ori_kv = torch.rand(batch * (WIN // BLOCK_SIZE), BLOCK_SIZE, 1, HEAD_DIM) - 0.5
    current_kv = torch.rand(tokens, HEAD_DIM) - 0.5
    swa_sources = torch.empty(tokens, WIN, dtype=torch.int32)
    for query in range(tokens):
        request = query // S
        swa_sources[query] = torch.arange(WIN, dtype=torch.int32) + request * WIN
        local = query % S
        for overlay_local in range(local + 1):
            overlay_query = request * S + overlay_local
            swa_sources[query, WIN - local - 1 + overlay_local] = (
                SWA_SOURCE_OVERLAY_BASE - overlay_query
            )
    if case == "causal_overlay":
        # Make the causal boundary numerically unmistakable: q0 can only see
        # itself, while q1 sees q0 and itself.  No future q1 source is present
        # in q0's row.
        swa_sources.fill_(SWA_SOURCE_INVALID)
        for request in range(batch):
            first = request * S
            for local in range(S):
                for overlay_local in range(local + 1):
                    swa_sources[first + local, WIN - local - 1 + overlay_local] = (
                        SWA_SOURCE_OVERLAY_BASE - (first + overlay_local)
                    )
        current_kv.zero_()
        for request in range(batch):
            current_kv[request * S] = 4.0
            for local in range(1, S):
                current_kv[request * S + local] = -4.0
    cmp_kv = torch.rand(max(physical_page, 1), BLOCK_SIZE, 1, HEAD_DIM) - 0.5
    angles = torch.arange(tokens * HALF_ROPE).reshape(tokens, HALF_ROPE) * 1e-3
    cos_half, sin_half = torch.cos(angles), torch.sin(angles)
    wo_a = (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN**0.5)
    wo_b_bf16 = (
        (torch.rand(D, O_GROUPS * O_LORA) - 0.5)
        / ((O_GROUPS * O_LORA) ** 0.5)
    ).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)
    inputs = {
        "q": q.to(torch.bfloat16),
        "ori_kv": ori_kv.to(torch.bfloat16),
        "current_kv": current_kv.to(torch.bfloat16),
        "swa_sources": swa_sources,
        "cmp_kv": cmp_kv.to(torch.bfloat16),
        "query_request_ids": torch.arange(batch, dtype=torch.int32).repeat_interleave(S),
        "hca_pages": torch.tensor(page_ids, dtype=torch.int32),
        "hca_page_offsets": torch.tensor(page_offsets, dtype=torch.int32),
        "hca_windows": torch.tensor(windows, dtype=torch.int32),
        "request_epochs": torch.full((batch,), 7, dtype=torch.int32),
        "hca_query_work_offsets": torch.tensor(work_offsets, dtype=torch.int32),
        "hca_work_query_ids": torch.tensor(work_query_ids, dtype=torch.int32),
        "hca_work_row_begin": torch.tensor(work_rows, dtype=torch.int32),
        "hca_work_valid_rows": torch.tensor(work_valid_rows, dtype=torch.int32),
        "attn_sink": torch.zeros(H),
        "freqs_cos": torch.cat([cos_half, cos_half], dim=-1).to(torch.bfloat16),
        "freqs_sin": torch.cat([sin_half, sin_half], dim=-1).to(torch.bfloat16),
        "wo_a": wo_a.to(torch.bfloat16),
        "wo_b": wo_b_i8,
        "wo_b_scale": wo_b_scale,
    }
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in inputs.items()
    ]
    specs.append(
        TensorSpec("attn_out", [tokens, D], torch.bfloat16, is_output=True)
    )
    return specs


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=list(_CASES), default="one_full_shard")
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()
    if args.batch < 4 or args.batch > B or args.batch % 4:
        parser.error(f"--batch must be a multiple of 4 in [4, {B}], got {args.batch}")
    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(args.case, batch=args.batch),
        golden_fn=golden_sparse_attn,
        compile_cfg={"dump_passes": args.dump_passes},
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_dep_gen": args.enable_dep_gen,
        },
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
