# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact fixed-width primitives for the decode CSA indexer Top-K forest."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    CSA_CANDIDATES_PER_LEAF,
    CSA_MAX_CANDIDATES,
    CSA_MAX_LEAVES_PER_QUERY,
    CSA_MAX_QUERIES,
    CSA_MAX_TOPK_TASKS,
    CSA_PAIR_WIDTH,
    CSA_TOPK,
    CSA_TOPK_INVALID_TASK_SLOT,
    CSA_TOPK_READY_FRONTIER_W,
    FLASH as M,
    FP32_NEG_INF,
)
from decode_metadata import (
    PHASE_D_LEAF_BEGIN,
    PHASE_D_LEAF_FIELDS,
    PHASE_D_LEAF_QUERY,
    PHASE_D_LEAF_VALID,
    PHASE_D_PAIR_CREDIT_SLOT,
    PHASE_D_PAIR_FIELDS,
    PHASE_D_PAIR_LEFT_LEAF,
    PHASE_D_PAIR_LEFT_SLOT,
    PHASE_D_PAIR_OUTPUT_SLOT,
    PHASE_D_PAIR_RIGHT_LEAF,
    PHASE_D_PAIR_RIGHT_SLOT,
    PHASE_D_ROOT_DEPENDENCY_SLOT,
    PHASE_D_ROOT_FIELDS,
    PHASE_D_ROOT_SLOT,
    PHASE_D_SINGLETON_CREDIT_SLOT,
    PHASE_D_SINGLETON_FIELDS,
    PHASE_D_SINGLETON_LEAF,
    PHASE_D_SINGLETON_SLOT,
    PHASE_D_UPPER_FIELDS,
    PHASE_D_UPPER_LEFT_SLOT,
    PHASE_D_UPPER_OUTPUT_SLOT,
    PHASE_D_UPPER_RIGHT_SLOT,
)


CSA_TOPK_NODE_LEAF = 0
CSA_TOPK_NODE_MERGE = 1
POC_LEAVES = 4
POC_NODES = 2 * POC_LEAVES - 1
POC_LEVEL1_BASE = POC_LEAVES
POC_ROOT_SLOT = POC_NODES - 1
POC_READY_FRONTIER = 1

LEAF_DYN = pl.dynamic("LEAF_DYN")
QUERY_DYN = pl.dynamic("QUERY_DYN")
ARENA_DYN = pl.dynamic("ARENA_DYN")
PAIR_GROUP_DYN = pl.dynamic("PAIR_GROUP_DYN")
SINGLETON_DYN = pl.dynamic("SINGLETON_DYN")
UPPER_MERGE_DYN = pl.dynamic("UPPER_MERGE_DYN")
REQUEST_DYN = pl.dynamic("REQUEST_DYN")
REQUEST_OFFSET_DYN = pl.dynamic("REQUEST_OFFSET_DYN")
PAGE_DYN = pl.dynamic("PAGE_DYN")
IDX_ROW_DYN = pl.dynamic("IDX_ROW_DYN")

IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
CSA_PAGE_ROWS = BLOCK_SIZE
CSA_SCORE_TILE = 64
CSA_LEAF_SCORE_FRAGMENTS = CSA_CANDIDATES_PER_LEAF // CSA_SCORE_TILE
CSA_MAX_CANDIDATES_FP32 = 262144.0
CSA_TOPK_PAIR_GRID_W = 16
CSA_TOPK_MAX_PAIR_GROUPS = CSA_MAX_QUERIES * CSA_MAX_LEAVES_PER_QUERY // 2
CSA_TOPK_MAX_PAIR_WAVES = (
    CSA_TOPK_MAX_PAIR_GROUPS + CSA_TOPK_PAIR_GRID_W - 1
) // CSA_TOPK_PAIR_GRID_W
assert CSA_MAX_CANDIDATES_FP32 == CSA_MAX_CANDIDATES


@pl.jit.incore
def select_2k_top512(
    leaf_scores: pl.Tensor[[LEAF_DYN, CSA_CANDIDATES_PER_LEAF], pl.FP32],
    leaf_valid_candidates: pl.Tensor[[LEAF_DYN], pl.INT32],
    leaf_logical_begins: pl.Tensor[[LEAF_DYN], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    leaf_id: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32]:
    valid_candidates = pl.cast(pl.read(leaf_valid_candidates, [leaf_id]), pl.INDEX)
    logical_begin = pl.read(leaf_logical_begins, [leaf_id])
    score_row = leaf_scores[leaf_id : leaf_id + 1, :]
    score_valid = pl.fillpad(
        pl.set_validshape(score_row, 1, valid_candidates),
        pad_value=pl.PadValue.min,
    )
    score_valid = pl.maximum(score_valid, FP32_NEG_INF)
    idx_init = pl.arange(
        pl.cast(pl.cast(logical_begin, pl.INDEX), pl.UINT32),
        [1, CSA_CANDIDATES_PER_LEAF],
        dtype=pl.UINT32,
    )
    idx_init = pl.fillpad(
        pl.set_validshape(idx_init, 1, valid_candidates),
        pad_value=pl.PadValue.max,
    )
    sorted_pairs = pl.sort32(score_valid, idx_init)
    sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
    sorted_pairs = pl.mrgsort(sorted_pairs, block_len=256)
    sorted_pairs = pl.mrgsort(sorted_pairs, block_len=1024)
    pair_arena[output_slot : output_slot + 1, :] = sorted_pairs[:, 0:CSA_PAIR_WIDTH]
    return pair_arena


@pl.jit.inline
def score_select_2k_top512(
    query_vectors: pl.Tensor[[QUERY_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    query_scales: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[QUERY_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[REQUEST_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    leaf_id: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32]:
    """Score one ragged 2K leaf and immediately retain its exact Top-512."""
    query = pl.cast(
        pl.read(leaf_descriptors, [leaf_id, PHASE_D_LEAF_QUERY]),
        pl.INDEX,
    )
    request = pl.cast(pl.read(query_request_ids, [query]), pl.INDEX)
    logical_begin = pl.read(
        leaf_descriptors,
        [leaf_id, PHASE_D_LEAF_BEGIN],
    )
    valid_candidates = pl.cast(
        pl.read(leaf_descriptors, [leaf_id, PHASE_D_LEAF_VALID]),
        pl.INDEX,
    )
    request_epoch = pl.read(request_epochs, [request])
    page_begin = pl.read(idx_page_offsets, [request])
    page_end = pl.read(idx_page_offsets, [request + 1])
    page_total = page_end - page_begin
    valid_begin = pl.read(idx_windows, [request, 0])
    valid_end = pl.read(idx_windows, [request, 1])
    head = pl.read(idx_windows, [request, 2])
    logical_page_base = valid_begin // CSA_PAGE_ROWS
    cache_rows = pl.tensor.dim(idx_kv_cache_flat, 0)
    block_count = cache_rows // CSA_PAGE_ROWS
    page_count = pl.tensor.dim(idx_pages, 0)
    query_vector = query_vectors[query : query + 1, :, :]
    query_vector = pl.reshape(query_vector, [IDX_N_HEADS, IDX_HEAD_DIM])
    query_scale = query_scales[query : query + 1, :]
    query_weight = query_weights[query : query + 1, :]
    # The first fragment seeds the rolling Top-512.  The initial carry is never
    # observed and therefore needs no invalid-pair sort, saving enough Vec space
    # to keep the 64-candidate score tile within the A2/A3 limit.
    running_pairs = pl.create_tensor([1, CSA_PAIR_WIDTH], dtype=pl.FP32)
    fragment_count = (
        valid_candidates + CSA_SCORE_TILE - 1
    ) // CSA_SCORE_TILE

    for fragment, (running_pairs_iter,) in pl.range(
        fragment_count,
        init_values=(running_pairs,),
    ):
        fragment_begin = fragment * CSA_SCORE_TILE
        fragment_valid = pl.min(
            CSA_SCORE_TILE,
            valid_candidates - fragment_begin,
        )
        # ``gather_row`` keeps the paged loads in the mat operand and avoids
        # assembling differently laid-out vector subviews across the dynamic
        # fragment loop.  Scales remain a vector accumulator because they feed
        # a row-wise FP32 multiply rather than a mat operand.
        kv_tile = pl.create_l1([CSA_SCORE_TILE, IDX_HEAD_DIM], pl.INT8)
        scale_tile = pl.create_tensor([CSA_SCORE_TILE, 1], dtype=pl.FP32)
        for lane in pl.range(CSA_SCORE_TILE):
            local_candidate = fragment_begin + lane
            logical_candidate = logical_begin + local_candidate
            source_valid = pl.cast(0, pl.INT32)
            source_row = pl.cast(0, pl.INDEX)
            if lane < fragment_valid:
                if logical_candidate >= valid_begin:
                    if logical_candidate < valid_end:
                        relative_page = (
                            logical_candidate // CSA_PAGE_ROWS
                            - logical_page_base
                        )
                        if page_total > 0:
                            if relative_page >= 0:
                                if relative_page < page_total:
                                    page_index = (head + relative_page) % page_total
                                    page_entry = page_begin + page_index
                                    if page_entry >= 0:
                                        if page_entry < page_count:
                                            physical_page = pl.read(
                                                idx_pages,
                                                [page_entry, 0],
                                            )
                                            page_epoch = pl.read(
                                                idx_pages,
                                                [page_entry, 1],
                                            )
                                            if physical_page >= 0:
                                                if page_epoch == request_epoch:
                                                    if physical_page < block_count:
                                                        source_valid = pl.cast(
                                                            1,
                                                            pl.INT32,
                                                        )
                                                        source_row = pl.cast(
                                                            physical_page
                                                            * CSA_PAGE_ROWS
                                                            + logical_candidate
                                                            % CSA_PAGE_ROWS,
                                                            pl.INDEX,
                                                        )
            if source_valid == 1:
                kv_tile = pl.gather_row(
                    kv_tile,
                    idx_kv_cache_flat,
                    [lane, 0],
                    [source_row, 0],
                    [1, IDX_HEAD_DIM],
                )
            else:
                kv_tile = pl.gather_row(
                    kv_tile,
                    idx_kv_cache_flat,
                    [lane, 0],
                    [0, 0],
                    [1, IDX_HEAD_DIM],
                )
            # Phase-D metadata validates every active leaf/page/epoch before it
            # is submitted.  Invalid tail lanes use row zero here and are then
            # removed by ``fragment_valid`` below.
            scale_value = pl.read(idx_kv_scale_flat, [source_row, 0])
            pl.write(scale_tile, [lane, 0], scale_value)
        score_i32 = pl.matmul(
            kv_tile,
            query_vector,
            b_trans=True,
            out_dtype=pl.INT32,
        )
        score_fp32 = pl.cast(score_i32, target_type=pl.FP32, mode="none")
        score_fp32 = pl.col_expand_mul(score_fp32, query_scale)
        score_fp32 = pl.maximum(score_fp32, 0.0)
        score_fp32 = pl.col_expand_mul(score_fp32, query_weight)
        score_fragment = pl.reshape(
            pl.mul(pl.row_sum(score_fp32), scale_tile),
            [1, CSA_SCORE_TILE],
        )
        score_padded = pl.concat(
            score_fragment,
            pl.full(
                [1, CSA_TOPK - CSA_SCORE_TILE],
                dtype=pl.FP32,
                value=FP32_NEG_INF,
            ),
        )
        score_padded = pl.fillpad(
            pl.set_validshape(score_padded, 1, fragment_valid),
            pad_value=pl.PadValue.min,
        )
        score_padded = pl.maximum(score_padded, FP32_NEG_INF)
        idx_init = pl.arange(
            pl.cast(
                pl.cast(logical_begin + fragment_begin, pl.INDEX),
                pl.UINT32,
            ),
            [1, CSA_TOPK],
            dtype=pl.UINT32,
        )
        idx_init = pl.fillpad(
            pl.set_validshape(idx_init, 1, fragment_valid),
            pad_value=pl.PadValue.max,
        )
        fragment_pairs = pl.sort32(score_padded, idx_init)
        fragment_pairs = pl.mrgsort(fragment_pairs, block_len=64)
        fragment_pairs = pl.mrgsort(fragment_pairs, block_len=256)
        if fragment == 0:
            merged_top = pl.maximum(fragment_pairs, FP32_NEG_INF)
        else:
            combined_pairs = pl.concat(running_pairs_iter, fragment_pairs)
            merged_pairs = pl.mrgsort(combined_pairs, block_len=512)
            merged_top = pl.maximum(
                merged_pairs[:, 0:CSA_PAIR_WIDTH],
                FP32_NEG_INF,
            )
        running_pairs_result = pl.yield_(merged_top)

    pair_arena[output_slot : output_slot + 1, :] = running_pairs_result[
        :,
        0:CSA_PAIR_WIDTH,
    ]
    return pair_arena


@pl.jit.incore
def merge2_top512(
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    left_slot: pl.Scalar[pl.INDEX],
    right_slot: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32]:
    merged = pl.mrgsort(
        pair_arena[left_slot : left_slot + 1, :],
        pair_arena[right_slot : right_slot + 1, :],
    )
    pair_arena[output_slot : output_slot + 1, :] = merged[:, 0:CSA_PAIR_WIDTH]
    return pair_arena


@pl.jit.incore
def init_topk_roots(
    topk_scores: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
) -> tuple[
    pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
]:
    query_count = pl.tensor.dim(topk_scores, 0)
    for query in pl.range(query_count):
        topk_scores[query : query + 1, :] = pl.full(
            [1, CSA_TOPK], dtype=pl.FP32, value=FP32_NEG_INF
        )
        topk_indices[query : query + 1, :] = pl.full(
            [1, CSA_TOPK], dtype=pl.INT32, value=-1
        )
    return topk_scores, topk_indices


@pl.jit.incore
def materialize_topk_root(
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    root_slot_raw: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
]:
    if root_slot_raw >= 0:
        root_slot = pl.cast(root_slot_raw, pl.INDEX)
        root_pairs = pair_arena[root_slot : root_slot + 1, :]
        topk_scores[query : query + 1, :] = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P0101,
        )
        root_indices = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
        index_fp32 = pl.cast(root_indices, target_type=pl.FP32, mode="none")
        invalid_flag = pl.cast(
            pl.div(
                pl.minimum(index_fp32, CSA_MAX_CANDIDATES_FP32),
                CSA_MAX_CANDIDATES_FP32,
            ),
            target_type=pl.INT32,
            mode="trunc",
        )
        root_indices = pl.sub(
            pl.sub(root_indices, pl.mul(root_indices, invalid_flag)),
            invalid_flag,
        )
        topk_indices[query : query + 1, :] = root_indices
    return topk_scores, topk_indices


@pl.jit.inline
def active_topk_forest(
    leaf_scores: pl.Tensor[[LEAF_DYN, CSA_CANDIDATES_PER_LEAF], pl.FP32],
    leaf_valid_candidates: pl.Tensor[[LEAF_DYN], pl.INT32],
    leaf_logical_begins: pl.Tensor[[LEAF_DYN], pl.INT32],
    pair_left_leaf_ids: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_right_leaf_ids: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_left_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_right_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_output_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_credit_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    singleton_leaf_ids: pl.Tensor[[SINGLETON_DYN], pl.INT32],
    singleton_slots: pl.Tensor[[SINGLETON_DYN], pl.INT32],
    singleton_credit_slots: pl.Tensor[[SINGLETON_DYN], pl.INT32],
    upper_left_slots: pl.Tensor[[UPPER_MERGE_DYN], pl.INT32],
    upper_right_slots: pl.Tensor[[UPPER_MERGE_DYN], pl.INT32],
    upper_output_slots: pl.Tensor[[UPPER_MERGE_DYN], pl.INT32],
    root_slots: pl.Tensor[[QUERY_DYN], pl.INT32],
    root_dependency_slots: pl.Tensor[[QUERY_DYN], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
    completion: pl.Array[1, pl.TASK_ID],
):
    pair_group_count = pl.tensor.dim(pair_left_leaf_ids, 0)
    singleton_count = pl.tensor.dim(singleton_leaf_ids, 0)
    upper_merge_count = pl.tensor.dim(upper_left_slots, 0)
    query_count = pl.tensor.dim(root_slots, 0)
    node_tids = pl.array.create(CSA_MAX_TOPK_TASKS + 1, pl.TASK_ID)
    root_tids = pl.array.create(CSA_MAX_QUERIES, pl.TASK_ID)

    with pl.manual_scope():
        with pl.spmd(1) as init_tid:
            _init_scores, _init_indices = init_topk_roots(topk_scores, topk_indices)

        for group, (node_tids_iter,) in pl.range(
            pair_group_count,
            init_values=(node_tids,),
        ):
            left_leaf_id = pl.cast(pl.read(pair_left_leaf_ids, [group]), pl.INDEX)
            right_leaf_id = pl.cast(pl.read(pair_right_leaf_ids, [group]), pl.INDEX)
            left_slot = pl.cast(pl.read(pair_left_slots, [group]), pl.INDEX)
            right_slot = pl.cast(pl.read(pair_right_slots, [group]), pl.INDEX)
            output_slot = pl.cast(pl.read(pair_output_slots, [group]), pl.INDEX)
            credit_slot = pl.cast(pl.read(pair_credit_slots, [group]), pl.INDEX)
            with pl.spmd(
                1,
                deps=[node_tids_iter[credit_slot]],
                allow_early_resolve=True,
            ) as left_tid:
                _left_pair_arena = select_2k_top512(
                    leaf_scores,
                    leaf_valid_candidates,
                    leaf_logical_begins,
                    pair_arena,
                    left_leaf_id,
                    left_slot,
                )
            node_tids_iter[left_slot] = left_tid
            with pl.spmd(
                1,
                deps=[node_tids_iter[credit_slot]],
                allow_early_resolve=True,
            ) as right_tid:
                _right_pair_arena = select_2k_top512(
                    leaf_scores,
                    leaf_valid_candidates,
                    leaf_logical_begins,
                    pair_arena,
                    right_leaf_id,
                    right_slot,
                )
            node_tids_iter[right_slot] = right_tid
            with pl.spmd(
                1,
                deps=[left_tid, right_tid],
                allow_early_resolve=True,
            ) as merge_tid:
                merged = pl.mrgsort(
                    pair_arena[left_slot : left_slot + 1, :],
                    pair_arena[right_slot : right_slot + 1, :],
                )
                pair_arena[output_slot : output_slot + 1, :] = merged[
                    :, 0:CSA_PAIR_WIDTH
                ]
            node_tids_iter[output_slot] = merge_tid
            node_tids_after_pairs = pl.yield_(node_tids_iter)

        for singleton, (node_tids_iter,) in pl.range(
            singleton_count,
            init_values=(node_tids_after_pairs,),
        ):
            leaf_id = pl.cast(pl.read(singleton_leaf_ids, [singleton]), pl.INDEX)
            output_slot = pl.cast(pl.read(singleton_slots, [singleton]), pl.INDEX)
            credit_slot = pl.cast(
                pl.read(singleton_credit_slots, [singleton]),
                pl.INDEX,
            )
            with pl.spmd(
                1,
                deps=[node_tids_iter[credit_slot]],
                allow_early_resolve=True,
            ) as singleton_tid:
                _singleton_pair_arena = select_2k_top512(
                    leaf_scores,
                    leaf_valid_candidates,
                    leaf_logical_begins,
                    pair_arena,
                    leaf_id,
                    output_slot,
                )
            node_tids_iter[output_slot] = singleton_tid
            node_tids_after_singletons = pl.yield_(node_tids_iter)

        for merge, (node_tids_iter,) in pl.range(
            upper_merge_count,
            init_values=(node_tids_after_singletons,),
        ):
            left_slot = pl.cast(pl.read(upper_left_slots, [merge]), pl.INDEX)
            right_slot = pl.cast(pl.read(upper_right_slots, [merge]), pl.INDEX)
            output_slot = pl.cast(pl.read(upper_output_slots, [merge]), pl.INDEX)
            with pl.spmd(
                1,
                deps=[node_tids_iter[left_slot], node_tids_iter[right_slot]],
                allow_early_resolve=True,
            ) as merge_tid:
                _upper_merge_arena = merge2_top512(
                    pair_arena,
                    left_slot,
                    right_slot,
                    output_slot,
                )
            node_tids_iter[output_slot] = merge_tid
            node_tids_after_merges = pl.yield_(node_tids_iter)

        for query, (root_tids_iter,) in pl.range(
            query_count,
            init_values=(root_tids,),
        ):
            root_slot_raw = pl.read(root_slots, [query])
            root_dependency_slot = pl.cast(
                pl.read(root_dependency_slots, [query]),
                pl.INDEX,
            )
            with pl.spmd(
                1,
                deps=[init_tid, node_tids_after_merges[root_dependency_slot]],
                allow_early_resolve=True,
            ) as root_tid:
                _root_scores, _root_indices = materialize_topk_root(
                    pair_arena,
                    topk_scores,
                    topk_indices,
                    query,
                    root_slot_raw,
                )
            root_tids_iter[query] = root_tid
            root_tids_after_roots = pl.yield_(root_tids_iter)

    completion[0] = pl.system.task_dummy(
        deps=[root_tids_after_roots[query] for query in range(CSA_MAX_QUERIES)]
    )
    return topk_scores, topk_indices


@pl.jit.inline
def active_score_topk_forest(
    query_vectors: pl.Tensor[[QUERY_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    query_scales: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[QUERY_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[REQUEST_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_descriptors: pl.Tensor[[PAIR_GROUP_DYN, PHASE_D_PAIR_FIELDS], pl.INT32],
    singleton_descriptors: pl.Tensor[[SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32],
    upper_descriptors: pl.Tensor[[UPPER_MERGE_DYN, PHASE_D_UPPER_FIELDS], pl.INT32],
    root_descriptors: pl.Tensor[[QUERY_DYN, PHASE_D_ROOT_FIELDS], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
    index_commit_dep: pl.Scalar[pl.TASK_ID],
    completion: pl.Array[1, pl.TASK_ID],
):
    """Submit only active score/select leaves and their exact merge forest."""
    pair_group_count = pl.tensor.dim(pair_descriptors, 0)
    singleton_count = pl.tensor.dim(singleton_descriptors, 0)
    upper_merge_count = pl.tensor.dim(upper_descriptors, 0)
    query_count = pl.tensor.dim(root_descriptors, 0)
    root_tids = pl.array.create(CSA_MAX_QUERIES, pl.TASK_ID)
    pair_wave_tids = pl.array.create(CSA_TOPK_MAX_PAIR_WAVES, pl.TASK_ID)
    pair_wave_count = (
        pair_group_count + CSA_TOPK_PAIR_GRID_W - 1
    ) // CSA_TOPK_PAIR_GRID_W
    singleton_grid_count = pl.max(singleton_count, 1)
    upper_grid_count = pl.max(query_count, 1)

    with pl.manual_scope():
        with pl.spmd(1, deps=[index_commit_dep]) as init_tid:
            _init_scores, _init_indices = init_topk_roots(
                topk_scores,
                topk_indices,
            )

        for pair_wave, (pair_wave_tids_iter,) in pl.range(
            pair_wave_count,
            init_values=(pair_wave_tids,),
        ):
            pair_begin = pair_wave * CSA_TOPK_PAIR_GRID_W
            pair_grid_count = pl.min(
                CSA_TOPK_PAIR_GRID_W,
                pair_group_count - pair_begin,
            )
            with pl.spmd(
                pair_grid_count,
                deps=[index_commit_dep],
                optimizations=[pl.cross_core_slot(slot_num=1)],
            ) as left_wave_tid:
                group = pair_begin + pl.tile.get_block_idx()
                left_leaf_id = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_LEFT_LEAF]),
                    pl.INDEX,
                )
                left_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_LEFT_SLOT]),
                    pl.INDEX,
                )
                _left_pair_arena = score_select_2k_top512(
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
                    pair_arena,
                    left_leaf_id,
                    left_slot,
                )
            with pl.spmd(
                pair_grid_count,
                deps=[index_commit_dep],
                optimizations=[pl.cross_core_slot(slot_num=1)],
            ) as right_wave_tid:
                group = pair_begin + pl.tile.get_block_idx()
                right_leaf_id = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_RIGHT_LEAF]),
                    pl.INDEX,
                )
                right_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_RIGHT_SLOT]),
                    pl.INDEX,
                )
                _right_pair_arena = score_select_2k_top512(
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
                    pair_arena,
                    right_leaf_id,
                    right_slot,
                )
            with pl.spmd(
                pair_grid_count,
                deps=[left_wave_tid, right_wave_tid],
            ) as pair_wave_tid:
                group = pair_begin + pl.tile.get_block_idx()
                left_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_LEFT_SLOT]),
                    pl.INDEX,
                )
                right_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_RIGHT_SLOT]),
                    pl.INDEX,
                )
                output_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_OUTPUT_SLOT]),
                    pl.INDEX,
                )
                merged = pl.mrgsort(
                    pair_arena[left_slot : left_slot + 1, :],
                    pair_arena[right_slot : right_slot + 1, :],
                )
                pair_arena[output_slot : output_slot + 1, :] = merged[
                    :, 0:CSA_PAIR_WIDTH
                ]
            pair_wave_tids_iter[pair_wave] = pair_wave_tid
            pair_wave_tids_after = pl.yield_(pair_wave_tids_iter)
        pair_tid = pl.system.task_dummy(deps=[pair_wave_tids_after])

        with pl.spmd(
            singleton_grid_count,
            deps=[index_commit_dep],
            optimizations=[pl.cross_core_slot(slot_num=1)],
        ) as singleton_tid:
            singleton = pl.tile.get_block_idx()
            if singleton < singleton_count:
                leaf_id = pl.cast(
                    pl.read(
                        singleton_descriptors,
                        [singleton, PHASE_D_SINGLETON_LEAF],
                    ),
                    pl.INDEX,
                )
                output_slot = pl.cast(
                    pl.read(
                        singleton_descriptors,
                        [singleton, PHASE_D_SINGLETON_SLOT],
                    ),
                    pl.INDEX,
                )
                _singleton_pair_arena = score_select_2k_top512(
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
                    pair_arena,
                    leaf_id,
                    output_slot,
                )

        with pl.spmd(
            upper_grid_count,
            deps=[pair_tid, singleton_tid],
        ) as upper_tid:
            query = pl.tile.get_block_idx()
            if query < query_count:
                root_slot_raw = pl.read(
                    root_descriptors,
                    [query, PHASE_D_ROOT_SLOT],
                )
                previous_root_slot = pl.cast(-1, pl.INT32)
                for previous_query in pl.range(query):
                    previous_root_slot = pl.max(
                        previous_root_slot,
                        pl.read(
                            root_descriptors,
                            [previous_query, PHASE_D_ROOT_SLOT],
                        ),
                    )
                for merge in pl.range(upper_merge_count):
                    output_slot_raw = pl.read(
                        upper_descriptors,
                        [merge, PHASE_D_UPPER_OUTPUT_SLOT],
                    )
                    if output_slot_raw > previous_root_slot:
                        if output_slot_raw <= root_slot_raw:
                            left_slot = pl.cast(
                                pl.read(
                                    upper_descriptors,
                                    [merge, PHASE_D_UPPER_LEFT_SLOT],
                                ),
                                pl.INDEX,
                            )
                            right_slot = pl.cast(
                                pl.read(
                                    upper_descriptors,
                                    [merge, PHASE_D_UPPER_RIGHT_SLOT],
                                ),
                                pl.INDEX,
                            )
                            output_slot = pl.cast(output_slot_raw, pl.INDEX)
                            merged = pl.mrgsort(
                                pair_arena[left_slot : left_slot + 1, :],
                                pair_arena[right_slot : right_slot + 1, :],
                            )
                            pair_arena[
                                output_slot : output_slot + 1, :
                            ] = merged[:, 0:CSA_PAIR_WIDTH]

        for query, (root_tids_iter,) in pl.range(
            query_count,
            init_values=(root_tids,),
        ):
            root_slot_raw = pl.read(
                root_descriptors,
                [query, PHASE_D_ROOT_SLOT],
            )
            with pl.spmd(
                1,
                deps=[init_tid, upper_tid],
                allow_early_resolve=True,
            ) as root_tid:
                _root_scores, _root_indices = materialize_topk_root(
                    pair_arena,
                    topk_scores,
                    topk_indices,
                    query,
                    root_slot_raw,
                )
            root_tids_iter[query] = root_tid
            root_tids_after_roots = pl.yield_(root_tids_iter)

    completion[0] = pl.system.task_dummy(
        deps=[root_tids_after_roots[query] for query in range(CSA_MAX_QUERIES)]
    )
    return topk_scores, topk_indices


@pl.jit
def packed_forest_test(
    leaf_scores: pl.Tensor[[LEAF_DYN, CSA_CANDIDATES_PER_LEAF], pl.FP32],
    leaf_valid_candidates: pl.Tensor[[LEAF_DYN], pl.INT32],
    leaf_logical_begins: pl.Tensor[[LEAF_DYN], pl.INT32],
    pair_left_leaf_ids: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_right_leaf_ids: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_left_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_right_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_output_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    pair_credit_slots: pl.Tensor[[PAIR_GROUP_DYN], pl.INT32],
    singleton_leaf_ids: pl.Tensor[[SINGLETON_DYN], pl.INT32],
    singleton_slots: pl.Tensor[[SINGLETON_DYN], pl.INT32],
    singleton_credit_slots: pl.Tensor[[SINGLETON_DYN], pl.INT32],
    upper_left_slots: pl.Tensor[[UPPER_MERGE_DYN], pl.INT32],
    upper_right_slots: pl.Tensor[[UPPER_MERGE_DYN], pl.INT32],
    upper_output_slots: pl.Tensor[[UPPER_MERGE_DYN], pl.INT32],
    root_slots: pl.Tensor[[QUERY_DYN], pl.INT32],
    root_dependency_slots: pl.Tensor[[QUERY_DYN], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Out[pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32]],
):
    leaf_scores.bind_dynamic(0, LEAF_DYN)
    leaf_valid_candidates.bind_dynamic(0, LEAF_DYN)
    leaf_logical_begins.bind_dynamic(0, LEAF_DYN)
    pair_left_leaf_ids.bind_dynamic(0, PAIR_GROUP_DYN)
    pair_right_leaf_ids.bind_dynamic(0, PAIR_GROUP_DYN)
    pair_left_slots.bind_dynamic(0, PAIR_GROUP_DYN)
    pair_right_slots.bind_dynamic(0, PAIR_GROUP_DYN)
    pair_output_slots.bind_dynamic(0, PAIR_GROUP_DYN)
    pair_credit_slots.bind_dynamic(0, PAIR_GROUP_DYN)
    singleton_leaf_ids.bind_dynamic(0, SINGLETON_DYN)
    singleton_slots.bind_dynamic(0, SINGLETON_DYN)
    singleton_credit_slots.bind_dynamic(0, SINGLETON_DYN)
    upper_left_slots.bind_dynamic(0, UPPER_MERGE_DYN)
    upper_right_slots.bind_dynamic(0, UPPER_MERGE_DYN)
    upper_output_slots.bind_dynamic(0, UPPER_MERGE_DYN)
    root_slots.bind_dynamic(0, QUERY_DYN)
    root_dependency_slots.bind_dynamic(0, QUERY_DYN)
    pair_arena.bind_dynamic(0, ARENA_DYN)
    topk_scores.bind_dynamic(0, QUERY_DYN)
    topk_indices.bind_dynamic(0, QUERY_DYN)
    completion = pl.array.create(1, pl.TASK_ID)
    active_topk_forest(
        leaf_scores,
        leaf_valid_candidates,
        leaf_logical_begins,
        pair_left_leaf_ids,
        pair_right_leaf_ids,
        pair_left_slots,
        pair_right_slots,
        pair_output_slots,
        pair_credit_slots,
        singleton_leaf_ids,
        singleton_slots,
        singleton_credit_slots,
        upper_left_slots,
        upper_right_slots,
        upper_output_slots,
        root_slots,
        root_dependency_slots,
        pair_arena,
        topk_scores,
        topk_indices,
        completion,
    )
    return topk_scores, topk_indices


@pl.jit
def score_select_ragged_test(
    query_vectors: pl.Tensor[[QUERY_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    query_scales: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[QUERY_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[REQUEST_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_descriptors: pl.Tensor[[PAIR_GROUP_DYN, PHASE_D_PAIR_FIELDS], pl.INT32],
    singleton_descriptors: pl.Tensor[[SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32],
    upper_descriptors: pl.Tensor[[UPPER_MERGE_DYN, PHASE_D_UPPER_FIELDS], pl.INT32],
    root_descriptors: pl.Tensor[[QUERY_DYN, PHASE_D_ROOT_FIELDS], pl.INT32],
    pair_arena: pl.Tensor[[ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Out[pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32]],
):
    query_vectors.bind_dynamic(0, QUERY_DYN)
    query_scales.bind_dynamic(0, QUERY_DYN)
    query_weights.bind_dynamic(0, QUERY_DYN)
    idx_kv_cache_flat.bind_dynamic(0, IDX_ROW_DYN)
    idx_kv_scale_flat.bind_dynamic(0, IDX_ROW_DYN)
    query_request_ids.bind_dynamic(0, QUERY_DYN)
    idx_pages.bind_dynamic(0, PAGE_DYN)
    idx_page_offsets.bind_dynamic(0, REQUEST_OFFSET_DYN)
    idx_windows.bind_dynamic(0, REQUEST_DYN)
    request_epochs.bind_dynamic(0, REQUEST_DYN)
    leaf_descriptors.bind_dynamic(0, LEAF_DYN)
    pair_descriptors.bind_dynamic(0, PAIR_GROUP_DYN)
    singleton_descriptors.bind_dynamic(0, SINGLETON_DYN)
    upper_descriptors.bind_dynamic(0, UPPER_MERGE_DYN)
    root_descriptors.bind_dynamic(0, QUERY_DYN)
    pair_arena.bind_dynamic(0, ARENA_DYN)
    topk_scores.bind_dynamic(0, QUERY_DYN)
    topk_indices.bind_dynamic(0, QUERY_DYN)
    index_commit_dep = pl.system.task_dummy(deps=[])
    completion = pl.array.create(1, pl.TASK_ID)
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
def pair_roundtrip_test(
    scores: pl.Tensor[[2, CSA_CANDIDATES_PER_LEAF], pl.FP32],
    pair_store: pl.Out[pl.Tensor[[2, CSA_PAIR_WIDTH], pl.FP32]],
    topk_scores: pl.Out[pl.Tensor[[1, CSA_TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[1, CSA_TOPK], pl.INT32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_pair_roundtrip_producer") as producer_tid:
        for leaf in pl.range(2):
            leaf_scores = scores[leaf : leaf + 1, :]
            idx_init = pl.arange(
                pl.cast(leaf * CSA_CANDIDATES_PER_LEAF, pl.UINT32),
                [1, CSA_CANDIDATES_PER_LEAF],
                dtype=pl.UINT32,
            )
            sorted_pairs = pl.sort32(leaf_scores, idx_init)
            sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
            sorted_pairs = pl.mrgsort(sorted_pairs, block_len=256)
            sorted_pairs = pl.mrgsort(sorted_pairs, block_len=1024)
            pair_store[leaf : leaf + 1, :] = sorted_pairs[:, 0:CSA_PAIR_WIDTH]

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="csa_pair_roundtrip_consumer",
        deps=[producer_tid],
    ):
        merged = pl.mrgsort(pair_store[0:1, :], pair_store[1:2, :])
        root_pairs = merged[:, 0:CSA_PAIR_WIDTH]
        topk_scores[:, :] = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P0101,
        )
        topk_indices[:, :] = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
    return pair_store, topk_scores, topk_indices


@pl.jit
def four_leaf_dag_test(
    scores: pl.Tensor[[POC_LEAVES, CSA_CANDIDATES_PER_LEAF], pl.FP32],
    topk_scores: pl.Out[pl.Tensor[[1, CSA_TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[1, CSA_TOPK], pl.INT32]],
):
    pair_arena = pl.create_tensor([POC_NODES, CSA_PAIR_WIDTH], dtype=pl.FP32)
    node_tids = pl.array.create(POC_NODES + 1, pl.TASK_ID)
    with pl.manual_scope():
        for group in pl.range(POC_LEAVES // 2):
            credit_slot = POC_NODES - group * (POC_NODES - POC_LEVEL1_BASE)
            credit_tid = node_tids[credit_slot]

            left_leaf = group * 2
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="csa_topk_leaf_left",
                deps=[credit_tid],
                allow_early_resolve=True,
            ) as left_tid:
                left_scores = scores[left_leaf : left_leaf + 1, :]
                left_idx = pl.arange(
                    pl.cast(left_leaf * CSA_CANDIDATES_PER_LEAF, pl.UINT32),
                    [1, CSA_CANDIDATES_PER_LEAF],
                    dtype=pl.UINT32,
                )
                left_pairs = pl.sort32(left_scores, left_idx)
                left_pairs = pl.mrgsort(left_pairs, block_len=64)
                left_pairs = pl.mrgsort(left_pairs, block_len=256)
                left_pairs = pl.mrgsort(left_pairs, block_len=1024)
                pair_arena[left_leaf : left_leaf + 1, :] = left_pairs[:, 0:CSA_PAIR_WIDTH]
            node_tids[left_leaf] = left_tid

            right_leaf = left_leaf + 1
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="csa_topk_leaf_right",
                deps=[credit_tid],
                allow_early_resolve=True,
            ) as right_tid:
                right_scores = scores[right_leaf : right_leaf + 1, :]
                right_idx = pl.arange(
                    pl.cast(right_leaf * CSA_CANDIDATES_PER_LEAF, pl.UINT32),
                    [1, CSA_CANDIDATES_PER_LEAF],
                    dtype=pl.UINT32,
                )
                right_pairs = pl.sort32(right_scores, right_idx)
                right_pairs = pl.mrgsort(right_pairs, block_len=64)
                right_pairs = pl.mrgsort(right_pairs, block_len=256)
                right_pairs = pl.mrgsort(right_pairs, block_len=1024)
                pair_arena[right_leaf : right_leaf + 1, :] = right_pairs[:, 0:CSA_PAIR_WIDTH]
            node_tids[right_leaf] = right_tid

            level1_slot = POC_LEVEL1_BASE + group
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="csa_topk_merge_level1",
                deps=[left_tid, right_tid],
                allow_early_resolve=True,
            ) as level1_tid:
                level1_pairs = pl.mrgsort(
                    pair_arena[left_leaf : left_leaf + 1, :],
                    pair_arena[right_leaf : right_leaf + 1, :],
                )
                pair_arena[level1_slot : level1_slot + 1, :] = level1_pairs[
                    :, 0:CSA_PAIR_WIDTH
                ]
            node_tids[level1_slot] = level1_tid

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="csa_topk_merge_root",
            deps=[node_tids[POC_LEVEL1_BASE], node_tids[POC_LEVEL1_BASE + 1]],
            allow_early_resolve=True,
        ):
            merged = pl.mrgsort(
                pair_arena[POC_LEVEL1_BASE : POC_LEVEL1_BASE + 1, :],
                pair_arena[POC_LEVEL1_BASE + 1 : POC_LEVEL1_BASE + 2, :],
            )
            root_pairs = merged[:, 0:CSA_PAIR_WIDTH]
            pair_arena[POC_ROOT_SLOT : POC_ROOT_SLOT + 1, :] = root_pairs
            topk_scores[:, :] = pl.gather(
                root_pairs,
                mask_pattern=pl.tile.MaskPattern.P0101,
            )
            topk_indices[:, :] = pl.gather(
                root_pairs,
                mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
    return topk_scores, topk_indices


def _pair_specs():
    import torch

    from golden import TensorSpec

    def init_scores():
        scores = torch.randn(2, CSA_CANDIDATES_PER_LEAF, dtype=torch.float32)
        scores[0, 0:32] = 7.0
        scores[1, 0:32] = 7.0
        return scores

    return [
        TensorSpec("scores", [2, CSA_CANDIDATES_PER_LEAF], torch.float32, init_value=init_scores),
        TensorSpec("pair_store", [2, CSA_PAIR_WIDTH], torch.float32, is_output=True),
        TensorSpec("topk_scores", [1, CSA_TOPK], torch.float32, is_output=True),
        TensorSpec("topk_indices", [1, CSA_TOPK], torch.int32, is_output=True),
    ]


def _dag_specs():
    import torch

    from golden import TensorSpec

    def init_scores():
        scores = torch.randn(POC_LEAVES, CSA_CANDIDATES_PER_LEAF, dtype=torch.float32)
        scores[:, 0:16] = 11.0
        return scores

    return [
        TensorSpec(
            "scores",
            [POC_LEAVES, CSA_CANDIDATES_PER_LEAF],
            torch.float32,
            init_value=init_scores,
        ),
        TensorSpec("topk_scores", [1, CSA_TOPK], torch.float32, is_output=True),
        TensorSpec("topk_indices", [1, CSA_TOPK], torch.int32, is_output=True),
    ]


def _golden_topk(tensors):
    import torch

    flat = tensors["scores"].reshape(-1).float()
    logical_indices = torch.arange(flat.numel(), dtype=torch.int64)
    order = sorted(
        range(flat.numel()),
        key=lambda i: (-float(flat[i].item()), int(logical_indices[i].item())),
    )[:CSA_TOPK]
    root_indices = torch.tensor(order, dtype=torch.int64)
    tensors["topk_indices"][:] = root_indices.to(torch.int32).reshape(1, CSA_TOPK)
    tensors["topk_scores"][:] = flat[root_indices].reshape(1, CSA_TOPK)


def _golden_pair_roundtrip(tensors):
    import torch

    _golden_topk(tensors)
    scores = tensors["scores"].float()
    pair_bits = torch.empty(2, CSA_PAIR_WIDTH, dtype=torch.int32)
    for leaf in range(2):
        leaf_order = sorted(
            range(CSA_CANDIDATES_PER_LEAF),
            key=lambda i: (-float(scores[leaf, i].item()), i),
        )[:CSA_TOPK]
        values = scores[leaf, leaf_order]
        indices = torch.tensor(leaf_order, dtype=torch.int32) + leaf * CSA_CANDIDATES_PER_LEAF
        pair_bits[leaf, 0::2] = values.view(torch.int32)
        pair_bits[leaf, 1::2] = indices
    tensors["pair_store"][:] = pair_bits.view(torch.float32)


def _score_select_specs():
    import torch

    from golden import TensorSpec

    logical_begin = 1
    valid_candidates = CSA_CANDIDATES_PER_LEAF
    page_count = 17
    physical_pages = list(reversed(range(page_count)))
    cache = torch.zeros(
        page_count,
        CSA_PAGE_ROWS,
        1,
        IDX_HEAD_DIM,
        dtype=torch.int8,
    )
    for logical in range(logical_begin, logical_begin + valid_candidates):
        relative_page = logical // CSA_PAGE_ROWS
        physical_page = physical_pages[relative_page]
        cache[physical_page, logical % CSA_PAGE_ROWS, 0, :] = logical % 7 - 3
    pages = torch.tensor(
        [[physical_page, 11] for physical_page in physical_pages],
        dtype=torch.int32,
    )
    return [
        TensorSpec(
            "query_vectors",
            [1, IDX_N_HEADS, IDX_HEAD_DIM],
            torch.int8,
            init_value=lambda: torch.ones(
                1,
                IDX_N_HEADS,
                IDX_HEAD_DIM,
                dtype=torch.int8,
            ),
        ),
        TensorSpec(
            "query_scales",
            [1, IDX_N_HEADS],
            torch.float32,
            init_value=lambda: torch.ones(1, IDX_N_HEADS),
        ),
        TensorSpec(
            "query_weights",
            [1, IDX_N_HEADS],
            torch.float32,
            init_value=lambda: torch.ones(1, IDX_N_HEADS),
        ),
        TensorSpec(
            "idx_kv_cache_flat",
            [page_count * CSA_PAGE_ROWS, IDX_HEAD_DIM],
            torch.int8,
            init_value=lambda: cache.reshape(
                page_count * CSA_PAGE_ROWS, IDX_HEAD_DIM
            ),
        ),
        TensorSpec(
            "idx_kv_scale_flat",
            [page_count * CSA_PAGE_ROWS, 1],
            torch.float32,
            init_value=lambda: torch.ones(page_count * CSA_PAGE_ROWS, 1),
        ),
        TensorSpec(
            "query_request_ids",
            [1],
            torch.int32,
            init_value=lambda: torch.tensor([0], dtype=torch.int32),
        ),
        TensorSpec(
            "idx_pages",
            list(pages.shape),
            torch.int32,
            init_value=lambda: pages,
        ),
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
            [1, PHASE_D_LEAF_FIELDS],
            torch.int32,
            init_value=lambda: torch.tensor(
                [[0, logical_begin, valid_candidates, 0, CSA_TOPK_INVALID_TASK_SLOT]],
                dtype=torch.int32,
            ),
        ),
        TensorSpec(
            "pair_descriptors",
            [0, PHASE_D_PAIR_FIELDS],
            torch.int32,
            init_value=lambda: torch.empty(0, PHASE_D_PAIR_FIELDS, dtype=torch.int32),
        ),
        TensorSpec(
            "singleton_descriptors",
            [1, PHASE_D_SINGLETON_FIELDS],
            torch.int32,
            init_value=lambda: torch.tensor(
                [[0, 0, CSA_TOPK_INVALID_TASK_SLOT]],
                dtype=torch.int32,
            ),
        ),
        TensorSpec(
            "upper_descriptors",
            [0, PHASE_D_UPPER_FIELDS],
            torch.int32,
            init_value=lambda: torch.empty(0, PHASE_D_UPPER_FIELDS, dtype=torch.int32),
        ),
        TensorSpec(
            "root_descriptors",
            [1, PHASE_D_ROOT_FIELDS],
            torch.int32,
            init_value=lambda: torch.tensor([[0, 0]], dtype=torch.int32),
        ),
        TensorSpec(
            "pair_arena",
            [1, CSA_PAIR_WIDTH],
            torch.float32,
            init_value=lambda: torch.zeros(1, CSA_PAIR_WIDTH),
        ),
        TensorSpec("topk_scores", [1, CSA_TOPK], torch.float32, is_output=True),
        TensorSpec("topk_indices", [1, CSA_TOPK], torch.int32, is_output=True),
    ]


def _golden_score_select(tensors):
    import torch

    logical_begin = int(
        tensors["leaf_descriptors"][0, PHASE_D_LEAF_BEGIN].item()
    )
    valid_candidates = int(
        tensors["leaf_descriptors"][0, PHASE_D_LEAF_VALID].item()
    )
    scores = []
    for logical in range(logical_begin, logical_begin + valid_candidates):
        value = float(logical % 7 - 3)
        score = max(value * IDX_HEAD_DIM, 0.0) * IDX_N_HEADS
        scores.append((score, logical))
    scores.sort(key=lambda item: (-item[0], item[1]))
    out_scores = torch.full((1, CSA_TOPK), FP32_NEG_INF, dtype=torch.float32)
    out_indices = torch.full((1, CSA_TOPK), -1, dtype=torch.int32)
    kept = min(CSA_TOPK, len(scores))
    if kept:
        out_scores[0, :kept] = torch.tensor(
            [score for score, _ in scores[:kept]],
            dtype=torch.float32,
        )
        out_indices[0, :kept] = torch.tensor(
            [index for _, index in scores[:kept]],
            dtype=torch.int32,
        )
    tensors["topk_scores"][:] = out_scores
    tensors["topk_indices"][:] = out_indices


def _forest_case_config(case):
    configs = {
        "leaf_random": ([2048], "random"),
        "leaf_ascending": ([2048], "ascending"),
        "leaf_descending": ([2048], "descending"),
        # Deliberately poison the non-valid tail with a dominant score.  The
        # selector must still mask those lanes before sorting.
        "leaf_tail_poison": ([511], "tail_poison"),
        "leaf_tail_1": ([1], "random"),
        "leaf_tail_511": ([511], "random"),
        "leaf_tail_512": ([512], "random"),
        "leaf_tail_513": ([513], "random"),
        "leaf_tail_2047": ([2047], "random"),
        "merge_disjoint": ([4096], "disjoint"),
        "merge_interleaved": ([4096], "interleaved"),
        "merge_duplicate_scores": ([4096], "duplicate"),
        "forest_zero_leaf": ([0], "random"),
        "forest_one_leaf": ([2048], "random"),
        "forest_two_leaves": ([4096], "random"),
        "forest_three_leaves": ([4097], "random"),
        "forest_eight_leaves": ([16384], "random"),
        # 127 leaves exercises odd carries at every upper level while keeping
        # the ready-frontier credit chain one node below the 1M ceiling.
        "forest_one_hundred_twenty_seven_leaves": (
            [127 * CSA_CANDIDATES_PER_LEAF],
            "random",
        ),
        "forest_one_hundred_twenty_eight_leaves": ([262144], "random"),
        "forest_heterogeneous": ([0, 32, 4096, 262144], "random"),
    }
    return configs[case]


def _build_forest_fixture(case):
    import torch

    candidate_counts, pattern = _forest_case_config(case)
    generator = torch.Generator().manual_seed(20260810)
    leaf_scores = []
    leaf_valid_candidates = []
    leaf_logical_begins = []
    node_kinds = []
    node_leaf_ids = []
    node_left_slots = []
    node_right_slots = []
    node_credit_slots = []
    root_slots = []

    def append_leaf(query, local_leaf, valid, credit_slot):
        leaf_id = len(leaf_scores)
        logical_begin = local_leaf * CSA_CANDIDATES_PER_LEAF
        logical = torch.arange(
            logical_begin,
            logical_begin + CSA_CANDIDATES_PER_LEAF,
            dtype=torch.float32,
        )
        if pattern == "ascending":
            scores = logical
        elif pattern == "descending":
            scores = -logical
        elif pattern == "disjoint":
            scores = logical + local_leaf * 1000000.0
        elif pattern == "interleaved":
            scores = logical.remainder(2) * 1000000.0 + logical // 2
        elif pattern == "duplicate":
            scores = logical.remainder(17)
        elif pattern == "tail_poison":
            scores = torch.randn(CSA_CANDIDATES_PER_LEAF, generator=generator)
            scores[valid:] = 1.0e9
        else:
            scores = torch.randn(CSA_CANDIDATES_PER_LEAF, generator=generator)
            scores[0:8] = 3.0 + query
        leaf_scores.append(scores)
        leaf_valid_candidates.append(valid)
        leaf_logical_begins.append(logical_begin)
        slot = len(node_kinds)
        node_kinds.append(CSA_TOPK_NODE_LEAF)
        node_leaf_ids.append(leaf_id)
        node_left_slots.append(-1)
        node_right_slots.append(-1)
        node_credit_slots.append(credit_slot)
        return slot

    def append_merge(left_slot, right_slot):
        slot = len(node_kinds)
        node_kinds.append(CSA_TOPK_NODE_MERGE)
        node_leaf_ids.append(-1)
        node_left_slots.append(left_slot)
        node_right_slots.append(right_slot)
        node_credit_slots.append(CSA_TOPK_INVALID_TASK_SLOT)
        return slot

    for query, candidates in enumerate(candidate_counts):
        leaves = (candidates + CSA_CANDIDATES_PER_LEAF - 1) // CSA_CANDIDATES_PER_LEAF
        if leaves == 0:
            root_slots.append(-1)
            continue

        level1_slots = []
        for group in range(leaves // 2):
            credit_slot = CSA_TOPK_INVALID_TASK_SLOT
            if group >= CSA_TOPK_READY_FRONTIER_W:
                credit_slot = level1_slots[group - CSA_TOPK_READY_FRONTIER_W]
            left_leaf = group * 2
            left_valid = min(
                CSA_CANDIDATES_PER_LEAF,
                candidates - left_leaf * CSA_CANDIDATES_PER_LEAF,
            )
            left_slot = append_leaf(query, left_leaf, left_valid, credit_slot)
            right_leaf = left_leaf + 1
            right_valid = min(
                CSA_CANDIDATES_PER_LEAF,
                candidates - right_leaf * CSA_CANDIDATES_PER_LEAF,
            )
            right_slot = append_leaf(query, right_leaf, right_valid, credit_slot)
            level1_slots.append(append_merge(left_slot, right_slot))

        if leaves % 2:
            local_leaf = leaves - 1
            credit_slot = CSA_TOPK_INVALID_TASK_SLOT
            if len(level1_slots) >= CSA_TOPK_READY_FRONTIER_W:
                credit_slot = level1_slots[-CSA_TOPK_READY_FRONTIER_W]
            valid = min(
                CSA_CANDIDATES_PER_LEAF,
                candidates - local_leaf * CSA_CANDIDATES_PER_LEAF,
            )
            level1_slots.append(append_leaf(query, local_leaf, valid, credit_slot))

        current = level1_slots
        while len(current) > 1:
            next_level = []
            for pair in range(len(current) // 2):
                next_level.append(append_merge(current[2 * pair], current[2 * pair + 1]))
            if len(current) % 2:
                next_level.append(current[-1])
            current = next_level
        root_slots.append(current[0])

    if leaf_scores:
        scores_tensor = torch.stack(leaf_scores).to(torch.float32)
    else:
        scores_tensor = torch.empty(0, CSA_CANDIDATES_PER_LEAF, dtype=torch.float32)
    pair_left_leaf_ids = []
    pair_right_leaf_ids = []
    pair_left_slots = []
    pair_right_slots = []
    pair_output_slots = []
    pair_credit_slots = []
    paired_leaf_slots = set()
    upper_left_slots = []
    upper_right_slots = []
    upper_output_slots = []
    for output_slot, kind in enumerate(node_kinds):
        if kind != CSA_TOPK_NODE_MERGE:
            continue
        left_slot = node_left_slots[output_slot]
        right_slot = node_right_slots[output_slot]
        if (
            node_kinds[left_slot] == CSA_TOPK_NODE_LEAF
            and node_kinds[right_slot] == CSA_TOPK_NODE_LEAF
        ):
            pair_left_leaf_ids.append(node_leaf_ids[left_slot])
            pair_right_leaf_ids.append(node_leaf_ids[right_slot])
            pair_left_slots.append(left_slot)
            pair_right_slots.append(right_slot)
            pair_output_slots.append(output_slot)
            pair_credit_slots.append(node_credit_slots[left_slot])
            paired_leaf_slots.update((left_slot, right_slot))
        else:
            upper_left_slots.append(left_slot)
            upper_right_slots.append(right_slot)
            upper_output_slots.append(output_slot)
    singleton_slots = [
        slot
        for slot, kind in enumerate(node_kinds)
        if kind == CSA_TOPK_NODE_LEAF and slot not in paired_leaf_slots
    ]
    fixture = {
        "leaf_scores": scores_tensor,
        "leaf_valid_candidates": torch.tensor(leaf_valid_candidates, dtype=torch.int32),
        "leaf_logical_begins": torch.tensor(leaf_logical_begins, dtype=torch.int32),
        "pair_left_leaf_ids": torch.tensor(pair_left_leaf_ids, dtype=torch.int32),
        "pair_right_leaf_ids": torch.tensor(pair_right_leaf_ids, dtype=torch.int32),
        "pair_left_slots": torch.tensor(pair_left_slots, dtype=torch.int32),
        "pair_right_slots": torch.tensor(pair_right_slots, dtype=torch.int32),
        "pair_output_slots": torch.tensor(pair_output_slots, dtype=torch.int32),
        "pair_credit_slots": torch.tensor(pair_credit_slots, dtype=torch.int32),
        "singleton_leaf_ids": torch.tensor(
            [node_leaf_ids[slot] for slot in singleton_slots],
            dtype=torch.int32,
        ),
        "singleton_slots": torch.tensor(singleton_slots, dtype=torch.int32),
        "singleton_credit_slots": torch.tensor(
            [node_credit_slots[slot] for slot in singleton_slots],
            dtype=torch.int32,
        ),
        "upper_left_slots": torch.tensor(upper_left_slots, dtype=torch.int32),
        "upper_right_slots": torch.tensor(upper_right_slots, dtype=torch.int32),
        "upper_output_slots": torch.tensor(upper_output_slots, dtype=torch.int32),
        "root_slots": torch.tensor(root_slots, dtype=torch.int32),
        "root_dependency_slots": torch.tensor(
            [root if root >= 0 else CSA_TOPK_INVALID_TASK_SLOT for root in root_slots],
            dtype=torch.int32,
        ),
        "candidate_counts": candidate_counts,
        "node_count": len(node_kinds),
    }
    assert len(node_kinds) == sum(max(2 * ((n + 2047) // 2048) - 1, 0) for n in candidate_counts)
    assert len(node_kinds) <= CSA_MAX_TOPK_TASKS
    return fixture


def _forest_specs(case):
    import torch

    from golden import TensorSpec

    fixture = _build_forest_fixture(case)
    query_count = len(fixture["candidate_counts"])
    arena_nodes = max(int(fixture["node_count"]), 1)
    return [
        TensorSpec(
            "leaf_scores",
            list(fixture["leaf_scores"].shape),
            torch.float32,
            init_value=lambda: fixture["leaf_scores"],
        ),
        TensorSpec(
            "leaf_valid_candidates",
            list(fixture["leaf_valid_candidates"].shape),
            torch.int32,
            init_value=lambda: fixture["leaf_valid_candidates"],
        ),
        TensorSpec(
            "leaf_logical_begins",
            list(fixture["leaf_logical_begins"].shape),
            torch.int32,
            init_value=lambda: fixture["leaf_logical_begins"],
        ),
        *[
            TensorSpec(
                name,
                list(fixture[name].shape),
                torch.int32,
                init_value=lambda name=name: fixture[name],
            )
            for name in (
                "pair_left_leaf_ids",
                "pair_right_leaf_ids",
                "pair_left_slots",
                "pair_right_slots",
                "pair_output_slots",
                "pair_credit_slots",
                "singleton_leaf_ids",
                "singleton_slots",
                "singleton_credit_slots",
                "upper_left_slots",
                "upper_right_slots",
                "upper_output_slots",
            )
        ],
        TensorSpec("root_slots", [query_count], torch.int32, init_value=lambda: fixture["root_slots"]),
        TensorSpec(
            "root_dependency_slots",
            [query_count],
            torch.int32,
            init_value=lambda: fixture["root_dependency_slots"],
        ),
        TensorSpec(
            "pair_arena",
            [arena_nodes, CSA_PAIR_WIDTH],
            torch.float32,
            init_value=lambda: torch.zeros(arena_nodes, CSA_PAIR_WIDTH, dtype=torch.float32),
        ),
        TensorSpec("topk_scores", [query_count, CSA_TOPK], torch.float32, is_output=True),
        TensorSpec("topk_indices", [query_count, CSA_TOPK], torch.int32, is_output=True),
    ]


def _golden_packed_forest(tensors):
    import torch

    query_count = tensors["root_slots"].numel()
    topk_scores = torch.full((query_count, CSA_TOPK), FP32_NEG_INF, dtype=torch.float32)
    topk_indices = torch.full((query_count, CSA_TOPK), -1, dtype=torch.int32)
    leaf_by_slot = {}
    children_by_slot = {}
    for group in range(tensors["pair_output_slots"].numel()):
        left_slot = int(tensors["pair_left_slots"][group].item())
        right_slot = int(tensors["pair_right_slots"][group].item())
        output_slot = int(tensors["pair_output_slots"][group].item())
        leaf_by_slot[left_slot] = int(tensors["pair_left_leaf_ids"][group].item())
        leaf_by_slot[right_slot] = int(tensors["pair_right_leaf_ids"][group].item())
        children_by_slot[output_slot] = (left_slot, right_slot)
    for singleton in range(tensors["singleton_slots"].numel()):
        leaf_by_slot[int(tensors["singleton_slots"][singleton].item())] = int(
            tensors["singleton_leaf_ids"][singleton].item()
        )
    for merge in range(tensors["upper_output_slots"].numel()):
        children_by_slot[int(tensors["upper_output_slots"][merge].item())] = (
            int(tensors["upper_left_slots"][merge].item()),
            int(tensors["upper_right_slots"][merge].item()),
        )
    for query in range(query_count):
        root = int(tensors["root_slots"][query].item())
        query_leaves = []
        stack = [] if root < 0 else [root]
        while stack:
            slot = stack.pop()
            if slot in leaf_by_slot:
                query_leaves.append(leaf_by_slot[slot])
            else:
                left_slot, right_slot = children_by_slot[slot]
                stack.append(left_slot)
                stack.append(right_slot)
        candidates = []
        for leaf_id in query_leaves:
            valid = int(tensors["leaf_valid_candidates"][leaf_id].item())
            begin = int(tensors["leaf_logical_begins"][leaf_id].item())
            for local in range(valid):
                candidates.append((float(tensors["leaf_scores"][leaf_id, local].item()), begin + local))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for rank, (score, index) in enumerate(candidates[:CSA_TOPK]):
            topk_scores[query, rank] = score
            topk_indices[query, rank] = index
    tensors["topk_scores"][:] = topk_scores
    tensors["topk_indices"][:] = topk_indices


CASES = {
    "pair_roundtrip": (pair_roundtrip_test, _pair_specs, _golden_pair_roundtrip),
    "dag_four_leaf": (four_leaf_dag_test, _dag_specs, _golden_topk),
    "score_select_ragged": (
        score_select_ragged_test,
        _score_select_specs,
        _golden_score_select,
    ),
}
for _case_name in (
    "leaf_random",
    "leaf_ascending",
    "leaf_descending",
    "leaf_tail_poison",
    "leaf_tail_1",
    "leaf_tail_511",
    "leaf_tail_512",
    "leaf_tail_513",
    "leaf_tail_2047",
    "merge_disjoint",
    "merge_interleaved",
    "merge_duplicate_scores",
    "forest_zero_leaf",
    "forest_one_leaf",
    "forest_two_leaves",
    "forest_three_leaves",
    "forest_eight_leaves",
    "forest_one_hundred_twenty_seven_leaves",
    "forest_one_hundred_twenty_eight_leaves",
    "forest_heterogeneous",
):
    CASES[_case_name] = (
        packed_forest_test,
        lambda case=_case_name: _forest_specs(case),
        _golden_packed_forest,
    )


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=list(CASES), default="pair_roundtrip")
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    fn, specs_builder, golden_fn = CASES[args.case]
    result = run_jit(
        fn=fn,
        specs=specs_builder(),
        golden_fn=golden_fn,
        compile_cfg={"dump_passes": args.dump_passes},
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_dep_gen": args.enable_dep_gen,
            "enable_l2_swimlane": args.enable_l2_swimlane,
        },
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
