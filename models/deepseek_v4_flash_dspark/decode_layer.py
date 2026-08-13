# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run-055 Phase E single-layer decode orchestration.

The public layer surface is attention-kind specialized. SWA, HCA, and CSA use
one device child per rank. CSA inlines the Phase-D production work below into
that rank-local child without importing the legacy dense 16K wrapper. All
three paths converge on the same distributed MoE activation and epoch protocol.
"""

import argparse
import os

# The 1M-tail HCA path keeps partial accumulators in ring 1 and its packed
# gather/QK workspace in ring 2.  This environment contract must be installed
# before importing PyPTO: the distributed runtime may cache its allocator
# configuration during import.
_MIB = 1024 * 1024
DECODE_LAYER_RING_HEAP = (256 * _MIB, 2048 * _MIB, 2048 * _MIB, 256 * _MIB)
os.environ.setdefault(
    "PTO2_RING_HEAP", ",".join(str(value) for value in DECODE_LAYER_RING_HEAP)
)

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import (
    BLOCK_SIZE,
    FLASH as MODEL_CONFIG,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_PAGES_PER_REQUEST,
)
from config import (
    CSA_INNER_STATE_BLOCK_SIZE,
    CSA_INNER_STATE_PAGES_PER_REQUEST,
    CSA_INNER_STATE_POOL_PAGES,
    CSA_INNER_STATE_ROWS_PER_REQUEST,
    CSA_CANDIDATES_PER_LEAF,
    CSA_MAX_NODES_PER_QUERY,
    CSA_PAIR_WIDTH,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_PAGES_PER_REQUEST,
    CSA_STATE_POOL_PAGES,
    CSA_STATE_ROWS_PER_REQUEST,
    CSA_TOPK as CSA_INDEX_TOPK,
    CSA_TOPK_INVALID_TASK_SLOT,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    SWA_SOURCE_OVERLAY_BASE,
)
from decode_compressor_ratio4 import (
    B_DYN as CSA_MAIN_B_DYN,
    CMP_BLOCK_NUM_DYN as CSA_MAIN_CACHE_DYN,
    COMPRESS_STATE_DIM as CSA_MAIN_STATE_DIM,
    COMPRESS_STATE_BLOCK_NUM_DYN as CSA_MAIN_STATE_POOL_DYN,
    EVENT_DYN as CSA_MAIN_EVENT_DYN,
    OUT_DIM as CSA_MAIN_OUT_DIM,
    T_DYN as CSA_MAIN_T_DYN,
    build_tensor_specs as build_csa_main_compressor_specs,
    compressor_test as csa_main_compressor_stage,
    compressor_ratio4,
)
from decode_indexer import (
    ARENA_DYN as CSA_INDEXER_ARENA_DYN,
    B_DYN as CSA_INDEXER_B_DYN,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_ROW_DYN as CSA_INDEXER_ROW_DYN,
    LEAF_DYN as CSA_INDEXER_LEAF_DYN,
    PAGE_DYN as CSA_INDEXER_PAGE_DYN,
    PAIR_GROUP_DYN as CSA_INDEXER_PAIR_DYN,
    PHASE_D_LEAF_FIELDS,
    PHASE_D_PAIR_FIELDS,
    PHASE_D_ROOT_FIELDS,
    PHASE_D_SINGLETON_FIELDS,
    PHASE_D_UPPER_FIELDS,
    PHASE_D_TRACE_CASE_LENGTHS,
    REQUEST_OFFSET_DYN as CSA_INDEXER_REQUEST_OFFSET_DYN,
    SINGLETON_DYN as CSA_INDEXER_SINGLETON_DYN,
    T_DYN as CSA_INDEXER_T_DYN,
    UPPER_MERGE_DYN as CSA_INDEXER_UPPER_DYN,
    build_phase_d_indexer_specs,
    indexer,
    phase_d_indexer_test as csa_indexer_stage,
)
from decode_indexer_compressor import (
    B_DYN as CSA_INNER_B_DYN,
    COMPRESS_STATE_DIM as CSA_INNER_STATE_DIM,
    COMPRESS_STATE_BLOCK_NUM_DYN as CSA_INNER_STATE_POOL_DYN,
    EVENT_DYN as CSA_INNER_EVENT_DYN,
    IDX_CACHE_BLOCK_NUM_DYN as CSA_INNER_CACHE_DYN,
    IDX_CACHE_ROW_NUM_DYN as CSA_INNER_CACHE_ROWS_DYN,
    OUT_DIM as CSA_INNER_OUT_DIM,
    T_DYN as CSA_INNER_T_DYN,
    build_tensor_specs as build_csa_inner_compressor_specs,
    compressor_test as csa_inner_compressor_stage,
    indexer_compressor,
)
from decode_metadata import (
    PHASE_D_LEAF_QUERY,
    PHASE_D_LEAF_VALID,
    PHASE_D_ROOT_DEPENDENCY_SLOT,
    PHASE_D_ROOT_SLOT,
)
from decode_sparse_attn_csa import (
    B_DYN as CSA_SPARSE_B_DYN,
    CMP_BLOCK_NUM_DYN as CSA_SPARSE_CMP_BLOCKS_DYN,
    ORI_BLOCK_NUM_DYN as CSA_SPARSE_RAW_BLOCKS_DYN,
    PAGE_DYN as CSA_SPARSE_PAGE_DYN,
    REQUEST_OFFSET_DYN as CSA_SPARSE_REQUEST_OFFSET_DYN,
    T_DYN as CSA_SPARSE_T_DYN,
    sparse_attn_csa,
    sparse_attn_test as csa_sparse_value_stage,
)
from decode_hca import (
    B_DYN as HCA_B_DYN,
    CMP_BLOCK_NUM_DYN as HCA_CMP_BLOCK_NUM_DYN,
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_DIM as HCA_COMPRESS_STATE_DIM,
    EVENT_DYN as HCA_EVENT_DYN,
    HCA_PAGES_DYN,
    HCA_QUERY_OFFSETS_DYN,
    HCA_REQUEST_OFFSETS_DYN,
    HCA_WORK_DYN,
    ORI_BLOCK_NUM_DYN as HCA_ORI_BLOCK_NUM_DYN,
    STATE_BLOCK_NUM_DYN as HCA_STATE_BLOCK_NUM_DYN,
    T_DYN as HCA_T_DYN,
    attention_hca,
    build_tensor_specs as build_hca_tensor_specs,
    golden_attention_hca,
)
from decode_swa import (
    B,
    D,
    H,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    MIX_HC,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCK_NUM_DYN as SWA_ORI_BLOCK_NUM_DYN,
    Q_LORA,
    ROPE_HEAD_DIM,
    S,
    T_DYN as SWA_T_DYN,
    WIN,
    attention_swa,
    build_tensor_specs as build_swa_tensor_specs,
    golden_attention_swa,
)
from moe import (
    AUX_PAD,
    IDX_PAD,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe,
    moe,
)
from hc_post import hc_post
from hc_pre import hc_pre
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm


assert T == B * S, "Phase E must cover the rank-local D-Spark decode rows"

# The Phase-D indexer accepts at most 16 queries. Use its full capacity so two
# complete D-Spark requests (one committed token plus seven draft tokens each)
# share one rank-local CSA program. This compile-time granularity keeps each
# request intact; leaf/merge work is still derived from exact visible candidate
# descriptors.
CSA_CHUNK_T = 16
CSA_CHUNKS = T // CSA_CHUNK_T
CSA_CHUNK_B = CSA_CHUNK_T // S
# The full43 program retains its flat persistent-storage ABI. The standalone
# CSA layer below uses explicit [N_RANKS, CSA_CHUNKS, ...] tensor layouts.
CSA_LAYER_SHARDS = N_RANKS * CSA_CHUNKS
CSA_EVENT_CAP = CSA_CHUNK_B * ((S + 3) // 4)
CSA_GLOBAL_STATE_HIGH_PAGE = 64
# The normal layer submits all compile-time chunks. Trace fixtures set this
# trace-time constant to two so their graph is exactly B=4/S=8 rather than a
# B=16 graph whose trailing chunks happen to be inactive.
CSA_SUBMISSION_CHUNKS = CSA_CHUNKS
assert T % CSA_CHUNK_T == 0
assert CSA_CHUNK_T % S == 0
assert CSA_STATE_POOL_PAGES > CSA_GLOBAL_STATE_HIGH_PAGE
assert CSA_INNER_STATE_POOL_PAGES > CSA_GLOBAL_STATE_HIGH_PAGE

CSA_MAIN_BLOCKS_DYN = pl.dynamic("LAYER_CSA_MAIN_BLOCKS_DYN")
CSA_IDX_BLOCKS_DYN = pl.dynamic("LAYER_CSA_IDX_BLOCKS_DYN")
CSA_IDX_ROWS_DYN = pl.dynamic("LAYER_CSA_IDX_ROWS_DYN")
CSA_RAW_BLOCKS_DYN = pl.dynamic("LAYER_CSA_RAW_BLOCKS_DYN")
CSA_EVENT_DYN = pl.dynamic("LAYER_CSA_EVENT_DYN")
CSA_PAGE_DYN = pl.dynamic("LAYER_CSA_PAGE_DYN")
CSA_REQUEST_OFFSET_DYN = pl.dynamic("LAYER_CSA_REQUEST_OFFSET_DYN")
CSA_REQUEST_DYN = pl.dynamic("LAYER_CSA_REQUEST_DYN")
CSA_LEAF_DYN = pl.dynamic("LAYER_CSA_LEAF_DYN")
CSA_PAIR_DYN = pl.dynamic("LAYER_CSA_PAIR_DYN")
CSA_SINGLETON_DYN = pl.dynamic("LAYER_CSA_SINGLETON_DYN")
CSA_UPPER_DYN = pl.dynamic("LAYER_CSA_UPPER_DYN")
CSA_ARENA_DYN = pl.dynamic("LAYER_CSA_ARENA_DYN")


def attention_kind_for_layer(layer_id):
    """Return the configured main-model attention kind for ``layer_id``."""
    if not 0 <= int(layer_id) < MODEL_CONFIG.num_hidden_layers:
        raise ValueError(
            f"layer_id must be in [0, {MODEL_CONFIG.num_hidden_layers - 1}], "
            f"got {layer_id}"
        )
    ratio = MODEL_CONFIG.compress_ratios[int(layer_id)]
    if ratio == 0:
        return "swa"
    if ratio == 4:
        return "csa"
    if ratio == 128:
        return "hca"
    raise ValueError(f"unsupported compression ratio {ratio} for layer {layer_id}")


def alloc_moe_signal_window_buffers():
    """Allocate the enclosing forward's shared MoE signal storage."""
    return (
        pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32),
        pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8),
        pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32),
        pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32),
        pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32),
        pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32),
        pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16),
        pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32),
    )


def open_moe_signal_windows(
    recv_meta_buf,
    recv_x_buf,
    recv_aux_buf,
    recv_route_buf,
    arrived_buf,
    data_arrived_buf,
    routed_y_buf_buf,
    combine_arrived_buf,
):
    """Open this rank's views of one forward-owned MoE signal window set."""
    return (
        pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32),
        pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8),
        pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32),
        pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32),
        pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32),
        pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32),
        pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16),
        pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32),
    )


@pl.jit(auto_scope=False)
def decode_layer_swa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[
        pl.Tensor[[SWA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    swa_write_slots: pl.Tensor[[T], pl.INT64],
    swa_sources: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    x_attn_workspace: pl.InOut[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    # The caller owns this attention-to-MoE bridge.  Full43 reuses one
    # rank-local workspace across all 43 layers so a dynamic layer loop does
    # not retain one 8-MiB root-scope allocation per iteration.
    x_attn = x_attn_workspace
    with pl.scope():
        attention_swa(
            x_hc,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_w,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            gamma_cq,
            gamma_ckv,
            rope_cos,
            rope_sin,
            kv_cache,
            swa_write_slots,
            swa_sources,
            swa_lens,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            x_attn,
        )
    num_tokens = pl.read(num_tokens_per_owner, [my_rank])
    with pl.scope():
        moe(
            x_attn,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            norm_w,
            gate_w,
            gate_bias,
            tid2eid,
            input_ids,
            routed_w1,
            routed_w1_scale,
            routed_w3,
            routed_w3_scale,
            routed_w2,
            routed_w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w3,
            shared_w3_scale,
            shared_w2,
            shared_w2_scale,
            x_next,
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            num_tokens,
            my_rank,
            moe_epoch,
        )
    # Do not clear the shared EP signal windows at a layer seam.  A multi-layer
    # forward reuses them with monotonically increasing ``moe_epoch`` values and
    # clears once after the final layer, matching the baseline decode_fwd
    # protocol.  Standalone execution allocates fresh windows for this one call.
    return x_next


@pl.jit.host
def l3_decode_layer_swa(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[N_RANKS, T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin: pl.Tensor[[N_RANKS, T, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[
        pl.Tensor[
            [N_RANKS, SWA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    swa_write_slots: pl.Tensor[[N_RANKS, T], pl.INT64],
    swa_sources: pl.Tensor[[N_RANKS, T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[N_RANKS, T], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    x_attn_workspace: pl.InOut[
        pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]
    ],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
):
    # Decorated host entries must expose window allocation and typed openings
    # directly: the frontend does not discover ordinary Python helpers.
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32] = pld.window(
            recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32
        )
        recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8] = pld.window(
            recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
        )
        recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32] = pld.window(
            recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
        )
        recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32] = pld.window(
            recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
        )
        arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16] = pld.window(
            routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16
        )
        combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        decode_layer_swa(
            x_hc[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            gamma_cq[rank],
            gamma_ckv[rank],
            rope_cos[rank],
            rope_sin[rank],
            kv_cache[rank],
            swa_write_slots[rank],
            swa_sources[rank],
            swa_lens[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank],
            gate_bias[rank],
            tid2eid[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            num_tokens_per_owner,
            x_attn_workspace[rank],
            x_next[rank],
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            rank,
            pl.const(1, pl.INT32),
            device=rank,
        )
    return x_next


def submit_decode_layer_swa(
    x_hc,
    hc_attn_fn,
    hc_attn_scale,
    hc_attn_base,
    attn_norm_w,
    wq_a,
    wq_b,
    wq_b_scale,
    wkv,
    gamma_cq,
    gamma_ckv,
    rope_cos,
    rope_sin,
    kv_cache,
    swa_write_slots,
    swa_sources,
    swa_lens,
    attn_sink,
    wo_a,
    wo_b,
    wo_b_scale,
    hc_ffn_fn,
    hc_ffn_scale,
    hc_ffn_base,
    norm_w,
    gate_w,
    gate_bias,
    tid2eid,
    input_ids,
    routed_w1,
    routed_w1_scale,
    routed_w3,
    routed_w3_scale,
    routed_w2,
    routed_w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w3,
    shared_w3_scale,
    shared_w2,
    shared_w2_scale,
    num_tokens_per_owner,
    x_attn_workspace,
    x_next,
    recv_meta_buf,
    recv_x_buf,
    recv_aux_buf,
    recv_route_buf,
    arrived_buf,
    data_arrived_buf,
    routed_y_buf_buf,
    combine_arrived_buf,
    layer_id,
    moe_epoch,
):
    """Submit one SWA layer with caller-owned forward MoE signal windows.

    This is intentionally a host composition seam rather than a second
    attention implementation.  ``kv_cache`` and all slot/source descriptors
    are supplied by the enclosing allocator plan; no cache slice is inferred
    from a layer count.
    """
    for rank in pl.range(pld.world_size()):
        (
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
        ) = open_moe_signal_windows(
            recv_meta_buf,
            recv_x_buf,
            recv_aux_buf,
            recv_route_buf,
            arrived_buf,
            data_arrived_buf,
            routed_y_buf_buf,
            combine_arrived_buf,
        )
        decode_layer_swa(
            x_hc[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            gamma_cq[rank],
            gamma_ckv[rank],
            rope_cos[rank],
            rope_sin[rank],
            kv_cache[rank],
            swa_write_slots[rank],
            swa_sources[rank],
            swa_lens[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank],
            gate_bias[rank],
            tid2eid[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            num_tokens_per_owner,
            x_attn_workspace[rank],
            x_next[rank],
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            rank,
            moe_epoch,
            device=rank,
        )
    return x_next


@pl.jit(auto_scope=False)
def decode_layer_hca(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    query_rope_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    query_rope_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    request_event_indices: pl.Tensor[[HCA_B_DYN], pl.INT32],
    event_rope_cos: pl.Tensor[[HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    event_rope_sin: pl.Tensor[[HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    compress_state: pl.InOut[
        pl.Tensor[
            [HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    state_page_ids: pl.Tensor[[HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    state_valid_ranges: pl.Tensor[[HCA_B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[[HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    request_epochs: pl.Tensor[[HCA_B_DYN], pl.INT32],
    state_write_slots: pl.Tensor[[T], pl.INT64],
    kv_cache: pl.InOut[
        pl.Tensor[[HCA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    swa_write_slots: pl.Tensor[[T], pl.INT64],
    swa_sources: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.InOut[
        pl.Tensor[[HCA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    query_request_ids: pl.Tensor[[T], pl.INT32],
    hca_pages: pl.Tensor[[HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[HCA_B_DYN, 3], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    x_attn_workspace: pl.InOut[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    # See ``decode_layer_swa``: this bridge is forward-owned so a looped
    # Full43 graph can reuse one bounded allocation rather than growing the
    # root ring by 8 MiB for every HCA layer.
    x_attn = x_attn_workspace
    with pl.scope():
        attention_hca(
            x_hc,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_w,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            gamma_cq,
            gamma_ckv,
            query_rope_cos,
            query_rope_sin,
            cmp_wkv,
            cmp_wgate,
            cmp_ape,
            cmp_norm_w,
            request_event_indices,
            event_rope_cos,
            event_rope_sin,
            compress_state,
            state_page_ids,
            state_valid_ranges,
            state_page_epochs,
            request_epochs,
            state_write_slots,
            kv_cache,
            swa_write_slots,
            swa_sources,
            cmp_kv,
            cmp_slot_mapping,
            position_ids,
            query_request_ids,
            hca_pages,
            hca_page_offsets,
            hca_windows,
            hca_query_work_offsets,
            hca_work_query_ids,
            hca_work_row_begin,
            hca_work_valid_rows,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            x_attn,
        )
    num_tokens = pl.read(num_tokens_per_owner, [my_rank])
    with pl.scope():
        moe(
            x_attn,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            norm_w,
            gate_w,
            gate_bias,
            tid2eid,
            input_ids,
            routed_w1,
            routed_w1_scale,
            routed_w3,
            routed_w3_scale,
            routed_w2,
            routed_w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w3,
            shared_w3_scale,
            shared_w2,
            shared_w2_scale,
            x_next,
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            num_tokens,
            my_rank,
            moe_epoch,
        )
    # Signal-window ownership belongs to the enclosing forward, not this layer.
    return x_next


@pl.jit.host
def l3_decode_layer_hca(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    query_rope_cos: pl.Tensor[[N_RANKS, T, ROPE_HEAD_DIM], pl.BF16],
    query_rope_sin: pl.Tensor[[N_RANKS, T, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[N_RANKS, HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[N_RANKS, HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    request_event_indices: pl.Tensor[[N_RANKS, HCA_B_DYN], pl.INT32],
    event_rope_cos: pl.Tensor[
        [N_RANKS, HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    event_rope_sin: pl.Tensor[
        [N_RANKS, HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    compress_state: pl.InOut[
        pl.Tensor[
            [N_RANKS, HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    state_page_ids: pl.Tensor[
        [N_RANKS, HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    state_valid_ranges: pl.Tensor[[N_RANKS, HCA_B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[
        [N_RANKS, HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    request_epochs: pl.Tensor[[N_RANKS, HCA_B_DYN], pl.INT32],
    state_write_slots: pl.Tensor[[N_RANKS, T], pl.INT64],
    kv_cache: pl.InOut[
        pl.Tensor[
            [N_RANKS, HCA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    swa_write_slots: pl.Tensor[[N_RANKS, T], pl.INT64],
    swa_sources: pl.Tensor[[N_RANKS, T, WIN], pl.INT32],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [N_RANKS, HCA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    query_request_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    hca_pages: pl.Tensor[[N_RANKS, HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[N_RANKS, HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[N_RANKS, HCA_B_DYN, 3], pl.INT32],
    hca_query_work_offsets: pl.Tensor[
        [N_RANKS, HCA_QUERY_OFFSETS_DYN], pl.INT32
    ],
    hca_work_query_ids: pl.Tensor[[N_RANKS, HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[N_RANKS, HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[N_RANKS, HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    x_attn_workspace: pl.InOut[
        pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]
    ],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
):
    # Keep allocation and typed window openings in this decorated host entry.
    # The frontend does not discover these resources through ordinary helpers.
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32] = pld.window(
            recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32
        )
        recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8] = pld.window(
            recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
        )
        recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32] = pld.window(
            recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
        )
        recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32] = pld.window(
            recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
        )
        arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16] = pld.window(
            routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16
        )
        combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        decode_layer_hca(
            x_hc[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            gamma_cq[rank],
            gamma_ckv[rank],
            query_rope_cos[rank],
            query_rope_sin[rank],
            cmp_wkv[rank],
            cmp_wgate[rank],
            cmp_ape[rank],
            cmp_norm_w[rank],
            request_event_indices[rank],
            event_rope_cos[rank],
            event_rope_sin[rank],
            compress_state[rank],
            state_page_ids[rank],
            state_valid_ranges[rank],
            state_page_epochs[rank],
            request_epochs[rank],
            state_write_slots[rank],
            kv_cache[rank],
            swa_write_slots[rank],
            swa_sources[rank],
            cmp_kv[rank],
            cmp_slot_mapping[rank],
            position_ids[rank],
            query_request_ids[rank],
            hca_pages[rank],
            hca_page_offsets[rank],
            hca_windows[rank],
            hca_query_work_offsets[rank],
            hca_work_query_ids[rank],
            hca_work_row_begin[rank],
            hca_work_valid_rows[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank],
            gate_bias[rank],
            tid2eid[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            num_tokens_per_owner,
            x_attn_workspace[rank],
            x_next[rank],
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            rank,
            pl.const(1, pl.INT32),
            device=rank,
        )
    return x_next


def submit_decode_layer_hca(
    x_hc,
    hc_attn_fn,
    hc_attn_scale,
    hc_attn_base,
    attn_norm_w,
    wq_a,
    wq_b,
    wq_b_scale,
    wkv,
    gamma_cq,
    gamma_ckv,
    query_rope_cos,
    query_rope_sin,
    cmp_wkv,
    cmp_wgate,
    cmp_ape,
    cmp_norm_w,
    request_event_indices,
    event_rope_cos,
    event_rope_sin,
    compress_state,
    state_page_ids,
    state_valid_ranges,
    state_page_epochs,
    request_epochs,
    state_write_slots,
    kv_cache,
    swa_write_slots,
    swa_sources,
    cmp_kv,
    cmp_slot_mapping,
    position_ids,
    query_request_ids,
    hca_pages,
    hca_page_offsets,
    hca_windows,
    hca_query_work_offsets,
    hca_work_query_ids,
    hca_work_row_begin,
    hca_work_valid_rows,
    attn_sink,
    wo_a,
    wo_b,
    wo_b_scale,
    hc_ffn_fn,
    hc_ffn_scale,
    hc_ffn_base,
    norm_w,
    gate_w,
    gate_bias,
    tid2eid,
    input_ids,
    routed_w1,
    routed_w1_scale,
    routed_w3,
    routed_w3_scale,
    routed_w2,
    routed_w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w3,
    shared_w3_scale,
    shared_w2,
    shared_w2_scale,
    num_tokens_per_owner,
    x_attn_workspace,
    x_next,
    recv_meta_buf,
    recv_x_buf,
    recv_aux_buf,
    recv_route_buf,
    arrived_buf,
    data_arrived_buf,
    routed_y_buf_buf,
    combine_arrived_buf,
    layer_id,
    moe_epoch,
):
    """Submit one HCA layer with the enclosing forward's MoE windows."""
    for rank in pl.range(pld.world_size()):
        (
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
        ) = open_moe_signal_windows(
            recv_meta_buf,
            recv_x_buf,
            recv_aux_buf,
            recv_route_buf,
            arrived_buf,
            data_arrived_buf,
            routed_y_buf_buf,
            combine_arrived_buf,
        )
        decode_layer_hca(
            x_hc[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            gamma_cq[rank],
            gamma_ckv[rank],
            query_rope_cos[rank],
            query_rope_sin[rank],
            cmp_wkv[rank],
            cmp_wgate[rank],
            cmp_ape[rank],
            cmp_norm_w[rank],
            request_event_indices[rank],
            event_rope_cos[rank],
            event_rope_sin[rank],
            compress_state[rank],
            state_page_ids[rank],
            state_valid_ranges[rank],
            state_page_epochs[rank],
            request_epochs[rank],
            state_write_slots[rank],
            kv_cache[rank],
            swa_write_slots[rank],
            swa_sources[rank],
            cmp_kv[rank],
            cmp_slot_mapping[rank],
            position_ids[rank],
            query_request_ids[rank],
            hca_pages[rank],
            hca_page_offsets[rank],
            hca_windows[rank],
            hca_query_work_offsets[rank],
            hca_work_query_ids[rank],
            hca_work_row_begin[rank],
            hca_work_valid_rows[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank],
            gate_bias[rank],
            tid2eid[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            num_tokens_per_owner,
            x_attn_workspace[rank],
            x_next[rank],
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            rank,
            moe_epoch,
            device=rank,
        )
    return x_next


@pl.jit.inline
def decode_layer_csa_chunk(
    x_hc: pl.Tensor[[CSA_CHUNK_T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    main_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_ape: pl.Tensor[[4, CSA_MAIN_OUT_DIM], pl.FP32],
    main_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    main_state: pl.Tensor[
        [CSA_MAIN_STATE_POOL_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ],
    main_state_page_ids: pl.Tensor[
        [CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    main_state_valid_ranges: pl.Tensor[[CSA_CHUNK_B, 2], pl.INT32],
    main_state_page_epochs: pl.Tensor[
        [CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    compressor_request_epochs: pl.Tensor[[CSA_CHUNK_B], pl.INT32],
    request_event_indices: pl.Tensor[[CSA_CHUNK_B], pl.INT32],
    event_query_ids: pl.Tensor[[CSA_EVENT_CAP], pl.INT32],
    event_rope_cos: pl.Tensor[
        [CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    event_rope_sin: pl.Tensor[
        [CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    main_event_write_slots: pl.Tensor[[CSA_EVENT_CAP], pl.INT64],
    position_ids: pl.Tensor[[CSA_CHUNK_T], pl.INT32],
    main_state_write_slots: pl.Tensor[[CSA_CHUNK_T], pl.INT64],
    main_cache: pl.Tensor[
        [CSA_MAIN_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[4, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_state: pl.Tensor[
        [CSA_INNER_STATE_POOL_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    inner_state_page_ids: pl.Tensor[
        [CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_state_valid_ranges: pl.Tensor[[CSA_CHUNK_B, 2], pl.INT32],
    inner_state_page_epochs: pl.Tensor[
        [CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_event_write_slots: pl.Tensor[[CSA_EVENT_CAP], pl.INT64],
    inner_state_write_slots: pl.Tensor[[CSA_CHUNK_T], pl.INT64],
    idx_kv_cache_flat: pl.Tensor[[CSA_IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[CSA_IDX_ROWS_DYN, 1], pl.FP32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    idx_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_cos_il: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32],
    idx_sin_signed: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32],
    query_request_ids: pl.Tensor[[CSA_CHUNK_T], pl.INT32],
    idx_pages: pl.Tensor[[CSA_PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[CSA_REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[CSA_REQUEST_DYN, 3], pl.INT32],
    page_request_epochs: pl.Tensor[[CSA_REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[CSA_LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_descriptors: pl.Tensor[[CSA_PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32],
    singleton_descriptors: pl.Tensor[
        [CSA_SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32
    ],
    upper_descriptors: pl.Tensor[
        [CSA_UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    root_descriptors: pl.Tensor[
        [CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], pl.INT32
    ],
    csa_pages: pl.Tensor[[CSA_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[CSA_REQUEST_OFFSET_DYN], pl.INT32],
    csa_windows: pl.Tensor[[CSA_REQUEST_DYN, 3], pl.INT32],
    kv_cache: pl.Tensor[
        [CSA_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ],
    raw_write_slots: pl.Tensor[[CSA_CHUNK_T], pl.INT64],
    swa_sources: pl.Tensor[[CSA_CHUNK_T, WIN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    pair_arena: pl.Tensor[[CSA_ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32],
    x_mixed: pl.Tensor[[CSA_CHUNK_T, D], pl.BF16],
    x: pl.Tensor[[CSA_CHUNK_T, D], pl.BF16],
    q: pl.Tensor[[CSA_CHUNK_T, H, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[CSA_CHUNK_T, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[CSA_CHUNK_T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[CSA_CHUNK_T, 1], pl.FP32],
    post: pl.Tensor[[CSA_CHUNK_T, HC_MULT], pl.FP32],
    comb: pl.Tensor[[CSA_CHUNK_T, HC_MULT * HC_MULT], pl.FP32],
    main_overlay: pl.Tensor[[CSA_CHUNK_T, HEAD_DIM], pl.FP32],
    inner_overlay: pl.Tensor[[CSA_CHUNK_T, IDX_HEAD_DIM], pl.FP32],
    query_vectors: pl.Tensor[
        [CSA_CHUNK_T, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8
    ],
    query_scales: pl.Tensor[[CSA_CHUNK_T, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[CSA_CHUNK_T, IDX_N_HEADS], pl.FP32],
    topk_scores: pl.Tensor[[CSA_CHUNK_T, CSA_INDEX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[CSA_CHUNK_T, CSA_INDEX_TOPK], pl.INT32],
    attn_out: pl.Tensor[[CSA_CHUNK_T, D], pl.BF16],
    x_attn: pl.Tensor[[CSA_CHUNK_T, HC_MULT, D], pl.FP32],
):
    """Inline one fixed-size CSA shard into the rank-local layer child."""
    hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )
    rms_tid = rms_norm(x_mixed, attn_norm_w, x)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    qkv_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos,
        rope_sin,
        gamma_cq,
        gamma_ckv,
        q,
        current_kv,
        qr,
        qr_scale,
        late_dep,
    )

    # Cache/page extents remain allocator-specialized.  The query dimension is
    # the compile-time CSA micro-shard size and must stay lexical here: binding
    # a new dynamic symbol inside the enclosing full-forward layer loop would
    # make that symbol escape its IR scope after inline expansion.
    main_state.bind_dynamic(0, CSA_MAIN_STATE_POOL_DYN)
    main_state_page_ids.bind_dynamic(0, CSA_MAIN_B_DYN)
    main_state_valid_ranges.bind_dynamic(0, CSA_MAIN_B_DYN)
    main_state_page_epochs.bind_dynamic(0, CSA_MAIN_B_DYN)
    compressor_request_epochs.bind_dynamic(0, CSA_MAIN_B_DYN)
    request_event_indices.bind_dynamic(0, CSA_MAIN_B_DYN)
    event_query_ids.bind_dynamic(0, CSA_MAIN_EVENT_DYN)
    event_rope_cos.bind_dynamic(0, CSA_MAIN_EVENT_DYN)
    event_rope_sin.bind_dynamic(0, CSA_MAIN_EVENT_DYN)
    main_event_write_slots.bind_dynamic(0, CSA_MAIN_EVENT_DYN)
    main_cache.bind_dynamic(0, CSA_MAIN_CACHE_DYN)

    event_cos_il = pl.create_tensor(
        [CSA_EVENT_CAP, ROPE_HEAD_DIM], dtype=pl.FP32
    )
    event_sin_signed = pl.create_tensor(
        [CSA_EVENT_CAP, ROPE_HEAD_DIM], dtype=pl.FP32
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="csa_layer_event_rope_interleave",
        allow_early_resolve=True,
    ):
        rope_ones = pl.full([1, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(
            rope_ones,
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        rope_dup_f = pl.cast(
            pl.cast(
                pl.mul(rope_col, 0.5),
                target_type=pl.INT32,
                mode="trunc",
            ),
            target_type=pl.FP32,
        )
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        for event in pl.range(CSA_EVENT_CAP):
            event_cos_il[event : event + 1, :] = pl.gather(
                event_rope_cos[event : event + 1, :],
                dim=-1,
                index=rope_dup_idx,
            )
            event_sin_signed[event : event + 1, :] = pl.mul(
                pl.gather(
                    event_rope_sin[event : event + 1, :],
                    dim=-1,
                    index=rope_dup_idx,
                ),
                rope_sign,
            )
    # These are local fixed-capacity workspaces. Dynamic bindings belong to
    # decorated function parameters and are removed by the specializer; a
    # method call on a locally created tensor survives inline expansion and is
    # not a supported IR operation.

    # PyPTO's inline specializer currently keys inferred tensor metadata by
    # the callee parameter name.  Publish exact-name aliases for each Phase-D
    # production leaf; no copy or host round-trip is introduced.
    kv = main_overlay
    compress_state = main_state
    state_page_ids = main_state_page_ids
    state_valid_ranges = main_state_valid_ranges
    state_page_epochs = main_state_page_epochs
    request_epochs = compressor_request_epochs
    wkv = main_wkv
    wgate = main_wgate
    ape = main_ape
    norm_w = main_norm_w
    cos = event_cos_il
    sin = event_sin_signed
    cmp_kv_cache = main_cache
    event_write_slots = main_event_write_slots
    state_slot_mapping = main_state_write_slots
    main_cache_done, _main_state_done = compressor_ratio4(
        x,
        kv,
        compress_state,
        state_page_ids,
        state_valid_ranges,
        state_page_epochs,
        request_epochs,
        wkv,
        wgate,
        ape,
        norm_w,
        request_event_indices,
        event_query_ids,
        cos,
        sin,
        cmp_kv_cache,
        event_write_slots,
        position_ids,
        state_slot_mapping,
        late_dep,
    )
    kv = inner_overlay
    compress_state = inner_state
    state_page_ids = inner_state_page_ids
    state_valid_ranges = inner_state_valid_ranges
    state_page_epochs = inner_state_page_epochs
    request_epochs = compressor_request_epochs
    wkv = inner_wkv
    wgate = inner_wgate
    ape = inner_ape
    norm_w = inner_norm_w
    hadamard = inner_hadamard
    idx_kv_cache = idx_kv_cache_flat
    idx_kv_scale = idx_kv_scale_flat
    idx_kv_cache_flat.bind_dynamic(0, CSA_INNER_CACHE_ROWS_DYN)
    idx_kv_scale_flat.bind_dynamic(0, CSA_INNER_CACHE_ROWS_DYN)
    event_write_slots = inner_event_write_slots
    state_slot_mapping = inner_state_write_slots
    inner_cache_done, _inner_state_done = indexer_compressor(
        x,
        kv,
        compress_state,
        state_page_ids,
        state_valid_ranges,
        state_page_epochs,
        request_epochs,
        wkv,
        wgate,
        ape,
        norm_w,
        request_event_indices,
        event_query_ids,
        cos,
        sin,
        hadamard,
        idx_kv_cache,
        idx_kv_scale,
        event_write_slots,
        position_ids,
        state_slot_mapping,
        late_dep,
    )

    root_completion = pl.array.create(1, pl.TASK_ID)
    wq_b = idx_wq_b
    wq_b_scale = idx_wq_b_scale
    weights_proj = idx_weights_proj
    cos_il = idx_cos_il
    sin_signed = idx_sin_signed
    hadamard = idx_hadamard
    request_epochs = page_request_epochs
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
        inner_cache_done,
        late_dep,
        root_completion,
    )

    ori_kv = kv_cache
    cmp_kv = main_cache
    idx_topk = topk_indices
    freqs_cos = rope_cos
    freqs_sin = rope_sin
    request_epochs = page_request_epochs
    value_done = sparse_attn_csa(
        q,
        ori_kv,
        current_kv,
        swa_sources,
        cmp_kv,
        query_request_ids,
        csa_pages,
        csa_page_offsets,
        csa_windows,
        request_epochs,
        idx_topk,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    raw_blocks = pl.tensor.dim(kv_cache, 0)
    raw_flat = pl.reshape(kv_cache, [raw_blocks * BLOCK_SIZE, HEAD_DIM])
    with pl.spmd(
        CSA_CHUNK_T // 8,
        name_hint="csa_layer_raw_commit",
        deps=[value_done],
    ) as _raw_commit_tid:
        token0 = pl.tile.get_block_idx() * 8
        for local_token in pl.range(8):
            token = token0 + local_token
            row_i64 = pl.read(raw_write_slots, [token])
            if row_i64 >= 0:
                row = pl.cast(row_i64, target_type=pl.INDEX)
                raw_flat[row : row + 1, :] = current_kv[
                    token : token + 1, :
                ]
    hc_post(attn_out, x_hc, post, comb, x_attn)
    return x_attn


@pl.jit(auto_scope=False)
def decode_layer_csa(
    x_hc: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    main_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_ape: pl.Tensor[[4, CSA_MAIN_OUT_DIM], pl.FP32],
    main_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    main_state: pl.InOut[pl.Tensor[
        [CSA_MAIN_STATE_POOL_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ]],
    main_state_page_ids: pl.Tensor[
        [CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    main_state_valid_ranges: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_B, 2], pl.INT32],
    main_state_page_epochs: pl.Tensor[
        [CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    compressor_request_epochs: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_B], pl.INT32],
    request_event_indices: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_B], pl.INT32],
    event_query_ids: pl.Tensor[[CSA_CHUNKS, CSA_EVENT_CAP], pl.INT32],
    event_rope_cos: pl.Tensor[
        [CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    event_rope_sin: pl.Tensor[
        [CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    main_event_write_slots: pl.Tensor[[CSA_CHUNKS, CSA_EVENT_CAP], pl.INT64],
    position_ids: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T], pl.INT32],
    main_state_write_slots: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T], pl.INT64],
    main_cache: pl.InOut[pl.Tensor[
        [CSA_MAIN_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[4, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_state: pl.InOut[pl.Tensor[
        [CSA_INNER_STATE_POOL_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ]],
    inner_state_page_ids: pl.Tensor[
        [CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_state_valid_ranges: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_B, 2], pl.INT32],
    inner_state_page_epochs: pl.Tensor[
        [CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_event_write_slots: pl.Tensor[[CSA_CHUNKS, CSA_EVENT_CAP], pl.INT64],
    inner_state_write_slots: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T], pl.INT64],
    idx_cache: pl.InOut[pl.Tensor[[CSA_IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8]],
    idx_scale: pl.InOut[pl.Tensor[[CSA_IDX_ROWS_DYN, 1], pl.FP32]],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    idx_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_cos_il: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32],
    idx_sin_signed: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32],
    query_request_ids: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T], pl.INT32],
    idx_pages: pl.Tensor[[CSA_CHUNKS, CSA_PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[CSA_CHUNKS, CSA_REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[CSA_CHUNKS, CSA_REQUEST_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[CSA_CHUNKS, CSA_REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[
        [CSA_CHUNKS, CSA_LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    pair_descriptors: pl.Tensor[
        [CSA_CHUNKS, CSA_PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    singleton_descriptors: pl.Tensor[
        [CSA_CHUNKS, CSA_SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32
    ],
    upper_descriptors: pl.Tensor[
        [CSA_CHUNKS, CSA_UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    root_descriptors: pl.Tensor[
        [CSA_CHUNKS, CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], pl.INT32
    ],
    csa_pages: pl.Tensor[[CSA_CHUNKS, CSA_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[CSA_CHUNKS, CSA_REQUEST_OFFSET_DYN], pl.INT32],
    csa_windows: pl.Tensor[[CSA_CHUNKS, CSA_REQUEST_DYN, 3], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[
        [CSA_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    raw_write_slots: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T], pl.INT64],
    swa_sources: pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, WIN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    csa_x_attn_workspace: pl.Out[pl.Tensor[
        [CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D], pl.FP32
    ]],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run every CSA shard and the rank-local MoE in one L2 program."""
    for chunk in pl.range(CSA_SUBMISSION_CHUNKS):
        with pl.scope():
            # Give every writable chunk view a rank-child local Var before it
            # crosses the inline boundary. Passing ``workspace[chunk]``
            # directly makes the inline mutator substitute a TensorSlice for
            # writable parameters in hc_pre/indexer and breaks assignment
            # lowering. These aliases do not copy storage.
            page_rows = pl.tensor.dim(idx_pages, 1)
            request_offset_rows = pl.tensor.dim(idx_page_offsets, 1)
            request_rows = pl.tensor.dim(idx_windows, 1)
            leaf_rows = pl.tensor.dim(leaf_descriptors, 1)
            pair_rows = pl.tensor.dim(pair_descriptors, 1)
            singleton_rows = pl.tensor.dim(singleton_descriptors, 1)
            upper_rows = pl.tensor.dim(upper_descriptors, 1)
            chunk_pair_arena = pl.create_tensor(
                [CSA_CHUNK_T * CSA_MAX_NODES_PER_QUERY, CSA_PAIR_WIDTH],
                dtype=pl.FP32,
            )
            x_mixed = pl.create_tensor([CSA_CHUNK_T, D], dtype=pl.BF16)
            x = pl.create_tensor([CSA_CHUNK_T, D], dtype=pl.BF16)
            q = pl.create_tensor([CSA_CHUNK_T, H, HEAD_DIM], dtype=pl.BF16)
            current_kv = pl.create_tensor([CSA_CHUNK_T, HEAD_DIM], dtype=pl.BF16)
            qr = pl.create_tensor([CSA_CHUNK_T, Q_LORA], dtype=pl.INT8)
            qr_scale = pl.create_tensor([CSA_CHUNK_T, 1], dtype=pl.FP32)
            post = pl.create_tensor([CSA_CHUNK_T, HC_MULT], dtype=pl.FP32)
            comb = pl.create_tensor(
                [CSA_CHUNK_T, HC_MULT * HC_MULT], dtype=pl.FP32
            )
            main_overlay = pl.create_tensor(
                [CSA_CHUNK_T, HEAD_DIM], dtype=pl.FP32
            )
            inner_overlay = pl.create_tensor(
                [CSA_CHUNK_T, IDX_HEAD_DIM], dtype=pl.FP32
            )
            query_vectors = pl.create_tensor(
                [CSA_CHUNK_T, IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8
            )
            query_scales = pl.create_tensor(
                [CSA_CHUNK_T, IDX_N_HEADS], dtype=pl.FP32
            )
            query_weights = pl.create_tensor(
                [CSA_CHUNK_T, IDX_N_HEADS], dtype=pl.FP32
            )
            topk_scores = pl.create_tensor(
                [CSA_CHUNK_T, CSA_INDEX_TOPK], dtype=pl.FP32
            )
            topk_indices = pl.create_tensor(
                [CSA_CHUNK_T, CSA_INDEX_TOPK], dtype=pl.INT32
            )
            attn_out = pl.create_tensor([CSA_CHUNK_T, D], dtype=pl.BF16)
            x_attn_out = pl.create_tensor(
                [CSA_CHUNK_T, HC_MULT, D], dtype=pl.FP32
            )
            chunk_x_hc: pl.Tensor[[CSA_CHUNK_T, HC_MULT, D], pl.FP32] = pl.reshape(pl.slice(x_hc, [1, CSA_CHUNK_T, HC_MULT, D], [chunk, 0, 0, 0]), [CSA_CHUNK_T, HC_MULT, D])
            rope_cos_chunk = pl.create_tensor(
                [CSA_CHUNK_T, ROPE_HEAD_DIM], dtype=pl.BF16
            )
            rope_sin_chunk = pl.create_tensor(
                [CSA_CHUNK_T, ROPE_HEAD_DIM], dtype=pl.BF16
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="csa_layer_rope_chunk_copy",
                allow_early_resolve=True,
            ):
                rope_cos_chunk[:, :] = pl.reshape(
                    pl.slice(
                        rope_cos,
                        [1, CSA_CHUNK_T, ROPE_HEAD_DIM],
                        [chunk, 0, 0],
                    ),
                    [CSA_CHUNK_T, ROPE_HEAD_DIM],
                )
                rope_sin_chunk[:, :] = pl.reshape(
                    pl.slice(
                        rope_sin,
                        [1, CSA_CHUNK_T, ROPE_HEAD_DIM],
                        [chunk, 0, 0],
                    ),
                    [CSA_CHUNK_T, ROPE_HEAD_DIM],
                )
            chunk_main_state_page_ids: pl.Tensor[[CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32] = pl.reshape(pl.slice(main_state_page_ids, [1, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], [chunk, 0, 0]), [CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST])
            chunk_main_state_valid_ranges: pl.Tensor[[CSA_CHUNK_B, 2], pl.INT32] = pl.reshape(pl.slice(main_state_valid_ranges, [1, CSA_CHUNK_B, 2], [chunk, 0, 0]), [CSA_CHUNK_B, 2])
            chunk_main_state_page_epochs: pl.Tensor[[CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32] = pl.reshape(pl.slice(main_state_page_epochs, [1, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], [chunk, 0, 0]), [CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST])
            chunk_compressor_request_epochs: pl.Tensor[[CSA_CHUNK_B], pl.INT32] = pl.reshape(pl.slice(compressor_request_epochs, [1, CSA_CHUNK_B], [chunk, 0]), [CSA_CHUNK_B])
            chunk_request_event_indices: pl.Tensor[[CSA_CHUNK_B], pl.INT32] = pl.reshape(pl.slice(request_event_indices, [1, CSA_CHUNK_B], [chunk, 0]), [CSA_CHUNK_B])
            chunk_event_query_ids: pl.Tensor[[CSA_EVENT_CAP], pl.INT32] = pl.reshape(pl.slice(event_query_ids, [1, CSA_EVENT_CAP], [chunk, 0]), [CSA_EVENT_CAP])
            chunk_event_rope_cos: pl.Tensor[[CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32] = pl.reshape(pl.slice(event_rope_cos, [1, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], [chunk, 0, 0]), [CSA_EVENT_CAP, ROPE_HEAD_DIM // 2])
            chunk_event_rope_sin: pl.Tensor[[CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32] = pl.reshape(pl.slice(event_rope_sin, [1, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], [chunk, 0, 0]), [CSA_EVENT_CAP, ROPE_HEAD_DIM // 2])
            chunk_main_event_write_slots: pl.Tensor[[CSA_EVENT_CAP], pl.INT64] = pl.reshape(pl.slice(main_event_write_slots, [1, CSA_EVENT_CAP], [chunk, 0]), [CSA_EVENT_CAP])
            chunk_position_ids: pl.Tensor[[CSA_CHUNK_T], pl.INT32] = pl.reshape(pl.slice(position_ids, [1, CSA_CHUNK_T], [chunk, 0]), [CSA_CHUNK_T])
            chunk_main_state_write_slots: pl.Tensor[[CSA_CHUNK_T], pl.INT64] = pl.reshape(pl.slice(main_state_write_slots, [1, CSA_CHUNK_T], [chunk, 0]), [CSA_CHUNK_T])
            chunk_inner_state_page_ids: pl.Tensor[[CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32] = pl.reshape(pl.slice(inner_state_page_ids, [1, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], [chunk, 0, 0]), [CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST])
            chunk_inner_state_valid_ranges: pl.Tensor[[CSA_CHUNK_B, 2], pl.INT32] = pl.reshape(pl.slice(inner_state_valid_ranges, [1, CSA_CHUNK_B, 2], [chunk, 0, 0]), [CSA_CHUNK_B, 2])
            chunk_inner_state_page_epochs: pl.Tensor[[CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32] = pl.reshape(pl.slice(inner_state_page_epochs, [1, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], [chunk, 0, 0]), [CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST])
            chunk_inner_event_write_slots: pl.Tensor[[CSA_EVENT_CAP], pl.INT64] = pl.reshape(pl.slice(inner_event_write_slots, [1, CSA_EVENT_CAP], [chunk, 0]), [CSA_EVENT_CAP])
            chunk_inner_state_write_slots: pl.Tensor[[CSA_CHUNK_T], pl.INT64] = pl.reshape(pl.slice(inner_state_write_slots, [1, CSA_CHUNK_T], [chunk, 0]), [CSA_CHUNK_T])
            chunk_idx_cos_il: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32] = pl.reshape(pl.slice(idx_cos_il, [1, CSA_CHUNK_T, ROPE_HEAD_DIM], [chunk, 0, 0]), [CSA_CHUNK_T, ROPE_HEAD_DIM])
            chunk_idx_sin_signed: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32] = pl.reshape(pl.slice(idx_sin_signed, [1, CSA_CHUNK_T, ROPE_HEAD_DIM], [chunk, 0, 0]), [CSA_CHUNK_T, ROPE_HEAD_DIM])
            chunk_query_request_ids: pl.Tensor[[CSA_CHUNK_T], pl.INT32] = pl.reshape(pl.slice(query_request_ids, [1, CSA_CHUNK_T], [chunk, 0]), [CSA_CHUNK_T])
            chunk_idx_pages: pl.Tensor[[page_rows, 2], pl.INT32] = pl.reshape(pl.slice(idx_pages, [1, page_rows, 2], [chunk, 0, 0]), [page_rows, 2])
            chunk_idx_page_offsets: pl.Tensor[[request_offset_rows], pl.INT32] = pl.reshape(pl.slice(idx_page_offsets, [1, request_offset_rows], [chunk, 0]), [request_offset_rows])
            chunk_idx_windows: pl.Tensor[[request_rows, 3], pl.INT32] = pl.reshape(pl.slice(idx_windows, [1, request_rows, 3], [chunk, 0, 0]), [request_rows, 3])
            chunk_request_epochs: pl.Tensor[[request_rows], pl.INT32] = pl.reshape(pl.slice(request_epochs, [1, request_rows], [chunk, 0]), [request_rows])
            chunk_leaf_descriptors: pl.Tensor[[leaf_rows, PHASE_D_LEAF_FIELDS], pl.INT32] = pl.reshape(pl.slice(leaf_descriptors, [1, leaf_rows, PHASE_D_LEAF_FIELDS], [chunk, 0, 0]), [leaf_rows, PHASE_D_LEAF_FIELDS])
            chunk_pair_descriptors: pl.Tensor[[pair_rows, PHASE_D_PAIR_FIELDS], pl.INT32] = pl.reshape(pl.slice(pair_descriptors, [1, pair_rows, PHASE_D_PAIR_FIELDS], [chunk, 0, 0]), [pair_rows, PHASE_D_PAIR_FIELDS])
            chunk_singleton_descriptors: pl.Tensor[[singleton_rows, PHASE_D_SINGLETON_FIELDS], pl.INT32] = pl.reshape(pl.slice(singleton_descriptors, [1, singleton_rows, PHASE_D_SINGLETON_FIELDS], [chunk, 0, 0]), [singleton_rows, PHASE_D_SINGLETON_FIELDS])
            chunk_upper_descriptors: pl.Tensor[[upper_rows, PHASE_D_UPPER_FIELDS], pl.INT32] = pl.reshape(pl.slice(upper_descriptors, [1, upper_rows, PHASE_D_UPPER_FIELDS], [chunk, 0, 0]), [upper_rows, PHASE_D_UPPER_FIELDS])
            chunk_root_descriptors: pl.Tensor[[CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], pl.INT32] = pl.reshape(pl.slice(root_descriptors, [1, CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], [chunk, 0, 0]), [CSA_CHUNK_T, PHASE_D_ROOT_FIELDS])
            chunk_csa_pages: pl.Tensor[[page_rows, 2], pl.INT32] = pl.reshape(pl.slice(csa_pages, [1, page_rows, 2], [chunk, 0, 0]), [page_rows, 2])
            chunk_csa_page_offsets: pl.Tensor[[request_offset_rows], pl.INT32] = pl.reshape(pl.slice(csa_page_offsets, [1, request_offset_rows], [chunk, 0]), [request_offset_rows])
            chunk_csa_windows: pl.Tensor[[request_rows, 3], pl.INT32] = pl.reshape(pl.slice(csa_windows, [1, request_rows, 3], [chunk, 0, 0]), [request_rows, 3])
            chunk_raw_write_slots: pl.Tensor[[CSA_CHUNK_T], pl.INT64] = pl.reshape(pl.slice(raw_write_slots, [1, CSA_CHUNK_T], [chunk, 0]), [CSA_CHUNK_T])
            chunk_swa_sources: pl.Tensor[[CSA_CHUNK_T, WIN], pl.INT32] = pl.reshape(pl.slice(swa_sources, [1, CSA_CHUNK_T, WIN], [chunk, 0, 0]), [CSA_CHUNK_T, WIN])
            _csa_x_attn = decode_layer_csa_chunk(
                chunk_x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                rope_cos_chunk, rope_sin_chunk, main_wkv, main_wgate,
                main_ape, main_norm_w, main_state, chunk_main_state_page_ids,
                chunk_main_state_valid_ranges, chunk_main_state_page_epochs,
                chunk_compressor_request_epochs, chunk_request_event_indices,
                chunk_event_query_ids, chunk_event_rope_cos,
                chunk_event_rope_sin, chunk_main_event_write_slots,
                chunk_position_ids, chunk_main_state_write_slots, main_cache,
                inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_hadamard,
                inner_state, chunk_inner_state_page_ids,
                chunk_inner_state_valid_ranges, chunk_inner_state_page_epochs,
                chunk_inner_event_write_slots, chunk_inner_state_write_slots,
                idx_cache, idx_scale, idx_wq_b, idx_wq_b_scale, idx_weights_proj,
                idx_hadamard, chunk_idx_cos_il, chunk_idx_sin_signed,
                chunk_query_request_ids, chunk_idx_pages,
                chunk_idx_page_offsets, chunk_idx_windows, chunk_request_epochs,
                chunk_leaf_descriptors, chunk_pair_descriptors,
                chunk_singleton_descriptors, chunk_upper_descriptors,
                chunk_root_descriptors, chunk_csa_pages,
                chunk_csa_page_offsets, chunk_csa_windows, kv_cache,
                chunk_raw_write_slots, chunk_swa_sources, attn_sink, wo_a,
                wo_b, wo_b_scale, chunk_pair_arena, x_mixed, x, q,
                current_kv, qr, qr_scale, post, comb, main_overlay,
                inner_overlay, query_vectors, query_scales, query_weights,
                topk_scores, topk_indices, attn_out, x_attn_out,
            )
            csa_x_attn_workspace[
                chunk : chunk + 1, :, :, :
            ] = pl.reshape(
                x_attn_out, [1, CSA_CHUNK_T, HC_MULT, D]
            )

    x_attn = pl.reshape(csa_x_attn_workspace, [T, HC_MULT, D])
    num_tokens = pl.read(num_tokens_per_owner, [my_rank])
    with pl.scope():
        moe(
            x_attn, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w,
            gate_bias, tid2eid, input_ids, routed_w1, routed_w1_scale,
            routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale, x_next, recv_meta, recv_x, recv_aux,
            recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, num_tokens, my_rank, moe_epoch,
        )
    return x_next


@pl.jit(auto_scope=False)
def csa_layer_frontend(
    x_hc: pl.Tensor[[CSA_CHUNK_T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin: pl.Tensor[[CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    csa_x_mixed_workspace: pl.Out[pl.Tensor[[CSA_CHUNK_T, D], pl.BF16]],
    csa_x_normed_workspace: pl.Out[pl.Tensor[[CSA_CHUNK_T, D], pl.BF16]],
    csa_q_workspace: pl.Out[
        pl.Tensor[[CSA_CHUNK_T, H, HEAD_DIM], pl.BF16]
    ],
    csa_current_kv_workspace: pl.Out[
        pl.Tensor[[CSA_CHUNK_T, HEAD_DIM], pl.BF16]
    ],
    csa_qr_workspace: pl.Out[pl.Tensor[[CSA_CHUNK_T, Q_LORA], pl.INT8]],
    csa_qr_scale_workspace: pl.Out[
        pl.Tensor[[CSA_CHUNK_T, 1], pl.FP32]
    ],
    csa_post_workspace: pl.Out[
        pl.Tensor[[CSA_CHUNK_T, HC_MULT], pl.FP32]
    ],
    csa_comb_workspace: pl.Out[
        pl.Tensor[[CSA_CHUNK_T, HC_MULT * HC_MULT], pl.FP32]
    ],
):
    """Materialize the reusable CSA query frontend for one fixed micro-shard."""
    x_mixed = csa_x_mixed_workspace
    x_normed = csa_x_normed_workspace
    q = csa_q_workspace
    current_kv = csa_current_kv_workspace
    qr = csa_qr_workspace
    qr_scale = csa_qr_scale_workspace
    post = csa_post_workspace
    comb = csa_comb_workspace
    hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    qkv_proj_rope(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos,
        rope_sin,
        gamma_cq,
        gamma_ckv,
        q,
        current_kv,
        qr,
        qr_scale,
        late_dep,
    )
    return (
        csa_x_normed_workspace,
        csa_q_workspace,
        csa_current_kv_workspace,
        csa_qr_workspace,
        csa_qr_scale_workspace,
        csa_post_workspace,
        csa_comb_workspace,
    )


@pl.jit
def csa_main_compressor_layer_stage(
    csa_x_normed_workspace: pl.Tensor[[CSA_MAIN_T_DYN, D], pl.BF16],
    csa_main_overlay_workspace: pl.Out[
        pl.Tensor[[CSA_MAIN_T_DYN, HEAD_DIM], pl.FP32]
    ],
    main_state: pl.InOut[pl.Tensor[
        [CSA_MAIN_STATE_POOL_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32
    ]],
    main_state_page_ids: pl.Tensor[
        [CSA_MAIN_B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    main_state_valid_ranges: pl.Tensor[[CSA_MAIN_B_DYN, 2], pl.INT32],
    main_state_page_epochs: pl.Tensor[
        [CSA_MAIN_B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    compressor_request_epochs: pl.Tensor[[CSA_MAIN_B_DYN], pl.INT32],
    main_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_ape: pl.Tensor[[4, CSA_MAIN_OUT_DIM], pl.FP32],
    main_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    request_event_indices: pl.Tensor[[CSA_MAIN_B_DYN], pl.INT32],
    event_query_ids: pl.Tensor[[CSA_MAIN_EVENT_DYN], pl.INT32],
    event_rope_cos: pl.Tensor[[CSA_MAIN_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    event_rope_sin: pl.Tensor[[CSA_MAIN_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    main_cache: pl.InOut[pl.Tensor[
        [CSA_MAIN_CACHE_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    main_event_write_slots: pl.Tensor[[CSA_MAIN_EVENT_DYN], pl.INT64],
    position_ids: pl.Tensor[[CSA_MAIN_T_DYN], pl.INT32],
    main_state_write_slots: pl.Tensor[[CSA_MAIN_T_DYN], pl.INT64],
):
    x = csa_x_normed_workspace
    kv = csa_main_overlay_workspace
    compress_state = main_state
    state_page_ids = main_state_page_ids
    state_valid_ranges = main_state_valid_ranges
    state_page_epochs = main_state_page_epochs
    request_epochs = compressor_request_epochs
    wkv = main_wkv
    wgate = main_wgate
    ape = main_ape
    norm_w = main_norm_w
    cmp_kv_cache = main_cache
    event_write_slots = main_event_write_slots
    state_slot_mapping = main_state_write_slots
    cos_il = pl.create_tensor(
        [CSA_EVENT_CAP, ROPE_HEAD_DIM], dtype=pl.FP32
    )
    sin_signed = pl.create_tensor(
        [CSA_EVENT_CAP, ROPE_HEAD_DIM], dtype=pl.FP32
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="csa_layer_main_event_rope",
        allow_early_resolve=True,
    ):
        rope_ones = pl.full([1, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(
            rope_ones,
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        rope_dup_f = pl.cast(
            pl.cast(
                pl.mul(rope_col, 0.5),
                target_type=pl.INT32,
                mode="trunc",
            ),
            target_type=pl.FP32,
        )
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        for event in pl.range(CSA_EVENT_CAP):
            cos_il[event : event + 1, :] = pl.gather(
                event_rope_cos[event : event + 1, :],
                dim=-1,
                index=rope_dup_idx,
            )
            sin_signed[event : event + 1, :] = pl.mul(
                pl.gather(
                    event_rope_sin[event : event + 1, :],
                    dim=-1,
                    index=rope_dup_idx,
                ),
                rope_sign,
            )
    late_dep = pl.system.task_dummy(deps=[])
    cos = cos_il
    sin = sin_signed
    compressor_ratio4(
        x, kv, compress_state, state_page_ids, state_valid_ranges, state_page_epochs,
        request_epochs, wkv, wgate, ape, norm_w, request_event_indices,
        event_query_ids, cos, sin, cmp_kv_cache, event_write_slots,
        position_ids, state_slot_mapping, late_dep,
    )
    return kv, compress_state, cmp_kv_cache


@pl.jit
def csa_inner_compressor_layer_stage(
    csa_x_normed_workspace: pl.Tensor[[CSA_INNER_T_DYN, D], pl.BF16],
    csa_inner_overlay_workspace: pl.Out[
        pl.Tensor[[CSA_INNER_T_DYN, IDX_HEAD_DIM], pl.FP32]
    ],
    inner_state: pl.InOut[pl.Tensor[
        [CSA_INNER_STATE_POOL_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ]],
    inner_state_page_ids: pl.Tensor[
        [CSA_INNER_B_DYN, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_state_valid_ranges: pl.Tensor[[CSA_INNER_B_DYN, 2], pl.INT32],
    inner_state_page_epochs: pl.Tensor[
        [CSA_INNER_B_DYN, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    compressor_request_epochs: pl.Tensor[[CSA_INNER_B_DYN], pl.INT32],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[4, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    request_event_indices: pl.Tensor[[CSA_INNER_B_DYN], pl.INT32],
    event_query_ids: pl.Tensor[[CSA_INNER_EVENT_DYN], pl.INT32],
    event_rope_cos: pl.Tensor[[CSA_INNER_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    event_rope_sin: pl.Tensor[[CSA_INNER_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    inner_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_cache: pl.InOut[
        pl.Tensor[[CSA_IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_scale: pl.InOut[
        pl.Tensor[[CSA_IDX_ROWS_DYN, 1], pl.FP32]
    ],
    inner_event_write_slots: pl.Tensor[[CSA_INNER_EVENT_DYN], pl.INT64],
    position_ids: pl.Tensor[[CSA_INNER_T_DYN], pl.INT32],
    inner_state_write_slots: pl.Tensor[[CSA_INNER_T_DYN], pl.INT64],
):
    x = csa_x_normed_workspace
    kv = csa_inner_overlay_workspace
    compress_state = inner_state
    state_page_ids = inner_state_page_ids
    state_valid_ranges = inner_state_valid_ranges
    state_page_epochs = inner_state_page_epochs
    request_epochs = compressor_request_epochs
    wkv = inner_wkv
    wgate = inner_wgate
    ape = inner_ape
    norm_w = inner_norm_w
    hadamard = inner_hadamard
    idx_kv_cache_flat = idx_cache
    idx_kv_scale_flat = idx_scale
    event_write_slots = inner_event_write_slots
    state_slot_mapping = inner_state_write_slots
    cos_il = pl.create_tensor(
        [CSA_EVENT_CAP, ROPE_HEAD_DIM], dtype=pl.FP32
    )
    sin_signed = pl.create_tensor(
        [CSA_EVENT_CAP, ROPE_HEAD_DIM], dtype=pl.FP32
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="csa_layer_inner_event_rope",
        allow_early_resolve=True,
    ):
        rope_ones = pl.full([1, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(
            rope_ones,
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        rope_dup_f = pl.cast(
            pl.cast(
                pl.mul(rope_col, 0.5),
                target_type=pl.INT32,
                mode="trunc",
            ),
            target_type=pl.FP32,
        )
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        for event in pl.range(CSA_EVENT_CAP):
            cos_il[event : event + 1, :] = pl.gather(
                event_rope_cos[event : event + 1, :],
                dim=-1,
                index=rope_dup_idx,
            )
            sin_signed[event : event + 1, :] = pl.mul(
                pl.gather(
                    event_rope_sin[event : event + 1, :],
                    dim=-1,
                    index=rope_dup_idx,
                ),
                rope_sign,
            )
    late_dep = pl.system.task_dummy(deps=[])
    cos = cos_il
    sin = sin_signed
    indexer_compressor(
        x, kv, compress_state, state_page_ids, state_valid_ranges, state_page_epochs,
        request_epochs, wkv, wgate, ape, norm_w, request_event_indices,
        event_query_ids, cos, sin, hadamard,
        idx_kv_cache_flat, idx_kv_scale_flat,
        event_write_slots, position_ids, state_slot_mapping, late_dep,
    )
    return kv, compress_state, idx_kv_cache_flat, idx_kv_scale_flat


@pl.jit
def csa_indexer_layer_stage(
    csa_x_normed_workspace: pl.Tensor[[CSA_INDEXER_T_DYN, D], pl.BF16],
    csa_qr_workspace: pl.Tensor[[CSA_INDEXER_T_DYN, Q_LORA], pl.INT8],
    csa_qr_scale_workspace: pl.Tensor[[CSA_INDEXER_T_DYN, 1], pl.FP32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    idx_cos_il: pl.Tensor[[CSA_INDEXER_T_DYN, ROPE_HEAD_DIM], pl.FP32],
    idx_sin_signed: pl.Tensor[[CSA_INDEXER_T_DYN, ROPE_HEAD_DIM], pl.FP32],
    idx_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_query_vectors_workspace: pl.InOut[pl.Tensor[
        [CSA_INDEXER_T_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8
    ]],
    csa_query_scales_workspace: pl.InOut[
        pl.Tensor[[CSA_INDEXER_T_DYN, IDX_N_HEADS], pl.FP32]
    ],
    csa_query_weights_workspace: pl.InOut[
        pl.Tensor[[CSA_INDEXER_T_DYN, IDX_N_HEADS], pl.FP32]
    ],
    idx_cache: pl.InOut[
        pl.Tensor[[CSA_INDEXER_ROW_DYN, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_scale: pl.InOut[
        pl.Tensor[[CSA_INDEXER_ROW_DYN, 1], pl.FP32]
    ],
    query_request_ids: pl.Tensor[[CSA_INDEXER_T_DYN], pl.INT32],
    idx_pages: pl.Tensor[[CSA_INDEXER_PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[CSA_INDEXER_REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[CSA_INDEXER_B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[CSA_INDEXER_B_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[
        [CSA_INDEXER_LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    pair_descriptors: pl.Tensor[
        [CSA_INDEXER_PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    singleton_descriptors: pl.Tensor[
        [CSA_INDEXER_SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32
    ],
    upper_descriptors: pl.Tensor[
        [CSA_INDEXER_UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    root_descriptors: pl.Tensor[
        [CSA_INDEXER_T_DYN, PHASE_D_ROOT_FIELDS], pl.INT32
    ],
    pair_arena: pl.InOut[
        pl.Tensor[[CSA_INDEXER_ARENA_DYN, CSA_PAIR_WIDTH], pl.FP32]
    ],
    csa_topk_scores_workspace: pl.InOut[
        pl.Tensor[[CSA_INDEXER_T_DYN, CSA_INDEX_TOPK], pl.FP32]
    ],
    csa_topk_indices_workspace: pl.InOut[
        pl.Tensor[[CSA_INDEXER_T_DYN, CSA_INDEX_TOPK], pl.INT32]
    ],
):
    x = csa_x_normed_workspace
    qr = csa_qr_workspace
    qr_scale = csa_qr_scale_workspace
    wq_b = idx_wq_b
    wq_b_scale = idx_wq_b_scale
    weights_proj = idx_weights_proj
    cos_il = idx_cos_il
    sin_signed = idx_sin_signed
    hadamard = idx_hadamard
    query_vectors = csa_query_vectors_workspace
    query_scales = csa_query_scales_workspace
    query_weights = csa_query_weights_workspace
    idx_kv_cache_flat = idx_cache
    idx_kv_scale_flat = idx_scale
    topk_scores = csa_topk_scores_workspace
    topk_indices = csa_topk_indices_workspace
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
    return csa_topk_scores_workspace, csa_topk_indices_workspace


@pl.jit
def csa_sparse_value_layer_stage(
    csa_q_workspace: pl.Tensor[
        [CSA_SPARSE_T_DYN, H, HEAD_DIM], pl.BF16
    ],
    kv_cache: pl.InOut[pl.Tensor[
        [CSA_SPARSE_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    csa_current_kv_workspace: pl.Tensor[
        [CSA_SPARSE_T_DYN, HEAD_DIM], pl.BF16
    ],
    swa_sources: pl.Tensor[[CSA_SPARSE_T_DYN, WIN], pl.INT32],
    main_cache: pl.InOut[pl.Tensor[
        [CSA_SPARSE_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    query_request_ids: pl.Tensor[[CSA_SPARSE_T_DYN], pl.INT32],
    csa_pages: pl.Tensor[[CSA_SPARSE_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[CSA_SPARSE_REQUEST_OFFSET_DYN], pl.INT32],
    csa_windows: pl.Tensor[[CSA_SPARSE_B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[CSA_SPARSE_B_DYN], pl.INT32],
    csa_topk_indices_workspace: pl.Tensor[
        [CSA_SPARSE_T_DYN, CSA_INDEX_TOPK], pl.INT32
    ],
    attn_sink: pl.Tensor[[H], pl.FP32],
    rope_cos: pl.Tensor[[CSA_SPARSE_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    rope_sin: pl.Tensor[[CSA_SPARSE_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    csa_attn_out_workspace: pl.Out[
        pl.Tensor[[CSA_SPARSE_T_DYN, D], pl.BF16]
    ],
):
    sparse_attn_csa(
        csa_q_workspace,
        kv_cache,
        csa_current_kv_workspace,
        swa_sources,
        main_cache,
        query_request_ids,
        csa_pages,
        csa_page_offsets,
        csa_windows,
        request_epochs,
        csa_topk_indices_workspace,
        attn_sink,
        rope_cos,
        rope_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        csa_attn_out_workspace,
    )
    return csa_attn_out_workspace


@pl.jit(auto_scope=False)
def csa_layer_finalize(
    csa_attn_out_workspace: pl.Tensor[[CSA_CHUNK_T, D], pl.BF16],
    csa_current_kv_workspace: pl.Tensor[
        [CSA_CHUNK_T, HEAD_DIM], pl.BF16
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[CSA_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    raw_write_slots: pl.Tensor[[CSA_CHUNK_T], pl.INT64],
    x_hc: pl.Tensor[[CSA_CHUNK_T, HC_MULT, D], pl.FP32],
    csa_post_workspace: pl.Tensor[[CSA_CHUNK_T, HC_MULT], pl.FP32],
    csa_comb_workspace: pl.Tensor[
        [CSA_CHUNK_T, HC_MULT * HC_MULT], pl.FP32
    ],
    csa_x_attn_workspace: pl.Out[
        pl.Tensor[[CSA_CHUNK_T, HC_MULT, D], pl.FP32]
    ],
):
    """Join CSA value readiness, delayed raw commit, and attention HC-post."""
    attn_out = csa_attn_out_workspace
    current_kv = csa_current_kv_workspace
    post = csa_post_workspace
    comb = csa_comb_workspace
    x_attn = csa_x_attn_workspace
    raw_blocks = pl.tensor.dim(kv_cache, 0)
    raw_flat = pl.reshape(kv_cache, [raw_blocks * BLOCK_SIZE, HEAD_DIM])
    for block in pl.spmd(
        CSA_CHUNK_T // 8,
        name_hint="csa_layer_raw_commit",
        allow_early_resolve=True,
    ):
        token0 = block * 8
        for dt in pl.range(8):
            token = token0 + dt
            row_i64 = pl.read(raw_write_slots, [token])
            if row_i64 >= 0:
                row = pl.cast(row_i64, pl.INDEX)
                raw_flat[row : row + 1, :] = current_kv[token : token + 1, :]
    hc_post(attn_out, x_hc, post, comb, x_attn)
    return kv_cache, csa_x_attn_workspace


@pl.jit(auto_scope=False)
def csa_layer_moe(
    csa_x_attn_workspace: pl.Tensor[
        [CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D], pl.FP32
    ],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    x_hc = pl.reshape(csa_x_attn_workspace, [T, HC_MULT, D])
    num_tokens = pl.read(num_tokens_per_owner, [my_rank])
    moe(
        x_hc,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        x_next,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        routed_y_buf,
        combine_arrived,
        layer_id,
        num_tokens,
        my_rank,
        moe_epoch,
    )
    # Signal-window ownership belongs to the enclosing forward, not this layer.
    return x_next


def submit_decode_layer_csa_moe(
    csa_x_attn_workspace,
    hc_ffn_fn,
    hc_ffn_scale,
    hc_ffn_base,
    norm_w,
    gate_w,
    gate_bias,
    tid2eid,
    input_ids,
    routed_w1,
    routed_w1_scale,
    routed_w3,
    routed_w3_scale,
    routed_w2,
    routed_w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w3,
    shared_w3_scale,
    shared_w2,
    shared_w2_scale,
    num_tokens_per_owner,
    x_next,
    recv_meta_buf,
    recv_x_buf,
    recv_aux_buf,
    recv_route_buf,
    arrived_buf,
    data_arrived_buf,
    routed_y_buf_buf,
    combine_arrived_buf,
    layer_id,
    moe_epoch,
):
    """Submit the retained CSA MoE seam through caller-owned signal windows."""
    for rank in pl.range(N_RANKS):
        (
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
        ) = open_moe_signal_windows(
            recv_meta_buf,
            recv_x_buf,
            recv_aux_buf,
            recv_route_buf,
            arrived_buf,
            data_arrived_buf,
            routed_y_buf_buf,
            combine_arrived_buf,
        )
        csa_layer_moe(
            csa_x_attn_workspace[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank],
            gate_bias[rank],
            tid2eid[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            num_tokens_per_owner,
            x_next[rank],
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            rank,
            moe_epoch,
            device=rank,
        )
    return x_next


@pl.jit.host
def l3_decode_layer_csa(
    x_hc: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16],
    main_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    main_ape: pl.Tensor[[N_RANKS, 4, CSA_MAIN_OUT_DIM], pl.FP32],
    main_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    main_state: pl.InOut[pl.Tensor[
        [N_RANKS, CSA_MAIN_STATE_POOL_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ]],
    main_state_page_ids: pl.Tensor[
        [N_RANKS, CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    main_state_valid_ranges: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_B, 2], pl.INT32],
    main_state_page_epochs: pl.Tensor[
        [N_RANKS, CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    compressor_request_epochs: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_B], pl.INT32],
    request_event_indices: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_B], pl.INT32],
    event_query_ids: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_EVENT_CAP], pl.INT32],
    event_rope_cos: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32],
    event_rope_sin: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32],
    main_event_write_slots: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_EVENT_CAP], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T], pl.INT32],
    main_state_write_slots: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T], pl.INT64],
    main_cache: pl.InOut[pl.Tensor[
        [N_RANKS, CSA_MAIN_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    inner_wkv: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[N_RANKS, 4, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    inner_hadamard: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_state: pl.InOut[pl.Tensor[
        [N_RANKS, CSA_INNER_STATE_POOL_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ]],
    inner_state_page_ids: pl.Tensor[
        [N_RANKS, CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_state_valid_ranges: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_B, 2], pl.INT32],
    inner_state_page_epochs: pl.Tensor[
        [N_RANKS, CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_event_write_slots: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_EVENT_CAP], pl.INT64],
    inner_state_write_slots: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T], pl.INT64],
    idx_cache: pl.InOut[
        pl.Tensor[[N_RANKS, CSA_IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_scale: pl.InOut[
        pl.Tensor[[N_RANKS, CSA_IDX_ROWS_DYN, 1], pl.FP32]
    ],
    idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    idx_hadamard: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_cos_il: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32],
    idx_sin_signed: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32],
    query_request_ids: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T], pl.INT32],
    idx_pages: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_REQUEST_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_descriptors: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32],
    singleton_descriptors: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32],
    upper_descriptors: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32],
    root_descriptors: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], pl.INT32],
    csa_pages: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_REQUEST_OFFSET_DYN], pl.INT32],
    csa_windows: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_REQUEST_DYN, 3], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[
        [N_RANKS, CSA_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    raw_write_slots: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T], pl.INT64],
    swa_sources: pl.Tensor[[N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, WIN], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    csa_x_attn_workspace: pl.Out[pl.Tensor[
        [N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D], pl.FP32
    ]],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Submit one rank-local CSA device child per rank."""
    # Keep direct allocations visible to the decorated host frontend.  An
    # ordinary helper hides the window metadata required for compilation.
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32] = pld.window(
            recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32
        )
        recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8] = pld.window(
            recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
        )
        recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32] = pld.window(
            recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
        )
        recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32] = pld.window(
            recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
        )
        arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16] = pld.window(
            routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16
        )
        combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(
            combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        decode_layer_csa(
            x_hc[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            gamma_cq[rank],
            gamma_ckv[rank],
            rope_cos[rank],
            rope_sin[rank],
            main_wkv[rank],
            main_wgate[rank],
            main_ape[rank],
            main_norm_w[rank],
            main_state[rank],
            main_state_page_ids[rank],
            main_state_valid_ranges[rank],
            main_state_page_epochs[rank],
            compressor_request_epochs[rank],
            request_event_indices[rank],
            event_query_ids[rank],
            event_rope_cos[rank],
            event_rope_sin[rank],
            main_event_write_slots[rank],
            position_ids[rank],
            main_state_write_slots[rank],
            main_cache[rank],
            inner_wkv[rank],
            inner_wgate[rank],
            inner_ape[rank],
            inner_norm_w[rank],
            inner_hadamard[rank],
            inner_state[rank],
            inner_state_page_ids[rank],
            inner_state_valid_ranges[rank],
            inner_state_page_epochs[rank],
            inner_event_write_slots[rank],
            inner_state_write_slots[rank],
            idx_cache[rank],
            idx_scale[rank],
            idx_wq_b[rank],
            idx_wq_b_scale[rank],
            idx_weights_proj[rank],
            idx_hadamard[rank],
            idx_cos_il[rank],
            idx_sin_signed[rank],
            query_request_ids[rank],
            idx_pages[rank],
            idx_page_offsets[rank],
            idx_windows[rank],
            request_epochs[rank],
            leaf_descriptors[rank],
            pair_descriptors[rank],
            singleton_descriptors[rank],
            upper_descriptors[rank],
            root_descriptors[rank],
            csa_pages[rank],
            csa_page_offsets[rank],
            csa_windows[rank],
            kv_cache[rank],
            raw_write_slots[rank],
            swa_sources[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            csa_x_attn_workspace[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank],
            gate_bias[rank],
            tid2eid[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            num_tokens_per_owner,
            x_next[rank],
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            rank,
            moe_epoch,
            device=rank,
        )
    return x_next


def submit_decode_layer_csa(*args, **kwargs):
    """Run the typed standalone CSA host entry through its raw Python seam.

    The standalone entry owns fresh signal windows. A decorated L3 entry cannot
    accept an untyped Python tuple of caller-owned window buffers.
    """
    if "window_buffers" in kwargs:
        raise ValueError(
            "submit_decode_layer_csa does not accept Python window tuples"
        )
    return l3_decode_layer_csa._func(*args, **kwargs)


_ATTENTION_RESIDENT_NAMES = frozenset(
    {
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "attn_sink",
        "wo_a",
        "wo_b",
        "wo_b_scale",
        "cmp_wkv",
        "cmp_wgate",
        "cmp_ape",
        "cmp_norm_w",
    }
)
_ATTENTION_CACHE_NAMES = frozenset({"kv_cache", "cmp_kv", "compress_state"})


def _rank_attention_spec(spec, counts, attention_specs):
    import torch
    from golden import TensorSpec

    def init_ranked(base_spec=spec):
        value = base_spec.create_tensor()
        ranked = value.unsqueeze(0).expand(N_RANKS, *value.shape).contiguous()
        position_spec = next(
            (candidate for candidate in attention_specs if candidate.name == "position_ids"),
            None,
        )
        positions = position_spec.create_tensor() if position_spec is not None else None
        for rank, active_rows in enumerate(counts):
            active_rows = max(0, min(T, int(active_rows)))
            if spec.name == "x_hc":
                ranked[rank, active_rows:] = 0
            elif spec.name in {
                "rope_cos",
                "rope_sin",
                "query_rope_cos",
                "query_rope_sin",
            }:
                ranked[rank, active_rows:] = 0
            elif spec.name in {
                "swa_write_slots",
                "state_write_slots",
                "cmp_slot_mapping",
            }:
                ranked[rank, active_rows:] = -1
            elif spec.name == "swa_sources":
                ranked[rank, active_rows:] = -1
            elif spec.name == "swa_lens":
                ranked[rank, active_rows:] = 0
            elif spec.name == "position_ids":
                ranked[rank, active_rows:] = 0
            elif spec.name == "query_request_ids":
                ranked[rank, active_rows:] = 0
            elif spec.name == "hca_query_work_offsets":
                work_end = int(ranked[rank, active_rows].item())
                ranked[rank, active_rows + 1 :] = work_end
            elif spec.name == "request_event_indices" and positions is not None:
                for request in range(B):
                    event_index = int(ranked[rank, request].item())
                    if event_index < 0:
                        continue
                    request_begin = request * S
                    request_end = min(request_begin + S, active_rows)
                    event_is_active = any(
                        (int(positions[row].item()) + 1) % HCA_COMPRESS_RATIO == 0
                        for row in range(request_begin, request_end)
                    )
                    if not event_is_active:
                        ranked[rank, request] = -1
            elif spec.name == "input_ids":
                ranked[rank, active_rows:] = 0
        return ranked

    resident = None
    if spec.name in _ATTENTION_RESIDENT_NAMES or spec.name in _ATTENTION_CACHE_NAMES:
        resident = "stacked"
    return TensorSpec(
        spec.name,
        [N_RANKS, *spec.shape],
        spec.dtype,
        init_value=init_ranked,
        is_output=spec.is_output,
        resident=resident,
    )


def _normalize_counts(num_tokens_per_owner):
    if num_tokens_per_owner is None:
        return [T] * N_RANKS
    if isinstance(num_tokens_per_owner, int):
        return [max(0, min(T, num_tokens_per_owner))] * N_RANKS
    counts = [max(0, min(T, int(value))) for value in num_tokens_per_owner]
    if len(counts) != N_RANKS:
        raise ValueError(f"expected {N_RANKS} token counts, got {len(counts)}")
    return counts


def _build_layer_specs(attention_specs, layer_id, counts, balanced_routing=False):
    import torch
    from golden import ScalarSpec, TensorSpec

    specs = []
    for spec in attention_specs:
        if spec.name == "x_out":
            continue
        specs.append(_rank_attention_spec(spec, counts, attention_specs))

    moe_specs = build_moe_tensor_specs(
        layer_id=layer_id,
        num_tokens=max(counts),
        balanced_routing=balanced_routing,
    )
    existing = {spec.name for spec in specs}
    for spec in moe_specs:
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next"} or spec.name in existing:
            continue
        specs.append(spec)
        existing.add(spec.name)

    specs.extend(
        [
            TensorSpec(
                "num_tokens_per_owner",
                [N_RANKS],
                torch.int32,
                init_value=lambda: torch.tensor(counts, dtype=torch.int32),
            ),
            TensorSpec(
                "x_attn_workspace",
                [N_RANKS, T, HC_MULT, D],
                torch.float32,
                init_value=lambda: torch.zeros(
                    N_RANKS, T, HC_MULT, D, dtype=torch.float32
                ),
                is_output=True,
            ),
            TensorSpec(
                "x_next",
                [N_RANKS, T, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            ScalarSpec("layer_id", torch.int32, layer_id),
        ]
    )
    return specs


def build_swa_layer_specs(
    case="heterogeneous_lengths",
    layer_id=0,
    num_tokens_per_owner=None,
    balanced_routing=False,
):
    if attention_kind_for_layer(layer_id) != "swa":
        raise ValueError(f"layer {layer_id} is not an SWA layer")
    counts = _normalize_counts(num_tokens_per_owner)
    attention_specs = build_swa_tensor_specs(case=case, batch=B)
    return _build_layer_specs(
        attention_specs,
        layer_id,
        counts,
        balanced_routing=balanced_routing,
    )


def build_hca_layer_specs(
    case="heterogeneous_lengths",
    layer_id=3,
    num_tokens_per_owner=None,
    balanced_routing=False,
):
    if attention_kind_for_layer(layer_id) != "hca":
        raise ValueError(f"layer {layer_id} is not an HCA layer")
    counts = _normalize_counts(num_tokens_per_owner)
    attention_specs = build_hca_tensor_specs(case=case, batch=B)
    return _build_layer_specs(
        attention_specs,
        layer_id,
        counts,
        balanced_routing=balanced_routing,
    )


def build_csa_layer_specs(
    case="global_state_pool",
    layer_id=2,
    num_tokens_per_owner=None,
    balanced_routing=False,
):
    """Build the rank-by-chunk CSA layer fixture with allocator-owned pools."""
    import inspect

    import torch
    from golden import ScalarSpec, TensorSpec

    if attention_kind_for_layer(layer_id) != "csa":
        raise ValueError(f"layer {layer_id} is not a CSA layer")
    if case not in {
        "heterogeneous_lengths",
        "csa_forest_boundaries",
        "ratio4_rollover",
        "one_m_tail",
        "page_permutation",
        "global_state_pool",
        "full43_long_context_tail",
        *PHASE_D_TRACE_CASE_LENGTHS,
    }:
        raise ValueError(f"unknown Phase E CSA case: {case!r}")
    counts = _normalize_counts(num_tokens_per_owner)
    global CSA_SUBMISSION_CHUNKS
    CSA_SUBMISSION_CHUNKS = (
        2
        if case in PHASE_D_TRACE_CASE_LENGTHS
        and case != "full43_long_context_tail"
        else CSA_CHUNKS
    )

    def values(specs):
        return {spec.name: spec.create_tensor() for spec in specs}

    def ranked(value):
        return value.unsqueeze(0).expand(N_RANKS, *value.shape).contiguous()

    def chunked(value):
        return (
            value.unsqueeze(0)
            .unsqueeze(0)
            .expand(N_RANKS, CSA_CHUNKS, *value.shape)
            .contiguous()
        )

    def padded_events(value, fill_value):
        padded = torch.full(
            (CSA_EVENT_CAP, *value.shape[1:]),
            fill_value,
            dtype=value.dtype,
        )
        valid = min(CSA_EVENT_CAP, value.shape[0])
        if valid:
            padded[:valid] = value[:valid]
        return padded

    def global_state_pool(page_count, rows, width):
        return torch.zeros(
            N_RANKS,
            page_count,
            rows,
            width,
            dtype=torch.float32,
        )

    def global_state_descriptors(source, rows, block_size, pages_per_request):
        page_ids = torch.empty(
            N_RANKS,
            CSA_CHUNKS,
            CSA_CHUNK_B,
            pages_per_request,
            dtype=torch.int32,
        )
        valid_ranges = torch.empty(
            N_RANKS, CSA_CHUNKS, CSA_CHUNK_B, 2, dtype=torch.int32
        )
        request_epochs = chunked(source["request_epochs"]).clone()
        state_page_epochs = torch.empty_like(page_ids)
        position_ids = chunked(source["position_ids"]).clone()
        write_slots = torch.empty(
            N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, dtype=torch.int64
        )
        for rank in range(N_RANKS):
            for chunk in range(CSA_CHUNKS):
                page_template = torch.empty(
                    CSA_CHUNK_B, pages_per_request, dtype=torch.int32
                )
                for request in range(CSA_CHUNK_B):
                    request_ordinal = chunk * CSA_CHUNK_B + request
                    first_page = (
                        CSA_GLOBAL_STATE_HIGH_PAGE
                        - request_ordinal * pages_per_request
                    )
                    page_template[request] = torch.arange(
                        first_page,
                        first_page - pages_per_request,
                        -1,
                        dtype=torch.int32,
                    )
                position_ids[rank, chunk] += chunk * S
                request_epochs[rank, chunk] += chunk * CSA_CHUNK_B
                state_page_epochs[rank, chunk] = request_epochs[rank, chunk, :, None].expand(
                    CSA_CHUNK_B, pages_per_request
                )
                page_ids[rank, chunk] = page_template
                for request in range(CSA_CHUNK_B):
                    first_position = position_ids[rank, chunk, request * S]
                    valid_ranges[rank, chunk, request, 0] = torch.clamp(
                        first_position - rows,
                        min=0,
                    )
                    valid_ranges[rank, chunk, request, 1] = first_position
                for token in range(CSA_CHUNK_T):
                    request = token // S
                    ring_row = int(position_ids[rank, chunk, token].item()) % rows
                    relative_page = ring_row // block_size
                    write_slots[rank, chunk, token] = (
                        page_ids[rank, chunk, request, relative_page].to(torch.int64)
                        * block_size
                        + ring_row % block_size
                    )
        return {
            "page_ids": page_ids,
            "valid_ranges": valid_ranges,
            "state_page_epochs": state_page_epochs,
            "request_epochs": request_epochs,
            "position_ids": position_ids,
            "write_slots": write_slots,
        }

    def global_event_write_slots(value):
        slots = chunked(padded_events(value, -1)).clone()
        valid_slots = value[value >= 0]
        stride = (
            1
            if valid_slots.numel() == 0
            else int(valid_slots.max().item()) + 1
        )
        for rank in range(N_RANKS):
            for chunk in range(CSA_CHUNKS):
                slots[rank, chunk] = torch.where(
                    slots[rank, chunk] >= 0,
                    slots[rank, chunk] + chunk * stride,
                    slots[rank, chunk],
                )
        return slots

    specs = []

    def add(name, value, *, output=False, resident=None):
        specs.append(
            TensorSpec(
                name,
                list(value.shape),
                value.dtype,
                init_value=lambda value=value: value,
                is_output=output,
                resident=resident,
            )
        )

    attn = values(build_swa_tensor_specs(case="short_history", batch=CSA_CHUNK_B))
    add(
        "x_hc",
        torch.zeros(
            N_RANKS,
            CSA_CHUNKS,
            CSA_CHUNK_T,
            HC_MULT,
            D,
            dtype=torch.float32,
        ),
    )
    for name in (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "attn_sink",
        "wo_a",
        "wo_b_scale",
    ):
        add(name, ranked(attn[name]), resident="stacked")
    add("wo_b", torch.zeros_like(ranked(attn["wo_b"])), resident="stacked")
    add("rope_cos", chunked(attn["rope_cos"]))
    add("rope_sin", chunked(attn["rope_sin"]))

    stateful_case = (
        case in {"global_state_pool", "full43_long_context_tail"}
        or case in PHASE_D_TRACE_CASE_LENGTHS
    )
    compressor_case = "two_step_state" if stateful_case else "no_event"
    main = values(
        build_csa_main_compressor_specs(case=compressor_case, batch=CSA_CHUNK_B)
    )
    inner = values(
        build_csa_inner_compressor_specs(case=compressor_case, batch=CSA_CHUNK_B)
    )
    if stateful_case:
        main_state_fixture = global_state_descriptors(
            main,
            CSA_STATE_ROWS_PER_REQUEST,
            CSA_STATE_BLOCK_SIZE,
            CSA_STATE_PAGES_PER_REQUEST,
        )
        inner_state_fixture = global_state_descriptors(
            inner,
            CSA_INNER_STATE_ROWS_PER_REQUEST,
            CSA_INNER_STATE_BLOCK_SIZE,
            CSA_INNER_STATE_PAGES_PER_REQUEST,
        )
    else:
        main_state_fixture = {
            "page_ids": chunked(main["state_page_ids"]),
            "valid_ranges": chunked(main["state_valid_ranges"]),
            "state_page_epochs": chunked(main["state_page_epochs"]),
            "request_epochs": chunked(main["request_epochs"]),
            "position_ids": chunked(main["position_ids"]),
            "write_slots": torch.full(
                (N_RANKS, CSA_CHUNKS, CSA_CHUNK_T), -1, dtype=torch.int64
            ),
        }
        inner_state_fixture = {
            "page_ids": chunked(inner["state_page_ids"]),
            "valid_ranges": chunked(inner["state_valid_ranges"]),
            "state_page_epochs": chunked(inner["state_page_epochs"]),
            "write_slots": torch.full(
                (N_RANKS, CSA_CHUNKS, CSA_CHUNK_T), -1, dtype=torch.int64
            ),
        }
    main_event_write_slots = (
        global_event_write_slots(main["event_write_slots"])
        if stateful_case
        else chunked(padded_events(main["event_write_slots"], -1))
    )
    inner_event_write_slots = (
        global_event_write_slots(inner["event_write_slots"])
        if stateful_case
        else chunked(padded_events(inner["event_write_slots"], -1))
    )
    if case == "full43_long_context_tail":
        request_lengths = (
            S,
            128,
            16_384,
            1_048_576,
            64,
            256,
            512,
            1_024,
            2_048,
            4_096,
            8_192,
            32,
            48,
            96,
            192,
            384,
        )
        index_chunks_by_rank = []
        for rank in range(N_RANKS):
            active_requests = counts[rank] // S
            rank_chunks = []
            for chunk in range(CSA_CHUNKS):
                request_begin = chunk * CSA_CHUNK_B
                rank_chunks.append(
                    values(
                        build_phase_d_indexer_specs(
                            query_count=CSA_CHUNK_T,
                            case="full43_long_context_tail",
                            request_lengths=request_lengths[
                                request_begin : request_begin + CSA_CHUNK_B
                            ],
                            request_active=tuple(
                                request_begin + local < active_requests
                                for local in range(CSA_CHUNK_B)
                            ),
                        )
                    )
                )
            index_chunks_by_rank.append(rank_chunks)
        index_chunks = index_chunks_by_rank[0]
        index = index_chunks[0]
    else:
        index_case = case if case in PHASE_D_TRACE_CASE_LENGTHS else "phase_d_mixed_forest"
        index = values(
            build_phase_d_indexer_specs(query_count=CSA_CHUNK_T, case=index_case)
        )
        index_chunks = None
        index_chunks_by_rank = None
    for target, source in (
        ("main_wkv", "wkv"),
        ("main_wgate", "wgate"),
        ("main_ape", "ape"),
        ("main_norm_w", "norm_w"),
    ):
        add(target, ranked(main[source]), resident="stacked")
    add(
        "main_state",
        global_state_pool(
            CSA_STATE_POOL_PAGES,
            CSA_STATE_BLOCK_SIZE,
            CSA_MAIN_STATE_DIM,
        ),
        output=True,
        resident="stacked",
    )
    add("main_state_page_ids", main_state_fixture["page_ids"])
    add("main_state_valid_ranges", main_state_fixture["valid_ranges"])
    add("main_state_page_epochs", main_state_fixture["state_page_epochs"])
    add("compressor_request_epochs", main_state_fixture["request_epochs"])
    add("request_event_indices", chunked(main["request_event_indices"]))
    add(
        "event_query_ids",
        chunked(padded_events(main["event_query_ids"], -1)),
    )
    add("event_rope_cos", chunked(padded_events(main["cos"], 0.0)))
    add("event_rope_sin", chunked(padded_events(main["sin"], 0.0)))
    add("main_event_write_slots", main_event_write_slots)
    add("position_ids", main_state_fixture["position_ids"])
    add("main_state_write_slots", main_state_fixture["write_slots"])

    for target, source in (
        ("inner_wkv", "wkv"),
        ("inner_wgate", "wgate"),
        ("inner_ape", "ape"),
        ("inner_norm_w", "norm_w"),
        ("inner_hadamard", "hadamard"),
    ):
        add(target, ranked(inner[source]), resident="stacked")
    add(
        "inner_state",
        global_state_pool(
            CSA_INNER_STATE_POOL_PAGES,
            CSA_INNER_STATE_BLOCK_SIZE,
            CSA_INNER_STATE_DIM,
        ),
        output=True,
        resident="stacked",
    )
    add("inner_state_page_ids", inner_state_fixture["page_ids"])
    add("inner_state_valid_ranges", inner_state_fixture["valid_ranges"])
    add("inner_state_page_epochs", inner_state_fixture["state_page_epochs"])
    add("inner_event_write_slots", inner_event_write_slots)
    add("inner_state_write_slots", inner_state_fixture["write_slots"])

    def slots_capacity(slots):
        valid_slots = slots[slots >= 0]
        if valid_slots.numel() == 0:
            return 1
        return int(valid_slots.max().item()) + 1

    trace_chunks = 2 if case in PHASE_D_TRACE_CASE_LENGTHS else 1
    if index_chunks is not None:
        trace_chunks = CSA_CHUNKS
    index_rows = index["idx_kv_cache_flat"].shape[0]
    packed_index_rows = (
        max(
            sum(chunk["idx_kv_cache_flat"].shape[0] for chunk in rank_chunks)
            for rank_chunks in index_chunks_by_rank
        )
        if index_chunks_by_rank is not None
        else index_rows * trace_chunks
    )
    idx_rows = max(
        packed_index_rows,
        slots_capacity(inner_event_write_slots),
    )
    idx_blocks = max(
        1,
        (idx_rows + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (slots_capacity(main_event_write_slots) + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    idx_cache = torch.zeros(
        N_RANKS,
        idx_rows,
        IDX_HEAD_DIM,
        dtype=index["idx_kv_cache_flat"].dtype,
    )
    for rank in range(N_RANKS):
        row_begin = 0
        rank_chunks = (
            [index] * trace_chunks
            if index_chunks_by_rank is None
            else index_chunks_by_rank[rank]
        )
        for source in rank_chunks:
            rows = source["idx_kv_cache_flat"].shape[0]
            idx_cache[rank, row_begin : row_begin + rows] = source[
                "idx_kv_cache_flat"
            ]
            row_begin += rows
    idx_scale = torch.zeros(
        N_RANKS,
        idx_rows,
        1,
        dtype=index["idx_kv_scale_flat"].dtype,
    )
    for rank in range(N_RANKS):
        row_begin = 0
        rank_chunks = (
            [index] * trace_chunks
            if index_chunks_by_rank is None
            else index_chunks_by_rank[rank]
        )
        for source in rank_chunks:
            rows = source["idx_kv_scale_flat"].shape[0]
            idx_scale[rank, row_begin : row_begin + rows] = source[
                "idx_kv_scale_flat"
            ]
            row_begin += rows
    add("idx_cache", idx_cache, output=True, resident="stacked")
    add("idx_scale", idx_scale, output=True, resident="stacked")
    add(
        "main_cache",
        torch.zeros(
            N_RANKS,
            idx_blocks,
            BLOCK_SIZE,
            1,
            HEAD_DIM,
            dtype=torch.bfloat16,
        ),
        output=True,
        resident="stacked",
    )
    add(
        "kv_cache",
        torch.zeros(
            N_RANKS, 1, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16
        ),
        output=True,
        resident="stacked",
    )
    raw_slots = torch.arange(T, dtype=torch.int64).reshape(
        CSA_CHUNKS, CSA_CHUNK_T
    )
    add(
        "raw_write_slots",
        raw_slots.unsqueeze(0).expand(
            N_RANKS, CSA_CHUNKS, CSA_CHUNK_T
        ).contiguous(),
    )
    sources = torch.full(
        (CSA_CHUNKS, CSA_CHUNK_T, WIN), -1, dtype=torch.int32
    )
    for chunk in range(CSA_CHUNKS):
        for token in range(CSA_CHUNK_T):
            sources[chunk, token, -1] = SWA_SOURCE_OVERLAY_BASE - token
    add(
        "swa_sources",
        sources.unsqueeze(0).expand(
            N_RANKS, CSA_CHUNKS, CSA_CHUNK_T, WIN
        ).contiguous(),
    )

    for target, source in (
        ("idx_wq_b", "wq_b"),
        ("idx_wq_b_scale", "wq_b_scale"),
        ("idx_weights_proj", "weights_proj"),
        ("idx_hadamard", "hadamard"),
    ):
        add(target, ranked(index[source]), resident="stacked")

    def packed_index_chunks(source, fill_value):
        if index_chunks_by_rank is None:
            return chunked(index[source])
        max_shape = tuple(
            max(
                chunk[source].shape[axis]
                for rank_chunks in index_chunks_by_rank
                for chunk in rank_chunks
            )
            for axis in range(index_chunks_by_rank[0][0][source].ndim)
        )
        packed = torch.full(
            (N_RANKS, CSA_CHUNKS, *max_shape),
            fill_value,
            dtype=index_chunks_by_rank[0][0][source].dtype,
        )
        for rank, rank_chunks in enumerate(index_chunks_by_rank):
            page_base = 0
            for chunk_ordinal, chunk in enumerate(rank_chunks):
                value = chunk[source].clone()
                if source == "idx_pages":
                    valid = value[:, 0] >= 0
                    value[valid, 0] += page_base
                    page_base += chunk["idx_kv_cache_flat"].shape[0] // BLOCK_SIZE
                target_slices = (rank, chunk_ordinal) + tuple(
                    slice(0, extent) for extent in value.shape
                )
                packed[target_slices] = value
                if source == "idx_page_offsets" and value.shape[0] < max_shape[0]:
                    packed[rank, chunk_ordinal, value.shape[0] :] = value[-1]
        return packed

    index_fill_values = {
        "cos_il": 0.0,
        "sin_signed": 0.0,
        "query_request_ids": -1,
        "idx_pages": -1,
        "idx_page_offsets": 0,
        "idx_windows": 0,
        "request_epochs": -1,
        "leaf_descriptors": -1,
        "pair_descriptors": -1,
        "singleton_descriptors": -1,
        "upper_descriptors": -1,
        "root_descriptors": -1,
    }
    for target, source in (
        ("idx_cos_il", "cos_il"),
        ("idx_sin_signed", "sin_signed"),
        ("query_request_ids", "query_request_ids"),
        ("idx_pages", "idx_pages"),
        ("idx_page_offsets", "idx_page_offsets"),
        ("idx_windows", "idx_windows"),
        ("request_epochs", "request_epochs"),
        ("leaf_descriptors", "leaf_descriptors"),
        ("pair_descriptors", "pair_descriptors"),
        ("singleton_descriptors", "singleton_descriptors"),
        ("upper_descriptors", "upper_descriptors"),
        ("root_descriptors", "root_descriptors"),
        ("csa_pages", "idx_pages"),
        ("csa_page_offsets", "idx_page_offsets"),
        ("csa_windows", "idx_windows"),
    ):
        value = packed_index_chunks(source, index_fill_values[source])
        if case in PHASE_D_TRACE_CASE_LENGTHS and source == "idx_pages":
            pages_per_chunk = index_rows // BLOCK_SIZE
            for rank in range(N_RANKS):
                for chunk in range(CSA_CHUNKS):
                    valid_pages = value[rank, chunk, :, 0] >= 0
                    value[rank, chunk, valid_pages, 0] += chunk * pages_per_chunk
        add(target, value)

    add(
        "csa_x_attn_workspace",
        torch.zeros(
            N_RANKS,
            CSA_CHUNKS,
            CSA_CHUNK_T,
            HC_MULT,
            D,
            dtype=torch.float32,
        ),
    )

    def replace_tensor(name, value):
        for index, spec in enumerate(specs):
            if spec.name != name:
                continue
            specs[index] = TensorSpec(
                name,
                list(value.shape),
                value.dtype,
                init_value=lambda value=value: value,
                is_output=spec.is_output,
                resident=spec.resident,
            )
            return
        raise KeyError(name)

    def materialize(name):
        return next(spec for spec in specs if spec.name == name).create_tensor()

    def valid_rows_for_chunk(rank, chunk):
        return max(0, min(CSA_CHUNK_T, counts[rank] - chunk * CSA_CHUNK_T))

    for name, fill_value in (
        ("x_hc", 0),
        ("rope_cos", 0),
        ("rope_sin", 0),
        ("position_ids", 0),
        ("idx_cos_il", 0),
        ("idx_sin_signed", 0),
        ("query_request_ids", -1),
        ("raw_write_slots", -1),
        ("swa_sources", -1),
        ("main_state_write_slots", -1),
        ("inner_state_write_slots", -1),
    ):
        value = materialize(name)
        for rank in range(N_RANKS):
            for chunk in range(CSA_CHUNKS):
                valid_rows = valid_rows_for_chunk(rank, chunk)
                value[rank, chunk, valid_rows:] = fill_value
        replace_tensor(name, value)

    leaf_descriptors = materialize("leaf_descriptors")
    for rank in range(N_RANKS):
        for chunk in range(CSA_CHUNKS):
            valid_rows = valid_rows_for_chunk(rank, chunk)
            inactive = (
                leaf_descriptors[rank, chunk, :, PHASE_D_LEAF_QUERY]
                >= valid_rows
            )
            leaf_descriptors[rank, chunk, inactive, PHASE_D_LEAF_QUERY] = 0
            leaf_descriptors[rank, chunk, inactive, PHASE_D_LEAF_VALID] = 0
    replace_tensor("leaf_descriptors", leaf_descriptors)

    root_descriptors = materialize("root_descriptors")
    for rank in range(N_RANKS):
        for chunk in range(CSA_CHUNKS):
            valid_rows = valid_rows_for_chunk(rank, chunk)
            root_descriptors[rank, chunk, valid_rows:, PHASE_D_ROOT_SLOT] = -1
            root_descriptors[
                rank, chunk, valid_rows:, PHASE_D_ROOT_DEPENDENCY_SLOT
            ] = CSA_TOPK_INVALID_TASK_SLOT
    replace_tensor("root_descriptors", root_descriptors)

    request_event_indices = materialize("request_event_indices")
    event_query_ids = materialize("event_query_ids")
    main_event_write_slots = materialize("main_event_write_slots")
    inner_event_write_slots = materialize("inner_event_write_slots")
    for rank in range(N_RANKS):
        for chunk in range(CSA_CHUNKS):
            valid_rows = valid_rows_for_chunk(rank, chunk)
            for request in range(CSA_CHUNK_B):
                event = int(request_event_indices[rank, chunk, request].item())
                if event < 0:
                    continue
                if (
                    event >= CSA_EVENT_CAP
                    or int(event_query_ids[rank, chunk, event].item())
                    >= valid_rows
                ):
                    request_event_indices[rank, chunk, request] = -1
            inactive_events = (event_query_ids[rank, chunk] < 0) | (
                event_query_ids[rank, chunk] >= valid_rows
            )
            event_query_ids[rank, chunk, inactive_events] = -1
            main_event_write_slots[rank, chunk, inactive_events] = -1
            inner_event_write_slots[rank, chunk, inactive_events] = -1
    replace_tensor("request_event_indices", request_event_indices)
    replace_tensor("event_query_ids", event_query_ids)
    replace_tensor("main_event_write_slots", main_event_write_slots)
    replace_tensor("inner_event_write_slots", inner_event_write_slots)

    existing = {spec.name for spec in specs}
    for spec in build_moe_tensor_specs(
        layer_id=layer_id,
        num_tokens=max(counts),
        balanced_routing=balanced_routing,
    ):
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next"} or spec.name in existing:
            continue
        specs.append(spec)
        existing.add(spec.name)
    add(
        "num_tokens_per_owner",
        torch.tensor(counts, dtype=torch.int32),
    )
    add(
        "x_next",
        torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32),
        output=True,
    )
    specs.append(ScalarSpec("layer_id", torch.int32, layer_id))
    specs.append(ScalarSpec("moe_epoch", torch.int32, 1))
    specs_by_name = {spec.name: spec for spec in specs}
    parameter_names = [
        name
        for name, parameter in inspect.signature(
            l3_decode_layer_csa._func
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in parameter_names if name not in specs_by_name]
    extra = [name for name in specs_by_name if name not in parameter_names]
    if missing or extra:
        raise ValueError(
            f"Phase E CSA spec/signature mismatch: missing={missing}, extra={extra}"
        )
    return [specs_by_name[name] for name in parameter_names]


def _golden_layer(tensors, attention_golden, attention_names):
    import torch

    x_attn = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    for rank in range(N_RANKS):
        local = {name: tensors[name][rank] for name in attention_names}
        local["x_out"] = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
        attention_golden(local)
        x_attn[rank] = local["x_out"]

    tensors["x_attn_workspace"].copy_(x_attn)
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    moe_tensors["x_next"] = tensors["x_next"]
    golden_moe(moe_tensors)


def golden_decode_layer_swa(tensors):
    attention_names = [
        spec.name
        for spec in build_swa_tensor_specs(
            case="heterogeneous_lengths", batch=B
        )
        if spec.name != "x_out"
    ]
    _golden_layer(tensors, golden_attention_swa, attention_names)


def golden_decode_layer_hca(tensors):
    attention_names = [
        spec.name
        for spec in build_hca_tensor_specs(
            case="heterogeneous_lengths", batch=B
        )
        if spec.name != "x_out"
    ]
    _golden_layer(tensors, golden_attention_hca, attention_names)


def golden_decode_layer_csa(tensors):
    """Golden state/cache commits for the zero-activation CSA fixture."""
    import torch

    def write_state(state_name, slots_name, ape_name, rows, out_dim):
        state = tensors[state_name]
        slots = tensors[slots_name]
        positions = tensors["position_ids"]
        chunks_with_writes = set()
        high_page_written = False
        for rank in range(N_RANKS):
            for chunk in range(CSA_CHUNKS):
                for token in range(CSA_CHUNK_T):
                    slot = int(slots[rank, chunk, token].item())
                    if slot < 0:
                        continue
                    page, row = divmod(slot, rows)
                    if not 0 <= page < state.shape[1]:
                        raise AssertionError(
                            f"{state_name} write page {page} is outside its global pool"
                        )
                    position = int(positions[rank, chunk, token].item())
                    state[rank, page, row, :out_dim] = 0.0
                    state[rank, page, row, out_dim:] = tensors[ape_name][
                        rank, position % 4
                    ].float()
                    chunks_with_writes.add(chunk)
                    high_page_written = high_page_written or (
                        page == CSA_GLOBAL_STATE_HIGH_PAGE
                    )
        return chunks_with_writes, high_page_written

    main_chunks, main_high_page = write_state(
        "main_state",
        "main_state_write_slots",
        "main_ape",
        CSA_STATE_BLOCK_SIZE,
        CSA_MAIN_OUT_DIM,
    )
    inner_chunks, inner_high_page = write_state(
        "inner_state",
        "inner_state_write_slots",
        "inner_ape",
        CSA_INNER_STATE_BLOCK_SIZE,
        CSA_INNER_OUT_DIM,
    )
    if bool(torch.any(tensors["main_state_write_slots"] >= 0).item()):
        if len(main_chunks) < 2 or not main_high_page:
            raise AssertionError("global CSA main-state fixture must write page 64 across chunks")
        if len(inner_chunks) < 2 or not inner_high_page:
            raise AssertionError("global CSA inner-state fixture must write page 64 across chunks")
        for page_ids_name in ("main_state_page_ids", "inner_state_page_ids"):
            page_ids = tensors[page_ids_name]
            for rank in range(N_RANKS):
                rank_pages = page_ids[rank].reshape(-1)
                if torch.unique(rank_pages).numel() != rank_pages.numel():
                    raise AssertionError(
                        f"{page_ids_name} must not alias pages across CSA chunks"
                    )
        for slots_name in ("main_event_write_slots", "inner_event_write_slots"):
            slots = tensors[slots_name]
            for rank in range(N_RANKS):
                rank_slots = slots[rank].reshape(-1)
                valid_slots = rank_slots[rank_slots >= 0]
                if torch.unique(valid_slots).numel() != valid_slots.numel():
                    raise AssertionError(
                        f"{slots_name} must not alias event rows across CSA chunks"
                    )

    for rank in range(N_RANKS):
        for chunk in range(CSA_CHUNKS):
            for event in range(CSA_EVENT_CAP):
                main_slot = int(
                    tensors["main_event_write_slots"][rank, chunk, event].item()
                )
                if main_slot >= 0:
                    block, row = divmod(main_slot, BLOCK_SIZE)
                    tensors["main_cache"][rank, block, row, 0] = 0.0
                inner_slot = int(
                    tensors["inner_event_write_slots"][rank, chunk, event].item()
                )
                if inner_slot >= 0:
                    tensors["idx_cache"][rank, inner_slot] = 0
                    tensors["idx_scale"][rank, inner_slot, 0] = (
                        INT8_AMAX_EPS / INT8_SCALE_MAX
                    )
    tensors["x_next"].zero_()


def _parse_counts(raw):
    if raw is None:
        return [T] * N_RANKS
    values = [int(value) for value in raw.split(",") if value]
    if len(values) == 1:
        return values * N_RANKS
    if len(values) != N_RANKS:
        raise ValueError(
            f"--num-tokens-per-owner needs one or {N_RANKS} comma-separated values"
        )
    return values


G4_LAYER_CASES = (
    "full_active",
    "ragged_mixed",
    "all_inactive",
    "one_m_tail",
)

CSA_TRACE_CASES = tuple(PHASE_D_TRACE_CASE_LENGTHS)


def describe_csa_trace_case(case):
    """Return the exact runtime geometry represented by one trace fixture."""
    if case not in PHASE_D_TRACE_CASE_LENGTHS:
        raise ValueError(f"unknown CSA trace case {case!r}")
    chunk_lengths = PHASE_D_TRACE_CASE_LENGTHS[case]
    request_lengths = chunk_lengths * 2
    final_candidates = tuple(length // 4 for length in request_lengths)
    final_leaves = tuple(
        (candidates + CSA_CANDIDATES_PER_LEAF - 1)
        // CSA_CANDIDATES_PER_LEAF
        for candidates in final_candidates
    )
    pair_pages = sum(
        (candidates + BLOCK_SIZE - 1) // BLOCK_SIZE
        for candidates in final_candidates[:2]
    )
    per_shard_nodes = 0
    for length in chunk_lengths:
        first_position = length - S
        for row in range(S):
            candidates = (first_position + row + 1) // 4
            leaves = (
                candidates + CSA_CANDIDATES_PER_LEAF - 1
            ) // CSA_CANDIDATES_PER_LEAF
            per_shard_nodes += 0 if leaves == 0 else 2 * leaves - 1
    return {
        "case": case,
        "batch": 4,
        "speculative_rows": S,
        "active_rows_per_rank": 4 * S,
        "request_lengths": request_lengths,
        "total_history_tokens": sum(request_lengths),
        "final_candidates": final_candidates,
        "final_leaves": final_leaves,
        "submitted_chunks_per_rank": 2,
        "topk_nodes_per_shard": per_shard_nodes,
        "cache_pages_per_rank": pair_pages * 2,
    }


def _g4_layer_case(kind, case):
    """Return the attention fixture and exact owner counts for one G4 gate."""
    if case not in G4_LAYER_CASES:
        raise ValueError(f"unknown Phase G4 layer case {case!r}")
    attention_case = "one_m_tail" if case == "one_m_tail" else "heterogeneous_lengths"
    if kind == "csa" and attention_case == "heterogeneous_lengths":
        attention_case = "global_state_pool"
    if case == "full_active":
        counts = [T] * N_RANKS
    elif case == "ragged_mixed":
        pattern = (0, 1, T - 1, T)
        counts = [pattern[rank % len(pattern)] for rank in range(N_RANKS)]
    elif case == "all_inactive":
        counts = [0] * N_RANKS
    else:
        pattern = tuple(range(S, 0, -1))
        counts = [pattern[rank % len(pattern)] for rank in range(N_RANKS)]
    return attention_case, counts


def main():
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8, 16])
    parser.add_argument("-d", "--device", default=",".join(map(str, range(N_RANKS))))
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--case", default="heterogeneous_lengths")
    parser.add_argument("--g4-case", choices=G4_LAYER_CASES, default=None)
    parser.add_argument("--num-tokens-per-owner", default=None)
    parser.add_argument("--balanced-routing", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--enable-l2-swimlane", type=int, nargs="?", const=1, default=0
    )
    parser.add_argument(
        "--dump-args",
        type=int,
        nargs="?",
        const=3,
        default=0,
        choices=(0, 1, 2, 3),
        help="capture task argument metadata (bare flag selects metadata-only level 3)",
    )
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    args = parser.parse_args()

    global CSA_SUBMISSION_CHUNKS
    kind = attention_kind_for_layer(args.layer_id)
    if args.g4_case is None:
        attention_case = args.case
        if attention_case in CSA_TRACE_CASES:
            if kind != "csa":
                raise ValueError("CSA trace cases require a CSA layer id")
            if args.num_tokens_per_owner is not None:
                raise ValueError(
                    "CSA trace cases own their exact B=4/S=8 active-row count"
                )
            counts = [4 * S] * N_RANKS
            CSA_SUBMISSION_CHUNKS = 2
            summary = describe_csa_trace_case(attention_case)
            print(
                "[CSA TRACE] "
                + " ".join(f"{key}={value}" for key, value in summary.items())
            )
        else:
            counts = _parse_counts(args.num_tokens_per_owner)
            CSA_SUBMISSION_CHUNKS = CSA_CHUNKS
    else:
        if args.num_tokens_per_owner is not None:
            raise ValueError("--g4-case and --num-tokens-per-owner are exclusive")
        attention_case, counts = _g4_layer_case(kind, args.g4_case)
        CSA_SUBMISSION_CHUNKS = CSA_CHUNKS
    if kind == "swa":
        host_fn = l3_decode_layer_swa
        specs = build_swa_layer_specs(
            case=attention_case,
            layer_id=args.layer_id,
            num_tokens_per_owner=counts,
            balanced_routing=args.balanced_routing,
        )
        golden_fn = golden_decode_layer_swa
    elif kind == "hca":
        host_fn = l3_decode_layer_hca
        specs = build_hca_layer_specs(
            case=attention_case,
            layer_id=args.layer_id,
            num_tokens_per_owner=counts,
            balanced_routing=args.balanced_routing,
        )
        golden_fn = golden_decode_layer_hca
    else:
        host_fn = l3_decode_layer_csa
        specs = build_csa_layer_specs(
            case=attention_case,
            layer_id=args.layer_id,
            num_tokens_per_owner=counts,
            balanced_routing=args.balanced_routing,
        )
        golden_fn = golden_decode_layer_csa

    device_ids = [int(value) for value in args.device.split(",")]
    if len(device_ids) < N_RANKS:
        raise ValueError(f"need at least {N_RANKS} devices, got {device_ids}")
    result = run_jit(
        fn=host_fn,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg={
            "dump_passes": args.dump_passes,
            "distributed_config": DistributedConfig(
                device_ids=device_ids[:N_RANKS], num_sub_workers=0
            ),
        },
        runtime_cfg={
            "platform": args.platform,
            "enable_l2_swimlane": args.enable_l2_swimlane,
            "enable_dump_args": args.dump_args,
            "enable_scope_stats": args.enable_scope_stats,
            "ring_heap": DECODE_LAYER_RING_HEAP,
        },
        compare_fn={
            "x_attn_workspace": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "main_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "idx_cache": ratio_allclose(atol=0.0, rtol=0.0),
            "idx_scale": ratio_allclose(atol=1e-4, rtol=1e-4),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
