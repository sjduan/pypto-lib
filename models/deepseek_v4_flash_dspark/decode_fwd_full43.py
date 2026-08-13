# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One decorated, packed-pool 43-layer D-Spark decode entry.

This module intentionally keeps the full forward separate from
``decode_fwd.py``'s host metadata planner.  It submits the existing
kind-specialized Phase-E leaves directly, and owns exactly one MoE signal
window domain for the whole 43-layer transaction.

Persistent cache/state tensors follow the baseline packed-block ABI.  The L3
owns ``[rank, packed_blocks, block_size, 1, head]`` pools and passes one rank
slice to each rank-local L2.  That child derives the per-layer extent once and
passes a direct four-dimensional block slice to each attention leaf.  Packing
the layer ordinal into the block axis keeps every public tensor at five or
fewer dimensions without losing the allocator-owned, B-independent pool
semantics.  CSA Phase-D descriptors remain ``[csa_layer, shard, ...]`` because
a shard is the unit consumed by the leaf pipeline.
"""

from __future__ import annotations

import inspect
import os

# The full-forward module imports PyPTO before importing ``decode_fwd``.  Set
# the profiled T=128 layout here so import order cannot silently fall back to
# the runtime default.  Run 055 Phase F established that HCA's two nested
# attention lifetimes need the middle rings. The rank-child scope adds one
# level relative to standalone HCA, so its bounded query frontier uses ring 3;
# Full43 scope stats measured 535.3 MiB there before the next layer admission.
# EP4 1M-tail profiling exhausted both middle 2-GiB rings and the default
# task/dependency frontiers.  Four GiB is the validated platform ceiling for
# each middle ring; the CSA Top-K submission path must keep its task frontier
# bounded instead of relying on a larger address arena.
_MIB = 1024 * 1024
DECODE_FULL43_RING_HEAP = (256 * _MIB, 4096 * _MIB, 4096 * _MIB, 256 * _MIB)
DECODE_FULL43_RING_TASK_WINDOW = 262144
DECODE_FULL43_RING_DEP_POOL = 524288
os.environ.setdefault(
    "PTO2_RING_HEAP", ",".join(str(value) for value in DECODE_FULL43_RING_HEAP)
)
os.environ.setdefault(
    "PTO2_RING_TASK_WINDOW", str(DECODE_FULL43_RING_TASK_WINDOW)
)
os.environ.setdefault("PTO2_RING_DEP_POOL", str(DECODE_FULL43_RING_DEP_POOL))

import pypto.language as pl
import pypto.language.distributed as pld

from config import KV_ORI_BLOCK_NUM
from decode_fwd import (
    CSA_ACTIVATION_COPY_BLOCKS,
    CSA_CHUNKS,
    EMBED_VOCAB_DYN,
    LM_HEAD_COMM_EPOCH,
    MAIN_LAYER_COUNT,
    TWO_SWA_LAYERS,
    decode_fwd_clear_shared_moe_signals,
    decode_fwd_pack_active,
    decode_fwd_stage_csa_activation,
    decode_fwd_terminal_head_active,
)
from decode_layer import (
    AUX_PAD,
    BLOCK_SIZE,
    CSA_CHUNK_B,
    CSA_CHUNK_T,
    CSA_EVENT_CAP,
    CSA_INDEX_TOPK,
    CSA_INNER_OUT_DIM,
    CSA_INNER_STATE_DIM,
    CSA_INNER_STATE_BLOCK_SIZE,
    CSA_INNER_STATE_PAGES_PER_REQUEST,
    CSA_INNER_STATE_ROWS_PER_REQUEST,
    CSA_LAYER_SHARDS,
    CSA_LEAF_DYN,
    CSA_MAIN_OUT_DIM,
    CSA_MAIN_STATE_DIM,
    CSA_PAIR_DYN,
    CSA_PAGE_DYN,
    CSA_REQUEST_DYN,
    CSA_REQUEST_OFFSET_DYN,
    CSA_SINGLETON_DYN,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_PAGES_PER_REQUEST,
    CSA_STATE_ROWS_PER_REQUEST,
    CSA_UPPER_DYN,
    D,
    H,
    HCA_B_DYN,
    HCA_COMPRESS_RATIO,
    HCA_COMPRESS_STATE_DIM,
    HCA_EVENT_DYN,
    HCA_ORI_BLOCK_NUM_DYN,
    HCA_PAGES_DYN,
    HCA_QUERY_OFFSETS_DYN,
    HCA_REQUEST_OFFSETS_DYN,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_PAGES_PER_REQUEST,
    HCA_WORK_DYN,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_ROUTES,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    PHASE_D_LEAF_FIELDS,
    PHASE_D_PAIR_FIELDS,
    PHASE_D_ROOT_FIELDS,
    PHASE_D_SINGLETON_FIELDS,
    PHASE_D_UPPER_FIELDS,
    Q_LORA,
    RECV_MAX,
    ROPE_HEAD_DIM,
    TOPK,
    VOCAB,
    WIN,
    csa_indexer_layer_stage,
    csa_inner_compressor_layer_stage,
    csa_layer_finalize,
    csa_layer_frontend,
    csa_layer_moe,
    csa_main_compressor_layer_stage,
    csa_sparse_value_layer_stage,
    decode_layer_hca,
    decode_layer_csa,
    decode_layer_swa,
)
from lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
)
from moe import N_RANKS, T


# The same rank-local layer implementations serve two public boundaries:
# standalone layer runners submit them as chip orchestrations, while the
# full-forward rank child needs their bodies spliced into one L2 program.
# Re-wrapping the original Python functions keeps one implementation of the
# layer math and changes only the JIT boundary.
decode_layer_swa_inline = pl.jit.inline(auto_scope=False)(decode_layer_swa._func)
decode_layer_swa_inline.__name__ = "decode_layer_swa_inline"
decode_layer_hca_inline = pl.jit.inline(auto_scope=False)(decode_layer_hca._func)
decode_layer_hca_inline.__name__ = "decode_layer_hca_inline"
decode_layer_csa_inline = pl.jit.inline(auto_scope=False)(decode_layer_csa._func)
decode_layer_csa_inline.__name__ = "decode_layer_csa_inline"
decode_fwd_pack_active_inline = pl.jit.inline(auto_scope=False)(
    decode_fwd_pack_active._func
)
decode_fwd_pack_active_inline.__name__ = "decode_fwd_pack_active_inline"
decode_fwd_clear_shared_moe_signals_inline = pl.jit.inline(auto_scope=False)(
    decode_fwd_clear_shared_moe_signals._func
)
decode_fwd_clear_shared_moe_signals_inline.__name__ = (
    "decode_fwd_clear_shared_moe_signals_inline"
)
decode_fwd_terminal_head_active_inline = pl.jit.inline(auto_scope=False)(
    decode_fwd_terminal_head_active._func
)
decode_fwd_terminal_head_active_inline.__name__ = (
    "decode_fwd_terminal_head_active_inline"
)


CSA_FULL_LAYERS = 21
HCA_FULL_LAYERS = 20
LAST_CSA_MODEL_LAYER = MAIN_LAYER_COUNT - 1
FWD_WEIGHT_BANK_DYN = pl.dynamic("FWD_WEIGHT_BANK_DYN")
FWD_PACKED_RAW_BLOCKS_DYN = pl.dynamic("FWD_PACKED_RAW_BLOCKS_DYN")
FWD_PACKED_HCA_STATE_BLOCKS_DYN = pl.dynamic("FWD_PACKED_HCA_STATE_BLOCKS_DYN")
FWD_PACKED_HCA_CMP_BLOCKS_DYN = pl.dynamic("FWD_PACKED_HCA_CMP_BLOCKS_DYN")
FWD_PACKED_CSA_MAIN_STATE_BLOCKS_DYN = pl.dynamic(
    "FWD_PACKED_CSA_MAIN_STATE_BLOCKS_DYN"
)
FWD_PACKED_CSA_MAIN_BLOCKS_DYN = pl.dynamic("FWD_PACKED_CSA_MAIN_BLOCKS_DYN")
FWD_PACKED_CSA_INNER_STATE_BLOCKS_DYN = pl.dynamic(
    "FWD_PACKED_CSA_INNER_STATE_BLOCKS_DYN"
)
FWD_PACKED_CSA_IDX_ROWS_DYN = pl.dynamic("FWD_PACKED_CSA_IDX_ROWS_DYN")

assert MAIN_LAYER_COUNT == 43
assert TWO_SWA_LAYERS == 2
assert CSA_FULL_LAYERS == 21
assert HCA_FULL_LAYERS == 20
assert LAST_CSA_MODEL_LAYER == 42
assert CSA_ACTIVATION_COPY_BLOCKS * 1 > 0


@pl.jit(auto_scope=False)
def mark_full43_program_completion(
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    completion: pl.Out[pl.Tensor[[1], pl.INT32]],
):
    """Publish an externally observable marker for the completed main graph."""
    for core in pl.spmd(1, name_hint="mark_full43_program_completion"):
        # Anchor the marker after terminal sampling. The distributed program
        # return remains the join for every scheduled persistent writer.
        _sample_anchor = pl.read(sampled_ids, [0, 0])
        pl.write(completion, [core], _sample_anchor)
    return completion


mark_full43_program_completion_inline = pl.jit.inline(auto_scope=False)(
    mark_full43_program_completion._func
)
mark_full43_program_completion_inline.__name__ = (
    "mark_full43_program_completion_inline"
)


@pl.jit(auto_scope=False)
def l3_decode_fwd_full43_rank(
    x_ping: pl.InOut[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_pong: pl.InOut[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    # Main-model immutable tensors use a caller-owned weight bank. Production
    # binds 43 entries; the bounded G5 runtime witness may bind one immutable
    # entry and reuse it without changing the 43-layer execution/cache graph.
    hc_attn_fn: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_WEIGHT_BANK_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_WEIGHT_BANK_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[FWD_WEIGHT_BANK_DYN, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[FWD_WEIGHT_BANK_DYN, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[FWD_WEIGHT_BANK_DYN, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_WEIGHT_BANK_DYN, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_WEIGHT_BANK_DYN, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_WEIGHT_BANK_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[FWD_WEIGHT_BANK_DYN, H], pl.FP32],
    wo_a: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
    ],
    wo_b: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, D, O_GROUPS * O_LORA], pl.INT8
    ],
    wo_b_scale: pl.Tensor[[FWD_WEIGHT_BANK_DYN, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_WEIGHT_BANK_DYN, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_WEIGHT_BANK_DYN, D], pl.BF16],
    gate_w: pl.Tensor[[FWD_WEIGHT_BANK_DYN, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_WEIGHT_BANK_DYN, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_WEIGHT_BANK_DYN, VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_DYN, N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[FWD_WEIGHT_BANK_DYN, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[FWD_WEIGHT_BANK_DYN, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[FWD_WEIGHT_BANK_DYN, D], pl.FP32],
    # Model-layer ownership is packed on the block axis.  Page IDs and write
    # slots remain local to one layer; the child rebases only the pool slice.
    raw_kv_pool: pl.InOut[pl.Tensor[
        [FWD_PACKED_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM],
        pl.BF16,
    ]],
    swa_rope_cos: pl.Tensor[[TWO_SWA_LAYERS, T, ROPE_HEAD_DIM], pl.BF16],
    swa_rope_sin: pl.Tensor[[TWO_SWA_LAYERS, T, ROPE_HEAD_DIM], pl.BF16],
    swa_raw_write_slots: pl.Tensor[[TWO_SWA_LAYERS, T], pl.INT64],
    swa_sources: pl.Tensor[[TWO_SWA_LAYERS, T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[TWO_SWA_LAYERS, T], pl.INT32],
    # HCA metadata is rank-local and layer-major; persistent pools are packed.
    hca_query_rope_cos: pl.Tensor[
        [HCA_FULL_LAYERS, T, ROPE_HEAD_DIM], pl.BF16
    ],
    hca_query_rope_sin: pl.Tensor[
        [HCA_FULL_LAYERS, T, ROPE_HEAD_DIM], pl.BF16
    ],
    hca_cmp_wkv: pl.Tensor[[HCA_FULL_LAYERS, HEAD_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_FULL_LAYERS, HEAD_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32
    ],
    hca_cmp_norm_w: pl.Tensor[[HCA_FULL_LAYERS, HEAD_DIM], pl.BF16],
    hca_request_event_indices: pl.Tensor[[HCA_FULL_LAYERS, HCA_B_DYN], pl.INT32],
    hca_event_rope_cos: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    hca_event_rope_sin: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    hca_compress_state: pl.InOut[pl.Tensor[
        [
            FWD_PACKED_HCA_STATE_BLOCKS_DYN,
            HCA_STATE_BLOCK_SIZE,
            HCA_COMPRESS_STATE_DIM,
        ],
        pl.FP32,
    ]],
    hca_state_page_ids: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST],
        pl.INT32,
    ],
    hca_state_valid_ranges: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_B_DYN, 2], pl.INT32
    ],
    hca_state_page_epochs: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST],
        pl.INT32,
    ],
    hca_request_epochs: pl.Tensor[[HCA_FULL_LAYERS, HCA_B_DYN], pl.INT32],
    hca_state_write_slots: pl.Tensor[[HCA_FULL_LAYERS, T], pl.INT64],
    hca_swa_write_slots: pl.Tensor[[HCA_FULL_LAYERS, T], pl.INT64],
    hca_swa_sources: pl.Tensor[[HCA_FULL_LAYERS, T, WIN], pl.INT32],
    hca_cmp_kv: pl.InOut[pl.Tensor[
        [
            FWD_PACKED_HCA_CMP_BLOCKS_DYN,
            BLOCK_SIZE,
            1,
            HEAD_DIM,
        ],
        pl.BF16,
    ]],
    hca_cmp_slot_mapping: pl.Tensor[[HCA_FULL_LAYERS, T], pl.INT64],
    hca_position_ids: pl.Tensor[[HCA_FULL_LAYERS, T], pl.INT32],
    hca_query_request_ids: pl.Tensor[[HCA_FULL_LAYERS, T], pl.INT32],
    hca_pages: pl.Tensor[[HCA_FULL_LAYERS, HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_REQUEST_OFFSETS_DYN], pl.INT32
    ],
    hca_windows: pl.Tensor[[HCA_FULL_LAYERS, HCA_B_DYN, 3], pl.INT32],
    hca_query_work_offsets: pl.Tensor[
        [HCA_FULL_LAYERS, HCA_QUERY_OFFSETS_DYN], pl.INT32
    ],
    hca_work_query_ids: pl.Tensor[[HCA_FULL_LAYERS, HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_FULL_LAYERS, HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_FULL_LAYERS, HCA_WORK_DYN], pl.INT32],
    # CSA descriptors retain their leaf-facing [layer, shard, ...] layout.
    csa_rope_cos: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16
    ],
    csa_rope_sin: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16
    ],
    csa_main_wkv: pl.Tensor[[CSA_FULL_LAYERS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_main_wgate: pl.Tensor[[CSA_FULL_LAYERS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_main_ape: pl.Tensor[[CSA_FULL_LAYERS, 4, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_main_norm_w: pl.Tensor[[CSA_FULL_LAYERS, HEAD_DIM], pl.BF16],
    csa_main_state: pl.InOut[pl.Tensor[
        [
            FWD_PACKED_CSA_MAIN_STATE_BLOCKS_DYN,
            CSA_STATE_BLOCK_SIZE,
            CSA_MAIN_STATE_DIM,
        ],
        pl.FP32,
    ]],
    csa_main_state_page_ids: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_main_state_valid_ranges: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B, 2], pl.INT32
    ],
    csa_main_state_page_epochs: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_compressor_request_epochs: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B], pl.INT32
    ],
    csa_request_event_indices: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B], pl.INT32
    ],
    csa_event_query_ids: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP], pl.INT32
    ],
    csa_event_rope_cos: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    csa_event_rope_sin: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    csa_main_event_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP], pl.INT64
    ],
    csa_position_ids: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT32
    ],
    csa_main_state_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT64
    ],
    csa_main_cache: pl.InOut[pl.Tensor[
        [
            FWD_PACKED_CSA_MAIN_BLOCKS_DYN,
            BLOCK_SIZE,
            1,
            HEAD_DIM,
        ],
        pl.BF16,
    ]],
    csa_inner_wkv: pl.Tensor[[CSA_FULL_LAYERS, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_FULL_LAYERS, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_FULL_LAYERS, 4, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_FULL_LAYERS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_hadamard: pl.Tensor[
        [CSA_FULL_LAYERS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16
    ],
    csa_inner_state: pl.InOut[pl.Tensor[
        [
            FWD_PACKED_CSA_INNER_STATE_BLOCKS_DYN,
            CSA_INNER_STATE_BLOCK_SIZE,
            CSA_INNER_STATE_DIM,
        ],
        pl.FP32,
    ]],
    csa_inner_state_page_ids: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_INNER_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_inner_state_valid_ranges: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B, 2], pl.INT32
    ],
    csa_inner_state_page_epochs: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_INNER_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_inner_event_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP], pl.INT64
    ],
    csa_inner_state_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT64
    ],
    csa_idx_cache: pl.InOut[pl.Tensor[
        [FWD_PACKED_CSA_IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8
    ]],
    csa_idx_scale: pl.InOut[pl.Tensor[
        [FWD_PACKED_CSA_IDX_ROWS_DYN, 1], pl.FP32
    ]],
    csa_idx_wq_b: pl.Tensor[
        [CSA_FULL_LAYERS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [CSA_FULL_LAYERS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32
    ],
    csa_idx_weights_proj: pl.Tensor[
        [CSA_FULL_LAYERS, D, IDX_N_HEADS], pl.BF16
    ],
    csa_idx_hadamard: pl.Tensor[
        [CSA_FULL_LAYERS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16
    ],
    csa_idx_cos_il: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32
    ],
    csa_idx_sin_signed: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32
    ],
    csa_query_request_ids: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT32
    ],
    csa_idx_pages: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_PAGE_DYN, 2], pl.INT32
    ],
    csa_idx_page_offsets: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_OFFSET_DYN], pl.INT32
    ],
    csa_idx_windows: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_DYN, 3], pl.INT32
    ],
    csa_request_epochs: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_DYN], pl.INT32
    ],
    csa_leaf_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    csa_pair_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    csa_singleton_descriptors: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_SINGLETON_DYN,
            PHASE_D_SINGLETON_FIELDS,
        ],
        pl.INT32,
    ],
    csa_upper_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    csa_root_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], pl.INT32
    ],
    csa_pages: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_PAGE_DYN, 2], pl.INT32
    ],
    csa_page_offsets: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_OFFSET_DYN], pl.INT32
    ],
    csa_windows: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_DYN, 3], pl.INT32
    ],
    csa_raw_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT64
    ],
    csa_swa_sources: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, WIN], pl.INT32
    ],
    input_ids: pl.Tensor[[T], pl.INT64],
    active_row_mask: pl.Tensor[[T], pl.INT32],
    embed_weight: pl.Tensor[[EMBED_VOCAB_DYN, D], pl.BF16],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[
        pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]
    ],
    full43_completion: pl.InOut[pl.Tensor[[1], pl.INT32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[
        [GROUP_LOGIT_ROWS, D], pl.BF16
    ],
    lm_head_hidden_done: pld.DistributedTensor[
        [LM_HEAD_TP_SIZE, 1], pl.INT32
    ],
    lm_head_logits_window: pld.DistributedTensor[
        [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32
    ],
    lm_head_logits_done: pld.DistributedTensor[
        [LM_HEAD_TP_SIZE, 1], pl.INT32
    ],
    # TaskArgs requires all tensor arguments to be bound before scalars.
    # Keep both rank-child scalars at the end of the public ABI.
    weight_bank_size: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Submit ``SWA×2 + (CSA,HCA)×20 + CSA`` in one MoE window domain.

    The only data-dependent history inputs are the allocator-produced ragged
    descriptors passed to the HCA/CSA leaves.  This host never allocates or
    pads history to a 1M tensor; work follows the descriptor fields exactly.
    """
    rank = my_rank
    # Ordinary tensors in this child are already rank-local.  Named aliases
    # preserve the leaf call sites without child-side world-axis slicing.
    x_ping_rank = x_ping
    x_pong_rank = x_pong
    input_ids_rank = input_ids
    active_row_mask_rank = active_row_mask
    embed_weight_rank = embed_weight
    hc_head_fn_rank = hc_head_fn
    hc_head_scale_rank = hc_head_scale
    hc_head_base_rank = hc_head_base
    final_norm_w_rank = final_norm_w
    lm_head_weight_rank = lm_head_weight
    hidden_out_rank = hidden_out
    logits_rank = logits
    sampled_ids_rank = sampled_ids
    full43_completion_rank = full43_completion

    # Host admission proves divisibility and exact configured capacities.
    # Therefore every layer view below is a native contiguous block slice.
    raw_blocks_per_layer = pl.tensor.dim(raw_kv_pool, 0) // MAIN_LAYER_COUNT
    hca_state_blocks_per_layer = (
        pl.tensor.dim(hca_compress_state, 0) // HCA_FULL_LAYERS
    )
    hca_cmp_blocks_per_layer = pl.tensor.dim(hca_cmp_kv, 0) // HCA_FULL_LAYERS
    csa_main_state_blocks_per_layer = (
        pl.tensor.dim(csa_main_state, 0) // CSA_FULL_LAYERS
    )
    csa_main_blocks_per_layer = pl.tensor.dim(csa_main_cache, 0) // CSA_FULL_LAYERS
    csa_inner_state_blocks_per_layer = (
        pl.tensor.dim(csa_inner_state, 0) // CSA_FULL_LAYERS
    )
    csa_idx_rows_per_layer = pl.tensor.dim(csa_idx_cache, 0) // CSA_FULL_LAYERS
    decode_fwd_pack_active_inline(
        input_ids_rank,
        active_row_mask_rank,
        embed_weight_rank,
        x_ping_rank,
    )
    # Keep stage ownership aligned with the baseline: the two static SWA
    # stages own independent bridges, while all CSA iterations reuse one and
    # all HCA iterations reuse another.  Four bounded 8-MiB allocations avoid
    # both the old per-layer root-ring growth and cross-stage producer/fanout
    # aliasing.
    swa0_attention_workspace_rank = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32
    )
    swa1_attention_workspace_rank = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32
    )
    csa_attention_workspace_rank = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32
    )
    hca_attention_workspace_rank = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32
    )

    # The first two layers are static so the ping-pong direction and initial
    # epochs are visible even to conservative host frontends.  Pair loops
    # below then cover epochs 3..42; the terminal CSA uses epoch 43.
    # Keep rank and layer selections as separate named assignments.  The
    # PyPTO dependency specializer tracks each rank-reducing subscript,
    # while a nested ``tensor[rank][layer]`` expression is intentionally
    # not treated as a statically typed dependency argument.
    hc_attn_fn_rank = hc_attn_fn
    hc_attn_fn_layer = hc_attn_fn_rank[0]
    hc_attn_scale_rank = hc_attn_scale
    hc_attn_scale_layer = hc_attn_scale_rank[0]
    hc_attn_base_rank = hc_attn_base
    hc_attn_base_layer = hc_attn_base_rank[0]
    attn_norm_w_rank = attn_norm_w
    attn_norm_w_layer = attn_norm_w_rank[0]
    wq_a_rank = wq_a
    wq_a_layer = wq_a_rank[0]
    wq_b_rank = wq_b
    wq_b_layer = wq_b_rank[0]
    wq_b_scale_rank = wq_b_scale
    wq_b_scale_layer = wq_b_scale_rank[0]
    wkv_rank = wkv
    wkv_layer = wkv_rank[0]
    gamma_cq_rank = gamma_cq
    gamma_cq_layer = gamma_cq_rank[0]
    gamma_ckv_rank = gamma_ckv
    gamma_ckv_layer = gamma_ckv_rank[0]
    swa_rope_cos_rank = swa_rope_cos
    rope_cos_layer = swa_rope_cos_rank[0]
    swa_rope_sin_rank = swa_rope_sin
    rope_sin_layer = swa_rope_sin_rank[0]
    swa_raw_write_slots_rank = swa_raw_write_slots
    raw_write_slots_layer = swa_raw_write_slots_rank[0]
    swa_sources_rank = swa_sources
    swa_sources_layer = swa_sources_rank[0]
    swa_lens_rank = swa_lens
    swa_lens_layer = swa_lens_rank[0]
    attn_sink_rank = attn_sink
    attn_sink_layer = attn_sink_rank[0]
    wo_a_rank = wo_a
    wo_a_layer = wo_a_rank[0]
    wo_b_rank = wo_b
    wo_b_layer = wo_b_rank[0]
    wo_b_scale_rank = wo_b_scale
    wo_b_scale_layer = wo_b_scale_rank[0]
    hc_ffn_fn_rank = hc_ffn_fn
    hc_ffn_fn_layer = hc_ffn_fn_rank[0]
    hc_ffn_scale_rank = hc_ffn_scale
    hc_ffn_scale_layer = hc_ffn_scale_rank[0]
    hc_ffn_base_rank = hc_ffn_base
    hc_ffn_base_layer = hc_ffn_base_rank[0]
    norm_w_rank = norm_w
    norm_w_layer = norm_w_rank[0]
    gate_w_rank = gate_w
    gate_w_layer = gate_w_rank[0]
    gate_bias_rank = gate_bias
    gate_bias_layer = gate_bias_rank[0]
    tid2eid_rank = tid2eid
    tid2eid_layer = tid2eid_rank[0]
    routed_w1_rank = routed_w1
    routed_w1_layer = routed_w1_rank[0]
    routed_w1_scale_rank = routed_w1_scale
    routed_w1_scale_layer = routed_w1_scale_rank[0]
    routed_w3_rank = routed_w3
    routed_w3_layer = routed_w3_rank[0]
    routed_w3_scale_rank = routed_w3_scale
    routed_w3_scale_layer = routed_w3_scale_rank[0]
    routed_w2_rank = routed_w2
    routed_w2_layer = routed_w2_rank[0]
    routed_w2_scale_rank = routed_w2_scale
    routed_w2_scale_layer = routed_w2_scale_rank[0]
    shared_w1_rank = shared_w1
    shared_w1_layer = shared_w1_rank[0]
    shared_w1_scale_rank = shared_w1_scale
    shared_w1_scale_layer = shared_w1_scale_rank[0]
    shared_w3_rank = shared_w3
    shared_w3_layer = shared_w3_rank[0]
    shared_w3_scale_rank = shared_w3_scale
    shared_w3_scale_layer = shared_w3_scale_rank[0]
    shared_w2_rank = shared_w2
    shared_w2_layer = shared_w2_rank[0]
    shared_w2_scale_rank = shared_w2_scale
    shared_w2_scale_layer = shared_w2_scale_rank[0]
    raw_layer = pl.slice(
        raw_kv_pool,
        [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [0, 0, 0, 0],
    )
    decode_layer_swa_inline(
        x_ping_rank,
        hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
        attn_norm_w_layer, wq_a_layer, wq_b_layer, wq_b_scale_layer,
        wkv_layer, gamma_cq_layer, gamma_ckv_layer, rope_cos_layer,
        rope_sin_layer,
        raw_layer,
        raw_write_slots_layer, swa_sources_layer, swa_lens_layer,
        attn_sink_layer, wo_a_layer, wo_b_layer, wo_b_scale_layer,
        hc_ffn_fn_layer, hc_ffn_scale_layer, hc_ffn_base_layer,
        norm_w_layer, gate_w_layer, gate_bias_layer, tid2eid_layer,
        input_ids_rank, routed_w1_layer, routed_w1_scale_layer,
        routed_w3_layer, routed_w3_scale_layer, routed_w2_layer,
        routed_w2_scale_layer, shared_w1_layer, shared_w1_scale_layer,
        shared_w3_layer, shared_w3_scale_layer, shared_w2_layer,
        shared_w2_scale_layer,
        num_tokens_per_owner, swa0_attention_workspace_rank, x_pong_rank,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        routed_y_buf, combine_arrived, pl.const(0, pl.INT32), rank,
        pl.const(1, pl.INT32),
    )

    weight_layer_one = pl.cast(1, pl.INT32) % weight_bank_size
    hc_attn_fn_rank = hc_attn_fn
    hc_attn_fn_layer = hc_attn_fn_rank[weight_layer_one]
    hc_attn_scale_rank = hc_attn_scale
    hc_attn_scale_layer = hc_attn_scale_rank[weight_layer_one]
    hc_attn_base_rank = hc_attn_base
    hc_attn_base_layer = hc_attn_base_rank[weight_layer_one]
    attn_norm_w_rank = attn_norm_w
    attn_norm_w_layer = attn_norm_w_rank[weight_layer_one]
    wq_a_rank = wq_a
    wq_a_layer = wq_a_rank[weight_layer_one]
    wq_b_rank = wq_b
    wq_b_layer = wq_b_rank[weight_layer_one]
    wq_b_scale_rank = wq_b_scale
    wq_b_scale_layer = wq_b_scale_rank[weight_layer_one]
    wkv_rank = wkv
    wkv_layer = wkv_rank[weight_layer_one]
    gamma_cq_rank = gamma_cq
    gamma_cq_layer = gamma_cq_rank[weight_layer_one]
    gamma_ckv_rank = gamma_ckv
    gamma_ckv_layer = gamma_ckv_rank[weight_layer_one]
    swa_rope_cos_rank = swa_rope_cos
    rope_cos_layer = swa_rope_cos_rank[1]
    swa_rope_sin_rank = swa_rope_sin
    rope_sin_layer = swa_rope_sin_rank[1]
    swa_raw_write_slots_rank = swa_raw_write_slots
    raw_write_slots_layer = swa_raw_write_slots_rank[1]
    swa_sources_rank = swa_sources
    swa_sources_layer = swa_sources_rank[1]
    swa_lens_rank = swa_lens
    swa_lens_layer = swa_lens_rank[1]
    attn_sink_rank = attn_sink
    attn_sink_layer = attn_sink_rank[weight_layer_one]
    wo_a_rank = wo_a
    wo_a_layer = wo_a_rank[weight_layer_one]
    wo_b_rank = wo_b
    wo_b_layer = wo_b_rank[weight_layer_one]
    wo_b_scale_rank = wo_b_scale
    wo_b_scale_layer = wo_b_scale_rank[weight_layer_one]
    hc_ffn_fn_rank = hc_ffn_fn
    hc_ffn_fn_layer = hc_ffn_fn_rank[weight_layer_one]
    hc_ffn_scale_rank = hc_ffn_scale
    hc_ffn_scale_layer = hc_ffn_scale_rank[weight_layer_one]
    hc_ffn_base_rank = hc_ffn_base
    hc_ffn_base_layer = hc_ffn_base_rank[weight_layer_one]
    norm_w_rank = norm_w
    norm_w_layer = norm_w_rank[weight_layer_one]
    gate_w_rank = gate_w
    gate_w_layer = gate_w_rank[weight_layer_one]
    gate_bias_rank = gate_bias
    gate_bias_layer = gate_bias_rank[weight_layer_one]
    tid2eid_rank = tid2eid
    tid2eid_layer = tid2eid_rank[weight_layer_one]
    routed_w1_rank = routed_w1
    routed_w1_layer = routed_w1_rank[weight_layer_one]
    routed_w1_scale_rank = routed_w1_scale
    routed_w1_scale_layer = routed_w1_scale_rank[weight_layer_one]
    routed_w3_rank = routed_w3
    routed_w3_layer = routed_w3_rank[weight_layer_one]
    routed_w3_scale_rank = routed_w3_scale
    routed_w3_scale_layer = routed_w3_scale_rank[weight_layer_one]
    routed_w2_rank = routed_w2
    routed_w2_layer = routed_w2_rank[weight_layer_one]
    routed_w2_scale_rank = routed_w2_scale
    routed_w2_scale_layer = routed_w2_scale_rank[weight_layer_one]
    shared_w1_rank = shared_w1
    shared_w1_layer = shared_w1_rank[weight_layer_one]
    shared_w1_scale_rank = shared_w1_scale
    shared_w1_scale_layer = shared_w1_scale_rank[weight_layer_one]
    shared_w3_rank = shared_w3
    shared_w3_layer = shared_w3_rank[weight_layer_one]
    shared_w3_scale_rank = shared_w3_scale
    shared_w3_scale_layer = shared_w3_scale_rank[weight_layer_one]
    shared_w2_rank = shared_w2
    shared_w2_layer = shared_w2_rank[weight_layer_one]
    shared_w2_scale_rank = shared_w2_scale
    shared_w2_scale_layer = shared_w2_scale_rank[weight_layer_one]
    raw_layer = pl.slice(
        raw_kv_pool,
        [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [raw_blocks_per_layer, 0, 0, 0],
    )
    decode_layer_swa_inline(
        x_pong_rank,
        hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
        attn_norm_w_layer, wq_a_layer, wq_b_layer, wq_b_scale_layer,
        wkv_layer, gamma_cq_layer, gamma_ckv_layer, rope_cos_layer,
        rope_sin_layer,
        raw_layer,
        raw_write_slots_layer, swa_sources_layer, swa_lens_layer,
        attn_sink_layer, wo_a_layer, wo_b_layer, wo_b_scale_layer,
        hc_ffn_fn_layer, hc_ffn_scale_layer, hc_ffn_base_layer,
        norm_w_layer, gate_w_layer, gate_bias_layer, tid2eid_layer,
        input_ids_rank,
        routed_w1_layer, routed_w1_scale_layer,
        routed_w3_layer, routed_w3_scale_layer,
        routed_w2_layer, routed_w2_scale_layer,
        shared_w1_layer, shared_w1_scale_layer,
        shared_w3_layer, shared_w3_scale_layer,
        shared_w2_layer, shared_w2_scale_layer,
        num_tokens_per_owner, swa1_attention_workspace_rank, x_ping_rank,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        routed_y_buf, combine_arrived, pl.const(1, pl.INT32), rank,
        pl.const(2, pl.INT32),
    )

    # Each pair is ordered CSA(2i+2) -> HCA(2i+3).  The type-specialized
    # leaves are called directly; only their rank/layer selections happen at
    # this orchestration boundary.
    for csa_ordinal in pl.range(CSA_FULL_LAYERS):
        csa_model_layer = pl.cast(csa_ordinal * 2 + 2, pl.INT32)
        hca_model_layer = pl.cast(csa_ordinal * 2 + 3, pl.INT32)
        csa_weight_layer = csa_model_layer % weight_bank_size
        hca_weight_layer = hca_model_layer % weight_bank_size
        csa_moe_epoch = pl.cast(csa_ordinal * 2 + 3, pl.INT32)
        hca_moe_epoch = pl.cast(csa_ordinal * 2 + 4, pl.INT32)

        # A CSA descriptor is intentionally selected by ordinal before its
        # fixed submission shard is selected below.  This retains the Phase-D
        # [layer, shard, ...] ABI and lets the leaf drive real ragged work.
        csa_rope_cos_layer = csa_rope_cos[csa_ordinal]
        csa_rope_sin_layer = csa_rope_sin[csa_ordinal]
        csa_main_state_page_ids_layer = csa_main_state_page_ids[csa_ordinal]
        csa_main_state_valid_ranges_layer = csa_main_state_valid_ranges[csa_ordinal]
        csa_main_state_page_epochs_layer = csa_main_state_page_epochs[csa_ordinal]
        csa_compressor_request_epochs_layer = csa_compressor_request_epochs[csa_ordinal]
        csa_request_event_indices_layer = csa_request_event_indices[csa_ordinal]
        csa_event_query_ids_layer = csa_event_query_ids[csa_ordinal]
        csa_event_rope_cos_layer = csa_event_rope_cos[csa_ordinal]
        csa_event_rope_sin_layer = csa_event_rope_sin[csa_ordinal]
        csa_main_event_write_slots_layer = csa_main_event_write_slots[csa_ordinal]
        csa_position_ids_layer = csa_position_ids[csa_ordinal]
        csa_main_state_write_slots_layer = csa_main_state_write_slots[csa_ordinal]
        csa_inner_state_page_ids_layer = csa_inner_state_page_ids[csa_ordinal]
        csa_inner_state_valid_ranges_layer = csa_inner_state_valid_ranges[csa_ordinal]
        csa_inner_state_page_epochs_layer = csa_inner_state_page_epochs[csa_ordinal]
        csa_inner_event_write_slots_layer = csa_inner_event_write_slots[csa_ordinal]
        csa_inner_state_write_slots_layer = csa_inner_state_write_slots[csa_ordinal]
        csa_idx_cos_il_layer = csa_idx_cos_il[csa_ordinal]
        csa_idx_sin_signed_layer = csa_idx_sin_signed[csa_ordinal]
        csa_query_request_ids_layer = csa_query_request_ids[csa_ordinal]
        csa_idx_pages_layer = csa_idx_pages[csa_ordinal]
        csa_idx_page_offsets_layer = csa_idx_page_offsets[csa_ordinal]
        csa_idx_windows_layer = csa_idx_windows[csa_ordinal]
        csa_request_epochs_layer = csa_request_epochs[csa_ordinal]
        csa_leaf_descriptors_layer = csa_leaf_descriptors[csa_ordinal]
        csa_pair_descriptors_layer = csa_pair_descriptors[csa_ordinal]
        csa_singleton_descriptors_layer = csa_singleton_descriptors[csa_ordinal]
        csa_upper_descriptors_layer = csa_upper_descriptors[csa_ordinal]
        csa_root_descriptors_layer = csa_root_descriptors[csa_ordinal]
        csa_pages_layer = csa_pages[csa_ordinal]
        csa_page_offsets_layer = csa_page_offsets[csa_ordinal]
        csa_windows_layer = csa_windows[csa_ordinal]
        csa_raw_write_slots_layer = csa_raw_write_slots[csa_ordinal]
        csa_swa_sources_layer = csa_swa_sources[csa_ordinal]

        # The full-forward ABI keeps allocator descriptors packed by
        # ``rank * CSA_CHUNKS + chunk``.  Recover this rank's bounded view
        # once, then splice the complete CSA layer into this L2 program.
        shard_begin = rank * CSA_CHUNKS
        page_rows = pl.tensor.dim(csa_idx_pages_layer, 1)
        request_offset_rows = pl.tensor.dim(csa_idx_page_offsets_layer, 1)
        request_rows = pl.tensor.dim(csa_idx_windows_layer, 1)
        leaf_rows = pl.tensor.dim(csa_leaf_descriptors_layer, 1)
        pair_rows = pl.tensor.dim(csa_pair_descriptors_layer, 1)
        singleton_rows = pl.tensor.dim(csa_singleton_descriptors_layer, 1)
        upper_rows = pl.tensor.dim(csa_upper_descriptors_layer, 1)

        csa_rope_cos_rank = pl.reshape(pl.slice(csa_rope_cos_layer, [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM])
        csa_rope_sin_rank = pl.reshape(pl.slice(csa_rope_sin_layer, [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM])
        csa_main_state_page_ids_rank = pl.reshape(pl.slice(csa_main_state_page_ids_layer, [CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST])
        csa_main_state_valid_ranges_rank = pl.reshape(pl.slice(csa_main_state_valid_ranges_layer, [CSA_CHUNKS, CSA_CHUNK_B, 2], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_B, 2])
        csa_main_state_page_epochs_rank = pl.reshape(pl.slice(csa_main_state_page_epochs_layer, [CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_B, CSA_STATE_PAGES_PER_REQUEST])
        csa_compressor_request_epochs_rank = pl.reshape(pl.slice(csa_compressor_request_epochs_layer, [CSA_CHUNKS, CSA_CHUNK_B], [shard_begin, 0]), [CSA_CHUNKS, CSA_CHUNK_B])
        csa_request_event_indices_rank = pl.reshape(pl.slice(csa_request_event_indices_layer, [CSA_CHUNKS, CSA_CHUNK_B], [shard_begin, 0]), [CSA_CHUNKS, CSA_CHUNK_B])
        csa_event_query_ids_rank = pl.reshape(pl.slice(csa_event_query_ids_layer, [CSA_CHUNKS, CSA_EVENT_CAP], [shard_begin, 0]), [CSA_CHUNKS, CSA_EVENT_CAP])
        csa_event_rope_cos_rank = pl.reshape(pl.slice(csa_event_rope_cos_layer, [CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2])
        csa_event_rope_sin_rank = pl.reshape(pl.slice(csa_event_rope_sin_layer, [CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2])
        csa_main_event_write_slots_rank = pl.reshape(pl.slice(csa_main_event_write_slots_layer, [CSA_CHUNKS, CSA_EVENT_CAP], [shard_begin, 0]), [CSA_CHUNKS, CSA_EVENT_CAP])
        csa_position_ids_rank = pl.reshape(pl.slice(csa_position_ids_layer, [CSA_CHUNKS, CSA_CHUNK_T], [shard_begin, 0]), [CSA_CHUNKS, CSA_CHUNK_T])
        csa_main_state_write_slots_rank = pl.reshape(pl.slice(csa_main_state_write_slots_layer, [CSA_CHUNKS, CSA_CHUNK_T], [shard_begin, 0]), [CSA_CHUNKS, CSA_CHUNK_T])
        csa_inner_state_page_ids_rank = pl.reshape(pl.slice(csa_inner_state_page_ids_layer, [CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST])
        csa_inner_state_valid_ranges_rank = pl.reshape(pl.slice(csa_inner_state_valid_ranges_layer, [CSA_CHUNKS, CSA_CHUNK_B, 2], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_B, 2])
        csa_inner_state_page_epochs_rank = pl.reshape(pl.slice(csa_inner_state_page_epochs_layer, [CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_B, CSA_INNER_STATE_PAGES_PER_REQUEST])
        csa_inner_event_write_slots_rank = pl.reshape(pl.slice(csa_inner_event_write_slots_layer, [CSA_CHUNKS, CSA_EVENT_CAP], [shard_begin, 0]), [CSA_CHUNKS, CSA_EVENT_CAP])
        csa_inner_state_write_slots_rank = pl.reshape(pl.slice(csa_inner_state_write_slots_layer, [CSA_CHUNKS, CSA_CHUNK_T], [shard_begin, 0]), [CSA_CHUNKS, CSA_CHUNK_T])
        csa_idx_cos_il_rank = pl.reshape(pl.slice(csa_idx_cos_il_layer, [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM])
        csa_idx_sin_signed_rank = pl.reshape(pl.slice(csa_idx_sin_signed_layer, [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_T, ROPE_HEAD_DIM])
        csa_query_request_ids_rank = pl.reshape(pl.slice(csa_query_request_ids_layer, [CSA_CHUNKS, CSA_CHUNK_T], [shard_begin, 0]), [CSA_CHUNKS, CSA_CHUNK_T])
        csa_idx_pages_rank = pl.reshape(pl.slice(csa_idx_pages_layer, [CSA_CHUNKS, page_rows, 2], [shard_begin, 0, 0]), [CSA_CHUNKS, page_rows, 2])
        csa_idx_page_offsets_rank = pl.reshape(pl.slice(csa_idx_page_offsets_layer, [CSA_CHUNKS, request_offset_rows], [shard_begin, 0]), [CSA_CHUNKS, request_offset_rows])
        csa_idx_windows_rank = pl.reshape(pl.slice(csa_idx_windows_layer, [CSA_CHUNKS, request_rows, 3], [shard_begin, 0, 0]), [CSA_CHUNKS, request_rows, 3])
        csa_request_epochs_rank = pl.reshape(pl.slice(csa_request_epochs_layer, [CSA_CHUNKS, request_rows], [shard_begin, 0]), [CSA_CHUNKS, request_rows])
        csa_leaf_descriptors_rank = pl.reshape(pl.slice(csa_leaf_descriptors_layer, [CSA_CHUNKS, leaf_rows, PHASE_D_LEAF_FIELDS], [shard_begin, 0, 0]), [CSA_CHUNKS, leaf_rows, PHASE_D_LEAF_FIELDS])
        csa_pair_descriptors_rank = pl.reshape(pl.slice(csa_pair_descriptors_layer, [CSA_CHUNKS, pair_rows, PHASE_D_PAIR_FIELDS], [shard_begin, 0, 0]), [CSA_CHUNKS, pair_rows, PHASE_D_PAIR_FIELDS])
        csa_singleton_descriptors_rank = pl.reshape(pl.slice(csa_singleton_descriptors_layer, [CSA_CHUNKS, singleton_rows, PHASE_D_SINGLETON_FIELDS], [shard_begin, 0, 0]), [CSA_CHUNKS, singleton_rows, PHASE_D_SINGLETON_FIELDS])
        csa_upper_descriptors_rank = pl.reshape(pl.slice(csa_upper_descriptors_layer, [CSA_CHUNKS, upper_rows, PHASE_D_UPPER_FIELDS], [shard_begin, 0, 0]), [CSA_CHUNKS, upper_rows, PHASE_D_UPPER_FIELDS])
        csa_root_descriptors_rank = pl.reshape(pl.slice(csa_root_descriptors_layer, [CSA_CHUNKS, CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_T, PHASE_D_ROOT_FIELDS])
        csa_pages_rank = pl.reshape(pl.slice(csa_pages_layer, [CSA_CHUNKS, page_rows, 2], [shard_begin, 0, 0]), [CSA_CHUNKS, page_rows, 2])
        csa_page_offsets_rank = pl.reshape(pl.slice(csa_page_offsets_layer, [CSA_CHUNKS, request_offset_rows], [shard_begin, 0]), [CSA_CHUNKS, request_offset_rows])
        csa_windows_rank = pl.reshape(pl.slice(csa_windows_layer, [CSA_CHUNKS, request_rows, 3], [shard_begin, 0, 0]), [CSA_CHUNKS, request_rows, 3])
        csa_raw_write_slots_rank = pl.reshape(pl.slice(csa_raw_write_slots_layer, [CSA_CHUNKS, CSA_CHUNK_T], [shard_begin, 0]), [CSA_CHUNKS, CSA_CHUNK_T])
        csa_swa_sources_rank = pl.reshape(pl.slice(csa_swa_sources_layer, [CSA_CHUNKS, CSA_CHUNK_T, WIN], [shard_begin, 0, 0]), [CSA_CHUNKS, CSA_CHUNK_T, WIN])

        hc_attn_fn_rank = hc_attn_fn
        hc_attn_fn_layer = hc_attn_fn_rank[csa_weight_layer]
        hc_attn_scale_rank = hc_attn_scale
        hc_attn_scale_layer = hc_attn_scale_rank[csa_weight_layer]
        hc_attn_base_rank = hc_attn_base
        hc_attn_base_layer = hc_attn_base_rank[csa_weight_layer]
        attn_norm_w_rank = attn_norm_w
        attn_norm_w_layer = attn_norm_w_rank[csa_weight_layer]
        wq_a_rank = wq_a
        wq_a_layer = wq_a_rank[csa_weight_layer]
        wq_b_rank = wq_b
        wq_b_layer = wq_b_rank[csa_weight_layer]
        wq_b_scale_rank = wq_b_scale
        wq_b_scale_layer = wq_b_scale_rank[csa_weight_layer]
        wkv_rank = wkv
        wkv_layer = wkv_rank[csa_weight_layer]
        gamma_cq_rank = gamma_cq
        gamma_cq_layer = gamma_cq_rank[csa_weight_layer]
        gamma_ckv_rank = gamma_ckv
        gamma_ckv_layer = gamma_ckv_rank[csa_weight_layer]
        attn_sink_rank = attn_sink
        attn_sink_layer = attn_sink_rank[csa_weight_layer]
        wo_a_rank = wo_a
        wo_a_layer = wo_a_rank[csa_weight_layer]
        wo_b_rank = wo_b
        wo_b_layer = wo_b_rank[csa_weight_layer]
        wo_b_scale_rank = wo_b_scale
        wo_b_scale_layer = wo_b_scale_rank[csa_weight_layer]

        csa_main_wkv_rank = csa_main_wkv
        csa_main_wkv_layer = csa_main_wkv_rank[csa_ordinal]
        csa_main_wgate_rank = csa_main_wgate
        csa_main_wgate_layer = csa_main_wgate_rank[csa_ordinal]
        csa_main_ape_rank = csa_main_ape
        csa_main_ape_layer = csa_main_ape_rank[csa_ordinal]
        csa_main_norm_w_rank = csa_main_norm_w
        csa_main_norm_w_layer = csa_main_norm_w_rank[csa_ordinal]
        csa_main_state_layer = pl.slice(
            csa_main_state,
            [csa_main_state_blocks_per_layer, CSA_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
            [csa_ordinal * csa_main_state_blocks_per_layer, 0, 0],
        )
        csa_main_cache_layer = pl.slice(
            csa_main_cache,
            [csa_main_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [csa_ordinal * csa_main_blocks_per_layer, 0, 0, 0],
        )
        csa_inner_wkv_rank = csa_inner_wkv
        csa_inner_wkv_layer = csa_inner_wkv_rank[csa_ordinal]
        csa_inner_wgate_rank = csa_inner_wgate
        csa_inner_wgate_layer = csa_inner_wgate_rank[csa_ordinal]
        csa_inner_ape_rank = csa_inner_ape
        csa_inner_ape_layer = csa_inner_ape_rank[csa_ordinal]
        csa_inner_norm_w_rank = csa_inner_norm_w
        csa_inner_norm_w_layer = csa_inner_norm_w_rank[csa_ordinal]
        csa_inner_hadamard_rank = csa_inner_hadamard
        csa_inner_hadamard_layer = csa_inner_hadamard_rank[csa_ordinal]
        csa_inner_state_layer = pl.slice(
            csa_inner_state,
            [
                csa_inner_state_blocks_per_layer,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            [csa_ordinal * csa_inner_state_blocks_per_layer, 0, 0],
        )
        csa_idx_cache_layer = pl.slice(
            csa_idx_cache,
            [csa_idx_rows_per_layer, IDX_HEAD_DIM],
            [csa_ordinal * csa_idx_rows_per_layer, 0],
        )
        csa_idx_scale_layer = pl.slice(
            csa_idx_scale,
            [csa_idx_rows_per_layer, 1],
            [csa_ordinal * csa_idx_rows_per_layer, 0],
        )
        csa_idx_wq_b_rank = csa_idx_wq_b
        csa_idx_wq_b_layer = csa_idx_wq_b_rank[csa_ordinal]
        csa_idx_wq_b_scale_rank = csa_idx_wq_b_scale
        csa_idx_wq_b_scale_layer = csa_idx_wq_b_scale_rank[csa_ordinal]
        csa_idx_weights_proj_rank = csa_idx_weights_proj
        csa_idx_weights_proj_layer = csa_idx_weights_proj_rank[csa_ordinal]
        csa_idx_hadamard_rank = csa_idx_hadamard
        csa_idx_hadamard_layer = csa_idx_hadamard_rank[csa_ordinal]
        csa_raw_layer = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [csa_model_layer * raw_blocks_per_layer, 0, 0, 0],
        )

        hc_ffn_fn_rank = hc_ffn_fn
        hc_ffn_fn_layer = hc_ffn_fn_rank[csa_weight_layer]
        hc_ffn_scale_rank = hc_ffn_scale
        hc_ffn_scale_layer = hc_ffn_scale_rank[csa_weight_layer]
        hc_ffn_base_rank = hc_ffn_base
        hc_ffn_base_layer = hc_ffn_base_rank[csa_weight_layer]
        norm_w_rank = norm_w
        norm_w_layer = norm_w_rank[csa_weight_layer]
        gate_w_rank = gate_w
        gate_w_layer = gate_w_rank[csa_weight_layer]
        gate_bias_rank = gate_bias
        gate_bias_layer = gate_bias_rank[csa_weight_layer]
        tid2eid_rank = tid2eid
        tid2eid_layer = tid2eid_rank[csa_weight_layer]
        routed_w1_rank = routed_w1
        routed_w1_layer = routed_w1_rank[csa_weight_layer]
        routed_w1_scale_rank = routed_w1_scale
        routed_w1_scale_layer = routed_w1_scale_rank[csa_weight_layer]
        routed_w3_rank = routed_w3
        routed_w3_layer = routed_w3_rank[csa_weight_layer]
        routed_w3_scale_rank = routed_w3_scale
        routed_w3_scale_layer = routed_w3_scale_rank[csa_weight_layer]
        routed_w2_rank = routed_w2
        routed_w2_layer = routed_w2_rank[csa_weight_layer]
        routed_w2_scale_rank = routed_w2_scale
        routed_w2_scale_layer = routed_w2_scale_rank[csa_weight_layer]
        shared_w1_rank = shared_w1
        shared_w1_layer = shared_w1_rank[csa_weight_layer]
        shared_w1_scale_rank = shared_w1_scale
        shared_w1_scale_layer = shared_w1_scale_rank[csa_weight_layer]
        shared_w3_rank = shared_w3
        shared_w3_layer = shared_w3_rank[csa_weight_layer]
        shared_w3_scale_rank = shared_w3_scale
        shared_w3_scale_layer = shared_w3_scale_rank[csa_weight_layer]
        shared_w2_rank = shared_w2
        shared_w2_layer = shared_w2_rank[csa_weight_layer]
        shared_w2_scale_rank = shared_w2_scale
        shared_w2_scale_layer = shared_w2_scale_rank[csa_weight_layer]
        csa_x_hc_rank = pl.reshape(
            x_ping_rank, [CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D]
        )
        csa_x_attn_rank = pl.reshape(
            csa_attention_workspace_rank,
            [CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D],
        )
        decode_layer_csa_inline(
            csa_x_hc_rank,
            hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
            attn_norm_w_layer, wq_a_layer, wq_b_layer, wq_b_scale_layer,
            wkv_layer, gamma_cq_layer, gamma_ckv_layer,
            csa_rope_cos_rank, csa_rope_sin_rank,
            csa_main_wkv_layer, csa_main_wgate_layer, csa_main_ape_layer,
            csa_main_norm_w_layer, csa_main_state_layer,
            csa_main_state_page_ids_rank, csa_main_state_valid_ranges_rank,
            csa_main_state_page_epochs_rank,
            csa_compressor_request_epochs_rank,
            csa_request_event_indices_rank, csa_event_query_ids_rank,
            csa_event_rope_cos_rank, csa_event_rope_sin_rank,
            csa_main_event_write_slots_rank, csa_position_ids_rank,
            csa_main_state_write_slots_rank, csa_main_cache_layer,
            csa_inner_wkv_layer, csa_inner_wgate_layer, csa_inner_ape_layer,
            csa_inner_norm_w_layer, csa_inner_hadamard_layer,
            csa_inner_state_layer, csa_inner_state_page_ids_rank,
            csa_inner_state_valid_ranges_rank,
            csa_inner_state_page_epochs_rank,
            csa_inner_event_write_slots_rank,
            csa_inner_state_write_slots_rank, csa_idx_cache_layer,
            csa_idx_scale_layer, csa_idx_wq_b_layer,
            csa_idx_wq_b_scale_layer, csa_idx_weights_proj_layer,
            csa_idx_hadamard_layer, csa_idx_cos_il_rank,
            csa_idx_sin_signed_rank, csa_query_request_ids_rank,
            csa_idx_pages_rank, csa_idx_page_offsets_rank,
            csa_idx_windows_rank, csa_request_epochs_rank,
            csa_leaf_descriptors_rank, csa_pair_descriptors_rank,
            csa_singleton_descriptors_rank, csa_upper_descriptors_rank,
            csa_root_descriptors_rank, csa_pages_rank,
            csa_page_offsets_rank, csa_windows_rank, csa_raw_layer,
            csa_raw_write_slots_rank, csa_swa_sources_rank,
            attn_sink_layer, wo_a_layer, wo_b_layer, wo_b_scale_layer,
            csa_x_attn_rank,
            hc_ffn_fn_layer, hc_ffn_scale_layer, hc_ffn_base_layer,
            norm_w_layer, gate_w_layer, gate_bias_layer, tid2eid_layer,
            input_ids_rank,
            routed_w1_layer, routed_w1_scale_layer,
            routed_w3_layer, routed_w3_scale_layer,
            routed_w2_layer, routed_w2_scale_layer,
            shared_w1_layer, shared_w1_scale_layer,
            shared_w3_layer, shared_w3_scale_layer,
            shared_w2_layer, shared_w2_scale_layer,
            num_tokens_per_owner, x_pong_rank, recv_meta, recv_x, recv_aux,
            recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
            csa_model_layer, rank, csa_moe_epoch,
        )

        # The 21st CSA is model layer 42 and is the terminal main layer.  Keep
        # the HCA body under a positive condition: the PyPTO host generator
        # cannot lower a loop-local ``continue`` into valid orchestration.
        if csa_ordinal < HCA_FULL_LAYERS:
            # HCA consumes the CSA ping-pong output and restores x_ping for the
            # next pair. Persistent storage is selected from packed rank-local
            # pools, while every descriptor remains local to one layer.
            raw_layer = pl.slice(
                raw_kv_pool,
                [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
                [hca_model_layer * raw_blocks_per_layer, 0, 0, 0],
            )
            hca_state_layer = pl.slice(
                hca_compress_state,
                [
                    hca_state_blocks_per_layer,
                    HCA_STATE_BLOCK_SIZE,
                    HCA_COMPRESS_STATE_DIM,
                ],
                [csa_ordinal * hca_state_blocks_per_layer, 0, 0],
            )
            hca_cmp_kv_layer = pl.slice(
                hca_cmp_kv,
                [hca_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
                [csa_ordinal * hca_cmp_blocks_per_layer, 0, 0, 0],
            )
            hc_attn_fn_rank = hc_attn_fn
            hc_attn_fn_layer = hc_attn_fn_rank[hca_weight_layer]
            hc_attn_scale_rank = hc_attn_scale
            hc_attn_scale_layer = hc_attn_scale_rank[hca_weight_layer]
            hc_attn_base_rank = hc_attn_base
            hc_attn_base_layer = hc_attn_base_rank[hca_weight_layer]
            attn_norm_w_rank = attn_norm_w
            attn_norm_w_layer = attn_norm_w_rank[hca_weight_layer]
            wq_a_rank = wq_a
            wq_a_layer = wq_a_rank[hca_weight_layer]
            wq_b_rank = wq_b
            wq_b_layer = wq_b_rank[hca_weight_layer]
            wq_b_scale_rank = wq_b_scale
            wq_b_scale_layer = wq_b_scale_rank[hca_weight_layer]
            wkv_rank = wkv
            wkv_layer = wkv_rank[hca_weight_layer]
            gamma_cq_rank = gamma_cq
            gamma_cq_layer = gamma_cq_rank[hca_weight_layer]
            gamma_ckv_rank = gamma_ckv
            gamma_ckv_layer = gamma_ckv_rank[hca_weight_layer]
            hca_query_rope_cos_rank = hca_query_rope_cos
            hca_query_rope_cos_layer = hca_query_rope_cos_rank[csa_ordinal]
            hca_query_rope_sin_rank = hca_query_rope_sin
            hca_query_rope_sin_layer = hca_query_rope_sin_rank[csa_ordinal]
            hca_cmp_wkv_rank = hca_cmp_wkv
            hca_cmp_wkv_layer = hca_cmp_wkv_rank[csa_ordinal]
            hca_cmp_wgate_rank = hca_cmp_wgate
            hca_cmp_wgate_layer = hca_cmp_wgate_rank[csa_ordinal]
            hca_cmp_ape_rank = hca_cmp_ape
            hca_cmp_ape_layer = hca_cmp_ape_rank[csa_ordinal]
            hca_cmp_norm_w_rank = hca_cmp_norm_w
            hca_cmp_norm_w_layer = hca_cmp_norm_w_rank[csa_ordinal]
            hca_request_event_indices_rank = hca_request_event_indices
            hca_request_event_indices_layer = hca_request_event_indices_rank[csa_ordinal]
            hca_event_rope_cos_rank = hca_event_rope_cos
            hca_event_rope_cos_layer = hca_event_rope_cos_rank[csa_ordinal]
            hca_event_rope_sin_rank = hca_event_rope_sin
            hca_event_rope_sin_layer = hca_event_rope_sin_rank[csa_ordinal]
            hca_state_page_ids_rank = hca_state_page_ids
            hca_state_page_ids_layer = hca_state_page_ids_rank[csa_ordinal]
            hca_state_valid_ranges_rank = hca_state_valid_ranges
            hca_state_valid_ranges_layer = hca_state_valid_ranges_rank[csa_ordinal]
            hca_state_page_epochs_rank = hca_state_page_epochs
            hca_state_page_epochs_layer = hca_state_page_epochs_rank[csa_ordinal]
            hca_request_epochs_rank = hca_request_epochs
            hca_request_epochs_layer = hca_request_epochs_rank[csa_ordinal]
            hca_state_write_slots_rank = hca_state_write_slots
            hca_state_write_slots_layer = hca_state_write_slots_rank[csa_ordinal]
            hca_swa_write_slots_rank = hca_swa_write_slots
            hca_swa_write_slots_layer = hca_swa_write_slots_rank[csa_ordinal]
            hca_swa_sources_rank = hca_swa_sources
            hca_swa_sources_layer = hca_swa_sources_rank[csa_ordinal]
            hca_cmp_slot_mapping_rank = hca_cmp_slot_mapping
            hca_cmp_slot_mapping_layer = hca_cmp_slot_mapping_rank[csa_ordinal]
            hca_position_ids_rank = hca_position_ids
            hca_position_ids_layer = hca_position_ids_rank[csa_ordinal]
            hca_query_request_ids_rank = hca_query_request_ids
            hca_query_request_ids_layer = hca_query_request_ids_rank[csa_ordinal]
            hca_pages_rank = hca_pages
            hca_pages_layer = hca_pages_rank[csa_ordinal]
            hca_page_offsets_rank = hca_page_offsets
            hca_page_offsets_layer = hca_page_offsets_rank[csa_ordinal]
            hca_windows_rank = hca_windows
            hca_windows_layer = hca_windows_rank[csa_ordinal]
            hca_query_work_offsets_rank = hca_query_work_offsets
            hca_query_work_offsets_layer = hca_query_work_offsets_rank[csa_ordinal]
            hca_work_query_ids_rank = hca_work_query_ids
            hca_work_query_ids_layer = hca_work_query_ids_rank[csa_ordinal]
            hca_work_row_begin_rank = hca_work_row_begin
            hca_work_row_begin_layer = hca_work_row_begin_rank[csa_ordinal]
            hca_work_valid_rows_rank = hca_work_valid_rows
            hca_work_valid_rows_layer = hca_work_valid_rows_rank[csa_ordinal]
            attn_sink_rank = attn_sink
            attn_sink_layer = attn_sink_rank[hca_weight_layer]
            wo_a_rank = wo_a
            wo_a_layer = wo_a_rank[hca_weight_layer]
            wo_b_rank = wo_b
            wo_b_layer = wo_b_rank[hca_weight_layer]
            wo_b_scale_rank = wo_b_scale
            wo_b_scale_layer = wo_b_scale_rank[hca_weight_layer]
            hc_ffn_fn_rank = hc_ffn_fn
            hc_ffn_fn_layer = hc_ffn_fn_rank[hca_weight_layer]
            hc_ffn_scale_rank = hc_ffn_scale
            hc_ffn_scale_layer = hc_ffn_scale_rank[hca_weight_layer]
            hc_ffn_base_rank = hc_ffn_base
            hc_ffn_base_layer = hc_ffn_base_rank[hca_weight_layer]
            norm_w_rank = norm_w
            norm_w_layer = norm_w_rank[hca_weight_layer]
            gate_w_rank = gate_w
            gate_w_layer = gate_w_rank[hca_weight_layer]
            gate_bias_rank = gate_bias
            gate_bias_layer = gate_bias_rank[hca_weight_layer]
            tid2eid_rank = tid2eid
            tid2eid_layer = tid2eid_rank[hca_weight_layer]
            routed_w1_rank = routed_w1
            routed_w1_layer = routed_w1_rank[hca_weight_layer]
            routed_w1_scale_rank = routed_w1_scale
            routed_w1_scale_layer = routed_w1_scale_rank[hca_weight_layer]
            routed_w3_rank = routed_w3
            routed_w3_layer = routed_w3_rank[hca_weight_layer]
            routed_w3_scale_rank = routed_w3_scale
            routed_w3_scale_layer = routed_w3_scale_rank[hca_weight_layer]
            routed_w2_rank = routed_w2
            routed_w2_layer = routed_w2_rank[hca_weight_layer]
            routed_w2_scale_rank = routed_w2_scale
            routed_w2_scale_layer = routed_w2_scale_rank[hca_weight_layer]
            shared_w1_rank = shared_w1
            shared_w1_layer = shared_w1_rank[hca_weight_layer]
            shared_w1_scale_rank = shared_w1_scale
            shared_w1_scale_layer = shared_w1_scale_rank[hca_weight_layer]
            shared_w3_rank = shared_w3
            shared_w3_layer = shared_w3_rank[hca_weight_layer]
            shared_w3_scale_rank = shared_w3_scale
            shared_w3_scale_layer = shared_w3_scale_rank[hca_weight_layer]
            shared_w2_rank = shared_w2
            shared_w2_layer = shared_w2_rank[hca_weight_layer]
            shared_w2_scale_rank = shared_w2_scale
            shared_w2_scale_layer = shared_w2_scale_rank[hca_weight_layer]
            decode_layer_hca_inline(
                x_pong_rank,
                hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
                attn_norm_w_layer, wq_a_layer, wq_b_layer, wq_b_scale_layer,
                wkv_layer, gamma_cq_layer, gamma_ckv_layer,
                hca_query_rope_cos_layer, hca_query_rope_sin_layer,
                hca_cmp_wkv_layer, hca_cmp_wgate_layer, hca_cmp_ape_layer,
                hca_cmp_norm_w_layer, hca_request_event_indices_layer,
                hca_event_rope_cos_layer, hca_event_rope_sin_layer,
                hca_state_layer,
                hca_state_page_ids_layer, hca_state_valid_ranges_layer,
                hca_state_page_epochs_layer, hca_request_epochs_layer,
                hca_state_write_slots_layer,
                raw_layer,
                hca_swa_write_slots_layer, hca_swa_sources_layer,
                hca_cmp_kv_layer,
                hca_cmp_slot_mapping_layer, hca_position_ids_layer,
                hca_query_request_ids_layer, hca_pages_layer,
                hca_page_offsets_layer, hca_windows_layer,
                hca_query_work_offsets_layer, hca_work_query_ids_layer,
                hca_work_row_begin_layer, hca_work_valid_rows_layer,
                attn_sink_layer, wo_a_layer, wo_b_layer, wo_b_scale_layer,
                hc_ffn_fn_layer, hc_ffn_scale_layer, hc_ffn_base_layer,
                norm_w_layer, gate_w_layer, gate_bias_layer, tid2eid_layer,
                input_ids_rank,
                routed_w1_layer, routed_w1_scale_layer,
                routed_w3_layer, routed_w3_scale_layer,
                routed_w2_layer, routed_w2_scale_layer,
                shared_w1_layer, shared_w1_scale_layer,
                shared_w3_layer, shared_w3_scale_layer,
                shared_w2_layer, shared_w2_scale_layer,
                num_tokens_per_owner, hca_attention_workspace_rank, x_ping_rank,
                recv_meta, recv_x, recv_aux,
                recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
                hca_model_layer, rank, hca_moe_epoch,
            )

    # Layer 42 wrote ``x_pong`` at epoch 43.  Clear the single shared MoE
    # signal domain only after that completion anchor, then run the independent
    # terminal LM-head collective with its own epoch-one signal buffers.
    decode_fwd_clear_shared_moe_signals_inline(
        x_pong_rank, arrived, data_arrived, combine_arrived
    )
    decode_fwd_terminal_head_active_inline(
        x_pong_rank,
        active_row_mask_rank,
        hc_head_fn_rank,
        hc_head_scale_rank,
        hc_head_base_rank,
        final_norm_w_rank,
        lm_head_weight_rank,
        hidden_out_rank,
        logits_rank,
        sampled_ids_rank,
        lm_head_hidden_window,
        lm_head_hidden_done,
        lm_head_logits_window,
        lm_head_logits_done,
        rank,
    )
    mark_full43_program_completion_inline(
        sampled_ids_rank, full43_completion_rank
    )
    return full43_completion

@pl.jit.host
def l3_decode_fwd_full43(
    x_ping: pl.InOut[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    x_pong: pl.InOut[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    # Main-model immutable tensors use a caller-owned weight bank. Production
    # binds 43 entries; the bounded G5 runtime witness may bind one immutable
    # entry and reuse it without changing the 43-layer execution/cache graph.
    hc_attn_fn: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, H], pl.FP32],
    wo_a: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
    ],
    wo_b: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, D, O_GROUPS * O_LORA], pl.INT8
    ],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_DYN, N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_DYN, D], pl.FP32],
    # Persistent cache/state uses one packed block axis per rank.
    raw_kv_pool: pl.InOut[pl.Tensor[
        [N_RANKS, FWD_PACKED_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM],
        pl.BF16,
    ]],
    swa_rope_cos: pl.Tensor[[N_RANKS, TWO_SWA_LAYERS, T, ROPE_HEAD_DIM], pl.BF16],
    swa_rope_sin: pl.Tensor[[N_RANKS, TWO_SWA_LAYERS, T, ROPE_HEAD_DIM], pl.BF16],
    swa_raw_write_slots: pl.Tensor[[N_RANKS, TWO_SWA_LAYERS, T], pl.INT64],
    swa_sources: pl.Tensor[[N_RANKS, TWO_SWA_LAYERS, T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[N_RANKS, TWO_SWA_LAYERS, T], pl.INT32],
    # HCA resources are rank -> HCA ordinal stacked and never share a pool
    # with another HCA layer.
    hca_query_rope_cos: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, T, ROPE_HEAD_DIM], pl.BF16
    ],
    hca_query_rope_sin: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, T, ROPE_HEAD_DIM], pl.BF16
    ],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HEAD_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HEAD_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32
    ],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HEAD_DIM], pl.BF16],
    hca_request_event_indices: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HCA_B_DYN], pl.INT32],
    hca_event_rope_cos: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    hca_event_rope_sin: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    hca_compress_state: pl.InOut[pl.Tensor[
        [
            N_RANKS,
            FWD_PACKED_HCA_STATE_BLOCKS_DYN,
            HCA_STATE_BLOCK_SIZE,
            HCA_COMPRESS_STATE_DIM,
        ],
        pl.FP32,
    ]],
    hca_state_page_ids: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST],
        pl.INT32,
    ],
    hca_state_valid_ranges: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_B_DYN, 2], pl.INT32
    ],
    hca_state_page_epochs: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_B_DYN, HCA_STATE_PAGES_PER_REQUEST],
        pl.INT32,
    ],
    hca_request_epochs: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HCA_B_DYN], pl.INT32],
    hca_state_write_slots: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, T], pl.INT64],
    hca_swa_write_slots: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, T], pl.INT64],
    hca_swa_sources: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, T, WIN], pl.INT32],
    hca_cmp_kv: pl.InOut[pl.Tensor[
        [
            N_RANKS,
            FWD_PACKED_HCA_CMP_BLOCKS_DYN,
            BLOCK_SIZE,
            1,
            HEAD_DIM,
        ],
        pl.BF16,
    ]],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, T], pl.INT64],
    hca_position_ids: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, T], pl.INT32],
    hca_query_request_ids: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, T], pl.INT32],
    hca_pages: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_REQUEST_OFFSETS_DYN], pl.INT32
    ],
    hca_windows: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HCA_B_DYN, 3], pl.INT32],
    hca_query_work_offsets: pl.Tensor[
        [N_RANKS, HCA_FULL_LAYERS, HCA_QUERY_OFFSETS_DYN], pl.INT32
    ],
    hca_work_query_ids: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[N_RANKS, HCA_FULL_LAYERS, HCA_WORK_DYN], pl.INT32],
    # CSA descriptors retain their leaf-facing [layer, shard, ...] layout.
    csa_rope_cos: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16
    ],
    csa_rope_sin: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.BF16
    ],
    csa_main_wkv: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_main_wgate: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_main_ape: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, 4, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_main_norm_w: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, HEAD_DIM], pl.BF16],
    csa_main_state: pl.InOut[pl.Tensor[
        [
            N_RANKS,
            FWD_PACKED_CSA_MAIN_STATE_BLOCKS_DYN,
            CSA_STATE_BLOCK_SIZE,
            CSA_MAIN_STATE_DIM,
        ],
        pl.FP32,
    ]],
    csa_main_state_page_ids: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_main_state_valid_ranges: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B, 2], pl.INT32
    ],
    csa_main_state_page_epochs: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_compressor_request_epochs: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B], pl.INT32
    ],
    csa_request_event_indices: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B], pl.INT32
    ],
    csa_event_query_ids: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP], pl.INT32
    ],
    csa_event_rope_cos: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    csa_event_rope_sin: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    csa_main_event_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP], pl.INT64
    ],
    csa_position_ids: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT32
    ],
    csa_main_state_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT64
    ],
    csa_main_cache: pl.InOut[pl.Tensor[
        [
            N_RANKS,
            FWD_PACKED_CSA_MAIN_BLOCKS_DYN,
            BLOCK_SIZE,
            1,
            HEAD_DIM,
        ],
        pl.BF16,
    ]],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, 4, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_FULL_LAYERS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_hadamard: pl.Tensor[
        [N_RANKS, CSA_FULL_LAYERS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16
    ],
    csa_inner_state: pl.InOut[pl.Tensor[
        [
            N_RANKS,
            FWD_PACKED_CSA_INNER_STATE_BLOCKS_DYN,
            CSA_INNER_STATE_BLOCK_SIZE,
            CSA_INNER_STATE_DIM,
        ],
        pl.FP32,
    ]],
    csa_inner_state_page_ids: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_INNER_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_inner_state_valid_ranges: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_B, 2], pl.INT32
    ],
    csa_inner_state_page_epochs: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_CHUNK_B,
            CSA_INNER_STATE_PAGES_PER_REQUEST,
        ],
        pl.INT32,
    ],
    csa_inner_event_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_EVENT_CAP], pl.INT64
    ],
    csa_inner_state_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT64
    ],
    csa_idx_cache: pl.InOut[pl.Tensor[
        [N_RANKS, FWD_PACKED_CSA_IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8
    ]],
    csa_idx_scale: pl.InOut[pl.Tensor[
        [N_RANKS, FWD_PACKED_CSA_IDX_ROWS_DYN, 1], pl.FP32
    ]],
    csa_idx_wq_b: pl.Tensor[
        [N_RANKS, CSA_FULL_LAYERS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [N_RANKS, CSA_FULL_LAYERS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32
    ],
    csa_idx_weights_proj: pl.Tensor[
        [N_RANKS, CSA_FULL_LAYERS, D, IDX_N_HEADS], pl.BF16
    ],
    csa_idx_hadamard: pl.Tensor[
        [N_RANKS, CSA_FULL_LAYERS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16
    ],
    csa_idx_cos_il: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32
    ],
    csa_idx_sin_signed: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, ROPE_HEAD_DIM], pl.FP32
    ],
    csa_query_request_ids: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT32
    ],
    csa_idx_pages: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_PAGE_DYN, 2], pl.INT32
    ],
    csa_idx_page_offsets: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_OFFSET_DYN], pl.INT32
    ],
    csa_idx_windows: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_DYN, 3], pl.INT32
    ],
    csa_request_epochs: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_DYN], pl.INT32
    ],
    csa_leaf_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    csa_pair_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    csa_singleton_descriptors: pl.Tensor[
        [
            CSA_FULL_LAYERS,
            CSA_LAYER_SHARDS,
            CSA_SINGLETON_DYN,
            PHASE_D_SINGLETON_FIELDS,
        ],
        pl.INT32,
    ],
    csa_upper_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    csa_root_descriptors: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, PHASE_D_ROOT_FIELDS], pl.INT32
    ],
    csa_pages: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_PAGE_DYN, 2], pl.INT32
    ],
    csa_page_offsets: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_OFFSET_DYN], pl.INT32
    ],
    csa_windows: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_REQUEST_DYN, 3], pl.INT32
    ],
    csa_raw_write_slots: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T], pl.INT64
    ],
    csa_swa_sources: pl.Tensor[
        [CSA_FULL_LAYERS, CSA_LAYER_SHARDS, CSA_CHUNK_T, WIN], pl.INT32
    ],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    active_row_mask: pl.Tensor[[N_RANKS, T], pl.INT32],
    embed_weight: pl.Tensor[[N_RANKS, EMBED_VOCAB_DYN, D], pl.BF16],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[
        pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]
    ],
    full43_completion: pl.InOut[pl.Tensor[[N_RANKS, 1], pl.INT32]],
    # Public binding follows the same tensors-first TaskArgs contract as the
    # rank child so runtime submission never adds a tensor after a scalar.
    weight_bank_size: pl.Scalar[pl.INT32],
):
    # One forward owns one communication domain.  The rank-local children
    # reuse these windows for epochs 1..43 and clear them exactly once after
    # the final layer, matching the baseline decode-forward topology.
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
    lm_head_hidden_buf = pld.alloc_window_buffer(
        [GROUP_LOGIT_ROWS, D], dtype=pl.BF16
    )
    lm_head_hidden_done_buf = pld.alloc_window_buffer(
        [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
    )
    lm_head_logits_buf = pld.alloc_window_buffer(
        [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32
    )
    lm_head_logits_done_buf = pld.alloc_window_buffer(
        [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
    )

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
        lm_head_hidden_window: pld.DistributedTensor[
            [GROUP_LOGIT_ROWS, D], pl.BF16
        ] = pld.window(
            lm_head_hidden_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16
        )
        lm_head_hidden_done: pld.DistributedTensor[
            [LM_HEAD_TP_SIZE, 1], pl.INT32
        ] = pld.window(
            lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        lm_head_logits_window: pld.DistributedTensor[
            [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32
        ] = pld.window(
            lm_head_logits_buf,
            [MAX_LOGIT_ROWS, LM_HEAD_VOCAB],
            dtype=pl.FP32,
        )
        lm_head_logits_done: pld.DistributedTensor[
            [LM_HEAD_TP_SIZE, 1], pl.INT32
        ] = pld.window(
            lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        l3_decode_fwd_full43_rank(
        x_ping=x_ping[rank],
        x_pong=x_pong[rank],
        hc_attn_fn=hc_attn_fn[rank],
        hc_attn_scale=hc_attn_scale[rank],
        hc_attn_base=hc_attn_base[rank],
        attn_norm_w=attn_norm_w[rank],
        wq_a=wq_a[rank],
        wq_b=wq_b[rank],
        wq_b_scale=wq_b_scale[rank],
        wkv=wkv[rank],
        gamma_cq=gamma_cq[rank],
        gamma_ckv=gamma_ckv[rank],
        attn_sink=attn_sink[rank],
        wo_a=wo_a[rank],
        wo_b=wo_b[rank],
        wo_b_scale=wo_b_scale[rank],
        hc_ffn_fn=hc_ffn_fn[rank],
        hc_ffn_scale=hc_ffn_scale[rank],
        hc_ffn_base=hc_ffn_base[rank],
        norm_w=norm_w[rank],
        gate_w=gate_w[rank],
        gate_bias=gate_bias[rank],
        tid2eid=tid2eid[rank],
        routed_w1=routed_w1[rank],
        routed_w1_scale=routed_w1_scale[rank],
        routed_w3=routed_w3[rank],
        routed_w3_scale=routed_w3_scale[rank],
        routed_w2=routed_w2[rank],
        routed_w2_scale=routed_w2_scale[rank],
        shared_w1=shared_w1[rank],
        shared_w1_scale=shared_w1_scale[rank],
        shared_w3=shared_w3[rank],
        shared_w3_scale=shared_w3_scale[rank],
        shared_w2=shared_w2[rank],
        shared_w2_scale=shared_w2_scale[rank],
        weight_bank_size=weight_bank_size,
        raw_kv_pool=raw_kv_pool[rank],
        swa_rope_cos=swa_rope_cos[rank],
        swa_rope_sin=swa_rope_sin[rank],
        swa_raw_write_slots=swa_raw_write_slots[rank],
        swa_sources=swa_sources[rank],
        swa_lens=swa_lens[rank],
        hca_query_rope_cos=hca_query_rope_cos[rank],
        hca_query_rope_sin=hca_query_rope_sin[rank],
        hca_cmp_wkv=hca_cmp_wkv[rank],
        hca_cmp_wgate=hca_cmp_wgate[rank],
        hca_cmp_ape=hca_cmp_ape[rank],
        hca_cmp_norm_w=hca_cmp_norm_w[rank],
        hca_request_event_indices=hca_request_event_indices[rank],
        hca_event_rope_cos=hca_event_rope_cos[rank],
        hca_event_rope_sin=hca_event_rope_sin[rank],
        hca_compress_state=hca_compress_state[rank],
        hca_state_page_ids=hca_state_page_ids[rank],
        hca_state_valid_ranges=hca_state_valid_ranges[rank],
        hca_state_page_epochs=hca_state_page_epochs[rank],
        hca_request_epochs=hca_request_epochs[rank],
        hca_state_write_slots=hca_state_write_slots[rank],
        hca_swa_write_slots=hca_swa_write_slots[rank],
        hca_swa_sources=hca_swa_sources[rank],
        hca_cmp_kv=hca_cmp_kv[rank],
        hca_cmp_slot_mapping=hca_cmp_slot_mapping[rank],
        hca_position_ids=hca_position_ids[rank],
        hca_query_request_ids=hca_query_request_ids[rank],
        hca_pages=hca_pages[rank],
        hca_page_offsets=hca_page_offsets[rank],
        hca_windows=hca_windows[rank],
        hca_query_work_offsets=hca_query_work_offsets[rank],
        hca_work_query_ids=hca_work_query_ids[rank],
        hca_work_row_begin=hca_work_row_begin[rank],
        hca_work_valid_rows=hca_work_valid_rows[rank],
        csa_rope_cos=csa_rope_cos,
        csa_rope_sin=csa_rope_sin,
        csa_main_wkv=csa_main_wkv[rank],
        csa_main_wgate=csa_main_wgate[rank],
        csa_main_ape=csa_main_ape[rank],
        csa_main_norm_w=csa_main_norm_w[rank],
        csa_main_state=csa_main_state[rank],
        csa_main_state_page_ids=csa_main_state_page_ids,
        csa_main_state_valid_ranges=csa_main_state_valid_ranges,
        csa_main_state_page_epochs=csa_main_state_page_epochs,
        csa_compressor_request_epochs=csa_compressor_request_epochs,
        csa_request_event_indices=csa_request_event_indices,
        csa_event_query_ids=csa_event_query_ids,
        csa_event_rope_cos=csa_event_rope_cos,
        csa_event_rope_sin=csa_event_rope_sin,
        csa_main_event_write_slots=csa_main_event_write_slots,
        csa_position_ids=csa_position_ids,
        csa_main_state_write_slots=csa_main_state_write_slots,
        csa_main_cache=csa_main_cache[rank],
        csa_inner_wkv=csa_inner_wkv[rank],
        csa_inner_wgate=csa_inner_wgate[rank],
        csa_inner_ape=csa_inner_ape[rank],
        csa_inner_norm_w=csa_inner_norm_w[rank],
        csa_inner_hadamard=csa_inner_hadamard[rank],
        csa_inner_state=csa_inner_state[rank],
        csa_inner_state_page_ids=csa_inner_state_page_ids,
        csa_inner_state_valid_ranges=csa_inner_state_valid_ranges,
        csa_inner_state_page_epochs=csa_inner_state_page_epochs,
        csa_inner_event_write_slots=csa_inner_event_write_slots,
        csa_inner_state_write_slots=csa_inner_state_write_slots,
        csa_idx_cache=csa_idx_cache[rank],
        csa_idx_scale=csa_idx_scale[rank],
        csa_idx_wq_b=csa_idx_wq_b[rank],
        csa_idx_wq_b_scale=csa_idx_wq_b_scale[rank],
        csa_idx_weights_proj=csa_idx_weights_proj[rank],
        csa_idx_hadamard=csa_idx_hadamard[rank],
        csa_idx_cos_il=csa_idx_cos_il,
        csa_idx_sin_signed=csa_idx_sin_signed,
        csa_query_request_ids=csa_query_request_ids,
        csa_idx_pages=csa_idx_pages,
        csa_idx_page_offsets=csa_idx_page_offsets,
        csa_idx_windows=csa_idx_windows,
        csa_request_epochs=csa_request_epochs,
        csa_leaf_descriptors=csa_leaf_descriptors,
        csa_pair_descriptors=csa_pair_descriptors,
        csa_singleton_descriptors=csa_singleton_descriptors,
        csa_upper_descriptors=csa_upper_descriptors,
        csa_root_descriptors=csa_root_descriptors,
        csa_pages=csa_pages,
        csa_page_offsets=csa_page_offsets,
        csa_windows=csa_windows,
        csa_raw_write_slots=csa_raw_write_slots,
        csa_swa_sources=csa_swa_sources,
        input_ids=input_ids[rank],
        active_row_mask=active_row_mask[rank],
        embed_weight=embed_weight[rank],
        num_tokens_per_owner=num_tokens_per_owner,
        hc_head_fn=hc_head_fn[rank],
        hc_head_scale=hc_head_scale[rank],
        hc_head_base=hc_head_base[rank],
        final_norm_w=final_norm_w[rank],
        lm_head_weight=lm_head_weight[rank],
        hidden_out=hidden_out[rank],
        logits=logits[rank],
        sampled_ids=sampled_ids[rank],
        full43_completion=full43_completion[rank],
        recv_meta=recv_meta,
        recv_x=recv_x,
        recv_aux=recv_aux,
        recv_route=recv_route,
        arrived=arrived,
        data_arrived=data_arrived,
        routed_y_buf=routed_y_buf,
        combine_arrived=combine_arrived,
        lm_head_hidden_window=lm_head_hidden_window,
        lm_head_hidden_done=lm_head_hidden_done,
        lm_head_logits_window=lm_head_logits_window,
        lm_head_logits_done=lm_head_logits_done,
        my_rank=rank,
        device=rank,
    )
    return full43_completion



decode_fwd_full43 = l3_decode_fwd_full43


FULL43_PACKED_POOL_LAYER_COUNTS = {
    "raw_kv_pool": MAIN_LAYER_COUNT,
    "hca_compress_state": HCA_FULL_LAYERS,
    "hca_cmp_kv": HCA_FULL_LAYERS,
    "csa_main_state": CSA_FULL_LAYERS,
    "csa_main_cache": CSA_FULL_LAYERS,
    "csa_inner_state": CSA_FULL_LAYERS,
    "csa_idx_cache": CSA_FULL_LAYERS,
    "csa_idx_scale": CSA_FULL_LAYERS,
}

FULL43_PACKED_POOL_BLOCK_SIZES = {
    "raw_kv_pool": BLOCK_SIZE,
    "hca_compress_state": HCA_STATE_BLOCK_SIZE,
    "hca_cmp_kv": BLOCK_SIZE,
    "csa_main_state": CSA_STATE_BLOCK_SIZE,
    "csa_main_cache": BLOCK_SIZE,
    "csa_inner_state": CSA_INNER_STATE_BLOCK_SIZE,
    "csa_idx_cache": 1,
    "csa_idx_scale": 1,
}


def build_full43_packed_pool_layout(per_layer_extents):
    """Return deterministic, non-overlapping layer slices for every pool."""
    unknown = set(per_layer_extents) - set(FULL43_PACKED_POOL_LAYER_COUNTS)
    missing = set(FULL43_PACKED_POOL_LAYER_COUNTS) - set(per_layer_extents)
    if unknown or missing:
        raise ValueError(
            f"packed pool extent keys mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    layout = {}
    for name, layer_count in FULL43_PACKED_POOL_LAYER_COUNTS.items():
        extent = int(per_layer_extents[name])
        if extent <= 0:
            raise ValueError(f"{name} per-layer extent must be positive, got {extent}")
        slices = tuple(
            (ordinal * extent, (ordinal + 1) * extent)
            for ordinal in range(layer_count)
        )
        layout[name] = {
            "layer_count": layer_count,
            "per_layer_extent": extent,
            "total_extent": layer_count * extent,
            "block_size": FULL43_PACKED_POOL_BLOCK_SIZES[name],
            "slices": slices,
        }
    return layout


def validate_full43_packed_pool_contract(
    total_extents,
    per_layer_extents,
    *,
    tensor_shapes=None,
):
    """Fail closed on packed extent, rank, stride, and dimensionality errors."""
    layout = build_full43_packed_pool_layout(per_layer_extents)
    if set(total_extents) != set(layout):
        raise ValueError("packed total extents must name every persistent pool exactly")
    for name, entry in layout.items():
        total = int(total_extents[name])
        layer_count = entry["layer_count"]
        if total % layer_count:
            raise ValueError(
                f"{name} total extent {total} is not divisible by {layer_count}"
            )
        if total != entry["total_extent"]:
            raise ValueError(
                f"{name} packed extent {total} does not match configured "
                f"{entry['total_extent']}"
            )
        previous_end = 0
        for begin, end in entry["slices"]:
            if begin != previous_end or begin >= end or end > total:
                raise ValueError(f"{name} layer slices overlap or leave a gap")
            previous_end = end

    if tensor_shapes is not None:
        for name, shape in tensor_shapes.items():
            dims = tuple(int(dim) for dim in shape)
            if len(dims) > 5:
                raise ValueError(f"{name} has {len(dims)} dimensions; maximum is 5")
            if name not in layout:
                continue
            if len(dims) < 2 or dims[0] != N_RANKS:
                raise ValueError(f"{name} must have a leading rank axis")
            if dims[1] != layout[name]["total_extent"]:
                raise ValueError(
                    f"{name} packed axis {dims[1]} does not match "
                    f"{layout[name]['total_extent']}"
                )
            if name not in {"csa_idx_cache", "csa_idx_scale"}:
                block_axis = dims[2]
                if block_axis != layout[name]["block_size"]:
                    raise ValueError(
                        f"{name} block size {block_axis} does not match "
                        f"{layout[name]['block_size']}"
                    )
    return layout


def validate_full43_local_slots(name, slots, per_layer_capacity, active_mask=None):
    """Validate that metadata addresses one layer-local pool and inactive rows."""
    import torch

    values = torch.as_tensor(slots)
    if bool((values < -1).any().item()):
        raise ValueError(f"{name} contains a slot below the -1 invalid sentinel")
    if bool((values >= int(per_layer_capacity)).any().item()):
        raise ValueError(
            f"{name} contains a non-local slot for capacity {per_layer_capacity}"
        )
    if active_mask is not None:
        active = torch.as_tensor(active_mask, dtype=torch.bool)
        while active.ndim < values.ndim:
            active = active.unsqueeze(-1)
        if bool((values.masked_select(~active) != -1).any().item()):
            raise ValueError(f"{name} retains a write slot for an inactive row")
    return values


def full43_csa_model_layer(ordinal):
    """Map CSA ordinal 0..20 to model layers 2,4,...,42."""
    ordinal = int(ordinal)
    if ordinal < 0 or ordinal >= CSA_FULL_LAYERS:
        raise ValueError(f"CSA ordinal must be in [0, 20], got {ordinal}")
    return 2 * ordinal + 2


def full43_hca_model_layer(ordinal):
    """Map HCA ordinal 0..19 to model layers 3,5,...,41."""
    ordinal = int(ordinal)
    if ordinal < 0 or ordinal >= HCA_FULL_LAYERS:
        raise ValueError(f"HCA ordinal must be in [0, 19], got {ordinal}")
    return 2 * ordinal + 3


def build_full43_device_specs(
    *, weight_bank_size=MAIN_LAYER_COUNT, runtime_case=None
):
    """Build a shape-complete compile fixture for the production 43-layer ABI.

    The full model tensors are intentionally zero-initialized and this fixture
    is accepted only by the compile-only CLI path.  Numerical validation uses
    the real single-layer and mixed-prefix fixtures, which carry meaningful
    allocator descriptors without materializing 43 layers of host weights.
    """
    import torch
    from golden import ScalarSpec, TensorSpec

    if weight_bank_size < 1 or weight_bank_size > MAIN_LAYER_COUNT:
        raise ValueError(
            f"weight_bank_size must be in [1, {MAIN_LAYER_COUNT}], "
            f"got {weight_bank_size}"
        )

    from decode_fwd import (
        _MIXED_QUAD_COMMON_NAMES,
        _MIXED_QUAD_INOUT_NAMES,
        build_mixed_quad_device_specs,
    )

    if runtime_case not in {
        None,
        "full_active",
        "ragged",
        "packed_pool_sentinel",
        "long_context_tail",
    }:
        raise ValueError(f"unknown full43 runtime case: {runtime_case!r}")
    if runtime_case == "full_active":
        fixture_counts = T
    elif runtime_case == "ragged":
        count_pattern = (T, T - 1, T // 2 + 1, 1)
        fixture_counts = [count_pattern[i % len(count_pattern)] for i in range(N_RANKS)]
    elif runtime_case == "packed_pool_sentinel":
        fixture_counts = T
    elif runtime_case == "long_context_tail":
        # B=16, S=8 per rank.  The row count remains bounded while metadata
        # below carries inactive, short, 16K and 1M histories independently.
        fixture_counts = [T, T - 8, T // 2, 0]
        fixture_counts = [fixture_counts[i % 4] for i in range(N_RANKS)]
    else:
        fixture_counts = T

    # Reuse only the mixed fixture's concrete dynamic shapes.  The upstream
    # D-Spark target verifies all eight rows per request, so the full-forward
    # capacity fixture must retain the complete rank-local T-row MoE workload.
    source_specs = build_mixed_quad_device_specs(
        num_tokens_per_owner=fixture_counts,
        context_case=(
            "one_m_tail" if runtime_case == "long_context_tail" else "default"
        ),
    )
    source_by_name = {
        spec.name: spec for spec in source_specs if isinstance(spec, TensorSpec)
    }

    shared_csa_workspaces = set()
    rank_owned_csa_tensors = {
        "csa_main_wkv",
        "csa_main_wgate",
        "csa_main_ape",
        "csa_main_norm_w",
        "csa_main_state",
        "csa_main_cache",
        "csa_inner_wkv",
        "csa_inner_wgate",
        "csa_inner_ape",
        "csa_inner_norm_w",
        "csa_inner_hadamard",
        "csa_inner_state",
        "csa_idx_cache",
        "csa_idx_scale",
        "csa_idx_wq_b",
        "csa_idx_wq_b_scale",
        "csa_idx_weights_proj",
        "csa_idx_hadamard",
    }
    pure_outputs = {"hidden_out", "logits", "sampled_ids"}
    pure_outputs.add("full43_completion")
    inout_names = set(_MIXED_QUAD_INOUT_NAMES)

    terminal_specs = {
        "hc_head_fn": ([N_RANKS, HC_MULT, HC_DIM], torch.float32),
        "hc_head_scale": ([N_RANKS, 1], torch.float32),
        "hc_head_base": ([N_RANKS, HC_MULT], torch.float32),
        "final_norm_w": ([N_RANKS, D], torch.bfloat16),
        "lm_head_weight": ([N_RANKS, VOCAB_PER_TP, D], torch.bfloat16),
        "hidden_out": ([N_RANKS, T, D], torch.bfloat16),
        "logits": ([N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], torch.float32),
        "sampled_ids": (
            [N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD],
            torch.int32,
        ),
        "full43_completion": ([N_RANKS, 1], torch.int32),
    }

    def packed_pool_sentinel(name, shape, dtype):
        layer_count = FULL43_PACKED_POOL_LAYER_COUNTS[name]
        per_layer_extent = shape[1] // layer_count
        value = torch.empty(shape, dtype=dtype)
        for ordinal in range(layer_count):
            sentinel = ordinal + 1
            value[:, ordinal * per_layer_extent : (ordinal + 1) * per_layer_extent].fill_(
                sentinel
            )
        return value

    runtime_source_values = {}
    if runtime_case is not None:
        # Only integer control/descriptor tensors need the concrete Phase-E
        # fixture. Immutable math tensors deliberately remain zero for the
        # bounded runtime witness; G4 owns their nonzero numerical goldens.
        for name, spec in source_by_name.items():
            if name in _MIXED_QUAD_COMMON_NAMES:
                continue
            if spec.dtype not in {torch.int32, torch.int64}:
                continue
            if name == "sampled_ids":
                continue
            runtime_source_values[name] = spec.create_tensor()

        swa_slots = runtime_source_values["swa_raw_write_slots"]
        valid_delta = (
            swa_slots[:, 1] - swa_slots[:, 0]
        )[swa_slots[:, 0] >= 0]
        if valid_delta.numel() == 0:
            raise ValueError("runtime SWA fixture has no valid raw write slots")
        swa_rows = int(valid_delta[0].item())
        if not bool((valid_delta == swa_rows).all().item()):
            raise ValueError("runtime SWA layer offsets are not uniform")

        def rebase_nonnegative(value, offset):
            return torch.where(value >= 0, value - offset, value)

        runtime_source_values["swa_raw_write_slots"][:, 1] = rebase_nonnegative(
            runtime_source_values["swa_raw_write_slots"][:, 1], swa_rows
        )
        runtime_source_values["swa_sources"][:, 1] = rebase_nonnegative(
            runtime_source_values["swa_sources"][:, 1], swa_rows
        )
        csa_raw_base = 2 * swa_rows
        hca_raw_base = csa_raw_base + BLOCK_SIZE
        runtime_source_values["csa_raw_write_slots"] = rebase_nonnegative(
            runtime_source_values["csa_raw_write_slots"], csa_raw_base
        )
        runtime_source_values["csa_swa_sources"] = rebase_nonnegative(
            runtime_source_values["csa_swa_sources"], csa_raw_base
        )
        runtime_source_values["hca_swa_write_slots"] = rebase_nonnegative(
            runtime_source_values["hca_swa_write_slots"], hca_raw_base
        )
        runtime_source_values["hca_swa_sources"] = rebase_nonnegative(
            runtime_source_values["hca_swa_sources"], hca_raw_base
        )

    def expanded_shape(name, base_shape):
        if name in _MIXED_QUAD_COMMON_NAMES:
            return [N_RANKS, weight_bank_size, *base_shape[2:]]
        if name == "raw_kv_pool":
            return [
                N_RANKS,
                MAIN_LAYER_COUNT * KV_ORI_BLOCK_NUM,
                base_shape[2],
                base_shape[3],
                base_shape[4],
            ]
        if name == "hca_cmp_kv":
            return [
                N_RANKS,
                HCA_FULL_LAYERS * base_shape[1],
                base_shape[2],
                base_shape[3],
                base_shape[4],
            ]
        if name == "hca_compress_state":
            return [N_RANKS, HCA_FULL_LAYERS * base_shape[1], *base_shape[2:]]
        if name.startswith("hca_"):
            return [N_RANKS, HCA_FULL_LAYERS, *base_shape[1:]]
        if name.startswith("csa_") and name not in shared_csa_workspaces:
            if name in rank_owned_csa_tensors:
                if name == "csa_main_cache":
                    return [
                        N_RANKS,
                        CSA_FULL_LAYERS * base_shape[1],
                        base_shape[2],
                        base_shape[3],
                        base_shape[4],
                    ]
                if name in {"csa_main_state", "csa_inner_state"}:
                    return [
                        N_RANKS,
                        CSA_FULL_LAYERS * base_shape[1],
                        *base_shape[2:],
                    ]
                if name in {"csa_idx_cache", "csa_idx_scale"}:
                    return [
                        N_RANKS,
                        CSA_FULL_LAYERS * base_shape[1],
                        *base_shape[2:],
                    ]
                return [N_RANKS, CSA_FULL_LAYERS, *base_shape[1:]]
            if base_shape[0] != N_RANKS:
                raise ValueError(f"CSA descriptor is not rank-owned: {name}")
            return [
                CSA_FULL_LAYERS,
                N_RANKS * base_shape[1],
                *base_shape[2:],
            ]
        return list(base_shape)

    def expanded_runtime_value(name, value):
        if name == "hca_cmp_kv":
            return value.repeat(1, HCA_FULL_LAYERS, 1, 1, 1)
        if name == "hca_compress_state":
            return value.repeat(1, HCA_FULL_LAYERS, 1, 1)
        if name.startswith("hca_"):
            if value.shape[0] != N_RANKS:
                raise ValueError(f"HCA runtime value is not rank-owned: {name}")
            return value.unsqueeze(1).expand(
                N_RANKS, HCA_FULL_LAYERS, *value.shape[1:]
            ).contiguous()
        if name.startswith("csa_") and name not in shared_csa_workspaces:
            if name in rank_owned_csa_tensors:
                if name == "csa_main_cache":
                    return value.repeat(1, CSA_FULL_LAYERS, 1, 1, 1)
                if name in {"csa_main_state", "csa_inner_state"}:
                    return value.repeat(1, CSA_FULL_LAYERS, 1, 1)
                if name in {"csa_idx_cache", "csa_idx_scale"}:
                    return value.repeat(1, CSA_FULL_LAYERS, 1)
                return value.unsqueeze(1).expand(
                    N_RANKS, CSA_FULL_LAYERS, *value.shape[1:]
                ).contiguous()
            if value.shape[0] != N_RANKS:
                raise ValueError(f"CSA descriptor is not rank-owned: {name}")
            flattened = value.reshape(
                N_RANKS * value.shape[1], *value.shape[2:]
            )
            return flattened.unsqueeze(0).expand(
                CSA_FULL_LAYERS, *flattened.shape
            ).contiguous()
        return value

    parameter_names = list(
        inspect.signature(l3_decode_fwd_full43._func).parameters
    )
    specs = []
    for name in parameter_names:
        if name == "weight_bank_size":
            specs.append(
                ScalarSpec("weight_bank_size", torch.int32, weight_bank_size)
            )
            continue
        if name in terminal_specs:
            shape, dtype = terminal_specs[name]
        else:
            try:
                source = source_by_name[name]
            except KeyError as exc:
                raise ValueError(
                    f"full43 fixture has no mixed-prefix source for {name}"
                ) from exc
            shape = expanded_shape(name, source.shape)
            dtype = source.dtype

        is_output = name in inout_names or name in pure_outputs
        init_value = None if name in pure_outputs else 0
        if name == "full43_completion":
            init_value = -1
        if runtime_case is not None:
            # Runtime readback is deliberately narrow: final ping-pong state,
            # terminal outputs, and the program completion marker. Persistent
            # cache mutation is already covered by the nonzero G4 leaf goldens.
            is_output = name in pure_outputs or name in {"x_ping", "x_pong"}
            if name in runtime_source_values:
                runtime_value = expanded_runtime_value(
                    name, runtime_source_values[name]
                )
                if list(runtime_value.shape) != list(shape):
                    raise ValueError(
                        f"runtime value shape mismatch for {name}: "
                        f"{list(runtime_value.shape)} != {list(shape)}"
                    )
                init_value = runtime_value
        if runtime_case == "packed_pool_sentinel" and name in FULL43_PACKED_POOL_LAYER_COUNTS:
            init_value = packed_pool_sentinel(name, shape, dtype)
            is_output = True
        resident = None
        if shape and shape[0] == N_RANKS and name != "num_tokens_per_owner":
            resident = "stacked"
        specs.append(
            TensorSpec(
                name,
                list(shape),
                dtype,
                init_value=init_value,
                is_output=is_output,
                resident=resident,
            )
        )

    if [spec.name for spec in specs] != parameter_names:
        raise AssertionError("full43 spec order diverged from the decorated ABI")

    tensor_spec_by_name = {
        spec.name: spec for spec in specs if isinstance(spec, TensorSpec)
    }
    per_layer_extents = {
        "raw_kv_pool": KV_ORI_BLOCK_NUM,
        "hca_compress_state": source_by_name["hca_compress_state"].shape[1],
        "hca_cmp_kv": source_by_name["hca_cmp_kv"].shape[1],
        "csa_main_state": source_by_name["csa_main_state"].shape[1],
        "csa_main_cache": source_by_name["csa_main_cache"].shape[1],
        "csa_inner_state": source_by_name["csa_inner_state"].shape[1],
        "csa_idx_cache": source_by_name["csa_idx_cache"].shape[1],
        "csa_idx_scale": source_by_name["csa_idx_scale"].shape[1],
    }
    total_extents = {
        name: tensor_spec_by_name[name].shape[1]
        for name in FULL43_PACKED_POOL_LAYER_COUNTS
    }
    validate_full43_packed_pool_contract(
        total_extents,
        per_layer_extents,
        tensor_shapes={
            name: spec.shape
            for name, spec in tensor_spec_by_name.items()
        },
    )

    if runtime_case is not None:
        active_rows = runtime_source_values["active_row_mask"]
        raw_capacity = per_layer_extents["raw_kv_pool"] * BLOCK_SIZE
        validate_full43_local_slots(
            "swa_raw_write_slots",
            runtime_source_values["swa_raw_write_slots"],
            raw_capacity,
            active_rows[:, None, :],
        )
        validate_full43_local_slots(
            "hca_swa_write_slots",
            runtime_source_values["hca_swa_write_slots"],
            raw_capacity,
            active_rows,
        )
        validate_full43_local_slots(
            "hca_state_write_slots",
            runtime_source_values["hca_state_write_slots"],
            per_layer_extents["hca_compress_state"] * HCA_STATE_BLOCK_SIZE,
            active_rows,
        )
        validate_full43_local_slots(
            "hca_cmp_slot_mapping",
            runtime_source_values["hca_cmp_slot_mapping"],
            per_layer_extents["hca_cmp_kv"] * BLOCK_SIZE,
            active_rows,
        )
        csa_active = active_rows.reshape(N_RANKS, CSA_CHUNKS, CSA_CHUNK_T)
        event_query_ids = runtime_source_values["csa_event_query_ids"]
        event_rows = event_query_ids.clamp(0, CSA_CHUNK_T - 1).to(dtype=torch.int64)
        csa_event_active = (
            (event_query_ids >= 0)
            & (event_query_ids < CSA_CHUNK_T)
            & (torch.gather(csa_active, 2, event_rows) != 0)
        )
        for name, capacity in (
            ("csa_raw_write_slots", raw_capacity),
            (
                "csa_main_state_write_slots",
                per_layer_extents["csa_main_state"] * CSA_STATE_BLOCK_SIZE,
            ),
            (
                "csa_inner_state_write_slots",
                per_layer_extents["csa_inner_state"] * CSA_INNER_STATE_BLOCK_SIZE,
            ),
        ):
            validate_full43_local_slots(
                name,
                runtime_source_values[name],
                capacity,
                csa_active,
            )
        for name, values, capacity in (
            (
                "hca_state_page_ids",
                runtime_source_values["hca_state_page_ids"],
                per_layer_extents["hca_compress_state"],
            ),
            (
                "hca_pages",
                runtime_source_values["hca_pages"][..., 0],
                per_layer_extents["hca_cmp_kv"],
            ),
            (
                "csa_main_state_page_ids",
                runtime_source_values["csa_main_state_page_ids"],
                per_layer_extents["csa_main_state"],
            ),
            (
                "csa_inner_state_page_ids",
                runtime_source_values["csa_inner_state_page_ids"],
                per_layer_extents["csa_inner_state"],
            ),
            (
                "csa_pages",
                runtime_source_values["csa_pages"][..., 0],
                per_layer_extents["csa_main_cache"],
            ),
            (
                "csa_idx_pages",
                runtime_source_values["csa_idx_pages"][..., 0],
                (
                    per_layer_extents["csa_idx_cache"]
                    + BLOCK_SIZE
                    - 1
                )
                // BLOCK_SIZE,
            ),
        ):
            validate_full43_local_slots(name, values, capacity)
        validate_full43_local_slots(
            "csa_main_event_write_slots",
            runtime_source_values["csa_main_event_write_slots"],
            per_layer_extents["csa_main_cache"] * BLOCK_SIZE,
            csa_event_active,
        )
        validate_full43_local_slots(
            "csa_inner_event_write_slots",
            runtime_source_values["csa_inner_event_write_slots"],
            per_layer_extents["csa_idx_cache"],
            csa_event_active,
        )
        if full43_csa_model_layer(20) != LAST_CSA_MODEL_LAYER:
            raise AssertionError("CSA ordinal 20 must map to model layer 42")
        if full43_hca_model_layer(19) != 41:
            raise AssertionError("HCA ordinal 19 must map to model layer 41")
    return specs


def golden_full43_runtime(tensors):
    """Reference for the zero-weight full43 topology/lifecycle witness."""
    import torch

    tensors["x_ping"].zero_()
    tensors["x_pong"].zero_()
    tensors["hidden_out"].zero_()
    tensors["logits"].zero_()
    tensors["sampled_ids"].fill_(-1)
    active = tensors["active_row_mask"] != 0
    for rank in range(N_RANKS):
        active_rows = torch.nonzero(active[rank], as_tuple=False).flatten()
        tensors["sampled_ids"][rank, active_rows] = 0
    tensors["full43_completion"].zero_()

    if not bool(torch.any(tensors["raw_kv_pool"] != 0).item()):
        return

    def zero_local_rows(pool_name, ordinal, rank, slots):
        pool = tensors[pool_name]
        layer_count = FULL43_PACKED_POOL_LAYER_COUNTS[pool_name]
        extent = pool.shape[1] // layer_count
        block_size = FULL43_PACKED_POOL_BLOCK_SIZES[pool_name]
        layer = pool[rank, ordinal * extent : (ordinal + 1) * extent]
        if block_size > 1:
            rows = layer.reshape(extent * block_size, *layer.shape[2:])
        else:
            rows = layer
        local = torch.as_tensor(slots, dtype=torch.int64).reshape(-1)
        local = local[(local >= 0) & (local < rows.shape[0])]
        if local.numel():
            rows[local] = 0

    for rank in range(N_RANKS):
        for swa_ordinal in range(TWO_SWA_LAYERS):
            zero_local_rows(
                "raw_kv_pool",
                swa_ordinal,
                rank,
                tensors["swa_raw_write_slots"][rank, swa_ordinal],
            )
        shard = slice(rank * CSA_CHUNKS, (rank + 1) * CSA_CHUNKS)
        for csa_ordinal in range(CSA_FULL_LAYERS):
            csa_model_layer = 2 * csa_ordinal + 2
            zero_local_rows(
                "raw_kv_pool",
                csa_model_layer,
                rank,
                tensors["csa_raw_write_slots"][csa_ordinal, shard],
            )
            zero_local_rows(
                "csa_main_state",
                csa_ordinal,
                rank,
                tensors["csa_main_state_write_slots"][csa_ordinal, shard],
            )
            zero_local_rows(
                "csa_main_cache",
                csa_ordinal,
                rank,
                tensors["csa_main_event_write_slots"][csa_ordinal, shard],
            )
            zero_local_rows(
                "csa_inner_state",
                csa_ordinal,
                rank,
                tensors["csa_inner_state_write_slots"][csa_ordinal, shard],
            )
            for pool_name in ("csa_idx_cache", "csa_idx_scale"):
                zero_local_rows(
                    pool_name,
                    csa_ordinal,
                    rank,
                    tensors["csa_inner_event_write_slots"][csa_ordinal, shard],
                )
            if csa_ordinal < HCA_FULL_LAYERS:
                hca_model_layer = 2 * csa_ordinal + 3
                zero_local_rows(
                    "raw_kv_pool",
                    hca_model_layer,
                    rank,
                    tensors["hca_swa_write_slots"][rank, csa_ordinal],
                )
                zero_local_rows(
                    "hca_compress_state",
                    csa_ordinal,
                    rank,
                    tensors["hca_state_write_slots"][rank, csa_ordinal],
                )
                zero_local_rows(
                    "hca_cmp_kv",
                    csa_ordinal,
                    rank,
                    tensors["hca_cmp_slot_mapping"][rank, csa_ordinal],
                )


def main() -> None:
    """Compile the complete production graph without allocating runtime data."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8, 16])
    parser.add_argument(
        "--tp", type=int, default=LM_HEAD_TP_SIZE, choices=[2, 4, 8, 16]
    )
    parser.add_argument("-d", "--device", default=",".join(map(str, range(N_RANKS))))
    parser.add_argument(
        "--weight-bank-size",
        type=int,
        default=MAIN_LAYER_COUNT,
        choices=range(1, MAIN_LAYER_COUNT + 1),
    )
    parser.add_argument(
        "--case",
        choices=(
            "production_compile",
            "full_active",
            "ragged",
            "packed_pool_sentinel",
            "long_context_tail",
        ),
        default="production_compile",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--print-spec", action="store_true", default=False)
    parser.add_argument("--runtime-dir")
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
    if args.ep != N_RANKS or args.tp != LM_HEAD_TP_SIZE:
        parser.error(
            "--ep/--tp must match the values parsed before kernel imports: "
            f"ep={N_RANKS}, tp={LM_HEAD_TP_SIZE}"
        )
    if args.ep % args.tp:
        parser.error(f"--ep must be a multiple of --tp, got {args.ep}/{args.tp}")

    runtime_case = None if args.case == "production_compile" else args.case
    if runtime_case is not None and args.weight_bank_size != 1:
        parser.error("full43 runtime witnesses require --weight-bank-size 1")
    specs = build_full43_device_specs(
        weight_bank_size=args.weight_bank_size,
        runtime_case=runtime_case,
    )
    if args.print_spec:
        import math
        import torch

        resident = sum(
            int(getattr(spec, "is_resident", False)) for spec in specs
        )
        dtype_bytes = {
            torch.float32: 4,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.int32: 4,
            torch.int64: 8,
        }
        tensor_specs = [spec for spec in specs if hasattr(spec, "shape")]
        tensor_sizes = [
            (dtype_bytes[spec.dtype] * math.prod(spec.shape), spec.name)
            for spec in tensor_specs
        ]
        fixture_bytes = sum(size for size, _ in tensor_sizes)
        print(
            f"full43 tensors={len(specs)} resident={resident} "
            f"layers={MAIN_LAYER_COUNT} weight_bank={args.weight_bank_size} "
            f"fixture_gib={fixture_bytes / 2**30:.3f} pattern=2/21/20"
        )
        for size, name in sorted(tensor_sizes, reverse=True)[:8]:
            print(f"  {name}: {size / 2**30:.3f} GiB")
        return
    if runtime_case is None and not args.compile_only:
        parser.error(
            "production_compile is compile-only; choose --case full_active "
            "or --case ragged with --weight-bank-size 1 for runtime"
        )

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    device_ids = [int(value) for value in args.device.split(",") if value]
    if len(device_ids) < N_RANKS:
        parser.error(f"need at least {N_RANKS} device ids, got {device_ids}")
    result = run_jit(
        fn=l3_decode_fwd_full43,
        specs=specs,
        golden_fn=(
            golden_full43_runtime if runtime_case is not None else None
        ),
        compile_only=args.compile_only,
        compile_cfg={
            "distributed_config": DistributedConfig(
                device_ids=device_ids[:N_RANKS], num_sub_workers=0
            )
        },
        runtime_cfg={
            "platform": args.platform,
            "log_level": "warn",
            # Match the standalone layer and the profiled mixed-prefix path.
            # Setting PTO2_RING_HEAP before importing PyPTO controls planning,
            # but execute_compiled still needs the concrete runtime capacities.
            "ring_task_window": DECODE_FULL43_RING_TASK_WINDOW,
            "ring_heap": DECODE_FULL43_RING_HEAP,
            "ring_dep_pool": DECODE_FULL43_RING_DEP_POOL,
            "enable_dump_args": args.dump_args,
            "enable_scope_stats": args.enable_scope_stats,
        },
        runtime_dir=args.runtime_dir,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
