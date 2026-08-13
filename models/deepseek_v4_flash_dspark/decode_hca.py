# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone elastic HCA decode orchestration for the Run-055 Phase C ABI."""

import os

# Runtime ring allocation for long-history HCA.
_MIB = 1024 * 1024
HCA_RUNTIME_RING_HEAP = (256 * _MIB, 2048 * _MIB, 2048 * _MIB, 256 * _MIB)
os.environ["PTO2_RING_HEAP"] = ",".join(str(value) for value in HCA_RUNTIME_RING_HEAP)

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    FLASH as M,
    HCA_ROWS_PER_SHARD,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_PAGES_PER_REQUEST,
    HCA_STATE_PHYSICAL_BLOCKS,
    HCA_STATE_ROWS_PER_REQUEST,
    SWA_PERSISTENT_PAGES_PER_REQUEST,
    SWA_SOURCE_OVERLAY_BASE,
    SWA_WINDOW_ROWS,
)
from decode_compressor_ratio128 import compressor_ratio128
from decode_sparse_attn_hca import sparse_attn_hca
from hc_post import hc_post
from hc_pre import hc_pre
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from rope_interleave import _rope_interleave_active_body


B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("HCA_STATE_BLOCK_NUM_DYN")
HCA_WORK_DYN = pl.dynamic("HCA_WORK_DYN")
HCA_PAGES_DYN = pl.dynamic("HCA_PAGES_DYN")
HCA_REQUEST_OFFSETS_DYN = pl.dynamic("HCA_REQUEST_OFFSETS_DYN")
HCA_QUERY_OFFSETS_DYN = pl.dynamic("HCA_QUERY_OFFSETS_DYN")
EVENT_DYN = pl.dynamic("HCA_EVENT_DYN")

B = DECODE_LOCAL_REQUESTS
S = DECODE_SEQ
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
WIN = M.sliding_window
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
COMPRESS_RATIO = 128
COMPRESS_STATE_DIM = 2 * HEAD_DIM
if HCA_STATE_ROWS_PER_REQUEST != COMPRESS_RATIO:
    raise ValueError("HCA state-ring capacity must match the compressor ratio")


@pl.jit.inline
def attention_hca(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    query_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    query_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    request_event_indices: pl.Tensor[[B_DYN], pl.INT32],
    event_rope_cos: pl.Tensor[[EVENT_DYN, ROPE_DIM // 2], pl.FP32],
    event_rope_sin: pl.Tensor[[EVENT_DYN, ROPE_DIM // 2], pl.FP32],
    compress_state: pl.Tensor[
        [STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32
    ],
    state_page_ids: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    state_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_cache: pl.Tensor[
        [ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ],
    swa_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[
        [CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    hca_pages: pl.Tensor[[HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    """Compose HC, QKV, state/event compression, packed attention, and commits."""
    t_dim = pl.tensor.dim(x_hc, 0)
    b_dim = pl.tensor.dim(request_epochs, 0)
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )
    x_normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    current_kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        query_rope_cos,
        query_rope_sin,
        gamma_cq,
        gamma_ckv,
        q,
        current_kv,
        qr,
        qr_scale,
        late_dep,
    )

    event_count = pl.tensor.dim(event_rope_cos, 0)
    event_cos_il = pl.create_tensor([event_count, ROPE_DIM], dtype=pl.FP32)
    event_sin_signed = pl.create_tensor([event_count, ROPE_DIM], dtype=pl.FP32)
    _rope_interleave_active_body(
        event_rope_cos,
        event_rope_sin,
        event_cos_il,
        event_sin_signed,
    )
    compressed_projection = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    cmp_done, state_done = compressor_ratio128(
        x_normed,
        compressed_projection,
        compress_state,
        state_page_ids,
        state_valid_ranges,
        state_page_epochs,
        request_epochs,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        request_event_indices,
        event_cos_il,
        event_sin_signed,
        cmp_kv,
        position_ids,
        cmp_slot_mapping,
        state_write_slots,
        late_dep,
    )

    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    attn_read_done = sparse_attn_hca(
        q,
        kv_cache,
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
        query_rope_cos,
        query_rope_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        cmp_done,
    )

    ori_blocks = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_blocks * BLOCK_SIZE, HEAD_DIM])
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="hca_raw_cache_commit",
        deps=[attn_read_done],
        allow_early_resolve=True,
    ):
        for token in pl.range(t_dim):
            slot_i64 = pl.read(swa_write_slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                kv_cache_flat[slot : slot + 1, 0:HEAD_DIM] = current_kv[
                    token : token + 1, 0:HEAD_DIM
                ]
    hc_post(attn_out, x_hc, post, comb, x_out)
    return x_out


@pl.jit
def attention_hca_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    query_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    query_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    request_event_indices: pl.Tensor[[B_DYN], pl.INT32],
    event_rope_cos: pl.Tensor[[EVENT_DYN, ROPE_DIM // 2], pl.FP32],
    event_rope_sin: pl.Tensor[[EVENT_DYN, ROPE_DIM // 2], pl.FP32],
    compress_state: pl.InOut[
        pl.Tensor[
            [STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    state_page_ids: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    state_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    swa_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.InOut[
        pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    hca_pages: pl.Tensor[[HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    query_rope_cos.bind_dynamic(0, T_DYN)
    query_rope_sin.bind_dynamic(0, T_DYN)
    request_event_indices.bind_dynamic(0, B_DYN)
    event_rope_cos.bind_dynamic(0, EVENT_DYN)
    event_rope_sin.bind_dynamic(0, EVENT_DYN)
    compress_state.bind_dynamic(0, STATE_BLOCK_NUM_DYN)
    state_page_ids.bind_dynamic(0, B_DYN)
    state_valid_ranges.bind_dynamic(0, B_DYN)
    state_page_epochs.bind_dynamic(0, B_DYN)
    request_epochs.bind_dynamic(0, B_DYN)
    state_write_slots.bind_dynamic(0, T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_write_slots.bind_dynamic(0, T_DYN)
    swa_sources.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    query_request_ids.bind_dynamic(0, T_DYN)
    hca_pages.bind_dynamic(0, HCA_PAGES_DYN)
    hca_page_offsets.bind_dynamic(0, HCA_REQUEST_OFFSETS_DYN)
    hca_windows.bind_dynamic(0, B_DYN)
    hca_query_work_offsets.bind_dynamic(0, HCA_QUERY_OFFSETS_DYN)
    hca_work_query_ids.bind_dynamic(0, HCA_WORK_DYN)
    hca_work_row_begin.bind_dynamic(0, HCA_WORK_DYN)
    hca_work_valid_rows.bind_dynamic(0, HCA_WORK_DYN)
    x_out.bind_dynamic(0, T_DYN)
    return attention_hca(
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
        x_out,
    )


def golden_attention_hca(tensors):
    """Torch reference preserving compressor, attention, and commit ordering."""
    import torch

    from decode_compressor_ratio128 import golden_compressor
    from decode_sparse_attn_hca import golden_sparse_attn
    from hc_post import golden_hc_post
    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm

    tokens = tensors["x_hc"].shape[0]
    mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post = torch.zeros(tokens, HC_MULT)
    comb = torch.zeros(tokens, HC_MULT * HC_MULT)
    golden_hc_pre(
        {
            "x": tensors["x_hc"],
            "hc_fn": tensors["hc_attn_fn"],
            "hc_scale": tensors["hc_attn_scale"],
            "hc_base": tensors["hc_attn_base"],
            "x_mixed": mixed,
            "post": post,
            "comb": comb,
        }
    )
    normed = golden_rms_norm(mixed, tensors["attn_norm_w"])
    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    current_kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    golden_qkv_proj_rope(
        {
            "x": normed,
            "wq_a": tensors["wq_a"],
            "wq_b": tensors["wq_b"],
            "wq_b_scale": tensors["wq_b_scale"],
            "wkv": tensors["wkv"],
            "rope_cos": tensors["query_rope_cos"],
            "rope_sin": tensors["query_rope_sin"],
            "gamma_cq": tensors["gamma_cq"],
            "gamma_ckv": tensors["gamma_ckv"],
            "q": q,
            "kv": current_kv,
            "qr": torch.zeros(tokens, Q_LORA, dtype=torch.int8),
            "qr_scale": torch.zeros(tokens, 1),
        }
    )
    compressed_projection = torch.zeros(tokens, HEAD_DIM)
    golden_compressor(
        {
            "x": normed,
            "kv": compressed_projection,
            "compress_state": tensors["compress_state"],
            "state_page_ids": tensors["state_page_ids"],
            "state_valid_ranges": tensors["state_valid_ranges"],
            "state_page_epochs": tensors["state_page_epochs"],
            "request_epochs": tensors["request_epochs"],
            "wkv": tensors["cmp_wkv"],
            "wgate": tensors["cmp_wgate"],
            "ape": tensors["cmp_ape"],
            "norm_w": tensors["cmp_norm_w"],
            "request_event_indices": tensors["request_event_indices"],
            "cos": tensors["event_rope_cos"],
            "sin": tensors["event_rope_sin"],
            "cmp_kv_cache": tensors["cmp_kv"],
            "position_ids": tensors["position_ids"],
            "cmp_slot_mapping": tensors["cmp_slot_mapping"],
            "state_slot_mapping": tensors["state_write_slots"],
        }
    )
    attn_out = torch.zeros(tokens, D, dtype=torch.bfloat16)
    golden_sparse_attn(
        {
            "q": q,
            "ori_kv": tensors["kv_cache"],
            "current_kv": current_kv,
            "swa_sources": tensors["swa_sources"],
            "cmp_kv": tensors["cmp_kv"],
            "query_request_ids": tensors["query_request_ids"],
            "hca_pages": tensors["hca_pages"],
            "hca_page_offsets": tensors["hca_page_offsets"],
            "hca_windows": tensors["hca_windows"],
            "request_epochs": tensors["request_epochs"],
            "hca_query_work_offsets": tensors["hca_query_work_offsets"],
            "hca_work_query_ids": tensors["hca_work_query_ids"],
            "hca_work_row_begin": tensors["hca_work_row_begin"],
            "hca_work_valid_rows": tensors["hca_work_valid_rows"],
            "attn_sink": tensors["attn_sink"],
            "freqs_cos": tensors["query_rope_cos"],
            "freqs_sin": tensors["query_rope_sin"],
            "wo_a": tensors["wo_a"],
            "wo_b": tensors["wo_b"],
            "wo_b_scale": tensors["wo_b_scale"],
            "attn_out": attn_out,
        }
    )
    flat_cache = tensors["kv_cache"].reshape(-1, HEAD_DIM)
    for token, slot in enumerate(tensors["swa_write_slots"].tolist()):
        if slot >= 0:
            flat_cache[slot] = current_kv[token]
    output = torch.zeros(tokens, HC_MULT, D)
    golden_hc_post(
        {
            "x": attn_out,
            "residual": tensors["x_hc"],
            "post": post,
            "comb": comb,
            "y": output,
        }
    )
    tensors["x_out"][:] = output


_CASES = {
    "short_history": [127, 127, 127, 127],
    "boundary_126_127": [128, 127, 127, 127],
    "state_rollover_127_128": [129, 127, 127, 127],
    "one_full_shard": [16_384, 127, 127, 127],
    "two_shards": [32_768, 127, 127, 127],
    "heterogeneous_lengths": [127, 128, 32_768, 1_048_576],
    "one_m_tail": [1_048_576, 127, 127, 127],
    "long_context_tail": [
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
    ],
}


def build_tensor_specs(case="short_history", batch=4):
    """Build elastic-S fixtures for a runtime batch, cycling the case pattern.

    The compiled HCA entry keeps its existing dynamic batch dimensions; this
    helper merely materializes a multiple-of-four runtime batch (up to the
    configured per-rank ``B``) without introducing another compile profile.
    """
    import torch
    from golden import TensorSpec
    from utils import quant_w_per_channel, token_local_rope

    if case not in _CASES:
        raise ValueError(f"unknown HCA case: {case!r}")
    if batch < 4 or batch > B or batch % 4:
        raise ValueError(f"batch must be a multiple of 4 in [4, {B}], got {batch}")

    pattern = _CASES[case]
    lengths = [pattern[index % len(pattern)] for index in range(batch)]
    tokens = batch * S
    positions = torch.tensor(
        [
            position
            for length in lengths
            for position in range(length - S, length)
        ],
        dtype=torch.int32,
    )
    request_ids = torch.arange(batch, dtype=torch.int32).repeat_interleave(S)
    page_entries = []
    page_offsets = [0]
    hca_windows = []
    work_offsets = [0]
    work_query_ids = []
    work_rows = []
    work_valid = []
    physical_page = 0
    for request, length in enumerate(lengths):
        max_rows = length // COMPRESS_RATIO
        page_count = max(1, (max_rows + BLOCK_SIZE - 1) // BLOCK_SIZE)
        pages = list(range(physical_page, physical_page + page_count))
        physical_page += page_count
        page_entries.extend((page, 7) for page in pages)
        page_offsets.append(len(page_entries))
        pre_end = max_rows - int(length % COMPRESS_RATIO == 0)
        hca_windows.append((0, pre_end, 0))
        for local in range(S):
            query = request * S + local
            visible_rows = (length - S + local + 1) // COMPRESS_RATIO
            row = 0
            while row < visible_rows:
                work_query_ids.append(query)
                work_rows.append(row)
                work_valid.append(min(HCA_ROWS_PER_SHARD, visible_rows - row))
                row += HCA_ROWS_PER_SHARD
            work_offsets.append(len(work_query_ids))

    required_state_pages = batch * HCA_STATE_PAGES_PER_REQUEST
    if required_state_pages > HCA_STATE_PHYSICAL_BLOCKS:
        raise ValueError("HCA fixture exceeds the physical state-page pool")
    state_page_ids = torch.arange(
        required_state_pages, dtype=torch.int32
    ).reshape(batch, HCA_STATE_PAGES_PER_REQUEST)
    state_ranges = []
    state = torch.zeros(
        HCA_STATE_PHYSICAL_BLOCKS,
        HCA_STATE_BLOCK_SIZE,
        COMPRESS_STATE_DIM,
    )
    generator = torch.Generator().manual_seed(5511)
    for request in range(batch):
        first = int(positions[request * S])
        begin = first // COMPRESS_RATIO * COMPRESS_RATIO
        state_ranges.append((begin, first))
        for position in range(begin, first):
            ring_row = position % HCA_STATE_ROWS_PER_REQUEST
            relative_page = ring_row // HCA_STATE_BLOCK_SIZE
            page = int(state_page_ids[request, relative_page])
            state[page, ring_row % HCA_STATE_BLOCK_SIZE] = (
                torch.randn(COMPRESS_STATE_DIM, generator=generator) * 0.02
            )
    state_ranges = torch.tensor(state_ranges, dtype=torch.int32)
    state_ring_rows = positions.reshape(batch, S).to(torch.int64) % HCA_STATE_ROWS_PER_REQUEST
    state_relative_pages = torch.div(
        state_ring_rows, HCA_STATE_BLOCK_SIZE, rounding_mode="floor"
    )
    state_write_pages = torch.gather(
        state_page_ids.to(torch.int64), 1, state_relative_pages
    )
    state_write_slots = (
        state_write_pages * HCA_STATE_BLOCK_SIZE
        + state_ring_rows % HCA_STATE_BLOCK_SIZE
    ).reshape(-1)
    cmp_slots = torch.full((tokens,), -1, dtype=torch.int64)
    for query, position in enumerate(positions.tolist()):
        if (position + 1) % COMPRESS_RATIO == 0:
            request = query // S
            row = position // COMPRESS_RATIO
            entry = page_offsets[request] + row // BLOCK_SIZE
            cmp_slots[query] = page_entries[entry][0] * BLOCK_SIZE + row % BLOCK_SIZE

    query_cos, query_sin = token_local_rope(
        M, COMPRESS_RATIO, positions.to(torch.int64), dtype=torch.bfloat16
    )
    request_event_indices = torch.full((batch,), -1, dtype=torch.int32)
    event_position_values = []
    for request in range(batch):
        request_positions = positions[request * S : (request + 1) * S]
        boundaries = request_positions[
            (request_positions.to(torch.int64) + 1) % COMPRESS_RATIO == 0
        ]
        if int(boundaries.numel()) != 0:
            request_event_indices[request] = len(event_position_values)
            event_position_values.append(
                int(boundaries[0]) // COMPRESS_RATIO * COMPRESS_RATIO
            )
    event_positions = torch.tensor(event_position_values, dtype=torch.int64)
    event_cos, event_sin = token_local_rope(
        M, COMPRESS_RATIO, event_positions, dtype=torch.float32
    )
    event_cos = event_cos[:, : ROPE_DIM // 2].contiguous()
    event_sin = event_sin[:, : ROPE_DIM // 2].contiguous()

    if SWA_PERSISTENT_PAGES_PER_REQUEST * BLOCK_SIZE < SWA_WINDOW_ROWS:
        raise ValueError("SWA persistent page descriptor cannot cover its window")
    kv_cache = torch.randn(
        batch * SWA_PERSISTENT_PAGES_PER_REQUEST,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
    ).to(torch.bfloat16)
    swa_sources = torch.empty(tokens, WIN, dtype=torch.int32)
    swa_write_slots = torch.empty(tokens, dtype=torch.int64)
    for query in range(tokens):
        request = query // S
        position = int(positions[query])
        window_begin = max(0, position + 1 - WIN)
        for lane in range(WIN):
            logical = window_begin + lane
            if logical >= position + 1:
                swa_sources[query, lane] = -1
            elif logical >= int(positions[request * S]) and logical <= position:
                overlay = request * S + logical - int(positions[request * S])
                swa_sources[query, lane] = SWA_SOURCE_OVERLAY_BASE - overlay
            else:
                swa_sources[query, lane] = (
                    request * SWA_WINDOW_ROWS + logical % SWA_WINDOW_ROWS
                )
        swa_write_slots[query] = (
            request * SWA_WINDOW_ROWS + position % SWA_WINDOW_ROWS
        )

    cmp_kv = torch.randn(
        max(physical_page, 1), BLOCK_SIZE, 1, HEAD_DIM
    ).to(torch.bfloat16)
    q_weight = torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA**0.5
    q_i8, q_scale = quant_w_per_channel(q_weight.to(torch.bfloat16).T)
    q_i8 = q_i8.T.contiguous()
    wo_weight = torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5
    wo_i8, wo_scale = quant_w_per_channel(wo_weight.to(torch.bfloat16))
    inputs = {
        "x_hc": torch.empty(tokens, HC_MULT, D).uniform_(-1, 1),
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM) * 0.0495,
        "hc_attn_scale": torch.tensor([0.079046, 0.04213, 0.121901]),
        "hc_attn_base": torch.randn(MIX_HC),
        "attn_norm_w": torch.ones(D, dtype=torch.bfloat16),
        "wq_a": (torch.randn(D, Q_LORA) / D**0.5).to(torch.bfloat16),
        "wq_b": q_i8,
        "wq_b_scale": q_scale,
        "wkv": (torch.randn(D, HEAD_DIM) / D**0.5).to(torch.bfloat16),
        "gamma_cq": torch.ones(Q_LORA, dtype=torch.bfloat16),
        "gamma_ckv": torch.ones(HEAD_DIM, dtype=torch.bfloat16),
        "query_rope_cos": query_cos,
        "query_rope_sin": query_sin,
        "cmp_wkv": (torch.randn(HEAD_DIM, D) * 0.0246).to(torch.bfloat16),
        "cmp_wgate": (torch.randn(HEAD_DIM, D) * 0.0316).to(torch.bfloat16),
        "cmp_ape": torch.randn(COMPRESS_RATIO, HEAD_DIM) * 0.034,
        "cmp_norm_w": (0.1001 + 0.0549 * torch.randn(HEAD_DIM)).to(torch.bfloat16),
        "request_event_indices": request_event_indices,
        "event_rope_cos": event_cos,
        "event_rope_sin": event_sin,
        "compress_state": state,
        "state_page_ids": state_page_ids,
        "state_valid_ranges": state_ranges,
        "state_page_epochs": torch.full(
            (batch, HCA_STATE_PAGES_PER_REQUEST), 7, dtype=torch.int32
        ),
        "request_epochs": torch.full((batch,), 7, dtype=torch.int32),
        "state_write_slots": state_write_slots,
        "kv_cache": kv_cache,
        "swa_write_slots": swa_write_slots,
        "swa_sources": swa_sources,
        "cmp_kv": cmp_kv,
        "cmp_slot_mapping": cmp_slots,
        "position_ids": positions,
        "query_request_ids": request_ids,
        "hca_pages": torch.tensor(page_entries, dtype=torch.int32),
        "hca_page_offsets": torch.tensor(page_offsets, dtype=torch.int32),
        "hca_windows": torch.tensor(hca_windows, dtype=torch.int32),
        "hca_query_work_offsets": torch.tensor(work_offsets, dtype=torch.int32),
        "hca_work_query_ids": torch.tensor(work_query_ids, dtype=torch.int32),
        "hca_work_row_begin": torch.tensor(work_rows, dtype=torch.int32),
        "hca_work_valid_rows": torch.tensor(work_valid, dtype=torch.int32),
        "attn_sink": torch.zeros(H),
        "wo_a": (torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN**0.5).to(torch.bfloat16),
        "wo_b": wo_i8,
        "wo_b_scale": wo_scale,
    }
    outputs = {"compress_state", "kv_cache", "cmp_kv"}
    specs = [
        TensorSpec(
            name,
            list(value.shape),
            value.dtype,
            init_value=value,
            is_output=name in outputs,
        )
        for name, value in inputs.items()
    ]
    specs.append(
        TensorSpec("x_out", [tokens, HC_MULT, D], torch.float32, is_output=True)
    )
    return specs


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=list(_CASES), default="short_history")
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()
    if args.batch < 4 or args.batch > B or args.batch % 4:
        parser.error(f"--batch must be a multiple of 4 in [4, {B}], got {args.batch}")
    result = run_jit(
        fn=attention_hca_test,
        specs=build_tensor_specs(args.case, batch=args.batch),
        golden_fn=golden_attention_hca,
        compile_cfg={"dump_passes": args.dump_passes},
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_dep_gen": args.enable_dep_gen,
        },
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
        },
        rtol=1e-2,
        atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
