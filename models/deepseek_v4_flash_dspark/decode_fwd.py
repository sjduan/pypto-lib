# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared kernels and fixtures for the 43-layer D-Spark decode forward.

The executable graph lives in :mod:`decode_fwd_full43`. This module contains
only its active-row packing, CSA activation staging, shared-MoE cleanup,
terminal head, and shape-complete fixture construction.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import config

# Install the profiled 1M-tail ring capacities before PyPTO imports its
# distributed runtime.  ``decode_layer`` exports the same value, but importing
# it below is too late for entry points that start here.
_MIB = 1024 * 1024
DECODE_FWD_RING_HEAP = (256 * _MIB, 2048 * _MIB, 2048 * _MIB, 256 * _MIB)
os.environ.setdefault(
    "PTO2_RING_HEAP", ",".join(str(value) for value in DECODE_FWD_RING_HEAP)
)

import pypto.language as pl
import pypto.language.distributed as pld

_EP_CHOICES = (2, 4, 8, 16)
_TP_CHOICES = (2, 4, 8, 16)
_EP_DEFAULT = config.EP
_TP_DEFAULT = config.TP


def _parse_parallel_argv(name, default):
    for index, token in enumerate(sys.argv):
        if token == name and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if token.startswith(f"{name}="):
            return int(token.split("=", 1)[1])
    return default


_REQUESTED_EP = _parse_parallel_argv("--ep", _EP_DEFAULT)
_REQUESTED_TP = _parse_parallel_argv("--tp", _TP_DEFAULT)
if _REQUESTED_EP not in _EP_CHOICES:
    raise ValueError(f"--ep must be one of {_EP_CHOICES}, got {_REQUESTED_EP}")
if _REQUESTED_TP not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES}, got {_REQUESTED_TP}")
if _REQUESTED_EP % _REQUESTED_TP:
    raise ValueError(
        f"--ep must be a multiple of --tp, got ep={_REQUESTED_EP}, tp={_REQUESTED_TP}"
    )

from config import BLOCK_SIZE, FLASH as MODEL_CONFIG
from decode_layer import (
    CSA_CHUNK_T,
    CSA_LAYER_SHARDS,
    D,
    HC_DIM,
    HC_MULT,
    attention_kind_for_layer,
)
from hc_head import hc_head
from lookup_embedding import VOCAB_DYN as EMBED_VOCAB_DYN
from lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    lm_head_with_sampling,
)
from moe import N_RANKS, T, clear_moe_signals
from rmsnorm import rms_norm

if N_RANKS != _REQUESTED_EP:
    raise ValueError(
        f"MoE parsed N_RANKS={N_RANKS}, expected --ep={_REQUESTED_EP}"
    )
if LM_HEAD_TP_SIZE != _REQUESTED_TP:
    raise ValueError(
        f"LM head parsed TP_SIZE={LM_HEAD_TP_SIZE}, expected --tp={_REQUESTED_TP}"
    )
if N_RANKS % LM_HEAD_TP_SIZE:
    raise ValueError(
        f"N_RANKS must be a multiple of LM_HEAD_TP_SIZE, got {N_RANKS} and {LM_HEAD_TP_SIZE}"
    )


MAIN_LAYER_COUNT = MODEL_CONFIG.num_hidden_layers
LM_HEAD_COMM_EPOCH = 1
FWD_RAW_BLOCKS_DYN = pl.dynamic("FWD_RAW_BLOCKS_DYN")
TWO_SWA_LAYERS = 2
MIXED_QUAD_LAYERS = 4
CSA_CHUNKS = T // CSA_CHUNK_T
CSA_ACTIVATION_COPY_TILE = 512
CSA_ACTIVATION_COPY_BLOCKS = 48
MIXED_QUAD_ATTENTION_KINDS = tuple(
    attention_kind_for_layer(layer_id) for layer_id in range(MIXED_QUAD_LAYERS)
)

assert MAIN_LAYER_COUNT == 43, "Phase F is defined for the 43-layer Flash model"
assert T % CSA_CHUNK_T == 0
assert D % CSA_ACTIVATION_COPY_TILE == 0
assert MIXED_QUAD_ATTENTION_KINDS == ("swa", "swa", "csa", "hca")
INACTIVE_TOKEN_ID = -1
FWD_PACK_HIDDEN_TILE = 512
FWD_PACK_SPMD_BLOCKS = 48
assert D % FWD_PACK_HIDDEN_TILE == 0
assert SAMPLED_IDS_PAD >= 1


@pl.jit.inline
def csa_activation_chunk_view(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D], pl.FP32]:
    """Return the bounded sixteen-query CSA activation view for one rank."""
    return pl.reshape(x_hc, [CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D])


@pl.jit.inline
def decode_fwd_pack_rows(
    input_ids: pl.Tensor[[T], pl.INT64],
    active_row_mask: pl.Tensor[[T], pl.INT32],
    embed_weight: pl.Tensor[[EMBED_VOCAB_DYN, D], pl.BF16],
    x_ping: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    """Embed active main-model rows and zero the bounded inactive capacity."""
    embed_weight.bind_dynamic(0, EMBED_VOCAB_DYN)
    x_flat = pl.reshape(x_ping, [T * HC_MULT, D])
    for block in pl.spmd(FWD_PACK_SPMD_BLOCKS, name_hint="decode_fwd_pack_rows"):
        for work_idx in pl.range(
            block,
            T * (D // FWD_PACK_HIDDEN_TILE),
            FWD_PACK_SPMD_BLOCKS,
        ):
            token_idx = work_idx // (D // FWD_PACK_HIDDEN_TILE)
            hidden_offset = (
                work_idx % (D // FWD_PACK_HIDDEN_TILE)
            ) * FWD_PACK_HIDDEN_TILE
            hidden_chunk = pl.full(
                [1, FWD_PACK_HIDDEN_TILE], dtype=pl.FP32, value=0.0
            )
            if pl.read(active_row_mask, [token_idx]) != 0:
                token_id = pl.read(input_ids, [token_idx])
                token_row = pl.cast(token_id, target_type=pl.INDEX)
                hidden_chunk = pl.cast(
                    embed_weight[
                        token_row : token_row + 1,
                        hidden_offset : hidden_offset + FWD_PACK_HIDDEN_TILE,
                    ],
                    target_type=pl.FP32,
                )
            for hc_idx in pl.range(HC_MULT):
                row = token_idx * HC_MULT + hc_idx
                x_flat[
                    row : row + 1,
                    hidden_offset : hidden_offset + FWD_PACK_HIDDEN_TILE,
                ] = hidden_chunk
    return x_ping


@pl.jit.inline
def build_main_logit_rows(
    active_row_mask: pl.Tensor[[T], pl.INT32],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
):
    """Map compact main-forward rows into the bounded LM-head input."""
    for row_core in pl.spmd(1, name_hint="build_main_logit_rows"):
        for row in pl.range(row_core, MAX_LOGIT_ROWS):
            source_row = pl.cast(-1, pl.INT32)
            if row < T:
                if pl.read(active_row_mask, [row]) != 0:
                    source_row = pl.cast(row, pl.INT32)
            pl.write(logit_row_indices, [row], source_row)
    return logit_row_indices


@pl.jit.inline
def mask_inactive_main_samples(
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Publish -1 for every fixed-capacity LM-head row without a source."""
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="mask_inactive_main_samples"):
        if pl.read(logit_row_indices, [row]) < 0:
            sampled_ids[row : row + 1, :] = pl.full(
                [1, SAMPLED_IDS_PAD],
                dtype=pl.INT32,
                value=INACTIVE_TOKEN_ID,
            )
    return sampled_ids


@pl.jit(auto_scope=False)
def decode_fwd_pack_active(
    input_ids: pl.Tensor[[T], pl.INT64],
    active_row_mask: pl.Tensor[[T], pl.INT32],
    embed_weight: pl.Tensor[[EMBED_VOCAB_DYN, D], pl.BF16],
    x_ping: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    """Materialize the bounded main activation without reading inactive ids."""
    return decode_fwd_pack_rows(input_ids, active_row_mask, embed_weight, x_ping)


@pl.jit(auto_scope=False)
def decode_fwd_stage_csa_activation(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    csa_x_hc_workspace: pl.Out[
        pl.Tensor[[CSA_CHUNKS, CSA_CHUNK_T, HC_MULT, D], pl.FP32]
    ],
):
    """Materialize the bounded CSA chunk layout inside a device leaf.

    Host-orchestration reshape lowering currently emits an unbound ``tensor``
    symbol.  This explicit device copy preserves the same contiguous row order
    without placing a shape reinterpretation in generated host code.
    """
    source_flat = pl.reshape(x_hc, [T * HC_MULT, D])
    destination_flat = pl.reshape(csa_x_hc_workspace, [T * HC_MULT, D])
    for block in pl.spmd(
        CSA_ACTIVATION_COPY_BLOCKS,
        name_hint="decode_fwd_stage_csa_activation",
    ):
        for work_idx in pl.range(
            block,
            T * HC_MULT * (D // CSA_ACTIVATION_COPY_TILE),
            CSA_ACTIVATION_COPY_BLOCKS,
        ):
            row = work_idx // (D // CSA_ACTIVATION_COPY_TILE)
            column = (work_idx % (D // CSA_ACTIVATION_COPY_TILE)) * CSA_ACTIVATION_COPY_TILE
            destination_flat[
                row : row + 1,
                column : column + CSA_ACTIVATION_COPY_TILE,
            ] = source_flat[
                row : row + 1,
                column : column + CSA_ACTIVATION_COPY_TILE,
            ]
    return csa_x_hc_workspace


@pl.jit(auto_scope=False)
def decode_fwd_clear_shared_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear the forward-owned MoE windows after the final submitted layer."""
    clear_moe_signals(
        completion_anchor,
        arrived,
        data_arrived,
        combine_arrived,
    )
    return completion_anchor


_MIXED_QUAD_COMMON_NAMES = (
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
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
)

_MIXED_QUAD_CSA_SOURCE_NAMES = {
    "csa_rope_cos": "rope_cos",
    "csa_rope_sin": "rope_sin",
}

_MIXED_QUAD_HCA_SOURCE_NAMES = {
    "hca_query_rope_cos": "query_rope_cos",
    "hca_query_rope_sin": "query_rope_sin",
}

_FULL43_HCA_FIXTURE_NAMES = (
    "hca_query_rope_cos",
    "hca_query_rope_sin",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_request_event_indices",
    "hca_event_rope_cos",
    "hca_event_rope_sin",
    "hca_compress_state",
    "hca_state_page_ids",
    "hca_state_valid_ranges",
    "hca_state_page_epochs",
    "hca_request_epochs",
    "hca_state_write_slots",
    "hca_swa_write_slots",
    "hca_swa_sources",
    "hca_cmp_kv",
    "hca_cmp_slot_mapping",
    "hca_position_ids",
    "hca_query_request_ids",
    "hca_pages",
    "hca_page_offsets",
    "hca_windows",
    "hca_query_work_offsets",
    "hca_work_query_ids",
    "hca_work_row_begin",
    "hca_work_valid_rows",
)

_FULL43_CSA_FIXTURE_NAMES = (
    "csa_x_attn_workspace",
    "csa_rope_cos",
    "csa_rope_sin",
    "csa_main_wkv",
    "csa_main_wgate",
    "csa_main_ape",
    "csa_main_norm_w",
    "csa_main_state",
    "csa_main_state_page_ids",
    "csa_main_state_valid_ranges",
    "csa_main_state_page_epochs",
    "csa_compressor_request_epochs",
    "csa_request_event_indices",
    "csa_event_query_ids",
    "csa_event_rope_cos",
    "csa_event_rope_sin",
    "csa_main_event_write_slots",
    "csa_position_ids",
    "csa_main_state_write_slots",
    "csa_main_cache",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_hadamard",
    "csa_inner_state",
    "csa_inner_state_page_ids",
    "csa_inner_state_valid_ranges",
    "csa_inner_state_page_epochs",
    "csa_inner_event_write_slots",
    "csa_inner_state_write_slots",
    "csa_idx_cache",
    "csa_idx_scale",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_idx_weights_proj",
    "csa_idx_hadamard",
    "csa_idx_cos_il",
    "csa_idx_sin_signed",
    "csa_query_request_ids",
    "csa_idx_pages",
    "csa_idx_page_offsets",
    "csa_idx_windows",
    "csa_request_epochs",
    "csa_leaf_descriptors",
    "csa_pair_descriptors",
    "csa_singleton_descriptors",
    "csa_upper_descriptors",
    "csa_root_descriptors",
    "csa_pages",
    "csa_page_offsets",
    "csa_windows",
    "csa_raw_write_slots",
    "csa_swa_sources",
)

_MIXED_QUAD_INOUT_NAMES = frozenset(
    {
        "x_ping",
        "x_pong",
        "raw_kv_pool",
        "hca_compress_state",
        "hca_cmp_kv",
        "csa_main_state",
        "csa_main_cache",
        "csa_inner_state",
        "csa_idx_cache",
        "csa_idx_scale",
        "csa_x_attn_workspace",
    }
)

# These pools have one leading rank shard and each child consumes only
# ``pool[rank]`` on ``device=rank``.  Keep them resident as stacked allocations
# so a rank never receives a replicated copy of another rank's mutable state.
_MIXED_QUAD_CSA_RESIDENT_POOLS = frozenset(
    {
        "main_state",
        "main_cache",
        "inner_state",
        "idx_cache",
        "idx_scale",
    }
)


def build_mixed_quad_device_specs(
    *,
    num_tokens_per_owner: Sequence[int] | int | None = None,
    context_case: str = "default",
):
    """Build the four-layer frontend-probe ABI from Phase-E layer fixtures.

    Immutable common tensors are stacked at the model-layer axis.  The raw
    cache is one physical pool whose slot/source descriptors are shifted to
    absolute rows before the four Phase-E children observe it.  CSA/HCA
    compressed/state pools remain distinct cache groups and retain their own
    ragged descriptors; no pool dimension is divided by a layer count.
    """
    import torch
    from golden import TensorSpec

    from decode_layer import (
        build_csa_layer_specs,
        build_hca_layer_specs,
        build_swa_layer_specs,
    )
    def tensor_values(specs):
        return {
            spec.name: spec.create_tensor()
            for spec in specs
            if isinstance(spec, TensorSpec)
        }

    def shift_absolute_rows(value, offset):
        if offset == 0:
            return value.clone()
        return torch.where(value >= 0, value + offset, value)

    def rank_expand(value):
        return value.unsqueeze(0).expand(N_RANKS, *value.shape).contiguous()

    if context_case not in {"default", "one_m_tail"}:
        raise ValueError(f"unknown mixed-quad context case: {context_case!r}")
    fixture_counts = num_tokens_per_owner
    if fixture_counts is None:
        fixture_counts = T

    swa = tensor_values(
        build_swa_layer_specs(
            case=(
                "long_context_tail"
                if context_case == "one_m_tail"
                else "short_history"
            ),
            layer_id=0,
            num_tokens_per_owner=fixture_counts,
        )
    )
    csa = tensor_values(
        build_csa_layer_specs(
            case=(
                "full43_long_context_tail"
                if context_case == "one_m_tail"
                else "global_state_pool"
            ),
            layer_id=2,
            num_tokens_per_owner=fixture_counts,
        )
    )
    hca = tensor_values(
        build_hca_layer_specs(
            case=(
                "long_context_tail"
                if context_case == "one_m_tail"
                else "heterogeneous_lengths"
            ),
            layer_id=3,
            num_tokens_per_owner=fixture_counts,
        )
    )
    # The main-model fixture owns its row mask directly. A rank's active rows
    # are a compact prefix whose length is exactly its MoE submission count.
    row_ids = torch.arange(T, dtype=torch.int32).unsqueeze(0)
    rank_active_rows = (
        row_ids < swa["num_tokens_per_owner"].to(dtype=torch.int32)[:, None]
    ).to(dtype=torch.int32)
    if not torch.equal(
        rank_active_rows.to(dtype=torch.int64).sum(dim=1),
        swa["num_tokens_per_owner"].to(dtype=torch.int64),
    ):
        raise AssertionError(
            "mixed-quad active rows must equal the rank-local MoE token count"
        )
    csa_active_rows = rank_active_rows.reshape(
        N_RANKS, CSA_CHUNKS, CSA_CHUNK_T
    ).contiguous()

    def row_mask(value, active_rows):
        if value.ndim < active_rows.ndim:
            raise ValueError(
                f"descriptor rank {value.ndim} is smaller than active-row rank "
                f"{active_rows.ndim}"
            )
        if tuple(value.shape[: active_rows.ndim]) != tuple(active_rows.shape):
            raise ValueError(
                "descriptor leading rows do not match the mixed active-row mask: "
                f"descriptor={tuple(value.shape)}, active={tuple(active_rows.shape)}"
            )
        return (active_rows != 0).reshape(
            (*active_rows.shape, *((1,) * (value.ndim - active_rows.ndim)))
        ).expand_as(value)

    def mask_inactive_rows(value, active_rows, inactive_value):
        return torch.where(
            row_mask(value, active_rows),
            value,
            torch.full_like(value, inactive_value),
        )

    def assert_inactive_rows_masked(name, value, active_rows, inactive_value):
        inactive_values = torch.where(
            row_mask(value, active_rows),
            torch.full_like(value, inactive_value),
            value,
        )
        if bool(torch.any(inactive_values != inactive_value).item()):
            raise AssertionError(
                f"mixed-quad inactive rows retain a valid {name} descriptor"
            )

    swa_write_slots = mask_inactive_rows(
        swa["swa_write_slots"], rank_active_rows, -1
    )
    swa_sources = mask_inactive_rows(swa["swa_sources"], rank_active_rows, -1)
    swa_lens = mask_inactive_rows(swa["swa_lens"], rank_active_rows, 0)
    assert_inactive_rows_masked(
        "SWA write-slot", swa_write_slots, rank_active_rows, -1
    )
    assert_inactive_rows_masked(
        "SWA source", swa_sources, rank_active_rows, -1
    )
    assert_inactive_rows_masked("SWA length", swa_lens, rank_active_rows, 0)

    hca_values = dict(hca)
    for name in ("state_write_slots", "swa_write_slots", "cmp_slot_mapping"):
        hca_values[name] = mask_inactive_rows(
            hca_values[name], rank_active_rows, -1
        )
        assert_inactive_rows_masked(
            f"HCA {name}", hca_values[name], rank_active_rows, -1
        )
    hca_values["swa_sources"] = mask_inactive_rows(
        hca_values["swa_sources"], rank_active_rows, -1
    )
    assert_inactive_rows_masked(
        "HCA swa_sources", hca_values["swa_sources"], rank_active_rows, -1
    )

    # HCA's packed work tensors are independent of the row-local write-slot
    # tensors above.  Compact them explicitly so an inactive long request
    # cannot retain its 1M-history QK/PV work behind a zero active-row mask.
    # Tensor extents stay compile-time compatible; only the reachable prefix
    # and per-query offsets change at runtime.
    source_work_offsets = hca_values["hca_query_work_offsets"]
    source_work_query_ids = hca_values["hca_work_query_ids"]
    source_work_row_begin = hca_values["hca_work_row_begin"]
    source_work_valid_rows = hca_values["hca_work_valid_rows"]
    compact_work_offsets = torch.zeros_like(source_work_offsets)
    compact_work_query_ids = torch.full_like(source_work_query_ids, -1)
    compact_work_row_begin = torch.zeros_like(source_work_row_begin)
    compact_work_valid_rows = torch.zeros_like(source_work_valid_rows)
    if source_work_offsets.shape != (N_RANKS, T + 1):
        raise ValueError(
            "HCA query-work offsets must be rank-owned with T+1 entries"
        )
    for rank in range(N_RANKS):
        cursor = 0
        previous = 0
        for query in range(T):
            begin = int(source_work_offsets[rank, query].item())
            end = int(source_work_offsets[rank, query + 1].item())
            if (
                begin < previous
                or end < begin
                or end > source_work_query_ids.shape[1]
            ):
                raise ValueError(
                    "HCA query-work offsets are not a valid packed prefix"
                )
            previous = end
            if int(rank_active_rows[rank, query].item()) != 0:
                count = end - begin
                if cursor + count > compact_work_query_ids.shape[1]:
                    raise ValueError(
                        "active HCA work exceeds its packed tensor extent"
                    )
                if count:
                    destination = slice(cursor, cursor + count)
                    source = slice(begin, end)
                    compact_work_query_ids[rank, destination] = (
                        source_work_query_ids[rank, source]
                    )
                    compact_work_row_begin[rank, destination] = (
                        source_work_row_begin[rank, source]
                    )
                    compact_work_valid_rows[rank, destination] = (
                        source_work_valid_rows[rank, source]
                    )
                    cursor += count
            compact_work_offsets[rank, query + 1] = cursor
        if cursor and bool(
            torch.any(
                compact_work_query_ids[rank, :cursor]
                >= T
            ).item()
        ):
            raise ValueError("compacted HCA work retains an out-of-range query id")
    hca_values["hca_query_work_offsets"] = compact_work_offsets
    hca_values["hca_work_query_ids"] = compact_work_query_ids
    hca_values["hca_work_row_begin"] = compact_work_row_begin
    hca_values["hca_work_valid_rows"] = compact_work_valid_rows

    csa_values = dict(csa)
    for name in (
        "raw_write_slots",
        "main_state_write_slots",
        "inner_state_write_slots",
    ):
        csa_values[name] = mask_inactive_rows(
            csa_values[name], csa_active_rows, -1
        )
        assert_inactive_rows_masked(
            f"CSA {name}", csa_values[name], csa_active_rows, -1
        )
    csa_values["swa_sources"] = mask_inactive_rows(
        csa_values["swa_sources"], csa_active_rows, -1
    )
    assert_inactive_rows_masked(
        "CSA swa_sources", csa_values["swa_sources"], csa_active_rows, -1
    )

    event_query_ids = csa_values["event_query_ids"]
    if event_query_ids.shape != csa_values["main_event_write_slots"].shape:
        raise ValueError("CSA event-query and write-slot fixtures must have equal shapes")
    event_rows = torch.gather(
        csa_active_rows,
        2,
        event_query_ids.clamp(0, CSA_CHUNK_T - 1).to(dtype=torch.int64),
    )
    event_active = (
        (event_query_ids >= 0)
        & (event_query_ids < CSA_CHUNK_T)
        & (event_rows != 0)
    )
    for name in ("main_event_write_slots", "inner_event_write_slots"):
        csa_values[name] = torch.where(
            event_active,
            csa_values[name],
            torch.full_like(csa_values[name], -1),
        )
        inactive_events = torch.where(
            event_active,
            torch.full_like(csa_values[name], -1),
            csa_values[name],
        )
        if bool(torch.any(inactive_events != -1).item()):
            raise AssertionError(
                f"mixed-quad inactive CSA event retains a valid {name} slot"
            )

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

    specs = []
    add("x_ping", swa["x_hc"], output=True)
    add("x_pong", torch.zeros_like(swa["x_hc"]), output=True)
    add(
        "csa_x_hc_workspace",
        torch.zeros(
            N_RANKS,
            CSA_CHUNKS,
            CSA_CHUNK_T,
            HC_MULT,
            D,
            dtype=torch.float32,
        ),
        output=True,
    )

    for name in _MIXED_QUAD_COMMON_NAMES:
        if name not in swa or name not in csa or name not in hca:
            raise ValueError(f"mixed-quad fixture is missing common tensor {name}")
        # The host selects ``[rank, layer]`` for every common tensor.  A
        # stacked device tensor only permits a whole leading rank shard, so
        # keep this compact mixed fixture host-bound until that resident ABI
        # supports layer-axis selection (the raw/cache pools remain rank-owned
        # stacked allocations below).
        add(
            name,
            torch.stack(
                (swa[name], swa[name].clone(), csa[name], hca[name]), dim=1
            ),
        )

    add("swa_rope_cos", torch.stack((swa["rope_cos"], swa["rope_cos"].clone()), dim=1))
    add("swa_rope_sin", torch.stack((swa["rope_sin"], swa["rope_sin"].clone()), dim=1))

    swa_blocks = swa["kv_cache"].shape[1]
    csa_blocks = csa["kv_cache"].shape[1]
    hca_blocks = hca["kv_cache"].shape[1]
    swa_rows = swa_blocks * BLOCK_SIZE
    csa_rows = csa_blocks * BLOCK_SIZE
    raw_kv_pool = torch.cat(
        (
            swa["kv_cache"],
            swa["kv_cache"].clone(),
            csa["kv_cache"],
            hca["kv_cache"],
        ),
        dim=1,
    )
    add("raw_kv_pool", raw_kv_pool, output=True, resident="stacked")
    add(
        "swa_raw_write_slots",
        torch.stack(
            (
                shift_absolute_rows(swa_write_slots, 0),
                shift_absolute_rows(swa_write_slots, swa_rows),
            ),
            dim=1,
        ),
    )
    add(
        "swa_sources",
        torch.stack(
            (
                shift_absolute_rows(swa_sources, 0),
                shift_absolute_rows(swa_sources, swa_rows),
            ),
            dim=1,
        ),
    )
    add("swa_lens", torch.stack((swa_lens, swa_lens.clone()), dim=1))

    for name in _FULL43_HCA_FIXTURE_NAMES:
        if name in {"hca_swa_write_slots", "hca_swa_sources"}:
            continue
        source_name = _MIXED_QUAD_HCA_SOURCE_NAMES.get(
            name,
            name if name in hca_values else name.removeprefix("hca_"),
        )
        if source_name not in hca_values:
            raise ValueError(f"mixed-quad HCA source is missing {source_name} for {name}")
        add(
            name,
            hca_values[source_name],
            output=name in _MIXED_QUAD_INOUT_NAMES,
            resident="stacked" if source_name.endswith(("wkv", "wgate", "ape", "norm_w")) else None,
        )
    add(
        "hca_swa_write_slots",
        shift_absolute_rows(
            hca_values["swa_write_slots"], 2 * swa_rows + csa_rows
        ),
    )
    add(
        "hca_swa_sources",
        shift_absolute_rows(hca_values["swa_sources"], 2 * swa_rows + csa_rows),
    )

    for name in _FULL43_CSA_FIXTURE_NAMES:
        # This activation scratch is supplied explicitly above, not by a
        # Phase-E CSA fixture.  The remaining CSA names map to fixture fields.
        if name in {
            "csa_raw_write_slots",
            "csa_swa_sources",
            "csa_x_hc_workspace",
        }:
            continue
        source_name = _MIXED_QUAD_CSA_SOURCE_NAMES.get(
            name,
            name if name in csa_values else name.removeprefix("csa_"),
        )
        if source_name not in csa_values:
            raise ValueError(f"mixed-quad CSA source is missing {source_name} for {name}")
        add(
            name,
            csa_values[source_name],
            output=name in _MIXED_QUAD_INOUT_NAMES,
            resident=(
                "stacked"
                if (
                    source_name.endswith(("wkv", "wgate", "ape", "norm_w"))
                    or source_name in _MIXED_QUAD_CSA_RESIDENT_POOLS
                )
                else None
            ),
        )
    add(
        "csa_raw_write_slots",
        shift_absolute_rows(csa_values["raw_write_slots"], 2 * swa_rows),
    )
    add(
        "csa_swa_sources",
        shift_absolute_rows(csa_values["swa_sources"], 2 * swa_rows),
    )

    input_ids = torch.arange(T, dtype=torch.int64).remainder(64).unsqueeze(0)
    input_ids = input_ids.expand(N_RANKS, T).contiguous()
    input_ids = torch.where(
        rank_active_rows != 0,
        input_ids,
        torch.full_like(input_ids, INACTIVE_TOKEN_ID),
    )
    embed_weight = torch.arange(64 * D, dtype=torch.float32).reshape(64, D)
    embed_weight = embed_weight.to(dtype=torch.bfloat16)
    add("input_ids", input_ids)
    add("active_row_mask", rank_active_rows)
    add("embed_weight", rank_expand(embed_weight), resident="stacked")
    add("num_tokens_per_owner", swa["num_tokens_per_owner"])
    add(
        "pre_hc_hidden_out",
        torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32),
        output=True,
    )
    return specs


@pl.jit(auto_scope=False)
def decode_fwd_terminal_head(
    pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    hidden_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run HC head, final RMSNorm, and active-row LM head sampling."""
    x_head = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(pre_hc_hidden, hc_head_fn, hc_head_scale, hc_head_base, x_head)
    rms_norm(x_head, final_norm_w, hidden_out)
    lm_head_with_sampling(
        hidden_out,
        lm_head_weight,
        logit_row_indices,
        logits,
        sampled_ids,
        lm_head_hidden_window,
        lm_head_hidden_done,
        lm_head_logits_window,
        lm_head_logits_done,
        my_rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
        my_rank % LM_HEAD_TP_SIZE,
        LM_HEAD_COMM_EPOCH,
    )
    return hidden_out, logits, sampled_ids


@pl.jit(auto_scope=False)
def decode_fwd_terminal_head_active(
    pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    active_row_mask: pl.Tensor[[T], pl.INT32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the terminal head with row indices built from the active mask."""
    with pl.scope():
        logit_row_indices = pl.create_tensor([MAX_LOGIT_ROWS], dtype=pl.INT32)
        build_main_logit_rows(active_row_mask, logit_row_indices)
        # Keep the active variant self-contained: PyPTO does not compose an
        # opaque device entry from another device entry.  This preserves the
        # baseline HC-head/RMSNorm/LM-head math exactly before the public mask.
        x_head = pl.create_tensor([T, D], dtype=pl.BF16)
        hc_head(pre_hc_hidden, hc_head_fn, hc_head_scale, hc_head_base, x_head)
        rms_norm(x_head, final_norm_w, hidden_out)
        lm_head_with_sampling(
            hidden_out,
            lm_head_weight,
            logit_row_indices,
            logits,
            sampled_ids,
            lm_head_hidden_window,
            lm_head_hidden_done,
            lm_head_logits_window,
            lm_head_logits_done,
            my_rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
            my_rank % LM_HEAD_TP_SIZE,
            LM_HEAD_COMM_EPOCH,
        )
        # ``-1`` row indices suppress the hidden-state gather in the LM head,
        # but greedy sampling of the zero-filled inactive logits would
        # otherwise emit token 0.  Preserve the input-pack sentinel for every
        # inactive row.
        mask_inactive_main_samples(logit_row_indices, sampled_ids)
    return hidden_out, logits, sampled_ids
