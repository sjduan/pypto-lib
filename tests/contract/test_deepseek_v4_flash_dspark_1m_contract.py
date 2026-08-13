# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Main-only contracts for the 1M D-Spark decode implementation."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO / "models" / "deepseek_v4_flash_dspark"
sys.path.insert(0, str(_MODEL_DIR))

import config  # noqa: E402
import context_geometry as geometry  # noqa: E402


_REMOVED_MAIN_EXCLUSIONS = (
    "decode_cache_transaction.py",
    "decode_fwd_mtp.py",
    "decode_fwd_terminal_validation.py",
    "decode_input_pack.py",
    "decode_mtp.py",
    "decode_mtp_verify.py",
    "long_context_validation.py",
)


def _source(name: str) -> str:
    return (_MODEL_DIR / name).read_text()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def _annotation_text(node: ast.FunctionDef, name: str) -> str:
    """Return one public parameter annotation in normalized AST form."""
    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        if argument.arg == name:
            if argument.annotation is None:
                raise AssertionError(f"{node.name}.{name} has no annotation")
            return ast.unparse(argument.annotation)
    raise AssertionError(f"missing {node.name}.{name} annotation")


def _tensor_shape(annotation: ast.AST) -> list[ast.AST] | None:
    """Extract the shape list from a ``Tensor``/``DistributedTensor`` annotation."""
    for child in ast.walk(annotation):
        if not isinstance(child, ast.Subscript):
            continue
        kind = ast.unparse(child.value).split(".")[-1]
        if kind not in {"Tensor", "DistributedTensor"}:
            continue
        payload = child.slice
        if isinstance(payload, ast.Tuple):
            payload = payload.elts[0]
        if isinstance(payload, ast.List):
            return payload.elts
    return None


def _annotation_ndim(annotation: ast.AST) -> int:
    shape = _tensor_shape(annotation)
    return len(shape) if shape is not None else 0


def _function_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (
            (isinstance(child.func, ast.Name) and child.func.id == name)
            or (
                isinstance(child.func, ast.Attribute) and child.func.attr == name
            )
        )
    ]


def _scope_calls(node: ast.AST) -> set[str]:
    """Names of direct calls enclosed by at least one ``with pl.scope``."""
    scoped: set[str] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "scope"
            for item in candidate.items
        ):
            continue
        for child in ast.walk(candidate):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                scoped.add(child.func.id)
    return scoped


def _contains_name(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id in names for child in ast.walk(node)
    )


def test_main_capacity_is_b16_s8_t128_at_one_million() -> None:
    assert config.MAX_CONTEXT_TOKENS == 1_048_576
    assert config.CACHE_BLOCK_SIZE == config.BLOCK_SIZE == 32
    assert config.DECODE_LOCAL_REQUESTS == 16
    assert config.DECODE_SEQ == 8
    assert config.DECODE_MAIN_ROWS_PER_RANK == 128
    assert config.DECODE_RANK_TOKENS == 128


def test_attention_capacity_is_not_a_runtime_length_profile() -> None:
    assert config.SWA_WINDOW_ROWS == 128
    assert config.MAX_HCA_ROWS == 8192
    assert config.MAX_HCA_SHARDS == 64
    assert config.MAX_CSA_CANDIDATES == 262144
    assert config.MAX_CSA_LEAVES == 128
    assert config.CSA_MAX_NODES_PER_QUERY == 255
    assert config.CSA_TOPK == 512
    source = _source("config.py")
    for forbidden in ("LENGTH_PROFILE", "CONTEXT_BUCKET", "SEQ_BUCKET"):
        assert forbidden not in source


def test_global_pool_capacity_is_not_partitioned_by_batch() -> None:
    assert config.SWA_KV_POOL_BLOCKS == 4
    assert config.HCA_KV_POOL_BLOCKS == 256
    assert config.CSA_MAIN_POOL_PAGES == 8192
    assert config.CSA_IDX_POOL_PAGES == 8192
    for name in (
        "SWA_KV_POOL_BLOCKS",
        "HCA_KV_POOL_BLOCKS",
        "CSA_MAIN_POOL_PAGES",
        "CSA_IDX_POOL_PAGES",
    ):
        assignment = next(
            node
            for node in ast.parse(_source("config.py")).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        )
        assert "DECODE_BATCH" not in ast.unparse(assignment.value)


def test_main_layer_plan_is_exactly_two_swa_twenty_one_csa_twenty_hca() -> None:
    plan = geometry.build_forward_layer_plan()
    assert len(plan) == 43
    assert tuple(layer.model_layer_id for layer in plan) == tuple(range(43))
    kinds = tuple(layer.attention_kind for layer in plan)
    assert kinds[:4] == ("swa", "swa", "csa", "hca")
    assert kinds.count("swa") == 2
    assert kinds.count("csa") == 21
    assert kinds.count("hca") == 20


def test_main_rows_are_compact_not_padded_to_context() -> None:
    cases = (
        ("full_active", 128),
        ("ragged_127", 127),
        ("single_active", 1),
        ("all_inactive", 0),
    )
    for case, active_rows in cases:
        fixture = geometry.build_phase_g4_main_step_case(case)
        assert fixture.step.active_query_count == active_rows
        assert fixture.step.request_query_offsets[-1] == active_rows
        assert len(fixture.token_ids) == active_rows


def test_one_m_tail_clamps_work_without_clamping_positions() -> None:
    fixture = geometry.build_phase_g4_main_step_case("one_m_tail")
    assert fixture.step.candidate_lane_counts[:9] == tuple(range(8, -1, -1))
    assert fixture.step.context_exhausted[8]
    assert all(
        lane.position < config.MAX_CONTEXT_TOKENS
        for lane in fixture.step.candidate_lanes
    )


def test_hca_and_csa_submit_exact_runtime_work() -> None:
    step = geometry.build_forward_step_geometry(
        (0, 127, config.MAX_CONTEXT_TOKENS - 8),
        requested_candidate_lanes=(0, 1, 8),
    )
    plan = geometry.build_forward_layer_plan()
    counters = geometry.build_per_layer_work_counters(
        step,
        layers=(plan[0], plan[2], plan[3]),
    )
    assert counters[0].active_rows == 9
    assert counters[1].csa_leaves < 9 * config.MAX_CSA_LEAVES
    assert counters[2].hca_shards < 9 * config.MAX_HCA_SHARDS


def test_hca_streaming_frontier_bounds_long_context_scratch() -> None:
    tree = ast.parse(_source("decode_sparse_attn_hca.py"))
    entry = _function(tree, "sparse_attn_hca")
    chunk = _function(tree, "sparse_attn_hca_chunk_heads")
    entry_source = ast.unparse(entry)
    chunk_source = ast.unparse(chunk)

    assert "HCA_QUERY_CHUNK_T = S" in _source("decode_sparse_attn_hca.py")
    assert "for chunk in pl.range(chunk_count)" in entry_source
    assert "with pl.scope():" in entry_source
    assert "work_begin_i32 = pl.read(hca_query_work_offsets" in entry_source
    assert "query_base + item" in chunk_source
    assert "work_base + item - t_dim" in chunk_source
    assert "- work_base" in chunk_source
    assert "pl.slice" not in entry_source
    assert "[item_count * ATTN_K_TILE, HEAD_DIM]" in chunk_source

    max_chunk_queries = config.DECODE_SEQ
    max_chunk_items = max_chunk_queries * (1 + config.MAX_HCA_SHARDS)
    per_buffer_bytes = (
        max_chunk_items
        * config.HCA_ROWS_PER_SHARD
        * config.FLASH.head_dim
        * 2
    )
    packed_scope_bytes = 2 * per_buffer_bytes
    partial_scope_bytes = per_buffer_bytes
    assert packed_scope_bytes < 2 * 1024**3
    assert partial_scope_bytes < 2 * 1024**3

    fixture_source = _source("decode_hca.py")
    assert "row += HCA_ROWS_PER_SHARD" in fixture_source
    assert "min(HCA_ROWS_PER_SHARD, visible_rows - row)" in fixture_source
    assert "row += BLOCK_SIZE" not in fixture_source


def test_csa_forest_capacity_is_exact_at_one_million() -> None:
    counts = geometry.csa_task_workspace_counts([config.MAX_CSA_LEAVES])
    forest = geometry.build_csa_binary_forest([config.MAX_CSA_LEAVES])
    assert counts.n_leaves == 128
    assert counts.n_merges == 127
    assert counts.n_nodes == 255
    assert forest.n_leaves == 128
    assert forest.n_merges == 127
    assert len(forest.root_slots) == 1


def test_attention_metadata_keeps_only_main_entries() -> None:
    source = _source("decode_metadata.py")
    for entry in (
        "phase_b_swa_metadata",
        "phase_c_hca_metadata",
        "phase_d_csa_metadata",
    ):
        assert f"def {entry}(" in source
    for forbidden in (
        "build_mtp_layer_plan",
        "SpeculativeUndoWriteSlots",
        "build_speculative_event_write_slots",
        "build_accepted_prefix_next_state",
        "decode_cache_transaction",
    ):
        assert forbidden not in source


def test_topk_implementation_is_bounded_and_branch_free() -> None:
    source = _source("decode_indexer_topk.py")
    for entry in (
        "select_2k_top512",
        "merge2_top512",
        "active_score_topk_forest",
    ):
        assert f"def {entry}(" in source
    assert "CSA_CANDIDATES_PER_LEAF" in source
    assert "CSA_TOPK_READY_FRONTIER_W" in source
    assert "MAX_CONTEXT_TOKENS" not in source


def test_csa_chunk_pair_arena_is_static_and_does_not_escape_scope() -> None:
    """The nested-inline chunk helper must not reference a module-scope DynVar.

    Phase H closes the EP4 Full43 ConvertToSSA blocker
    (``Variable 'LAYER_CSA_ARENA_DYN' used outside its defining scope``) by
    replacing the escaped ``CSA_ARENA_DYN`` annotation on
    ``decode_layer_csa_chunk.pair_arena`` with a static compile-time bound.
    A module-scope ``pl.dynamic`` symbol that is never ``bind_dynamic``-ed
    inside the inline boundary escapes its IR scope after inline expansion.
    """
    source = _source("decode_layer.py")
    tree = ast.parse(source)
    chunk = _function(tree, "decode_layer_csa_chunk")

    # The production nested-inline ABI must not carry the escaped DynVar.
    assert "CSA_ARENA_DYN" not in source
    assert "LAYER_CSA_ARENA_DYN" not in source

    # The arena bound is a static expression fixed by the chunk/request caps.
    assert "CSA_CHUNK_ARENA_ROWS = CSA_CHUNK_T * CSA_MAX_NODES_PER_QUERY" in source
    assert "assert CSA_CHUNK_ARENA_ROWS == 4080" in source

    pair_arena = _annotation_text(chunk, "pair_arena")
    assert "CSA_CHUNK_ARENA_ROWS" in pair_arena
    assert "CSA_ARENA_DYN" not in pair_arena

    # The descriptor extents (leaf/pair/singleton/upper) stay dynamic: only the
    # arena scratch is static, never the visible candidate/merge frontier.
    for name, dyn in (
        ("leaf_descriptors", "CSA_LEAF_DYN"),
        ("pair_descriptors", "CSA_PAIR_DYN"),
        ("singleton_descriptors", "CSA_SINGLETON_DYN"),
        ("upper_descriptors", "CSA_UPPER_DYN"),
    ):
        annotation = _annotation_text(chunk, name)
        assert dyn in annotation, name

    # The enclosing layer allocates the arena with the same static shape so the
    # annotation and the created tensor agree after inline expansion.
    layer = _function(tree, "decode_layer_csa")
    layer_source = ast.unparse(layer)
    assert (
        "pl.create_tensor([CSA_CHUNK_ARENA_ROWS, CSA_PAIR_WIDTH]"
        in layer_source
    )

    # The public standalone diagnostic entry also uses the static chunk-wide
    # arena bound: it calls the ``indexer`` inline helper, whose ``pair_arena``
    # annotation is now static ``TOPK_ARENA_ROWS``.  A standalone DynVar that is
    # never ``bind_dynamic``-ed would escape the inline expansion it drives.
    standalone = _function(tree, "csa_indexer_layer_stage")
    standalone_arena = _annotation_text(standalone, "pair_arena")
    assert "TOPK_ARENA_ROWS" in standalone_arena
    assert "CSA_INDEXER_ARENA_DYN" not in standalone_arena


def test_active_topk_batches_the_long_context_submission_frontier() -> None:
    source = _source("decode_indexer_topk.py")
    tree = ast.parse(source)
    active = _function(tree, "active_score_topk_forest")
    active_source = ast.unparse(active)
    for name in (
        "pair_grid_count",
        "singleton_grid_count",
        "upper_grid_count",
    ):
        assert name in active_source
    assert "pl.spmd(pair_grid_count" in active_source
    assert "pl.spmd(singleton_grid_count" in active_source
    assert "pl.spmd(upper_grid_count" in active_source
    assert "for group" not in active_source
    assert "for singleton" not in active_source
    assert "init_values=(node_tids" not in active_source


def test_layer_surface_has_only_main_attention_kinds() -> None:
    source = _source("decode_layer.py")
    for entry in (
        "decode_layer_swa",
        "decode_layer_hca",
        "decode_layer_csa",
        "decode_layer_csa_chunk",
        "csa_layer_frontend",
        "csa_main_compressor_layer_stage",
        "csa_inner_compressor_layer_stage",
        "csa_indexer_layer_stage",
        "csa_sparse_value_layer_stage",
        "csa_layer_finalize",
        "csa_layer_moe",
    ):
        assert f"def {entry}(" in source
    assert "decode_mtp" not in source
    assert "decode_attention_cp" not in source


def test_csa_layer_restores_one_rank_one_l2_submission() -> None:
    source = _source("decode_layer.py")
    tree = ast.parse(source)
    entry = _function(tree, "l3_decode_layer_csa")
    child = _function(tree, "decode_layer_csa")
    chunk = _function(tree, "decode_layer_csa_chunk")

    entry_calls = _called_names(entry)
    assert "decode_layer_csa" in entry_calls
    assert not {
        "csa_layer_frontend",
        "csa_main_compressor_layer_stage",
        "csa_inner_compressor_layer_stage",
        "csa_indexer_layer_stage",
        "csa_sparse_value_layer_stage",
        "csa_layer_finalize",
        "csa_layer_moe",
    } & entry_calls
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_layer_csa"
        for node in ast.walk(entry)
    ) == 1

    entry_source = ast.unparse(entry)
    assert "CSA_LAYER_SHARDS" not in entry_source
    for name in (
        "x_hc",
        "rope_cos",
        "main_state_page_ids",
        "idx_pages",
        "raw_write_slots",
        "csa_x_attn_workspace",
    ):
        parameter = next(arg for arg in entry.args.args if arg.arg == name)
        assert "N_RANKS, CSA_CHUNKS" in ast.unparse(parameter.annotation)

    decorators = {ast.unparse(node) for node in child.decorator_list}
    assert "pl.jit(auto_scope=False)" in decorators
    assert {ast.unparse(node) for node in chunk.decorator_list} == {
        "pl.jit.inline"
    }
    child_calls = _called_names(child)
    assert {"decode_layer_csa_chunk", "moe"} <= child_calls
    assert not {
        "csa_layer_frontend",
        "csa_main_compressor_layer_stage",
        "csa_inner_compressor_layer_stage",
        "csa_indexer_layer_stage",
        "csa_sparse_value_layer_stage",
        "csa_layer_finalize",
        "csa_layer_moe",
    } & child_calls
    dead_split_workspaces = {
        "pair_arena",
        "csa_x_mixed_workspace",
        "csa_x_normed_workspace",
        "csa_q_workspace",
        "csa_current_kv_workspace",
        "csa_qr_workspace",
        "csa_qr_scale_workspace",
        "csa_post_workspace",
        "csa_comb_workspace",
        "csa_main_overlay_workspace",
        "csa_inner_overlay_workspace",
        "csa_query_vectors_workspace",
        "csa_query_scales_workspace",
        "csa_query_weights_workspace",
        "csa_topk_scores_workspace",
        "csa_topk_indices_workspace",
        "csa_attn_out_workspace",
    }
    assert not dead_split_workspaces & {
        argument.arg for argument in entry.args.args
    }
    assert not dead_split_workspaces & {
        argument.arg for argument in child.args.args
    }
    assert "pl.Out" in _annotation_text(entry, "csa_x_attn_workspace")
    assert "pl.Out" in _annotation_text(child, "csa_x_attn_workspace")


def test_full43_is_one_decorated_direct_device_graph() -> None:
    tree = ast.parse(_source("decode_fwd_full43.py"))
    entry = _function(tree, "l3_decode_fwd_full43")
    child = _function(tree, "l3_decode_fwd_full43_rank")
    decorators = {ast.unparse(node) for node in entry.decorator_list}
    assert "pl.jit.host" in decorators
    calls = _called_names(entry)
    assert {"l3_decode_fwd_full43_rank"} <= calls
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "l3_decode_fwd_full43_rank"
        for node in ast.walk(entry)
    ) == 1
    assert "pld.alloc_window_buffer" in ast.unparse(entry)

    child_calls = _called_names(child)
    required = {
        "decode_fwd_pack_active_inline",
        "decode_layer_swa_inline",
        "decode_layer_csa_inline",
        "decode_layer_hca_inline",
        "decode_fwd_terminal_head_active_inline",
        "decode_fwd_clear_shared_moe_signals_inline",
        "mark_full43_program_completion_inline",
    }
    assert required <= child_calls
    assert not {
        "csa_layer_frontend",
        "csa_main_compressor_layer_stage",
        "csa_inner_compressor_layer_stage",
        "csa_indexer_layer_stage",
        "csa_sparse_value_layer_stage",
        "csa_layer_finalize",
        "csa_layer_moe",
        "pld.alloc_window_buffer",
        "pld.window",
    } & child_calls
    assert not any(
        keyword.arg == "device"
        for call in ast.walk(child)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
    )

    # The host owns one shared communication-domain allocation and submits one
    # rank-local child per world rank.  The child itself must not create a
    # second window domain or use a device callback.
    assert not {"alloc_window_buffer", "window"} & child_calls
    rank_loops = [
        node
        for node in ast.walk(entry)
        if isinstance(node, ast.For)
        and "world_size" in ast.unparse(node.iter)
    ]
    assert len(rank_loops) == 1
    assert any(
        _function_calls(loop, "l3_decode_fwd_full43_rank") for loop in rank_loops
    )
    rank_loop_line = rank_loops[0].lineno
    assert all(
        call.lineno < rank_loop_line
        for call in _function_calls(entry, "alloc_window_buffer")
    )
    alloc_targets = [
        target.id
        for node in ast.walk(entry)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "alloc_window_buffer"
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert alloc_targets
    assert len(alloc_targets) == len(set(alloc_targets))
    assert {
        "arrived_buf",
        "data_arrived_buf",
        "routed_y_buf_buf",
        "combine_arrived_buf",
    } <= set(alloc_targets)
    for loop in rank_loops:
        assert not _function_calls(loop, "alloc_window_buffer")


def test_attention_and_head_scope_ownership_is_explicit() -> None:
    """Scope lifetimes belong to the baseline layer/head implementations."""
    layer_tree = ast.parse(_source("decode_layer.py"))
    for function_name, attention_name in (
        ("decode_layer_swa", "attention_swa"),
        ("decode_layer_hca", "attention_hca"),
    ):
        function = _function(layer_tree, function_name)
        scoped = _scope_calls(function)
        assert attention_name in scoped
        assert "moe" in scoped
        assert sum(
            isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "scope"
                for item in node.items
            )
            for node in ast.walk(function)
        ) == 2

    csa = _function(layer_tree, "decode_layer_csa")
    csa_scoped = _scope_calls(csa)
    assert "decode_layer_csa_chunk" in _called_names(csa)
    assert "moe" in csa_scoped
    assert sum(
        isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "scope"
            for item in node.items
        )
        for node in ast.walk(csa)
    ) == 2

    fwd_tree = ast.parse(_source("decode_fwd.py"))
    head = _function(fwd_tree, "decode_fwd_terminal_head_active")
    head_scopes = [
        node
        for node in ast.walk(head)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "scope"
            for item in node.items
        )
    ]
    assert len(head_scopes) == 1
    scoped_head_calls = _scope_calls(head)
    assert {
        "build_main_logit_rows",
        "hc_head",
        "rms_norm",
        "lm_head_with_sampling",
        "mask_inactive_main_samples",
    } <= scoped_head_calls


def test_full43_persistent_pools_are_packed_and_rank_local() -> None:
    source = _source("decode_fwd_full43.py")
    tree = ast.parse(source)
    host = _function(tree, "l3_decode_fwd_full43")
    child = _function(tree, "l3_decode_fwd_full43_rank")
    child_source = ast.unparse(child)
    child_submissions = _function_calls(host, "l3_decode_fwd_full43_rank")
    assert len(child_submissions) == 1
    child_submission = child_submissions[0]
    persistent = (
        "raw_kv_pool",
        "hca_compress_state",
        "hca_cmp_kv",
        "csa_main_state",
        "csa_main_cache",
        "csa_inner_state",
        "csa_idx_cache",
        "csa_idx_scale",
    )
    packed_axes = {
        "raw_kv_pool": "FWD_PACKED_RAW_BLOCKS_DYN",
        "hca_compress_state": "FWD_PACKED_HCA_STATE_BLOCKS_DYN",
        "hca_cmp_kv": "FWD_PACKED_HCA_CMP_BLOCKS_DYN",
        "csa_main_state": "FWD_PACKED_CSA_MAIN_STATE_BLOCKS_DYN",
        "csa_main_cache": "FWD_PACKED_CSA_MAIN_BLOCKS_DYN",
        "csa_inner_state": "FWD_PACKED_CSA_INNER_STATE_BLOCKS_DYN",
        "csa_idx_cache": "FWD_PACKED_CSA_IDX_ROWS_DYN",
        "csa_idx_scale": "FWD_PACKED_CSA_IDX_ROWS_DYN",
    }
    for pool, layer_count in (
        ("raw_kv_pool", "MAIN_LAYER_COUNT"),
        ("hca_compress_state", "HCA_FULL_LAYERS"),
        ("hca_cmp_kv", "HCA_FULL_LAYERS"),
        ("csa_main_state", "CSA_FULL_LAYERS"),
        ("csa_main_cache", "CSA_FULL_LAYERS"),
        ("csa_inner_state", "CSA_FULL_LAYERS"),
        ("csa_idx_cache", "CSA_FULL_LAYERS"),
    ):
        assert f"pl.tensor.dim({pool}, 0) // {layer_count}" in child_source
    for name in persistent:
        host_annotation = _annotation_text(host, name)
        child_annotation = _annotation_text(child, name)
        assert "N_RANKS" in host_annotation, name
        assert "N_RANKS" not in child_annotation, name
        assert "pl.InOut" in host_annotation, name
        assert "pl.InOut" in child_annotation, name
        assert packed_axes[name] in host_annotation, name
        assert packed_axes[name] in child_annotation, name
        assert _annotation_ndim(ast.parse(host_annotation, mode="eval").body) <= 5
        assert _annotation_ndim(ast.parse(child_annotation, mode="eval").body) <= 5
        rank_arg = next(
            keyword.value
            for keyword in child_submission.keywords
            if keyword.arg == name
        )
        assert isinstance(rank_arg, ast.Subscript), name
        assert isinstance(rank_arg.value, ast.Name), name
        assert rank_arg.value.id == name, name
        assert ast.unparse(rank_arg.slice) == "rank", name

        # ``pl.slice(pool, shape, offsets)`` is intentionally kept as a
        # direct call: unlike a subscript or a compact-cache reshape, this
        # preserves the packed block/row ABI while selecting one layer.
        assert any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "slice"
            and len(call.args) >= 3
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == name
            for call in ast.walk(child)
        ), name

    # Persistent block/row storage is sliced directly.  The old compact-cache
    # path first selected a 3-D cache and restored a singleton row axis with a
    # reshape; that silently changed the public ABI and is forbidden here.
    pool_names = set(persistent)
    layer_aliases = {
        "raw_layer",
        "hca_state_layer",
        "hca_cmp_kv_layer",
        "csa_main_state_layer",
        "csa_main_cache_layer",
        "csa_inner_state_layer",
        "csa_idx_cache_layer",
        "csa_idx_scale_layer",
    }
    for call in _function_calls(child, "reshape"):
        names = {
            node.id
            for node in ast.walk(call)
            if isinstance(node, ast.Name)
        }
        assert not names & pool_names
        assert not names & layer_aliases
        assert not any("layer_flat" in name for name in names)

    # Every rank-local persistent root is passed through in place.  No child
    # return tuple may make the host rebind an entire pool after the call.
    for node in ast.walk(child):
        if isinstance(node, ast.Return) and isinstance(
            node.value, (ast.Tuple, ast.List)
        ):
            assert not _contains_name(node.value, pool_names)
    for node in ast.walk(host):
        if isinstance(node, ast.Return) and isinstance(
            node.value, (ast.Tuple, ast.List)
        ):
            assert not _contains_name(node.value, pool_names)
    for node in ast.walk(host):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
            node.value, ast.Call
        ):
            continue
        if not (
            isinstance(node.value.func, ast.Name)
            and node.value.func.id == "l3_decode_fwd_full43_rank"
        ):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        assert not isinstance(target, (ast.Tuple, ast.List))


def test_full43_has_exact_layer_epochs_and_one_shared_clear() -> None:
    source = _source("decode_fwd_full43.py")
    tree = ast.parse(source)
    child = _function(tree, "l3_decode_fwd_full43_rank")
    assert "MAIN_LAYER_COUNT == 43" in source
    assert "CSA_FULL_LAYERS = 21" in source
    assert "HCA_FULL_LAYERS = 20" in source
    assert "TWO_SWA_LAYERS == 2" in source
    assert "for csa_ordinal in pl.range(CSA_FULL_LAYERS)" in source
    assert "csa_ordinal * 2 + 3" in source
    assert "csa_ordinal * 2 + 4" in source
    child_source = ast.unparse(child)
    assert "pl.const(0, pl.INT32)" in child_source
    assert "pl.const(1, pl.INT32)" in child_source
    assert "pl.const(2, pl.INT32)" in child_source
    assert "csa_moe_epoch = pl.cast(csa_ordinal * 2 + 3, pl.INT32)" in child_source
    assert "hca_moe_epoch = pl.cast(csa_ordinal * 2 + 4, pl.INT32)" in child_source
    clear_calls = _function_calls(
        child, "decode_fwd_clear_shared_moe_signals_inline"
    )
    assert len(clear_calls) == 1


def test_full43_public_tensor_arity_is_bounded() -> None:
    tree = ast.parse(_source("decode_fwd_full43.py"))
    for function_name in ("l3_decode_fwd_full43", "l3_decode_fwd_full43_rank"):
        function = _function(tree, function_name)
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ):
            if argument.annotation is None:
                continue
            assert _annotation_ndim(argument.annotation) <= 5, (
                function_name,
                argument.arg,
                ast.unparse(argument.annotation),
            )


def test_full43_pool_and_slot_admission_contract_is_exposed() -> None:
    source = _source("decode_fwd_full43.py")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    # Keep the helper naming intentionally semantic: a rename that removes
    # either admission check should fail this contract instead of silently
    # accepting a malformed packed pool or local slot map.
    assert "validate_full43_packed_pool_contract" in names
    assert "build_full43_packed_pool_layout" in names
    assert "validate_full43_local_slots" in names
    layout = _function(tree, "build_full43_packed_pool_layout")
    layout_source = ast.unparse(layout)
    assert "for ordinal in range(layer_count)" in layout_source
    validate = _function(tree, "validate_full43_packed_pool_contract")
    validate_source = ast.unparse(validate).lower()
    assert "divis" in validate_source
    assert "total % layer_count" in validate_source
    slot_validate = _function(tree, "validate_full43_local_slots")
    slot_source = ast.unparse(slot_validate).lower()
    assert "slot" in slot_source
    assert "active_mask" in slot_source
    assert "values < -1" in slot_source
    assert "values >= int(per_layer_capacity)" in slot_source
    for case in ("packed_pool_sentinel", "long_context_tail"):
        assert case in source
    specs = _function(tree, "build_full43_device_specs")
    specs_source = ast.unparse(specs)
    assert "'raw_kv_pool': KV_ORI_BLOCK_NUM" in specs_source
    assert "MAIN_LAYER_COUNT * KV_ORI_BLOCK_NUM" in specs_source
    for case in ("packed_pool_sentinel", "long_context_tail"):
        assert case in specs_source


def test_full43_runtime_uses_the_profiled_ring_heap() -> None:
    """Planning and execution must agree on the four ring extents."""
    source = _source("decode_fwd_full43.py")
    assert "DECODE_FULL43_RING_HEAP = (" in source
    assert "4096 * _MIB, 4096 * _MIB, 256 * _MIB" in source
    assert "DECODE_FULL43_RING_TASK_WINDOW = 262144" in source
    assert "DECODE_FULL43_RING_DEP_POOL = 524288" in source
    assert '"ring_task_window": DECODE_FULL43_RING_TASK_WINDOW' in source
    assert '"ring_heap": DECODE_FULL43_RING_HEAP' in source
    assert '"ring_dep_pool": DECODE_FULL43_RING_DEP_POOL' in source


def test_full43_reuses_stage_owned_attention_to_moe_workspaces() -> None:
    """Dynamic layers must reuse bounded, stage-owned root-ring bridges."""
    source = _source("decode_fwd_full43.py")
    tree = ast.parse(source)
    child = _function(tree, "l3_decode_fwd_full43_rank")
    child_source = ast.unparse(child)
    for name in (
        "swa0_attention_workspace_rank",
        "swa1_attention_workspace_rank",
        "csa_attention_workspace_rank",
        "hca_attention_workspace_rank",
    ):
        assert child_source.count(f"{name} = pl.create_tensor") == 1
    assert "csa_x_attn_rank = pl.reshape(csa_attention_workspace_rank" in child_source
    assert "csa_x_attn_rank = pl.create_tensor" not in child_source
    assert "num_tokens_per_owner, swa0_attention_workspace_rank" in child_source
    assert "num_tokens_per_owner, swa1_attention_workspace_rank" in child_source
    assert "num_tokens_per_owner, hca_attention_workspace_rank" in child_source
    layer_source = _source("decode_layer.py")
    assert "x_attn = x_attn_workspace" in layer_source


def test_full43_fixture_compacts_hca_work_for_inactive_rows() -> None:
    """Inactive long requests must not retain packed HCA attention work."""
    source = _source("decode_fwd.py")
    tree = ast.parse(source)
    builder = _function(tree, "build_mixed_quad_device_specs")
    builder_source = ast.unparse(builder)
    assert "source_work_offsets = hca_values['hca_query_work_offsets']" in builder_source
    assert "if int(rank_active_rows[rank, query].item()) != 0" in builder_source
    assert "hca_values['hca_query_work_offsets'] = compact_work_offsets" in builder_source
    assert "hca_values['hca_work_query_ids'] = compact_work_query_ids" in builder_source
    assert "hca_values['hca_work_valid_rows'] = compact_work_valid_rows" in builder_source


def test_phase_d_empty_page_fixture_preserves_descriptor_rank() -> None:
    """An all-inactive EP rank still exposes the canonical ``[0, 2]`` ABI."""
    source = _source("decode_indexer.py")
    tree = ast.parse(source)
    builder = _function(tree, "build_phase_d_indexer_specs")
    builder_source = ast.unparse(builder)
    assert "dtype=torch.int32).reshape(-1, 2)" in builder_source


def test_full43_has_rank_and_layer_owned_persistent_axes() -> None:
    """The L3 owns rank; its child owns packed layer pools locally."""
    source = _source("decode_fwd_full43.py")
    tree = ast.parse(source)
    host = _function(tree, "l3_decode_fwd_full43")
    child = _function(tree, "l3_decode_fwd_full43_rank")
    for name in (
        "raw_kv_pool",
        "hca_compress_state",
        "hca_cmp_kv",
        "csa_main_state",
        "csa_main_cache",
        "csa_inner_state",
        "csa_idx_cache",
        "csa_idx_scale",
    ):
        assert "N_RANKS" in _annotation_text(host, name)
        assert "N_RANKS" not in _annotation_text(child, name)
        assert "FWD_PACKED_" in _annotation_text(child, name)
    assert "for csa_ordinal in pl.range(CSA_FULL_LAYERS)" in source
    assert "if csa_ordinal < HCA_FULL_LAYERS" in source
    assert "CSA_FULL_LAYERS = 21" in source
    assert "HCA_FULL_LAYERS = 20" in source


def test_full43_and_shared_forward_do_not_reference_deferred_paths() -> None:
    combined = _source("decode_fwd.py") + _source("decode_fwd_full43.py")
    lowered = combined.lower()
    for forbidden in (
        "decode_mtp",
        "decode_fwd_mtp",
        "decode_attention_cp",
        "decode_o_projection_cp",
        "speculative",
        "accepted_prefix",
        "undo_log",
    ):
        assert forbidden not in lowered


def test_deferred_new_modules_are_removed() -> None:
    for name in _REMOVED_MAIN_EXCLUSIONS:
        assert not (_MODEL_DIR / name).exists()


def test_context_geometry_selftest_passes() -> None:
    geometry.selftest()
