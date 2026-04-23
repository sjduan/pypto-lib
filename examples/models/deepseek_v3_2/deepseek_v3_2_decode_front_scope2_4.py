# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

"""
DeepSeek V3.2-EXP decode front scope2 stage 4 (rotate_activation/Hadamard transform).

This module implements the Hadamard transformation for query and key tensors,
replacing the placeholder scalar Hadamard scale with the full matrix multiplication.

The Hadamard transform is applied after RoPE:
- q_idx_full [B, INDEX_HEADS * INDEX_HEAD_DIM] -> reshape to [B, INDEX_HEADS, INDEX_HEAD_DIM]
  then matmul with hadamard_q [INDEX_HEAD_DIM, INDEX_HEAD_DIM] per head
- k_idx [B, INDEX_HEAD_DIM] -> matmul with hadamard_k [INDEX_HEAD_DIM, INDEX_HEAD_DIM]

Reference: lightning_indexer_prolog_quant_impl.py lines 462-470 (Query-Hadamard) and 501-503 (Key-Hadamard).

Defaults are intentionally reduced for faster standalone validation.
"""

import pypto.language as pl

import os
os.environ.setdefault("PTO2_RING_TASK_WINDOW", "524288")
os.environ.setdefault("PTO2_RING_DEP_POOL", "1048576")
os.environ.setdefault("PTO2_RING_HEAP", "4294967296")


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
Q_LORA_RANK = 1536
QK_ROPE_HEAD_DIM = 64

INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_Q_OUT = INDEX_HEADS * INDEX_HEAD_DIM

EPS = 1e-6

CACHE_ROWS = BATCH * MAX_SEQ

# Tile sizes for Hadamard matmul
HADAMARD_K_CHUNK = 128  # K-dimension chunk for Hadamard matmul (head_dim tiles)
HADAMARD_N_CHUNK = 128  # N-dimension chunk for Hadamard matmul (head_dim tiles)
BATCH_TILE = 16
HEAD_TILE = 8  # Process heads in chunks


def build_deepseek_v3_2_decode_front_scope2_stage4_program(
    batch: int = BATCH,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
):
    BATCH_CFG = batch
    INDEX_HEADS_CFG = index_heads
    INDEX_HEAD_DIM_CFG = index_head_dim
    INDEX_Q_OUT_CFG = index_heads * index_head_dim

    HADAMARD_K_BLOCKS = (INDEX_HEAD_DIM_CFG + HADAMARD_K_CHUNK - 1) // HADAMARD_K_CHUNK
    HADAMARD_N_BLOCKS = (INDEX_HEAD_DIM_CFG + HADAMARD_N_CHUNK - 1) // HADAMARD_N_CHUNK

    @pl.program
    class DeepSeekV32DecodeFrontScope2Stage4:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope2_stage4(
            self,
            q_idx_full: pl.Tensor[[BATCH_CFG, INDEX_Q_OUT_CFG], pl.BF16],
            k_idx: pl.Tensor[[BATCH_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
            hadamard_q: pl.Tensor[[INDEX_HEAD_DIM_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
            hadamard_k: pl.Tensor[[INDEX_HEAD_DIM_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
            q_idx_out: pl.Tensor[[BATCH_CFG, INDEX_Q_OUT_CFG], pl.BF16],
            k_idx_out: pl.Tensor[[BATCH_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, INDEX_Q_OUT_CFG], pl.BF16]:
            q_hadamard_full = pl.create_tensor([BATCH_CFG, INDEX_Q_OUT_CFG], dtype=pl.BF16)
            k_hadamard = pl.create_tensor([BATCH_CFG, INDEX_HEAD_DIM_CFG], dtype=pl.BF16)

            # Stage 4a: Query Hadamard transformation.
            # q_idx_full is shaped as [B, INDEX_HEADS * INDEX_HEAD_DIM].
            # For each head h, slice q[b, h*D : (h+1)*D] and matmul with hadamard_q[D, D].
            # Result is assembled back into q_hadamard_full[b, h*D : (h+1)*D].
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for h in pl.parallel(0, INDEX_HEADS_CFG, 1, chunk=8):
                    h_offset = h * INDEX_HEAD_DIM_CFG
                    for nb in pl.range(HADAMARD_N_BLOCKS):
                        n0 = nb * HADAMARD_N_CHUNK
                        q_acc = pl.full([BATCH_TILE, HADAMARD_N_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HADAMARD_K_BLOCKS):
                            k0 = kb * HADAMARD_K_CHUNK
                            # Slice q for this head: [BATCH_TILE, K_CHUNK] from position [0, h*D + k0]
                            q_tile = pl.cast(
                                pl.slice(
                                    q_idx_full,
                                    [BATCH_TILE, HADAMARD_K_CHUNK],
                                    [0, h_offset + k0],
                                ),
                                target_type=pl.FP32,
                            )
                            hadamard_q_tile = pl.slice(
                                hadamard_q,
                                [HADAMARD_K_CHUNK, HADAMARD_N_CHUNK],
                                [k0, n0],
                            )
                            q_h_tile = pl.matmul(
                                pl.cast(q_tile, target_type=pl.BF16),
                                hadamard_q_tile,
                                out_dtype=pl.FP32,
                            )
                            q_acc = pl.add(q_acc, q_h_tile)
                        # Assemble into q_hadamard_full at position [0, h*D + n0]
                        q_hadamard_full = pl.assemble(
                            q_hadamard_full,
                            pl.cast(q_acc, target_type=pl.BF16),
                            [0, h_offset + n0],
                        )

            # Stage 4b: Key Hadamard transformation.
            # k_idx is shaped as [B, INDEX_HEAD_DIM], apply matmul with hadamard_k.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for nb in pl.parallel(0, HADAMARD_N_BLOCKS, 1, chunk=8):
                    n0 = nb * HADAMARD_N_CHUNK
                    k_acc = pl.full([BATCH_TILE, HADAMARD_N_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HADAMARD_K_BLOCKS):
                        k0 = kb * HADAMARD_K_CHUNK
                        k_tile = pl.cast(
                            pl.slice(k_idx, [BATCH_TILE, HADAMARD_K_CHUNK], [0, k0]),
                            target_type=pl.FP32,
                        )
                        hadamard_k_tile = pl.slice(
                            hadamard_k,
                            [HADAMARD_K_CHUNK, HADAMARD_N_CHUNK],
                            [k0, n0],
                        )
                        k_h_tile = pl.matmul(
                            pl.cast(k_tile, target_type=pl.BF16),
                            hadamard_k_tile,
                            out_dtype=pl.FP32,
                        )
                        k_acc = pl.add(k_acc, k_h_tile)
                    k_hadamard = pl.assemble(
                        k_hadamard,
                        pl.cast(k_acc, target_type=pl.BF16),
                        [0, n0],
                    )

            # Assemble outputs
            q_idx_out = pl.assemble(q_idx_out, q_hadamard_full, [0, 0])
            k_idx_out = pl.assemble(k_idx_out, k_hadamard, [0, 0])

            return q_idx_out

    return DeepSeekV32DecodeFrontScope2Stage4


def build_inputs(
    batch: int = BATCH,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
):
    import torch

    torch.manual_seed(4242)

    index_q_out = index_heads * index_head_dim

    # Generate Hadamard matrices (orthogonal-ish random matrices for testing)
    # In real usage, these would be pre-computed Hadamard matrices
    hadamard_q = (
        (torch.rand(index_head_dim, index_head_dim, dtype=torch.float32) - 0.5)
        / (index_head_dim ** 0.5)
    ).to(torch.bfloat16)
    hadamard_k = (
        (torch.rand(index_head_dim, index_head_dim, dtype=torch.float32) - 0.5)
        / (index_head_dim ** 0.5)
    ).to(torch.bfloat16)

    # Input tensors (simulating post-RoPE q and k)
    q_idx_full = (torch.rand(batch, index_q_out, dtype=torch.float32) - 0.5).to(torch.bfloat16)
    k_idx = (torch.rand(batch, index_head_dim, dtype=torch.float32) - 0.5).to(torch.bfloat16)

    # Output tensors (initialized to zero)
    q_idx_out = torch.zeros(batch, index_q_out, dtype=torch.bfloat16)
    k_idx_out = torch.zeros(batch, index_head_dim, dtype=torch.bfloat16)

    return (
        q_idx_full,
        k_idx,
        hadamard_q,
        hadamard_k,
        q_idx_out,
        k_idx_out,
    )


def golden_decode_front_scope2_stage4(tensors, params=None):
    del params

    import torch

    q_idx_full = tensors["q_idx_full"].float()
    k_idx = tensors["k_idx"].float()
    hadamard_q = tensors["hadamard_q"].float()
    hadamard_k = tensors["hadamard_k"].float()

    batch = q_idx_full.shape[0]
    index_head_dim = hadamard_q.shape[0]  # hadamard_q is [D, D], shared across all heads
    index_heads = q_idx_full.shape[1] // index_head_dim  # infer H from q_idx_full shape [B, H*D]

    # Query Hadamard: q_idx_full [B, H*D] -> reshape to [B, H, D] -> matmul per head
    q_view = q_idx_full.view(batch, index_heads, index_head_dim)
    # Apply Hadamard to each head: q_h[b, h, :] = q_view[b, h, :] @ hadamard_q
    q_hadamard = torch.einsum("bhd,dk->bhk", q_view, hadamard_q)
    q_out = q_hadamard.reshape(batch, index_heads * index_head_dim).to(torch.bfloat16)

    # Key Hadamard: k_idx [B, D] -> matmul with hadamard_k
    k_hadamard = k_idx @ hadamard_k
    k_out = k_hadamard.to(torch.bfloat16)

    tensors["q_idx_out"].copy_(q_out)
    tensors["k_idx_out"].copy_(k_out)


def compile_and_run(
    batch: int = BATCH,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    import time

    import torch

    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, RunResult, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    print(
        "Scope2 Stage4 profile:",
        {
            "batch": batch,
            "index_heads": index_heads,
            "index_head_dim": index_head_dim,
        },
    )

    program = build_deepseek_v3_2_decode_front_scope2_stage4_program(
        batch=batch,
        index_heads=index_heads,
        index_head_dim=index_head_dim,
    )

    (
        q_idx_full,
        k_idx,
        hadamard_q,
        hadamard_k,
        q_idx_out,
        k_idx_out,
    ) = build_inputs(
        batch=batch,
        index_heads=index_heads,
        index_head_dim=index_head_dim,
    )

    expected_tensors = {
        "q_idx_full": q_idx_full.detach().clone(),
        "k_idx": k_idx.detach().clone(),
        "hadamard_q": hadamard_q.detach().clone(),
        "hadamard_k": hadamard_k.detach().clone(),
        "q_idx_out": q_idx_out.detach().clone(),
        "k_idx_out": k_idx_out.detach().clone(),
    }
    golden_decode_front_scope2_stage4(expected_tensors, None)

    start = time.perf_counter()

    run(
        program,
        q_idx_full,
        k_idx,
        hadamard_q,
        hadamard_k,
        q_idx_out,
        k_idx_out,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=runtime_profiling,
        ),
    )
    execution_time = time.perf_counter() - start

    try:
        torch.testing.assert_close(q_idx_out, expected_tensors["q_idx_out"], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_idx_out, expected_tensors["k_idx_out"], rtol=1e-3, atol=1e-3)
    except AssertionError as exc:
        return RunResult(passed=False, error=str(exc), execution_time=execution_time)

    return RunResult(passed=True, execution_time=execution_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = compile_and_run(
        batch=BATCH,
        index_heads=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        platform=args.platform,
        device_id=args.device,
        dump_passes=True,
        runtime_profiling=args.runtime_profiling,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)