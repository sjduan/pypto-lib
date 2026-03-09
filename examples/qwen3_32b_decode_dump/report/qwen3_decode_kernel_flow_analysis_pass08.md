# Qwen3-32B Decode Kernel Flow Analysis (Pass 08)

基于 `passes_dump/08_after_ExpandMixedKernel.py`，当前 decode 已展开为 5 个 mixed kernel group + 1 个 solo InCore function：

- `qwen3_decode_layer_incore_0_group` (AIC+AIV): Q projection
- `qwen3_decode_layer_incore_1_group` (AIC+AIV): K/V projection
- `qwen3_decode_layer_incore_2_group` (AIC+AIV): RoPE + attention (Q RoPE on AIV, scores/oi on AIC)
- `qwen3_decode_layer_incore_3_group` (AIC+AIV): O projection + residual
- `qwen3_decode_layer_incore_4_group` (AIC+AIV): MLP down projection
- `qwen3_decode_layer_incore_5` (solo InCore): final residual add + output assemble

---

## 1) Top-Level Flow (Orchestration)

```mermaid
flowchart TD
    A[qwen3_decode_layer]
    A --> B["Scope-A: sq_sum + inv_rms (full batch, inline)"]
    B --> C["for b0 in range(0,16,4) — BATCH_TILE loop"]
    C --> D["Group0: incore_0_group ×20 (Q proj)"]
    D --> E["Group1: incore_1_group ×4 (K/V proj)"]
    E --> F["for b in parallel(0,16,chunk=4)"]
    F --> G["Scope-B: K RoPE + cache write (orchestration)"]
    G --> H["Group2: incore_2_group ×8 (Q RoPE + attention)"]
    H --> I["assemble attn_out[b]"]
    I --> J["for b0 in range(0,16,4) — output scope"]
    J --> K["Group3: incore_3_group ×10 (O proj + residual)"]
    K --> L["Scope-C: post-RMS sq_sum + inv_rms (inline)"]
    L --> M["Scope-D: post_norm_tile (inline)"]
    M --> N["MLP gate/up matmul + SwiGLU (orchestration, ×800)"]
    N --> O["Group4: incore_4_group ×20 (down proj)"]
    O --> P["incore_5 ×20 (final residual + output write)"]
    P --> Q[out]
```

### Key Structural Differences from Prefill

| Aspect | Decode | Prefill |
|--------|--------|---------|
| Token count | 1 per session (fixed) | 1..MAX_SEQ per session (variable) |
| Outer dynamic loop | None — batch loop is fixed | `for p0_idx in range(tok_blocks)` |
| Scope 1 (Q/K/V proj) | Single `auto_incore` over full batch | Per-token-tile `auto_incore` |
| Scope 2 (attention) | K RoPE at orchestration; Q RoPE in Group2 AIV | All RoPE + cache in Group2 AIV |
| MLP gate/up | Orchestration-level matmul (small [4,32] tiles) | Orchestration-level matmul ([4,256] tiles) |
| Final output | Solo InCore (incore_5) | Fused into Group4 (conditional `ob==MLP_OUT_BLOCKS-1`) |

---

## 2) Detailed Orchestration Trace

### Phase 1: RMSNorm + Q/K/V Projection

```
sq_sum: [16,1] ← inline loop kb=0..19, row_sum(x² chunks)
inv_rms: [16,1] ← rsqrt(sq_sum * HIDDEN_INV + EPS)

for b0 in range(0, 16, 4):             # BATCH_TILE=4
    inv_rms_tile = view(inv_rms, [4,1], [b0, 0])

    for ob_0_out in range(20):          # Q_OUT_BLOCKS/4 = 80/4 = 20
        call_group(incore_0_group)      # Q proj: inner parallel 4, total 80 chunks
                                        # AIV: load x_chunk [4,128], gamma, norm → push normed+wq to AIC
                                        # AIC: assemble [4,256] normed + [256,64] wq → matmul → push [2,64] to AIV
                                        # AIV: accumulate q_acc [2,64] over 20 kb → assemble q_proj

    for ob_1_out in range(4):           # KV_OUT_BLOCKS/8 = 32/8 = 4
        call_group(incore_1_group)      # K/V proj: inner parallel 8, total 32 chunks
                                        # AIV: load x_chunk, gamma, norm → push normed+wk+wv to AIC
                                        # AIC: matmul(normed, wk), matmul(normed, wv) → push to AIV
                                        # AIV: accumulate k_acc, v_acc → assemble k_proj, v_proj
```

### Phase 2: RoPE + Cache Update + Attention

```
for b in parallel(0, 16, chunk=4):
    ctx_len = tensor.read(seq_lens, [b])
    pos = ctx_len - 1
    ctx_blocks = ⌈ctx_len / 120⌉
    cos/sin views at position pos

    for kvh in parallel(0, 8, chunk=4):     # K RoPE + cache write (orchestration level)
        k_row = cast(view(k_proj, [1,128], [b, kv_col]))
        k_rot = RoPE(k_row, cos, sin)
        k_cache[b, kvh, pos] = cast(k_rot, BF16)
        v_cache[b, kvh, pos] = view(v_proj, [1,128], [b, kv_col])

    for h_0_out in range(8):                # 64 heads / 8 inner = 8 outer
        call_group(incore_2_group)          # Q RoPE + attention
                                            # AIV: Q RoPE → push q_rot to AIC
                                            # AIV: load k_tile [60,128], v_tile [60,128] → push to AIC
                                            # AIC: assemble [120,128] tiles, matmul(q, k^T), scores
                                            # AIV: softmax workaround (scores_valid, exp_pad)
                                            #      push exp_pad to AIC
                                            # AIC: matmul(exp_pad, v) → oi_tmp
                                            # AIV: online mi/li/oi update
                                            # AIV: ctx = oi / li → assemble attn_row

    attn_out[b] = assemble(attn_out, attn_row, [b, 0])
```

### Phase 3: Output Projection + MLP + Output

```
for b0 in range(0, 16, 4):
    for ob_4_out in range(10):              # Q_OUT_BLOCKS/8 = 80/8 = 10
        call_group(incore_3_group)          # O proj + residual
                                            # AIV: load attn_out chunk + wo chunk → push to AIC
                                            # AIC: matmul → push to AIV
                                            # AIV: accumulate o_acc + residual → assemble resid1_tile

    sq_sum, inv_rms (inline post-RMS)
    post_norm_tile (inline norm + assemble)

    for ob_5 in range(800):                 # MLP_OUT_BLOCKS = 25600/32 = 800
        gate_acc, up_acc = 0
        for kb in range(20):                # orchestration-level matmul
            matmul(post_chunk [4,256], wg [256,32]) → gate_acc
            matmul(post_chunk [4,256], wu [256,32]) → up_acc
        sigmoid = 1 / (1 + exp(-gate_acc))
        mlp_chunk = gate_acc * sigmoid * up_acc
        mlp_chunk_bf16 = cast(mlp_chunk, BF16)

        for dob_0_out in range(20):         # Q_OUT_BLOCKS/4 = 80/4 = 20
            call_group(incore_4_group)      # down proj
                                            # AIV: load w_down [16,64] → push to AIC
                                            # AIC: assemble [32,64] w_down, matmul(mlp [4,32], w_down)
                                            # AIC: push [2,64] → AIV
                                            # AIV: down_prev + partial → assemble down_proj_tile

    for ob_6_out in range(20):              # Q_OUT_BLOCKS/4 = 80/4 = 20
        incore_5(...)                       # solo InCore (no AIC/AIV split)
            down_acc = view(down_proj_tile) + view(resid1_tile)
            out[b0, o0] = cast(down_acc, BF16)
```

---

## 3) Group-by-Group Flow Charts

### Group 0: `qwen3_decode_layer_incore_0_group` — Q Projection

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop ob_0_out=0..19 / ob_0_in=0..3 (parallel)
        loop kb=0..19
            V0->>C: wq_chunk [128,64] + normed [2,256]
            V1->>C: wq_chunk [128,64] + normed [2,256]
            Note over C: assemble [256,64] wq + [4,256] normed
            Note over C: matmul [4,256]×[256,64] → [4,64]
            C->>V0: q partial [2,64]
            C->>V1: q partial [2,64]
            Note over V0,V1: q_acc += partial
        end
        Note over V0,V1: assemble q_proj_tile[b0+AIV*2, q0]
    end
```

### Group 1: `qwen3_decode_layer_incore_1_group` — K/V Projection

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop ob_1_out=0..3 / ob_1_in=0..7 (parallel)
        loop kb=0..19
            V0->>C: normed [2,256] + wk [128,32] + wv [128,32]
            V1->>C: normed [2,256] + wk [128,32] + wv [128,32]
            Note over C: assemble full tiles
            Note over C: matmul(normed,wk) + matmul(normed,wv)
            C->>V0: k partial [2,32] + v partial [2,32]
            C->>V1: k partial [2,32] + v partial [2,32]
            Note over V0,V1: k_acc += k_part, v_acc += v_part
        end
        Note over V0,V1: assemble k_proj/v_proj[b0+AIV*2, kv0]
    end
```

### Group 2: `qwen3_decode_layer_incore_2_group` — RoPE + Attention

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop h_0_out=0..7 / h_0_in=0..7 (parallel, 64 heads)
        Note over V0,V1: Q RoPE: view q_proj → lo/hi split → cos/sin multiply → assemble q_rot
        V0->>C: q_rot_bf16 [1,128]
        V1->>C: q_rot_bf16 [1,128] (discarded)
        loop sb=0..ctx_blocks-1 (dynamic)
            V0->>C: k_tile [60,128] + v_tile [60,128]
            V1->>C: k_tile [60,128] + v_tile [60,128]
            Note over C: assemble [120,128] k/v tiles
            Note over C: scores = matmul(q_rot, k^T) × scale
            Note over V0,V1: scores_valid view → row_max/row_sum
            Note over V0,V1: exp_pad = zeros + assemble(exp_scores)
            V0->>C: exp_pad [1,60]
            V1->>C: exp_pad [1,60]
            Note over C: assemble [1,120] exp_pad
            Note over C: oi_tmp = matmul(exp_pad, v_tile)
            Note over V0,V1: online softmax: mi/li/oi update
        end
        Note over V0,V1: ctx = oi/li → assemble attn_row[0, q_col]
    end
```

### Group 3: `qwen3_decode_layer_incore_3_group` — O Projection + Residual

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop ob_4_out=0..9 / ob_4_in=0..7 (parallel)
        loop kb=0..19
            V0->>C: attn chunk [2,256] + wo chunk [128,64]
            V1->>C: attn chunk [2,256] + wo chunk [128,64]
            Note over C: assemble + matmul [4,256]×[256,64]
            C->>V0: o partial [2,64]
            C->>V1: o partial [2,64]
            Note over V0,V1: o_acc += partial
        end
        Note over V0,V1: resid = hidden_states[b0+AIV*2, o0]
        Note over V0,V1: resid1_tile[AIV*2, o0] = o_acc + resid
    end
```

### Group 4: `qwen3_decode_layer_incore_4_group` — MLP Down Projection

```mermaid
sequenceDiagram
    participant V0 as AIV_0
    participant C as AIC
    participant V1 as AIV_1
    loop dob_0_out=0..19 / dob_0_in=0..3 (parallel)
        V0->>C: w_down chunk [16,64]
        V1->>C: w_down chunk [16,64]
        Note over C: assemble [32,64] w_down
        Note over C: matmul(mlp_chunk [4,32], w_down [32,64])
        C->>V0: down partial [2,64]
        C->>V1: down partial [2,64]
        Note over V0,V1: down_prev + partial → assemble down_proj_tile
    end
```

### Solo InCore 5: `qwen3_decode_layer_incore_5` — Final Residual + Output

```mermaid
sequenceDiagram
    participant IC as InCore (no AIC/AIV split)
    loop ob_6_out=0..19 / ob_6_in=0..3 (parallel)
        Note over IC: down_acc = view(down_proj_tile, [4,64], [0,o0])
        Note over IC: + view(resid1_tile, [4,64], [0,o0])
        Note over IC: out[b0, o0] = cast(down_acc, BF16)
    end
```

---

## 4) AIC/AIV Split Dimensions (SPMD 2-way)

| Group | AIC tile | AIV_0 tile | AIV_1 tile | Split axis |
|-------|---------|-----------|-----------|------------|
| Group 0 (Q proj) | normed [4,256], wq [256,64] | normed [2,256], wq [128,64] | normed [2,256], wq [128,64] | rows (batch) + rows (hidden) |
| Group 1 (K/V proj) | normed [4,256], wk/wv [256,32] | normed [2,256], wk/wv [128,32] | normed [2,256], wk/wv [128,32] | rows (batch) + rows (hidden) |
| Group 2 (attention) | k_tile [120,128], v_tile [120,128] | k_tile [60,128], v_tile [60,128] | k_tile [60,128], v_tile [60,128] | rows (seq) |
| Group 3 (O proj) | attn [4,256], wo [256,64] | attn [2,256], wo [128,64] | attn [2,256], wo [128,64] | rows (batch) + rows (hidden) |
| Group 4 (down) | w_down [32,64] | w_down [16,64] | w_down [16,64] | rows (intermediate) |
| incore_5 (solo) | N/A — no split | — | — | — |

---

## 5) Notes

- 当前 pass 8 结果下，gate/up MLP 计算（800 个 [4,32] matmul）保持在 orchestration 层级，因为 `MLP_OUT_CHUNK=32` 较小，未被分拆为 AIC+AIV group。
- `incore_5`（final residual + output write）也未被分拆，作为 solo InCore 函数运行——仅做简单的 add + cast + assemble。
- `incore_2_group` 仍是最复杂的 kernel（RoPE、cache tile DMA、两次 matmul、online softmax），是性能调优的优先点。
- K RoPE + cache write 在 orchestration 层完成（decode 只有 1 个新 token，不值得放入 kernel），而 Q RoPE 在 Group 2 的 AIV 中完成。
- 与 prefill 相比，decode 的 Group 4 输出和 final residual 是分离的（Group 4 → incore_5），而 prefill 将它们 fused 在 Group 4 内（通过 `ob==MLP_OUT_BLOCKS-1` 条件）。
