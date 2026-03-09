# Qwen3 Prefill/Decode AIV Splitting Analysis (Pass 08)

本文复核 `ExpandMixedKernel` 后（pass 08）两个程序在 AIV InCore 中的切分策略，重点回答：

1. `AIV_IDX` 是否把计算负载和 local tensor 数据量有效拆成两半；
2. duplicated 运算和 splitted 运算各有多少；
3. 切分后语义是否与 mixed kernel 切分前一致，是否存在计算错误风险。

分析输入：

- `examples/qwen3_32b_decode_dump/passes_dump/08_after_ExpandMixedKernel.py`
- `examples/qwen3_32b_prefill_dump/passes_dump/08_after_ExpandMixedKernel.py`

---

## 1) `AIV_IDX` 在 Pass 08 中的作用（证据）

在两个程序中，每个 mixed group 的 AIV 端函数都带有 `AIV_IDX` runtime 参数（取值 0/1）。  
`AIV_IDX` 的典型作用是：

- **分片读**：通过 `offset + AIV_IDX * stride` 读取一半输入；
- **分片写**：通过 `assemble(..., [base + AIV_IDX * stride, ...])` 写回各自负责的一半；
- **AIC/AIV 通信配对**：`tpush_to_aic(..., AIV_IDX)` / `tpop_from_aic(AIV_IDX)` 保证 lane0/lane1 各自回收自己的结果。

### Decode 中 `AIV_IDX` 的 stride 分布

- `*128`（8 次）：hidden/intermediate 维切半（256 -> 128）
- `*64`（1 次）：head_dim 切半（128 -> 64）
- `*60`（2 次）：seq tile 切半（120 -> 60）
- `*16`（1 次）：down matmul K 维切半（32 -> 16）
- `*2`（7 次）：token/batch 行切半（4 -> 2）

### Prefill 中 `AIV_IDX` 的 stride 分布

- `*128`（9 次）：hidden/intermediate 维切半
- `*64`（4 次）：head_dim 切半
- `*2`（9 次）：token/batch 行切半

结论：两程序都明确采用了 2-way SPMD（AIV0/AIV1）分担同一 kernel 的输入输出子块，结构上是“各干一半”。

---

## 2) Split vs Duplicate 运算计数

计数口径（静态语句级）：

- **splitted 运算**：AIV 函数体内 `pl.tensor.*` / `pl.comm.*` 且语句显式依赖 `AIV_IDX`；
- **duplicated 运算**：AIV 函数体内 `pl.tensor.*` / `pl.comm.*` 但语句不显式依赖 `AIV_IDX`（两 lane 都执行同构计算/控制）。

> 注：这是“语句计数”而非“动态执行次数”（动态次数还要乘循环 trip count）。

### Decode 计数

| AIV function | split_ops | duplicated_ops |
|---|---:|---:|
| `incore_0_aiv` | 8 | 8 |
| `incore_1_aiv` | 13 | 12 |
| `incore_2_aiv` | 6 | 23 |
| `incore_3_aiv` | 8 | 6 |
| `incore_4_aiv` | 5 | 2 |
| **TOTAL** | **40** | **51** |

### Prefill 计数

| AIV function | split_ops | duplicated_ops |
|---|---:|---:|
| `incore_0_aiv` | 8 | 8 |
| `incore_1_aiv` | 13 | 12 |
| `incore_2_aiv` | 7 | 39 |
| `incore_3_aiv` | 8 | 6 |
| `incore_4_aiv` | 7 | 5 |
| **TOTAL** | **43** | **70** |

为什么 duplicated 运算不少：

- Group2（attention）存在大量“每 lane 都要做”的控制/归约流程（online softmax 的 mi/li/oi 更新、loop 控制、中间 tensor create/mul）；
- duplicated 不等于重复错误，只要每 lane 处理的是本 lane 的 split 数据，或一个 lane 结果被明确 discard，即语义仍可能正确。

### 2.1 “确实重复浪费”与“分片不浪费”的分账

为回答“到底浪费了多少”这个问题，这里再做一次更细分口径：

- **未浪费（split-assigned）**：运算显式使用 `AIV_IDX`，两 lane 处理互斥切片（不是同一数据重复计算）。
- **确实重复浪费（confirmed waste）**：可以从 pass08 中明确证明“某 lane 计算结果被丢弃”。

当前可明确确认的“确实重复浪费”模式只有 1 类（decode/prefill 都有）：

- **Group2 Q-RoPE 路径**：AIV0/AIV1 都计算并 `tpush` `q_rot_bf16`，但 AIC 侧显式 `q_rot_bf16_0__discard = tpop_from_aiv(1)`，表示 lane1 该路结果不参与后续计算。
- 对应到 AIV 端，被丢弃结果的生产链是同一段 15 条操作（`view/cast/deep_view/.../assemble/cast/tpush`）。

据此给出“语句级静态”分账表：

| Program | 未浪费 split-assigned | 确实重复浪费（可确认） | duplicated 中其余未定性 |
|---|---:|---:|---:|
| Decode | 40 | 15 | 36 (= 51-15) |
| Prefill | 43 | 15 | 55 (= 70-15) |

说明：

- “其余未定性 duplicated”不等于浪费，里面混有大量必要控制流/归约操作；
- 这里“确实重复浪费=15”是**保守下界**（only confirmed），不是上界。

### 2.2 按 Group 拆分（`incore_0..4_aiv`）

下面把“未浪费 split / 可确认浪费 / duplicated 未定性”按 group 展开，便于定位优化优先级。

#### Decode（按 group）

| Group(AIV) | 未浪费 split-assigned | 确实重复浪费（可确认） | duplicated 中其余未定性 |
|---|---:|---:|---:|
| `incore_0_aiv` | 8 | 0 | 8 |
| `incore_1_aiv` | 13 | 0 | 12 |
| `incore_2_aiv` | 6 | 15 | 8 |
| `incore_3_aiv` | 8 | 0 | 6 |
| `incore_4_aiv` | 5 | 0 | 2 |
| **TOTAL** | **40** | **15** | **36** |

#### Prefill（按 group）

| Group(AIV) | 未浪费 split-assigned | 确实重复浪费（可确认） | duplicated 中其余未定性 |
|---|---:|---:|---:|
| `incore_0_aiv` | 8 | 0 | 8 |
| `incore_1_aiv` | 13 | 0 | 12 |
| `incore_2_aiv` | 7 | 15 | 24 |
| `incore_3_aiv` | 8 | 0 | 6 |
| `incore_4_aiv` | 7 | 0 | 5 |
| **TOTAL** | **43** | **15** | **55** |

注：

- “确实重复浪费=15”目前只在 `incore_2_aiv` 被确认，来源是 Q-RoPE 路径 lane1 结果被 AIC 显式 discard；
- 其余 group 目前未发现“结果被明确丢弃”的硬证据，因此归入“duplicated 未定性”。

### 2.3 “duplicated 未定性”的再细分 + 代码证据

为避免把正常 SPMD 并行误判成浪费，这里把“未定性”再分三类：

#### A) Lane-local 派生计算（通常不浪费）

定义：语句本身不写 `AIV_IDX`，但输入来自 `AIV_IDX` 分片读；两 lane 在算不同数据。  
结论：通常属于“结构 duplicated、数据不重复”。

```python
# decode: qwen3_decode_layer_incore_0_aiv
x_chunk_bf16_0 = pl.tensor.view(hidden_states_0, [4, 128], [b0_0, k0_7 + AIV_IDX * 128])
x_chunk_7 = pl.tensor.cast(x_chunk_bf16_0, target_type=pl.FP32, mode=2)
_t5 = pl.tensor.row_expand_mul(x_chunk_7, inv_rms_tile_0)   # 这句不含 AIV_IDX，但 x_chunk_7 已是分片数据
normed_0 = pl.tensor.col_expand_mul(_t5, gamma_0)
```

#### B) 控制/归约状态计算（通常必要开销）

定义：online softmax、loop/状态维护、临时 tensor 初始化等，两 lane 都执行同构流程。  
结论：通常是算法控制开销，不应直接算“重复浪费”。

```python
# decode: qwen3_decode_layer_incore_2_aiv
oi_0 = pl.tensor.create([1, 64], dtype=pl.FP32)
li_0 = pl.tensor.create([1, 1], dtype=pl.FP32)
mi_0 = pl.tensor.create([1, 1], dtype=pl.FP32)
oi_1 = pl.tensor.mul(oi_0, 0.0)
li_1 = pl.tensor.mul(li_0, 0.0)
mi_1 = pl.tensor.mul(mi_0, 0.0)
for sb_0, (li_iter_2, mi_iter_2, oi_iter_2) in pl.range(0, ctx_blocks_0, 1, init_values=(li_1, mi_1, oi_1)):
    ...
```

#### C) 运行时路径相关（需 trace 才能定性）

定义：是否执行/是否有效依赖运行时分支或循环末端条件。  
结论：静态文本不足以判断是浪费还是必要。

```python
# prefill: qwen3_prefill_layer_incore_4_aiv
if ob_3 == 100 - 1:
    _t60 = pl.tensor.deep_view(down_proj_tile_6, [2, 64], [0, d0_0])
    _t61 = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [2, 64], [0 + AIV_IDX * 2, d0_0])
    down_acc_0 = pl.tensor.add(_t60, _t61)
    out_9 = pl.tensor.assemble(out_iter_7_outer_l1, _t62, [b_0 + AIV_IDX * 2, p0_0, d0_0])
else:
    out_10 = pl.yield_(out_iter_7_outer_l1)
```

#### D) 可确认浪费（硬证据：结果被显式丢弃）

定义：某 lane 结果被 AIC 显式 `discard`，可直接判定为重复浪费。  
结论：这是当前报告中唯一“已确认浪费”来源（15 条生产链语句）。

```python
# decode: qwen3_decode_layer_incore_2_aic
q_rot_bf16_0 = pl.comm.tpop_from_aiv(0)
q_rot_bf16_0__discard = pl.comm.tpop_from_aiv(1)
pl.comm.tfree_to_aiv(1)
```

```python
# prefill: qwen3_prefill_layer_incore_2_aic
q_rot_bf16_0 = pl.comm.tpop_from_aiv(0)
q_rot_bf16_0__discard = pl.comm.tpop_from_aiv(1)
pl.comm.tfree_to_aiv(1)
```

小结：

- “duplicated 未定性”不是“已经确认浪费”，而是证据还不够；
- 目前只有 D 类（显式 discard）被确认为浪费并进入“15 条”统计；
- A/B/C 类需要结合数据流和 runtime trace 才能进一步压缩不确定区间。

---

## 3) 按 Group 的切分语义复核（Decode）

### Group0 / Group1 / Group3 / Group4

语义模式一致：

1. AIV 读取 half tile（如 `[2,*]` 或 `[* ,128]`）；
2. 通过 `AIV_IDX` 把 half 数据推到 AIC；
3. AIC 组装 full tile 做 matmul；
4. AIV 收 half 结果并按 `AIV_IDX` 写回 disjoint 区域（`+ AIV_IDX * 2` 等）。

这是标准 split-K 或 split-row 再合并流程，语义与切分前一致。

### Group2（attention）

- `k_tile/v_tile` 由 AIV 各取 `[60,128]`，AIC 组装 `[120,128]` 做 `q @ k^T` 与 `exp @ v`；
- `q` 路径中 AIC 从 AIV pop 两次，其中一路显式 `_discard`，用于双 lane 同步接口匹配；
- 对外结果仍按 head 维 assemble 回 `attn_row`。

结论：Decode 的 5 个 AIV group 在切分映射上自洽，未发现会直接破坏数学语义的异常索引。

---

## 4) 按 Group 的切分语义复核（Prefill）

### Group0 / Group1 / Group3 / Group4

与 decode 同类，切分映射是标准的 half-load / AIC-assemble / half-store，语义保持一致。

### Group2（RoPE + cache + attention）重点检查

> 以下两条是**修复前**（旧 pass08）的异常观察，用于保留问题历史。

发现一处**高风险不一致**：

- `k_cache` 写回：`assemble(..., [cache_row_0, 0])`（整行 128）；
- `v_cache` 写回：`assemble(..., [cache_row_0 + AIV_IDX * 64, 0])`（写入 `[1,64]` 但偏移到 row 维）。

从张量形状看，`v_cache` 目标是 `[CACHE_ROWS, 128]`。  
若写 `[1,64]` 的 half，直观应是列偏移（如 `[cache_row_0, AIV_IDX*64]`）或在 AIC 合成 `[1,128]` 后一次写整行；当前写法是 row 偏移，存在将同一 token 的 V 向量写到错误行的风险。

因此（修复前判断）：

- 对 prefill Group2，**目前不能严格确认“切分后语义完全等价且不会计算错误”**；
- 这处索引应作为优先级最高的核查点（建议对比 pass07/原始 mixed kernel 的 cache 写地址，或做最小 case 数值对拍）。

### 4.1 编译器根因定位（ExpandMixedKernel）

已在编译器源码中定位到直接根因，位于：

- `pypto/src/ir/transforms/expand_mixed_kernel_pass.cpp`
- `AIVSplitMutator::RewriteTensorAssemble`

问题本质：

- 该函数在给 `tensor.assemble` 注入 `AIV_IDX` 偏移时，原实现把偏移**硬编码在 offset[0]**：
  - `new_offset_elems[0] = MakeAivOffset(..., src_half_dim, ...)`
- 这隐含假设“所有 split 都发生在 axis 0”，但 Group2 的 `k/v/q` 子向量 split 发生在非 0 轴（如 head_dim 轴）。
- 因此会把本应列向拼接（axis=1）的半块，错误地变成行向偏移，导致 `v_cache` 写地址错位风险。

这与 prefill pass08 观察到的异常完全一致：

- `k_cache` 以整行写回（看起来正常）；
- `v_cache` 以 `[cache_row_0 + AIV_IDX * 64, 0]` 写回（行偏移可疑）。

### 4.2 修复策略与已实施修改

修复原则：`assemble` 的 `AIV_IDX` 偏移必须沿**真实 split_axis**，不能固定 axis 0。

已实施代码修复（同一文件）：

1. 引入 `VarSplitMeta {split_axis, half_dim}`，替代仅记录 `half_dim` 的旧状态；
2. 在 `RewriteTensorView/Create/Reshape/ElementWise/TpopFromAic` 中保存每个变量的 split 轴信息；
3. 在 `RewriteTensorAssemble` 中按 source tensor 的 `split_axis` 改写 offset：
   - 从 `new_offset_elems[0] = ...`
   - 改为 `new_offset_elems[src_split_axis] = ...`
4. 增加 axis 边界检查，避免 offset 维度不足时误改写。

修复后预期：

- 对 axis=0 的 split 行为保持不变；
- 对 axis=1（或其他轴）split，偏移落在正确维度，避免 `v_cache` 这类错位写回。

### 4.3 为什么 ExpandMixedKernel 会出现这个错误

从实现演进看，这属于“先支持主路径、后补通用轴处理”的典型问题：

- pass9d 的 AIC 双路 tpush/tpop 主要按 axis0 half/assemble 实现；
- pass9c 引入 chain-level split 后，实际 split_axis 已可能不是 0；
- 但 `RewriteTensorAssemble` 仍沿用早期 axis0 假设，形成了“分析支持任意轴，改写仍固定轴”的不一致。

即：**能力模型已升级到任意 split 轴，具体改写点没同步升级**。

### 4.4 重跑验证结果（修复后）

已执行：

- 重装编译器：`pip install -e .`（`pypto`）
- 重跑程序：
  - `examples/qwen3_32b_decode.py`
  - `examples/qwen3_32b_prefill.py`

从新生成的 prefill pass08（`examples/qwen3_32b_prefill_dump/passes_dump/08_after_ExpandMixedKernel.py`）可见：

```python
k_cache_9 = pl.tensor.assemble(k_cache_iter_7_outer_l1, _t23, [cache_row_0, 0])
v_cache_9 = pl.tensor.assemble(v_cache_iter_7_outer_l1, _t24, [cache_row_0, 0 + AIV_IDX * 64])
```

这说明 `v_cache` 偏移已从“错误的 row 轴偏移”修正为“列轴偏移”，与 split-axis 语义一致。  
因此，本文前述 prefill Group2 的高风险写回偏移问题已被修复。

附加说明（本轮运行状态）：

- 两个程序在 codegen 阶段都报：`No codegen registered for operation: comm.aic_initialize_pipe`；
- 但 pass dump 已成功写出，所以可以完成本次“pass08 级别”的修复验证；
- 该 codegen 报错是独立问题，不影响本次 `RewriteTensorAssemble` 修复是否生效的结论。

---

## 5) 总结结论

1. **负载/数据二分是否有效？**  
   是。两个程序都广泛使用 `AIV_IDX` 做 2-way 切分，AIV0/AIV1 各负担一半 local tile（hidden、seq、intermediate 等维度）。

2. **运算数量分账（语句级）**  
   - Decode: 未浪费 split 40；确实重复浪费 15；其余 duplicated 未定性 36  
   - Prefill: 未浪费 split 43；确实重复浪费 15；其余 duplicated 未定性 55

3. **语义一致性是否确认无误？**  
   - Decode: 可确认切分语义保持一致，未见明显错误映射。  
   - Prefill: Group2 的 `v_cache` 写回偏移问题在重跑后已修复（`[cache_row_0, 0 + AIV_IDX * 64]`）；pass08 级语义一致性已恢复。

---

## 6) 建议的后续验证（最小闭环）

1. 解决 codegen 缺失 `comm.aic_initialize_pipe` 注册（当前阻断端到端编译/运行）；
2. 在 codegen 问题解除后，做小规模数值对拍（`BATCH=1, MAX_SEQ=16, NUM_HEADS=8`）；
3. 对比 prefill/decode 的 `k_cache/v_cache` 写地址与高层预期，补一份地址 trace 附录；
4. 将本次修复加入 `ExpandMixedKernel` 的回归测试（覆盖非 axis0 split 的 `tensor.assemble` 改写）。
