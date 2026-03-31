# Reduce 类算子开发方法论

---

## 目录

- [零、API 使用规范](#零api-使用规范)
- [一、算子分析与设计](#一算子分析与设计)
  - [1.1 设计原则](#11-设计原则)
  - [1.2 识别 Reduce 维度和模式](#12-识别-reduce-维度和模式)
  - [1.3 多核切分策略](#13-多核切分策略)
  - [1.4 UB 切分策略](#14-ub-切分策略)
  - [1.5 处理模式判定](#15-处理模式判定)
  - [1.6 任意轴规约设计](#16-任意轴规约设计)
  - [1.7 非对齐场景处理](#17-非对齐场景处理)
- [二、分支详细实现](#二分支详细实现)
  - [2.1 分支导航](#21-分支导航)
  - [2.2 Tiling 参数计算原则](#22-tiling-参数计算原则)
- [三、关键设计说明](#三关键设计说明)
- [四、分支对照表](#四分支对照表)
- [五、AR vs ARA 模板对比](#五-ar-vs-ara-模板对比)
- [六、Reduce API 使用规范](#六reduce-api-使用规范)
- [七、常见问题与解决方案](#七常见问题与解决方案)

---

## 零、API 使用规范

> **快速参考**: [guide.md](guide.md)

### ⛔️ 禁止使用的 API

```cpp
WholeReduce*  /  BlockReduce*  /  PairReduce*  /  RepeatReduce*
```

### ⛔️ 数据搬运 API 黑名单

| API | 禁止场景 | 原因 |
|-----|---------|------|
| `DataCopy` | **GM ↔ UB 数据搬运** | 不支持非对齐数据，易导致隐蔽 bug |
| `GlobalTensor::SetValue` | 生产代码 | 效率极低，仅调试可用 |
| `GlobalTensor::GetValue` | 生产代码 | 效率极低，仅调试可用 |

### ✅ 强制使用的 API

```cpp
// 数据搬运：统一使用 DataCopyPad
AscendC::DataCopyPadParams padParams;
AscendC::DataCopyPad(dstLocal, srcGm, 
    {1, static_cast<uint16_t>(dataBytes), 0, 0}, padParams);
```

### ✅ 推荐使用的 API

```cpp
// 高阶 API - 任意轴归约
AscendC::ReduceMax<T, Pattern::Reduce::RA/AR>(dst, src, tmp, srcShape, isInit);
AscendC::ReduceSum<T, Pattern::Reduce::RA/AR>(dst, src, tmp, srcShape, isInit);

// 简化形式 - 仅支持 -1 轴，无对齐要求
AscendC::ReduceSum<T>(dst, src, tmp, count);
AscendC::ReduceMax<T>(dst, src, tmp, count);
```

**Pattern 说明**:
- `RA` - 沿第一维（行方向）Reduce，输入 (R, C) → 输出 (C,)
- `AR` - 沿最后一维（列方向）Reduce，输入 (R, C) → 输出 (R,)

---

## 一、算子分析与设计

### 1.1 设计原则

**核心原则：设计时必须考虑所有分支场景**

Reduce 类算子需要覆盖多种输入情况，设计阶段必须预先考虑：

```
设计时必须覆盖的场景：

1. 模板分支
   ├── AR 模板 (A0=1)
   └── ARA 模板 (A0>1)

2. 载入模式
   ├── 全载：UB 能容纳完整数据块
   └── 分载：需要分 chunk 处理

3. 任意轴规约
   └── 通过 3D 抽象统一为 AR/ARA 模板

4. 数据对齐
   ├── 32 字节对齐：直接使用
   └── 非对齐：使用 DataCopyPad 填充

5. 边界情况
   ├── 小 R：R ≤ threshold
   └── 大 R：R > threshold（需要 Split）
```

**设计检查清单**:
- [ ] 支持 AR 和 ARA 两种模板
- [ ] 处理全载和分载模式
- [ ] 支持任意 axis 参数
- [ ] 处理非 32 字节对齐的 A0
- [ ] 边界值测试覆盖

**代码组织建议**:
```
operator/
├── operator_ar_fullload.h    - AR 全载分支
├── operator_ar_colsplit.h    - AR 分载分支
├── operator_ara_fullload.h   - ARA 全载分支
├── operator_ara_rowsplit.h   - ARA 分载分支
└── operator.h                - 主入口，根据条件调用各分支
```

### 1.2 识别 Reduce 维度和模式

```
输入 Tensor: (d0, d1, ..., d_{axis}, ..., d_{n-1})
                    ↑
                Reduce 维度 R

抽象为 3D: (A1, R, A0)
  - A1 = d0 × ... × d_{axis-1}     (Reduce 前的维度乘积)
  - R  = d_{axis}                   (Reduce 维度)
  - A0 = d_{axis+1} × ... × d_{n-1} (Reduce 后的维度乘积)
```

**模板分类**:
| A0 值 | 模板名称 | Shape | 说明 |
|------|---------|-------|------|
| A0=1 | **AR 模板** | (A1, R) | 每行连续，归约为标量 |
| A0>1 | **ARA 模板** | (A1, R, A0) | 每列不连续（间隔 A0），归约为向量 |

### 1.3 多核切分策略

```
总任务 = A1 × A0Outer (每个 tile 处理 (R, A0Inner) 的数据块)

切分原则：
1. 负载均衡：每个核处理的 tile 数尽量相等
2. 数据局部性：相邻 tile 尽量分配给同一核
3. 粒度适中：tile 不能太小（调度开销），不能太大（并行度低）
```

### 1.4 UB 切分策略

```
UB 容量：
├── A2/A3 服务器: 192KB
└── A5 服务器: 248KB

UB 切分依据：
1. R_chunk_size: 单次能处理的 R 行数
2. A0Inner: 单次能处理的 A0 列数
3. 对齐要求: 32 字节对齐，buffer 按 a0TileBase 对齐
```

**Buffer 优化原则**:
- 生命周期分析：识别 buffer 的使用时机，判断是否可复用
- 减少Buffer数量：分析不同Buffer的使用时机，如果生命周期不重叠，可以复用
- 节省空间：复用 buffer 可节省 UB 空间，容纳更多数据

**GM↔UB 双向拷贝都必须使用 `DataCopyPad`**:
```cpp
// GM→UB
DataCopyPad(xLocal, xGm[offset], copyParams, padParams)
// UB→GM
DataCopyPad(yGm[offset], yLocal, copyParams)
```

### 1.5 处理模式判定

```
模板类型（由 A0 决定）：
├── A0 = 1 → AR 模板
└── A0 > 1 → ARA 模板

载入模式（由 UB 容量决定）：
├── 全载：UB 能容纳「加载 + 完整计算」一个 tile 所需的全部 buffer
└── 分载：UB 不能容纳，需要分 chunk 处理

全载条件（关键）：
├── AR 模板: UB 能容纳整行 + 计算过程所有 buffer
└── ARA 模板: UB 能容纳 R×a0TileBase + 计算过程所有 buffer

注意：全载判定要看 kernel 中实际 buffer 使用量，包括：
├── 输入 buffer（inQueue）
├── 输出 buffer（outQueue）
└── 计算过程中的中间 buffer（tmpBuf 等）

分载模式切分方向：
├── AR 模板 分载 → Col-Split（沿列方向切分，行分 chunk）
└── ARA 模板 分载 → Row-Split（沿行方向切分，R 分 chunk）
```

### 1.6 任意轴规约设计

**设计原则**：通过 Host 侧 3D 抽象，将任意轴归约统一为 AR 或 ARA 模板处理。

**关键认知**：
- 只有最后一维（axis=-1）的 A0=1，走 AR 模板
- 其他所有轴的 A0>1，走 ARA 模板

**示例推导**：shape = (A, B, C, D, E, F)

| axis | 轴名 | A1 | R | A0 | 分支 |
|------|-----|----|----|-----|------|
| -1 (5) | F | A×B×C×D×E | F | 1 | AR |
| -2 (4) | E | A×B×C×D | E | F | ARA |
| -3 (3) | D | A×B×C | D | E×F | ARA |
| 0 | A | 1 | A | B×C×D×E×F | ARA |

### 1.7 非对齐场景处理

**解决方案**：使用 `DataCopyPad` 进行自动填充。

```cpp
// 计算对齐后的列数
uint32_t alignedCols = ((a0Count * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// DataCopyPad 自动填充到 32 字节边界
AscendC::DataCopyExtParams copyParams;
copyParams.blockCount = rCount;
copyParams.blockLen = a0Count * sizeof(float);
copyParams.srcStride = (A0 - a0Count) * sizeof(float);

AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

// 后续 API 使用 alignedCols 而非 a0Count
uint32_t srcShape[] = {rCount, alignedCols};
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(dst, xLocal, tmpLocal, srcShape, true);
```

**注意事项**:
1. Reduce带pattern的接口中 `srcShape[1]` 在A2/A3上必须 32 字节对齐
2. 若每次只操作1行，且Pattern=AR时，使用不带pattern的简化Reduce类API
3. 写回 GM 时使用原始 `a0Count`

---

## 二、分支详细实现

> **重要**：以下 4 个分支的实现已拆分到独立文件，根据实际需要加载对应分支的文档。

### 2.1 分支导航

| 分支名称 | 适用场景 | 难度 | 详细文档 |
|---------|---------|------|---------|
| **AR-全载** | A0=1, R ≤ threshold | ★★ | [ar-fullload.md](ar-fullload.md) |
| **AR-Col-Split** | A0=1, R > threshold | ★★★ | [ar-colsplit.md](ar-colsplit.md) |
| **ARA-全载** | A0>1, R ≤ R_max | ★★★ | [ara-fullload.md](ara-fullload.md) |
| **ARA-Row-Split** | A0>1, R > R_max | ★★★★ | [ara-rowsplit.md](ara-rowsplit.md) |

### 2.2 Tiling 参数计算原则

**核心设计思想**:
1. **直接公式计算，不二分查找**：基于 UB 容量和 buffer 结构直接计算阈值
2. **约束取最小**：A0Inner = min(UB限制, A0维度限制, 多核均衡限制)
3. **基于 a0TileBase 估算**：用最小对齐单位估算，实际 tileA0Len ≤ 估算值

**全载 vs 分载判定**:

**全载条件 = 加载数据 + 计算过程所需全部 buffer ≤ UB_SIZE**

不同算子的计算逻辑不同，中间 buffer 数量和大小也不同，所以全载阈值计算公式**因算子而异**。

---

## 三、关键设计说明

### 3.1 为什么用 a0TileBase 估算而不是迭代？

1. **a0TileBase 是最小对齐单位**：`a0TileBase = VECTOR_REG_WIDTH / sizeof(T)`
2. **Buffer 按 a0TileBase 分配**：所有 buffer 大小都是 a0TileBase 的整数倍
3. **tileA0Len 自然对齐**：`tileA0Len = a0Inner × a0TileBase`，必然满足 32 字节对齐
4. **保守估算**：用 a0TileBase 估算 `ubPerTileBase`，实际 `tileA0Len` 可能更小，不会超出 UB 容量

### 3.2 为什么不需要二分查找？

1. **直接公式计算 R_max**：基于 UB 容量和 buffer 结构直接计算
2. **A0Inner 约束取最小**：三个约束都可直接计算，无需搜索
3. **a0TileBase 保守估算**：确保不会超出 UB 容量

### 3.3 全载判定的核心原则

**全载 = UB 能容纳「加载数据 + 计算过程全部 buffer」**

这要求：
1. 分析 kernel 中所有 buffer 的使用
2. 计算完整计算过程所需的 UB 总量
3. 与 UB_SIZE 比较判定是否全载

---

## 四、分支对照表

| 分支名称 | Shape | 载入模式 | 切分方向 | 适用条件 |
|---------|-------|---------|---------|---------|
| **AR-全载** | (A1, R) | 全载 | - | A0=1, R ≤ threshold |
| **AR-Col-Split** | (A1, R) | 分载 | 列方向 | A0=1, R > threshold |
| **ARA-全载** | (A1, R, A0) | 全载 | - | A0>1, R ≤ R_max |
| **ARA-Row-Split** | (A1, R, A0) | 分载 | 行方向 | A0>1, R > R_max |

### 4.1 分支决策流程图

```
                输入: shape, dtype, axis
                         │
                         ▼
               ┌───────────────────┐
               │  3D 抽象: (A1, R, A0) │
               └───────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  A0 == 1 ?  │
                  └─────────────┘
                    │         │
                   YES        NO
                    │         │
                    ▼         ▼
            ┌─────────┐  ┌──────────┐
            │ AR 模板 │  │ ARA 模板 │
            └─────────┘  └──────────┘
                 │            │
                 ▼            ▼
         ┌──────────────┐ ┌─────────────┐
         │R ≤ threshold?│ │ R ≤ R_max ? │
         └──────────────┘ └─────────────┘
           │        │       │        │
          YES       NO     YES       NO
           │        │       │        │
           ▼        ▼       ▼        ▼
        ┌─────┐ ┌───────┐ ┌─────┐ ┌───────┐
        │全载 │ │Col-   │ │全载 │ │Row-   │
        │     │ │Split  │ │     │ │Split  │
        └─────┘ └───────┘ └─────┘ └───────┘
```

---

## 五、AR vs ARA 模板对比

| 维度 | AR 模板 (A0=1) | ARA 模板 (A0>1) |
|-----|----------------|-----------------|
| **数据连续性** | 每行 R 个元素连续 | 每列 R 个元素不连续（间隔 A0） |
| **Reduce 结果** | 标量（1 个值） | 向量（A0 个值） |
| **多核切分** | 按 A1（行）切分 | 按 A1 × A0Outer 切分 |
| **内存访问** | `GM[row * R + col]` | `GM[a1 * R * A0 + r * A0 + a0]` |
| **DataCopy** | 连续块拷贝 | 需要 srcStride 跨越 |
| **UB 结构** | 1D 连续 buffer | 2D (R × alignedCols) buffer |
| **对齐处理** | 行对齐 | 列对齐，需要 alignedCols |

---

## 六、Reduce API 使用规范

### 6.1 API 分类

| API 类型 | 接口示例 | Pattern 支持 | 适用场景 | 限制条件 |
|---------|---------|-------------|---------|---------|
| **不带 Pattern** | `ReduceSum<T>` | 仅 AR (-1轴) | 只能对最后一维 Reduce | 无额外对齐要求 |
| **带 Pattern** | `ReduceSum<T, Pattern::RA>` | RA, AR | 任意轴 Reduce | A2/A3: shape 最后一维必须 32 字节对齐 |

### 6.2 tmpBufSize 计算

```cpp
uint32_t perRepeat = 256 / sizeof(float);      // 64 for FP32
uint32_t perBlock = 32 / sizeof(float);        // 8 for FP32
uint32_t repeats = (R * alignedCols + perRepeat - 1) / perRepeat;
uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * sizeof(float);
tmpBufSize = std::max(tmpBufSize, 4096u);      // 最小 4KB
```

### 6.3 API 参数限制对 Tiling 的影响

**关键原则**：如果 kernel 中用到的 API 的某个参数有上限限制，且该参数值与 tiling 参数相关，则在计算 tiling 参数时需要考虑此限制。

**示例**：
- 向量化 API 的 `repeatTimes` 有 255 上限
- 若 `repeatTimes` 传的是行数 R
- 则 R_max 计算时需要：`R_max = min(R_max, 255)`

### 6.4 逐行处理模式（AR 全载推荐）

**适用场景**：逐行独立计算的算子

**核心要点**:

1. **UB 数据布局（关键！）**
```
❌ 错误: rowOffset = rowIdx * rLength
✓ 正确: rowOffset = rowIdx * rLengthAlign

UB 布局:
  Row0: [有效数据 rLength][padding 到 rLengthAlign]
  Row1: [有效数据 rLength][padding 到 rLengthAlign]
```

2. **数据搬运**
```cpp
// blockLen 用 rLength，不是 rLengthAlign
AscendC::DataCopyPad(xLocal, xGm[offset], 
    {static_cast<uint16_t>(rows), static_cast<uint32_t>(rLength * sizeof(T)), 0, 0, 0}, 
    {false, 0, 0, 0});
```

3. **Reduce API 选择**
```cpp
// 使用 Level 2 接口（无 Pattern），只传有效数据个数
AscendC::ReduceMax<T>(rowTmp, xLocal[rowOffset], reduceTmp, 
    static_cast<int32_t>(rLength), false);
```

---

## 七、常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 精度过大 | srcShape 使用 a0Count 而非 alignedCols | 使用对齐后的列数 |
| FP16 精度差 | 中间计算精度不足 | 混合精度（FP32 计算） |
| 输出全 0 | Buffer 未正确初始化 | 检查 AllocTensor |
| 部分数据错误 | Split 模式 chunk 边界处理 | 检查 last_chunk_size |
| 崩溃 | UB 越界 | 检查 Buffer 大小计算 |
| 带宽利用率低 | 单 Buffer 无并行 | Double Buffer (depth=2) |
| 单行通过，多行失败 | rowOffset 用 rLength 而非 rLengthAlign | `rowOffset = rowIdx × rLengthAlign` |
| 非对齐场景全 0 | DataCopy 后未 EnQue/DeQue 同步 | 完整流水线同步 |
| 编译错误 | API 参数不匹配 | 使用 Level 2 接口 |
| 核心超时/挂起 | Buffer 泄漏 | 检查 FreeTensor 与 AllocTensor 成对 |
| 特定行数据错误 | 多核切分时尾核处理 | 检查 tailCoreRows 和 tailLoopRows |
| 多核负载不均 | tilesPerCore 计算错误 | 重新计算 tiling 参数 |
