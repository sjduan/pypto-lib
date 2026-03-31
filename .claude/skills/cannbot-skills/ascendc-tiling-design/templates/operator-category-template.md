# 算子类型最佳设计模板

> 本模板基于 Reduce 类算子的设计方法论提炼，适用于创建新的算子类别设计文档。

---

## 一、文件组织结构

```
ascendc-tiling-design/
├── SKILL.md                           # 主入口，算子分类体系 + 通用设计要素
└── references/
    ├── reduction/                     # Reduction 归约类
    │   ├── guide.md                   # 快速参考（1-2页）
    │   ├── methodology.md             # 完整方法论（5-10页）
    │   ├── ar-fullload.md             # AR 全载分支
    │   ├── ar-colsplit.md             # AR 分载分支
    │   ├── ara-fullload.md            # ARA 全载分支
    │   └── ara-rowsplit.md            # ARA 分载分支
    ├── elementwise/                   # Elementwise 逐元素类
    │   ├── guide.md
    │   ├── methodology.md
    │   └── {branch}.md
    ├── broadcast/                     # Broadcast 广播类
    ├── conversion/                    # Conversion 数据转换类
    ├── matmul/                        # MatMul 矩阵乘类
    └── ...                            # 其他类别
```

**命名规范**：
- 文件夹名：小写英文，如 `reduction`、`elementwise`、`matmul`
- 文件名：简洁描述性，如 `guide.md`、`methodology.md`、`ar-fullload.md`

---

## 二、SKILL.md 模板

```markdown
---
name: ascendc-tiling-design
description: Ascend C 算子 Tiling 设计指南。提供算子分类体系和 Tiling 核心要素（多核切分、UB切分、Buffer规划、分支覆盖）的详细设计方法。触发：算子设计阶段、设计 Tiling 策略（多核切分/UB切分）、规划 Buffer 分配、查阅某类算子的 Tiling 方法论时。
---

# Ascend C 算子 Tiling 设计指南

## 算子分类体系

| 类别 | 特征 | 典型算子 | 设计指南 |
|------|------|---------|---------|
| Reduction 归约类 | 沿轴归约，输出维度减少 | ReduceSum, Softmax, LayerNorm | ✅ [快速参考](references/reduction/guide.md) / [完整方法论](references/reduction/methodology.md) |
| Elementwise 逐元素类 | ... | ... | 📋 规划中 |
| Broadcast 广播类 | ... | ... | 📋 规划中 |
| ... | ... | ... | ... |

---

## 使用场景

本技能是**知识库型技能**，提供设计参考，不定义开发流程。

**典型使用方式**：
1. `ascendc-kernel-develop-workflow` 在设计阶段（阶段二）调用本技能
2. 开发者查阅算子分类和通用设计要素
3. 根据算子类别查阅对应的详细设计指南（references 文件夹）

---

## 通用设计要素（所有类别必须）

### 1. 多核切分策略

**核心问题**：任务如何分配给多个 AI Core？

**设计要点**：
- 负载均衡：每个核处理的任务量尽量相等
- 数据局部性：相邻数据尽量分配给同一核
- 粒度适中：tile 不能太小（调度开销大），不能太大（并行度低）

**输出**：
- [ ] 总任务切分方式（按哪个维度切）
- [ ] 每个 AI Core 处理的任务量
- [ ] 使用的 AI Core 数量

### 2. UB 切分策略

**核心问题**：单次能处理多少数据？

**设计要点**：
- UB 容量限制（A2/A3: 192KB, A5: 248KB）
- 单次处理数据量
- 是否需要分 chunk 处理

**输出**：
- [ ] 单次处理的数据量
- [ ] 是否需要分 chunk
- [ ] chunk 大小计算公式

### 3. Buffer 规划

**核心问题**：需要哪些 buffer？各多大？

**设计要点**：
- 输入 buffer（inQueue）
- 输出 buffer（outQueue）
- 中间计算 buffer（tmpBuf, workBuf 等）
- Double Buffer 优化

**输出**：
- [ ] Buffer 列表及用途
- [ ] 各 Buffer 大小计算公式
- [ ] 总 UB 使用量

### 4. 分支场景覆盖

**核心问题**：需要处理哪些不同场景？

**常见分支维度**：
- 数据类型：FP32 / FP16 / BF16 / INT8
- Shape 大小：大 shape / 小 shape
- 数据对齐：32字节对齐 / 非对齐
- 边界情况：最小值 / 最大值 / 特殊值

**输出**：
- [ ] 分支决策条件
- [ ] 各分支的处理策略
- [ ] 边界测试用例
```

---

## 三、快速参考模板 (references/{category}/guide.md)

```markdown
# {Category} 类算子快速参考

> **完整方法论**: [methodology.md](methodology.md)

---

## 目录

- [API 使用规范](#api-使用规范)
- [核心概念](#核心概念)
- [分支速查](#分支速查)
- [常见问题](#常见问题)
- [检查清单](#检查清单)

---

## API 使用规范

### ⛔️ 禁止使用的 API

```cpp
// 列出该类别算子不推荐使用的 API
```

### ✅ 推荐使用的 API

```cpp
// 列出该类别算子推荐使用的 API 及典型用法
```

### API 限制

| API | 适用场景 | 对齐要求 | 限制条件 |
|-----|---------|---------|---------|
| ... | ... | ... | ... |

---

## 核心概念

### {类别特有的核心概念 1}

```
// 概念说明
```

### {类别特有的核心概念 2}

```
// 概念说明
```

---

## 分支速查

| 分支 | 条件 | 难度 | 详细文档 |
|-----|------|------|---------|
| **分支1** | 条件1 | ★★ | [branch1.md](branch1.md) |
| **分支2** | 条件2 | ★★★ | [branch2.md](branch2.md) |

### 分支决策流程

```
输入: shape, dtype, 其他参数
        ↓
   核心判断 1
     /    \
   YES    NO
    |      |
  分支1   核心判断 2
          /    \
        YES    NO
         |      |
       分支2  分支3
```

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 问题1 | 原因1 | 解决方案1 |
| 问题2 | 原因2 | 解决方案2 |

---

## 检查清单

### 设计阶段

- [ ] 设计项1
- [ ] 设计项2

### 实现阶段

- [ ] 实现项1
- [ ] 实现项2

### 测试阶段

- [ ] 测试项1
- [ ] 测试项2
```

---

## 四、完整方法论模板 (references/{category}/methodology.md)

```markdown
# {Category} 类算子开发方法论

---

## 目录

- [零、API 使用规范](#零api-使用规范)
- [一、算子分析与设计](#一算子分析与设计)
  - [1.1 设计原则](#11-设计原则)
  - [1.2 {维度/模式识别}](#12-维度模式识别)
  - [1.3 多核切分策略](#13-多核切分策略)
  - [1.4 UB 切分策略](#14-ub-切分策略)
  - [1.5 处理模式判定](#15-处理模式判定)
  - [1.6 {类别特有设计点}](#16-类别特有设计点)
  - [1.7 非对齐场景处理](#17-非对齐场景处理)
- [二、分支详细实现](#二分支详细实现)
  - [2.1 分支导航](#21-分支导航)
  - [2.2 Tiling 参数计算原则](#22-tiling-参数计算原则)
- [三、关键设计说明](#三关键设计说明)
- [四、分支对照表](#四分支对照表)
- [五、{类别特有对比/说明}](#五类别特有对比说明)
- [六、{类别特有 API 规范}](#六类别特有-api-规范)
- [七、常见问题与解决方案](#七常见问题与解决方案)

---

## 零、API 使用规范

> **快速参考**: [guide.md](guide.md)

### ⛔️ 禁止使用的 API

```cpp
// API 列表
```

### ✅ 推荐使用的 API

```cpp
// API 列表及用法
```

---

## 一、算子分析与设计

### 1.1 设计原则

**核心原则：设计时必须考虑所有分支场景**

{类别}类算子需要覆盖多种输入情况，设计阶段必须预先考虑：

```
设计时必须覆盖的场景：

1. 模板分支
   ├── 分支1 (条件A)
   └── 分支2 (条件B)

2. 载入模式
   ├── 全载：UB 能容纳完整数据块
   └── 分载：需要分 chunk 处理

3. {类别特有场景3}

4. 数据对齐
   ├── 32 字节对齐：直接使用
   └── 非对齐：使用 DataCopyPad 填充

5. 边界情况
   ├── 小维度
   └── 大维度（需要 Split）
```

**设计检查清单**:
- [ ] 检查项1
- [ ] 检查项2

**代码组织建议**:
```
operator/
├── operator_branch1.h    - 分支1 实现
├── operator_branch2.h    - 分支2 实现
└── operator.h            - 主入口，根据条件调用各分支
```

### 1.2 {维度/模式识别}

```
输入 Tensor 分析

核心维度定义

模板/模式分类
```

### 1.3 多核切分策略

```
总任务 = {计算公式}

切分原则：
1. 负载均衡
2. 数据局部性
3. 粒度适中
```

### 1.4 UB 切分策略

```
UB 容量：
├── A2/A3 服务器: 192KB
└── A5 服务器: 248KB

UB 切分依据：
1. {维度1}_chunk_size
2. {维度2}_inner
3. 对齐要求
```

### 1.5 处理模式判定

```
模板类型（由 {条件1} 决定）：
├── 条件1 → 模板A
└── 其他 → 模板B

载入模式（由 UB 容量决定）：
├── 全载：UB 能容纳完整处理所需 buffer
└── 分载：需要分 chunk 处理
```

### 1.6 {类别特有设计点}

{针对该类别算子的特殊设计考虑}

### 1.7 非对齐场景处理

**解决方案**：使用 `DataCopyPad` 进行自动填充。

```cpp
// 非对齐处理代码示例
```

---

## 二、分支详细实现

> **重要**：以下分支的实现已拆分到独立文件，根据实际需要加载对应分支的文档。

### 2.1 分支导航

| 分支名称 | 适用场景 | 难度 | 详细文档 |
|---------|---------|------|---------|
| **分支1** | 条件1 | ★★ | [branch1.md](branch1.md) |
| **分支2** | 条件2 | ★★★ | [branch2.md](branch2.md) |

### 2.2 Tiling 参数计算原则

**核心设计思想**:
1. **直接公式计算，不二分查找**
2. **约束取最小**
3. **基于对齐单位估算**

**全载 vs 分载判定**:

**全载条件 = 加载数据 + 计算过程所需全部 buffer ≤ UB_SIZE**

---

## 三、关键设计说明

### 3.1 为什么用 {参数} 估算？

{解释说明}

### 3.2 为什么不需要二分查找？

{解释说明}

### 3.3 全载判定的核心原则

**全载 = UB 能容纳「加载数据 + 计算过程全部 buffer」**

---

## 四、分支对照表

| 分支名称 | Shape | 载入模式 | 切分方向 | 适用条件 |
|---------|-------|---------|---------|---------|
| **分支1** | ... | 全载 | - | 条件1 |
| **分支2** | ... | 分载 | 方向1 | 条件2 |

### 4.1 分支决策流程图

```
{决策流程图}
```

---

## 五、{类别特有对比/说明}

{如有多个模板/模式，在此对比说明}

---

## 六、{类别特有 API 规范}

### 6.1 API 分类

| API 类型 | 接口示例 | 适用场景 | 限制条件 |
|---------|---------|---------|---------|
| ... | ... | ... | ... |

### 6.2 Buffer 大小计算

```cpp
// Buffer 大小计算公式
```

### 6.3 API 参数限制对 Tiling 的影响

{关键原则说明}

---

## 七、常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 问题1 | 原因1 | 解决方案1 |
| 问题2 | 原因2 | 解决方案2 |
```

---

## 五、分支实现模板 (references/{category}/{branch}.md)

```markdown
# {分支名称} 分支

> **适用场景**: {条件描述}

---

## 目录

- [一、分支特征](#一分支特征)
- [二、Buffer 规划](#二buffer-规划)
- [三、Tiling 参数计算](#三tiling-参数计算)
- [四、Kernel 实现要点](#四kernel-实现要点)
- [五、测试用例](#五测试用例)
- [六、常见问题](#六常见问题)
- [七、性能优化建议](#七性能优化建议)

---

## 一、分支特征

| 特征 | 说明 |
|------|------|
| **模板类型** | {模板名称} |
| **Shape 抽象** | {Shape} |
| **载入模式** | 全载/分载 |
| **适用条件** | {条件} |
| **数据连续性** | {连续性说明} |
| **计算结果** | {结果说明} |

---

## 二、Buffer 规划

### 2.1 FP32 场景

```cpp
// Buffer 初始化代码
pipe->InitBuffer(inQueueX, 2, size1);
pipe->InitBuffer(outQueueY, 2, size2);
pipe->InitBuffer(tmpBuf, tmpBufSize);

// 总 UB = {公式}
```

### 2.2 FP16 场景（混合精度）

```cpp
// FP16 输入，FP32 计算场景的 Buffer 规划
```

### 2.3 tmpBufSize 计算

```cpp
uint32_t ComputeTmpBufSize(...) {
    // Buffer 大小计算公式
    return tmpBufSize;
}
```

---

## 三、Tiling 参数计算

### 3.1 全载阈值（如适用）

```cpp
// 全载条件计算
uint32_t threshold = {计算公式};
```

### 3.2 多核切分参数

```cpp
// 多核切分参数计算
uint32_t perCore = {计算公式};
uint32_t usedCoreNum = {计算公式};
uint32_t tailCore = {计算公式};
```

### 3.3 对齐处理

```cpp
// 计算对齐后的参数
uint32_t aligned = {对齐计算};
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM ({shape}) → UB ({buffer_shape})
    ↓
[{核心 API}] → result ({result_shape})
    ↓
UB ({output_shape}) → GM ({output_shape})
```

### 4.2 核心 API 调用

```cpp
// 核心计算代码
```

**关键要点**:
- 要点1
- 要点2

### 4.3 参数使用对照表

| 参数位置 | 用法1 | 用法2 |
|---------|------|------|
| API 参数 | ✓ | ✗ |
| Buffer 大小 | ✗ | ✓ |

### 4.4 流水线设计

**Double Buffer 模式** (depth=2):
```
Tile N:   CopyIn(data0) → Compute(data0) → CopyOut(data0)
Tile N+1:              CopyIn(data1) → Compute(data1) → CopyOut(data1)
```

---

## 五、测试用例

### 5.1 功能测试矩阵

| ID | Shape | 其他参数 | Dtype | 说明 |
|----|-------|---------|-------|------|
| F01 | ... | ... | ... | 基础场景 |
| F02 | ... | ... | ... | 多核场景 |
| F03 | ... | ... | ... | FP16 混合精度 |

### 5.2 边界测试用例

| ID | 场景 | 参数 | 预期结果 |
|----|------|------|---------|
| B01 | 边界1 | param=value | 预期结果 |
| B02 | 边界2 | param=value | 预期结果 |

### 5.3 精度要求

**相对误差阈值**: < 1e-5

```python
def check_precision(output, ref):
    abs_diff = np.abs(output - ref)
    rel_diff = abs_diff / (np.abs(ref) + 1e-8)
    return np.max(rel_diff) < 1e-5
```

---

## 六、常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 问题1 | 原因1 | 解决方案1 |
| 问题2 | 原因2 | 解决方案2 |

---

## 七、性能优化建议

1. **Double Buffer**: 使用 depth=2 的队列
2. **对齐处理**: 提前计算对齐参数，避免重复计算
3. **流水线**: CopyIn/Compute/CopyOut 并行
4. **FP16 混合精度**: FP16 输入，FP32 计算，FP16 输出
```

---

## 六、模板使用指南

### 6.1 创建新算子类别设计文档的步骤

1. **在 SKILL.md 中添加新类别**
   - 在"算子分类体系"表中添加新行
   - 填写类别名称、特征、典型算子

2. **创建类别文件夹** `references/{category}/`

3. **创建快速参考** `references/{category}/guide.md`
   - 确定该类别算子的核心 API
   - 识别核心概念
   - 初步划分分支

4. **创建完整方法论** `references/{category}/methodology.md`
   - 深入分析设计原则
   - 详细描述多核切分、UB切分、分支判定
   - 记录关键设计决策

5. **创建各分支实现文档** `references/{category}/{branch}.md`
   - 每个分支独立文件
   - 包含 Buffer 规划、Tiling 计算、Kernel 实现要点、测试用例

### 6.2 模板核心要素总结

| 要素 | guide.md | methodology.md | {branch}.md |
|------|---------|----------------|-------------|
| API 规范 | ✓ 简表 | ✓ 详细 | ✓ 示例 |
| 核心概念 | ✓ 定义 | ✓ 展开说明 | - |
| 分支速查 | ✓ 表格 | ✓ 流程图 | ✓ 特征表 |
| 设计原则 | ✓ 检查清单 | ✓ 完整说明 | - |
| Buffer 规划 | - | ✓ 原则 | ✓ 具体大小 |
| Tiling 计算 | - | ✓ 原则 | ✓ 公式 |
| 测试用例 | ✓ 检查清单 | - | ✓ 完整矩阵 |
| 常见问题 | ✓ 速查表 | ✓ 详细表 | ✓ 分支特有 |

### 6.3 关键设计模式

**1. 分支设计模式**
- 确定分支判定条件（通常是 Shape 特征）
- 每个分支独立文档
- 主文档提供导航

**2. Tiling 计算模式**
- 直接公式计算，不二分查找
- 多约束取最小值
- 基于对齐单位估算

**3. 文档分层模式**
- 快速参考（guide.md）：1-2 页，快速查阅
- 完整方法论（methodology.md）：5-10 页，深入理解
- 分支实现（{branch}.md）：每个分支 1 文件，具体实现参考

---

## 七、现有结构迁移方案

### 7.1 当前扁平结构

```
references/
├── reduction-ops-guide.md
├── reduction-ops-methodology.md
├── reduction-ops-ar-fullload.md
├── reduction-ops-ar-colsplit.md
├── reduction-ops-ara-fullload.md
└── reduction-ops-ara-rowsplit.md
```

### 7.2 目标文件夹结构

```
references/
└── reduction/
    ├── guide.md
    ├── methodology.md
    ├── ar-fullload.md
    ├── ar-colsplit.md
    ├── ara-fullload.md
    └── ara-rowsplit.md
```

### 7.3 迁移步骤

```bash
# 1. 创建文件夹
mkdir -p references/reduction

# 2. 移动并重命名文件
mv references/reduction-ops-guide.md references/reduction/guide.md
mv references/reduction-ops-methodology.md references/reduction/methodology.md
mv references/reduction-ops-ar-fullload.md references/reduction/ar-fullload.md
mv references/reduction-ops-ar-colsplit.md references/reduction/ar-colsplit.md
mv references/reduction-ops-ara-fullload.md references/reduction/ara-fullload.md
mv references/reduction-ops-ara-rowsplit.md references/reduction/ara-rowsplit.md

# 3. 更新 SKILL.md 中的链接
# 4. 更新各文件内部引用（使用相对路径）
```

### 7.4 引用路径变更

| 位置 | 旧路径 | 新路径 |
|------|--------|--------|
| SKILL.md | `references/reduction-ops-guide.md` | `references/reduction/guide.md` |
| guide.md | `{category}-ops-methodology.md` | `methodology.md` |
| methodology.md | `{category}-ops-branch1.md` | `branch1.md` |

---

## 八、示例：已完成的类别

| 类别 | 文件夹 | 状态 |
|------|--------|------|
| Reduction 归约类 | `references/reduction/` | ✅ 需迁移 |
| Elementwise 逐元素类 | `references/elementwise/` | 📋 规划中 |
| Broadcast 广播类 | `references/broadcast/` | 📋 规划中 |
| Conversion 数据转换类 | `references/conversion/` | 📋 规划中 |
| Random 随机类 | `references/random/` | 📋 规划中 |
| Advanced 高级算法类 | `references/advanced/` | 📋 规划中 |
| MatMul 矩阵乘类 | `references/matmul/` | 📋 规划中 |
| Convolution 卷积类 | `references/convolution/` | 📋 规划中 |
| NN 神经网络类 | `references/nn/` | 📋 规划中 |
