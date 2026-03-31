# Reduce 类算子快速参考

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
WholeReduce*  /  BlockReduce*  /  PairReduce*  /  RepeatReduce*
```

### ✅ 推荐使用的 API

```cpp
// 高阶 API - 任意轴归约，支持 RA/AR 两种 Pattern
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(dst, src, tmp, srcShape, isInit);
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(dst, src, tmp, srcShape, isInit);
AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(dst, src, tmp, srcShape, isInit);
AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(dst, src, tmp, srcShape, isInit);

// 简化形式 - 仅支持 -1 轴（最后一维），无对齐要求
AscendC::ReduceSum<T>(dst, src, tmp, count);
AscendC::ReduceMax<T>(dst, src, tmp, count);
```

### Pattern 说明

| Pattern | 方向 | 输入 Shape | 输出 Shape | 说明 |
|---------|------|-----------|-----------|------|
| **RA** | 行方向 | (R, C) | (C,) | 沿第一维 Reduce，保留列 |
| **AR** | 列方向 | (R, C) | (R,) | 沿最后一维 Reduce，保留行 |

### 使用示例

```cpp
// RA Pattern - 沿第一维（行方向）归约
uint32_t srcShape[] = {R, alignedCols};  // alignedCols 必须 32 字节对齐
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(maxLocal, srcLocal, tmpLocal, srcShape, true);

// AR Pattern - 沿最后一维（列方向）归约
uint32_t srcShape[] = {rows, alignedCols};
AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(sumLocal, srcLocal, tmpLocal, srcShape, true);

// 简化形式 - 仅支持 -1 轴，无对齐要求
AscendC::ReduceMax<T>(dstLocal, srcLocal, tmpLocal, count);
```

### API 限制

| API | Pattern | 对齐要求 | 限制条件 |
|-----|---------|---------|---------|
| `ReduceSum<T>` | 仅 AR (-1轴) | 无 | 单次操作1行无限制，多行有限制 |
| `ReduceSum<T, Pattern>` | RA, AR | A2/A3 需32字节对齐 | repeatTimes ≤ 255 |

---

## 核心概念

### 3D 抽象

```
输入 Tensor: (d0, d1, ..., d_{axis}, ..., d_{n-1})
                    ↑
                Reduce 维度 R

抽象为 3D: (A1, R, A0)
  - A1 = d0 × ... × d_{axis-1}     (Reduce 前的维度乘积)
  - R  = d_{axis}                   (Reduce 维度)
  - A0 = d_{axis+1} × ... × d_{n-1} (Reduce 后的维度乘积)
```

### 模板分类

| A0 值 | 模板名称 | Shape | 特点 |
|------|---------|-------|------|
| A0=1 | **AR 模板** | (A1, R) | 每行连续，归约为标量 |
| A0>1 | **ARA 模板** | (A1, R, A0) | 每列不连续，归约为向量 |

### ARA 模板下 Pattern::RA 的选择原因

**多核切分方式**：
- ARA 模板将 A0 维度切分为多个 `A0_inner`
- 每个核处理部分 `(R, A0_inner)` 的数据

**数据在 UB 中的布局**：
```
GM 中的数据: (A1, R, A0)
                ↓ 取一个 tile 到 UB
UB 中的数据: (R, A0_inner)
  布局: [row0的全部A0_inner, row1的全部A0_inner, ..., row{R-1}的全部A0_inner]
        └──────────────────────────────────────────────────────┘
                              共 R 行，每行 A0_inner 个元素
```

**Pattern::RA 的含义**：
- **R** = Reduce 维度（沿第一维归约）
- **A** = Align 维度（保留第二维）
- 输入 shape = `(R, alignedCols)`，输出 shape = `(alignedCols,)`
- 对于每个 `a0` 位置，取 R 个值归约，输出 A0_inner 个结果

**为什么必须用 Pattern::RA**：
- Level 2 API `Reduce<T>(dst, src, tmp, count)` 只能处理连续的 count 个元素
- ARA 模板的数据带 stride（间隔 A0_inner），必须用 Pattern API 处理
- `Reduce<T, Pattern::RA>` 会正确处理 stride，对每列的 R 个值归约

---

## 分支速查

| 分支 | 条件 | 难度 | 详细文档 |
|-----|------|------|---------|
| **AR-全载** | A0=1, R ≤ threshold | ★★ | [ar-fullload.md](ar-fullload.md) |
| **AR-Col-Split** | A0=1, R > threshold | ★★★ | [ar-colsplit.md](ar-colsplit.md) |
| **ARA-全载** | A0>1, R ≤ R_max | ★★★ | [ara-fullload.md](ara-fullload.md) |
| **ARA-Row-Split** | A0>1, R > R_max | ★★★★ | [ara-rowsplit.md](ara-rowsplit.md) |

### 分支决策流程

```
输入: shape, dtype, axis
        ↓
3D 抽象: (A1, R, A0)
        ↓
   A0 == 1 ?
    /    \
  YES    NO
   |      |
 AR 模板  ARA 模板
   |      |
 R ≤ threshold?  R ≤ R_max?
  /    \         /    \
 YES   NO      YES    NO
  |     |       |      |
 全载  Col-Split 全载  Row-Split
```

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 精度过大 | srcShape 使用 a0Count 而非 alignedCols | 使用对齐后的列数 |
| FP16 精度差 | 中间计算精度不足 | 混合精度（FP32 计算） |
| 输出全 0 | Buffer 未正确初始化 | 检查 AllocTensor/FreeTensor |
| UB 越界崩溃 | Buffer 大小计算错误 | 检查 UB 容量计算 |
| 带宽利用率低 | 单 Buffer 无并行 | Double Buffer (depth=2) |
| 编译错误：no matching ReduceMax | API 参数不匹配 | 使用正确接口：`ReduceMax(dst, src, tmp, count)` |
| 核心超时/挂起 | Buffer 泄漏 | 检查所有路径的 FreeTensor 与 AllocTensor 成对 |

---

## 检查清单

### 设计阶段

- [ ] 支持 AR 和 ARA 两种模板
- [ ] 处理全载和分载两种模式
- [ ] 支持任意 axis 参数
- [ ] 处理非 32 字节对齐的 A0
- [ ] 边界值测试覆盖

### 实现阶段

- [ ] Buffer 大小使用 `rLengthAlign` / `alignedCols`
- [ ] API count 参数使用 `rLength` / `tileA0Len`（有效数据）
- [ ] rowOffset 使用 `rLengthAlign`（UB 对齐存储）
- [ ] GM↔UB 双向使用 `DataCopyPad`
- [ ] Double Buffer (depth=2) 优化带宽

### 测试阶段

- [ ] R 边界测试（threshold ± 1, 2×threshold）
- [ ] 非对齐 A0 测试
- [ ] 小 A0 测试（A0 < a0TileBase）
- [ ] 多核负载均衡测试
- [ ] FP16 混合精度测试
- [ ] 精度达标（相对误差 < 1e-5）

---

## 快速链接

- **完整方法论**: [methodology.md](methodology.md)
- **分支实现**:
  - [AR-全载](ar-fullload.md)
  - [AR-Col-Split](ar-colsplit.md)
  - [ARA-全载](ara-fullload.md)
  - [ARA-Row-Split](ara-rowsplit.md)
- **相关参考**:
  - [广播计算优化](../../../ascendc-api-best-practices/references/api-arithmetic.md)
