# Index-Tracking 索引跟踪类算子设计指南

## ⚠️ 前置要求

**Index-Tracking 算子的设计方法论与 Reduction 完全相同**，请先阅读：
- [归约类算子快速参考](../reduction/guide.md)
- [归约类算子方法论](../reduction/methodology.md)

本指南**只补充**索引跟踪的特殊处理方法。

---

## 算子分类

| 算子 | 功能 | 典型API | 设计复杂度 | 详细文档 |
|------|------|---------|-----------|---------|
| **ArgMax/ArgMin** | 找最大/最小值索引 | ReduceMax(calIndex=true) 或 Compare+Select | ★★ | [argmax.md](argmax.md) |
| **Top-K** | 前K个值+索引 | TopK / TopKMask / Sort+切片 | ★★★ | [topk.md](topk.md) |
| **Sort/Argsort** | 排序 | Sort / BitonicSort | ★★★ | [sort.md](sort.md) |
| **Where/NonZero** | 条件查找索引 | Compare + 压缩 | ★★★★ | [conditional.md](conditional.md) |

---

## 共同特征

### 1. 输出包含索引

| 算子 | 值输出 | 索引输出 |
|------|--------|---------|
| ArgMax | 可选（可只要索引） | 必须 |
| TopK | 必须 | 必须 |
| Sort | 必须 | 可选（Argsort只要索引） |
| Where | 可选 | 必须 |

### 2. Buffer需求更多

| 算子类型 | Buffer数量 | 额外Buffer |
|---------|-----------|-----------|
| Reduction | 2-3个 | - |
| **Index-Tracking** | **5-7个** | 索引buffer、临时索引、mask等 |

### 3. 设计要素与Reduction相同

| 设计要素 | 与Reduction关系 |
|---------|----------------|
| 3D抽象 | ✅ 完全相同 |
| 多核切分 | ✅ 完全相同 |
| UB切分 | ✅ 完全相同 |
| Tiling参数 | ✅ 完全相同 |
| 分支覆盖 | ✅ 完全相同 |

---

## API 快速对照

### ArgMax/ArgMin

| 场景 | API | 约束 |
|------|-----|------|
| 最后一轴 | ReduceMax(calIndex=true) | 数据连续 |
| 非最后一轴 | Compare + Select | 需同步更新值和索引 |

### Top-K

| 场景 | API | 约束 |
|------|-----|------|
| K≤128 | TopKMask | 极小K专用 |
| K≤4096, inner≤4096 | TopK | Normal模式 |
| K很小(≤10) | 多次ArgMax | 性能较差 |
| 大K或inner>4096 | Sort + 切片 | 全排序开销 |

### Sort

| 场景 | API | 约束 |
|------|-----|------|
| 通用 | Sort | 升序，需类型转换读索引 |
| 长度=2^n | BitonicSort | 性能更优 |
| 降序 | 取反→Sort→取反 | Sort只支持升序 |

### 条件查找

| 场景 | API | 约束 |
|------|-----|------|
| 生成mask | Compare | mask是uint8_t |
| 选择值 | Select | 不压缩 |
| 压缩索引 | 手动实现 | 无专用API |

---

## 详细设计指南

每个算子有独立的详细设计文档，包含：
- 场景识别
- API选择决策树
- Buffer规划
- 实现示例
- 常见陷阱

| 算子 | 详细文档 |
|------|---------|
| ArgMax/ArgMin | [argmax.md](argmax.md) |
| Top-K | [topk.md](topk.md) |
| Sort | [sort.md](sort.md) |
| 条件查找 | [conditional.md](conditional.md) |

---

## 通用陷阱

| 陷阱 | 场景 | 解决方案 |
|------|------|---------|
| 索引类型转换 | ReduceMax/TopK读取 | reinterpret_cast |
| 索引不同步 | ArgMax非最后一轴 | 同一次Select更新 |
| mask类型 | Compare输出 | uint8_t，不是bool |
| Buffer不足 | Index-Tracking多4-5个buffer | 重新规划UB |
| API选错 | 不同算子场景不同 | 查阅对应详细文档 |

---

## 设计要素（与Reduction相同）

以下设计要素与 Reduction 完全相同，请参考对应文档：

| 设计要素 | 文档 | 说明 |
|---------|------|------|
| 3D抽象 | [reduction/methodology.md](../reduction/methodology.md) | A1, R, A0 计算 |
| 多核切分 | [reduction/methodology.md](../reduction/methodology.md) | A1×A0Outer 切分 |
| UB切分 | [reduction/methodology.md](../reduction/methodology.md) | 全载/分载判定 |
| Tiling参数 | [reduction/methodology.md](../reduction/methodology.md) | tileA0Len, a0Outer |
| 分支覆盖 | [reduction/guide.md](../reduction/guide.md) | AR/ARA模板选择 |

---

## 快速开始

1. **确定算子类型**：ArgMax / TopK / Sort / Where
2. **阅读对应详细文档**：见上表
3. **参考Reduction设计要素**：多核切分、UB切分、Tiling
4. **补充索引特殊处理**：Buffer规划、同步更新、类型转换
