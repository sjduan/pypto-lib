# 依赖关系约束表达模型规范

## 概述

本文档定义了一套专业的约束表达模型，用于描述算子参数间的依赖关系，支持自动化解析和测试因子值推导。

## 设计原则

1. **可计算性**：所有约束表达式必须是可执行的 Python 表达式
2. **可解析性**：约束结构清晰，易于 YAML 解析和图结构构建
3. **可推导性**：支持正向推导（从输入计算输出）和反向验证（检查输出是否满足约束）
4. **可组合性**：约束可以组合、嵌套，支持复杂依赖场景

---

## 约束设计原则

> **核心原则：仅描述资料中显式说明的参数间依赖关系**

### 何时需要添加约束

约束用于描述**参数之间的动态依赖关系**，而非静态取值范围。以下情况需要添加约束：

| 场景 | 是否需要约束 | 说明 |
|------|-------------|------|
| dtype 等值约束（out.dtype = self.dtype） | ✅ 需要 | 参数间的类型传递关系 |
| 类型推导（self.dtype 由 batch1/batch2 推导） | ✅ 需要 | 参数间的类型依赖 |
| 形状计算（out.shape 由 input.shape 计算） | ✅ 需要 | 参数间的形状依赖 |
| 广播约束（bias.shape 可广播到 out.shape） | ✅ 需要 | 参数间的形状兼容性 |
| 维度匹配（batch1[2] == batch2[1]） | ✅ 需要 | 参数间的维度依赖 |
| 类型可转换（scalar.dtype 可转换为 tensor.dtype） | ✅ 需要 | 参数间的类型兼容性 |
| 可选参数联动（bias 不存在时，bias.dtype = NA） | ✅ 需要 | 参数间的存在性依赖 |

### 何时无需添加约束

以下情况**不需要**添加约束，因为取值范围已在 `04_测试因子.yaml` 中定义：

| 场景 | 示例 | 说明 |
|------|------|------|
| 固定枚举取值 | `cubeMathType.value: [0, 1, 2, 3]` | 在测试因子中已定义 |
| 固定数据格式 | `format: ["ND"]` | 在测试因子中已定义 |
| 固定维度数 | `dimensions: [3]` (batch2仅支持3维) | 在测试因子中已定义 |
| 固定数据类型列表 | `dtype: ["float16", "float32", "bfloat16"]` | 在测试因子中已定义 |
| 必选参数的存在性 | `self.exist: [true]`, `batch1.exist: [true]` | 必选参数无需约束 |

### 存在性约束的正确使用

**错误示例**（不应添加）：
```yaml
# ❌ self 和 batch1 都是必选参数，不存在可选之间的依赖关系
- id: "C-EXIST-001"
  type: existential
  trigger: "self.exist"
  trigger_value: false
  effect:
    target: "batch1.exist"
    action: set_na
  description: "self不存在时batch1设为NA"  # 错误：两个都是必选参数
```

### 约束命名规范

约束ID采用 `{类型前缀}-{序号}` 格式：



| 前缀 | 类型 | 示例 |
|------|------|------|
| `C-TYPE` | 类型约束 | `C-TYPE-001` |
| `C-SHAPE` | 形状约束 | `C-SHAPE-001` |
| `C-DIM` | 维度约束 | `C-DIM-001` |
| `C-EXIST` | 存在性约束 | `C-EXIST-001` |
| `C-CAST` | 类型转换约束 | `C-CAST-001` |
| `C-BCAST` | 广播约束 | `C-BCAST-001` |

## 核心概念

### 因子节点 (Factor Node)

因子节点是约束系统的基本单元，代表一个可测试的属性。

```yaml
# 因子节点标识规范
{id}: "{parameter}.{attribute}"
```

**属性类型**：
| 属性 | 说明 | 示例 |
|------|------|------|
| `dtype` | 数据类型 | `self.dtype` |
| `shape` | 张量形状 | `batch1.shape` |
| `value`  | 枚举值   | `cubeMathType.value`   |
| `format` | 数据格式 | `input.format` |
| `exist` | 存在性 | `bias.exist` |
| `dimensions` | 维度值 | `input.dimensions` |

**io_type 类型**：
| io_type | 说明 | 示例 |
|---------|------|------|
| `input` | 输入参数 | `self.dtype: {type: string, param: self, io_type: input}` |
| `output` | 输出参数 | `out.dtype: {type: string, param: out, io_type: output}` |
| `intermediate` | 中间因子（派生值，用于两步约束） | `_broadcast_target.shape: {type: list, param: _broadcast_target, io_type: intermediate}` |

---

## YAML 结构定义

### 完整结构

```yaml
# 元信息
metadata:
  operator: "aclnnAddbmm"
  version: "1.0"
  description: "Addbmm算子约束关系定义"

# Tensor类型因子节点定义
factors:
  self.dtype: {type: dtype, param: self, io_type: input}
  self.shape: {type: shape, param: self, io_type: input}
  self.format: {type: format, param: self, io_type: input}
  self.exist: {type: exist, param: self, io_type: input}
  self.dimensions: {type: dimensions, param: self, io_type: input}
  out.dtype: {type: dtype, param: out, io_type: output}
  out.shape: {type: shape, param: out, io_type: output}
  out.format: {type: format, param: out, io_type: output}
  out.exist: {type: exist, param: out, io_type: output}
  out.dimensions: {type: dimensions, param: out, io_type: output}

# 枚举类型因子节点定义
factors:
  cubeMathType.dtype: {type: dtype, param: cubeMathType, io_type: input}
  cubeMathType.exist: {type: exist, param: cubeMathType, io_type: input}
  cubeMathType.value: {type: value, param: cubeMathType, io_type: input}

# 非枚举/tensor类型因子节点定义
factors:
  alpha.dtype: {type: dtype, param: alpha, io_type: input}
  alpha.exist: {type: exist, param: alpha, io_type: input}

# 约束定义
constraints:
  - id: "C001"
    type: equal
    # ... 详细配置
```
---

## 约束类型定义

### 1. 计算约束 (calculate)

**语义**：目标因子的值通过数学/逻辑计算得出。

**特化场景**：
- **等值计算**：`expression: "sources[0]"`
- **公式计算**：`expression: "sources[0][1] * sources[1][2]"`

```yaml
# 等值计算
- id: "C001"
  type: calculate
  sources: ["self.dtype"]
  target: "out.dtype"
  expression: "sources[0]"
  description: "输出类型等于self类型"

# 形状计算
- id: "C006"
  type: calculate
  sources: ["batch1.shape", "batch2.shape"]
  target: "out.shape"
  expression: "[sources[0][1], sources[1][2]]"  # [M, N]
  description: "out.shape = [batch1[1], batch2[2]]"
```
---

### 3. 指定维度广播约束 (broadcast_dim)

**语义**：两个张量的特定维度需满足广播关系。

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 约束唯一标识 |
| `type` | string | 是 | 固定值 `broadcast_dim` |
| `sources` | list | 是 | 源因子列表，通常为 `["param.shape"]` |
| `source_index` | int | 是 | 源形状的维度索引 |
| `target` | string | 是 | 目标因子，如 `"other.shape"` |
| `target_index` | int | 是 | 目标形状的维度索引 |
| `description` | string | 否 | 约束描述 |

**广播规则**（单维度）：
- 两维度相等 → 兼容
- 源维度为 1 → 可广播到任意目标维度
- 目标维度为 1 → 可接受任意源维度

```yaml
- id: "C-BCAST-001"
  type: broadcast_dim
  sources: ["batch1.shape"]
  source_index: 0  # batch1.shape[0]
  target: "batch2.shape"
  target_index: 1  # batch2.shape[1]
  description: "batch1.shape[0]与batch2.shape[1]需满足广播关系"
```
---

### 4. 张量广播约束 (broadcast_shape)

**语义**：两个张量的整个 shape 需满足广播关系。

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 约束唯一标识 |
| `type` | string | 是 | 固定值 `broadcast_shape` |
| `sources` | list | 是 | 源因子列表，通常为 `["param.shape"]` |
| `target` | string | 是 | 目标因子，如 `"other.shape"` |
| `mode` | string | 是 | 广播模式：`unidirectional` 或 `bidirectional` |
| `description` | string | 否 | 约束描述 |

**广播模式**：

| 模式 | 说明 | 典型场景 |
|------|------|----------|
| `unidirectional` | 单向广播：target → sources | `self.shape` 可广播到 `out.shape` |
| `bidirectional` | 双向广播：sources ↔ target | `x.shape` 和 `y.shape` 可相互广播 |

**重要限制**：
- `broadcast_shape` **不支持** `source_shape_expr` 字段（派生形状表达式）
- 如需表达"广播到派生形状"的约束，应使用**两步约束**模式（见下方示例）

```yaml
# 场景1: 完整Shape广播 - 单向
- id: "C-BCAST-002"
  type: broadcast_shape
  sources: ["out.shape"]
  target: "self.shape"
  mode: unidirectional
  description: "self.shape可广播到out.shape"

# 场景2: 完整Shape广播 - 双向
- id: "C-BCAST-003"
  type: broadcast_shape
  sources: ["x.shape"]
  target: "y.shape"
  mode: bidirectional
  description: "x.shape和y.shape双向广播"

# 场景3: 广播到派生形状（两步约束模式）
# 需求：self.shape 需广播到 [batch1.shape[1], batch2.shape[2]]

# ❌ 错误写法：broadcast_shape 不支持 source_shape_expr
# - id: "C-BCAST-004"
#   type: broadcast_shape
#   sources: ["batch1.shape", "batch2.shape"]
#   target: "self.shape"
#   mode: unidirectional
#   source_shape_expr: "[sources[0][1], sources[1][2]]"  # ❌ 不支持此字段

# ✅ 正确写法：使用两步约束
# 步骤1：计算派生形状
- id: "C-BCAST-004-CALC"
  type: calculate
  sources: ["batch1.shape", "batch2.shape"]
  target: "_broadcast_target.shape"
  expression: "[sources[0][1], sources[1][2]]"
  description: "计算广播目标形状 [M, N]"

# 步骤2：检查广播兼容性
- id: "C-BCAST-004-BCAST"
  type: broadcast_shape
  sources: ["_broadcast_target.shape"]
  target: "self.shape"
  mode: unidirectional
  description: "self.shape可广播到[M, N]"
```
---

### 5. 条件约束 (conditional)

**语义**：当条件满足时，应用特定约束。

```yaml
- id: "C007"
  type: conditional
  condition:
    factor: "beta.value"
    expression: "== 0"
  then:
    type: ignore
    target: "self"
  else:
    type: require
    target: "self"
  description: "beta=0时self被忽略"
```
---

### 6. 匹配约束 (match)

**语义**：源因子和目标因子的特定维度必须相等。

```yaml
# 维度匹配
- id: "C008"
  type: match
  sources: ["batch1.shape"]
  source_index: 2  # batch1.shape[2]
  target: "batch2.shape"
  target_index: 1  # batch2.shape[1]
  description: "K维度匹配: batch1[2] == batch2[1]"
```
---

### 7. 依赖存在性约束 (existential)

**语义**：当某个因子存在/不存在时，影响其他因子的状态。

> **重要**：此约束仅用于**可选参数之间的依赖关系**。必选参数（`required: true`）之间无需添加此约束。

**使用条件**：
- trigger 参数必须是**可选参数**（`required: false`）
- effect.target 必须是可选参数或其属性

**正确示例**：
```yaml
# ✅ bias 是可选参数，当 bias 不存在时，bias.dtype 设为 NA
- id: "C-EXIST-001"
  type: existential
  trigger: "bias.exist"
  trigger_value: false
  effect:
    target: "bias.dtype"
    action: set_na  # 设置为NA
  description: "bias不存在时，bias.dtype设为NA"
```

**错误示例**（不应添加）：
```yaml
# ❌ self 和 batch1 都是必选参数，不存在可选之间的依赖关系
- id: "C-EXIST-002"
  type: existential
  trigger: "self.exist"
  trigger_value: false
  effect:
    target: "batch1.exist"
    action: set_na
  description: "self不存在时batch1设为NA"  # 错误：两个都是必选参数
```
---

### 8. 可转换约束 (convertible)

**语义**：目标因子的类型必须能够转换到源因子的类型。用于约束标量参数与Tensor参数之间的类型兼容性。

**转换规则**（参考互转换关系）：
- 整数类型间可以转换，也支持往浮点、复数类型转换
- 浮点类型间可以转换，也支持往复数类型转换
- 复数类型间可以转换
- BOOL支持往整数、浮点、复数类型转换

```yaml
- id: "C011"
  type: convertible
  sources: ["self.dtype"]           # 目标类型（转换终点）
  target: "beta.dtype"               # 源类型（需要可转换到目标）
  target_domain: ["float32", "float16", "int32", "int64", "bool"]  # 可选：限制候选域
  description: "beta类型需要可转换成self类型"
```
---

### 10. 可推导筛选约束 (inferable_filter)

**语义**：用于**链式依赖场景**，从target_domain中筛选出与sources兼容的类型。适用于多Tensor类型互推导的链式求解。

**核心特性**：
1. **链式依赖**：选择一个锚点独立采样，其他因子逐层推导
2. **兼容性筛选**：从候选域中筛选出与sources互推导的类型
3. **单目标约束**：每个约束只描述一个target因子的推导关系

**适用场景**：
- 多个Tensor类型需满足互推导关系（如 batch1.dtype, batch2.dtype, self.dtype）
- 需要链式求解的依赖关系（Level 0 → Level 1 → Level 2）

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 约束唯一标识 |
| `type` | string | 是 | 固定值 `inferable_filter` |
| `sources` | list | 是 | 源因子列表（1个或多个） |
| `target` | string | 是 | 目标因子（需推导的因子） |
| `target_domain` | list | 是 | 目标因子的候选类型列表 |
| `description` | string | 否 | 约束描述 |

**链式依赖设计模式**：

对于三个Tensor类型的互推导约束，采用链式依赖：

```yaml
# 原约束（多变量互推导，不推荐）
- id: "C-TYPE-001"
  type: inferable
  mode: tensor_tensor
  sources: ["batch1.dtype", "batch2.dtype", "self.dtype"]
  description: "三个Tensor类型需满足互推导关系"

# 链式依赖（推荐）
# Step 1: 选择锚点（Level 0 独立采样）
# batch1.dtype 作为锚点

# Step 2: Level 1 推导
- id: "C-TYPE-001"
  type: inferable_filter
  sources: ["batch1.dtype"]
  target: "batch2.dtype"
  target_domain: ["float32", "float16", "bfloat16"]
  description: "batch2.dtype 需与 batch1.dtype 兼容"

# Step 3: Level 2 推导
- id: "C-TYPE-002"
  type: inferable_filter
  sources: ["batch1.dtype", "batch2.dtype"]
  target: "self.dtype"
  target_domain: ["float32", "float16", "bfloat16"]
  description: "self.dtype 需与 (batch1, batch2) 的推导结果兼容"
```
---

### 11. 可推导约束 (inferable)

**语义**：多个因子的类型必须能够相互推导，即推导结果不能为空（不兼容）。支持两种推导模式：

| 模式 | 说明 | 参考文档 |
|------|------|----------|
| `tensor_tensor` | Tensor-Tensor 推导，两个 Tensor 之间类型推导 | 互推导关系.md |
| `tensor_scalar` | Tensor-Scalar 推导，Tensor 和 Scalar 之间类型推导 | TensorScalar互推导关系.md |

**两种模式的差异**：
- `tensor_tensor`：遵循 Type Promotion 规则，取较大类型
- `tensor_scalar`：Tensor 类型优先，Scalar 转换到 Tensor 类型

**示例对比**：
| 输入组合 | tensor_tensor 结果 | tensor_scalar 结果 (Tensor=第一列) |
|----------|-------------------|-----------------------------------|
| (f32, f16) | f32 | f16 |
| (f16, f32) | f32 | f32 |
| (s8, s8) | s8 | s8 |

```yaml
# Tensor-Tensor 推导（默认模式）
- id: "C012"
  type: inferable
  mode: tensor_tensor            # 可选，默认为 tensor_tensor
  sources: ["batch1.dtype", "batch2.dtype"]
  target: "self.dtype"           # 可选：推导结果赋值给目标
  description: "batch1和batch2类型需要可相互推导"

# Tensor-Scalar 推导
- id: "C013"
  type: inferable
  mode: tensor_scalar            # Tensor-Scalar 模式
  sources: ["input.dtype", "scalar.dtype"]  # 第一个为Tensor类型，第二个为Scalar类型
  target: "out.dtype"
  description: "Tensor和Scalar类型推导"
```
---

## 完整示例

```yaml
metadata:
  operator: "aclnnAddbmm"
  version: "1.0"
  description: "Addbmm算子完整约束定义"

factors:
  # 输入Tensor
  self.dtype: {type: dtype, param: self, io_type: input}
  self.shape: {type: shape, param: self, io_type: input}
  self.format: {type: format, param: self, io_type: input}
  self.exist: {type: exist, param: self, io_type: input}
  self.dimensions: {type: dimensions, param: self, io_type: input}
  
  # 输出Tensor
  out.dtype: {type: dtype, param: out, io_type: output}
  out.shape: {type: shape, param: out, io_type: output}
  out.format: {type: format, param: out, io_type: output}
  out.exist: {type: exist, param: out, io_type: output}
  out.dimensions: {type: dimensions, param: out, io_type: output}
  
  # 非标量参数
  beta.dtype: {type: dtype, param: beta, io_type: input}
  beta.exist: {type: exist, param: beta, io_type: input}
  # 枚举类型参数
  cubeMathType.dtype: {type: dtype, param: cubeMathType, io_type: input}
  cubeMathType.exist: {type: exist, param: cubeMathType, io_type: input}
  cubeMathType.enum_values: {type: enum_values, param: cubeMathType, io_type: input}

constraints:
  # ========== 形状约束 ==========
  - id: "SHAPE-001"
    type: match
    sources: ["batch1.shape"]
    source_index: 2
    target: "batch2.shape"
    target_index: 1
    description: "K维度匹配: batch1[2] == batch2[1]"
  
  - id: "SHAPE-002"
    type: calculate
    sources: ["batch1.shape", "batch2.shape"]
    target: "out.shape"
    expression: "[sources[0][1], sources[1][2]]"
    description: "out.shape = [M, N] = [batch1[1], batch2[2]]"
---