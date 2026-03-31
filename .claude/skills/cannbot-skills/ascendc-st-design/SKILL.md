---
name: ascendc-st-design
description: Ascend C 算子测试设计技能，基于aclnn接口md资料完成算子测试用例设计。当需要（1）设计算子测试用例、（2）编写测试计划（plan.yaml）、（3）分析算子参数和依赖关系、（4）提取测试因子、（5）生成测试用例组合时使用此技能。
---

# Ascend C 算子测试设计

本技能提供 Ascend C 算子测试设计的完整工作流程，基于aclnn测试框架进行算子功能和精度验证。

## 核心工作流程

### 1. 输入文件校准

在项目根目录下查找算子资料 `{operator_name}.md` 文件；或者指定文件路径。

### 1.1 输出路径规范

**所有测试设计结果统一存放到项目根目录的 `result/` 下**：

```
{项目根目录}/
└── result/
    └── {operator_name}/
        ├── 03_参数定义.yaml
        ├── 04_测试因子.yaml
        ├── 05_约束定义.yaml
        ├── 06_求解配置.yaml
        ├── 07_因子值.csv
        ├── L0_test_cases.csv
        ├── L0_coverage_report.yaml
        └── L1_test_cases.csv
```

**路径变量**：
- `PROJECT_ROOT`：项目根目录（执行脚本时的工作目录）
- `RESULT_DIR`：`{PROJECT_ROOT}/result/{operator_name}/`

### 2. 算子参数定义

根据参数类型（aclTensor/aclScalar/int8_t等）定义参数规范。

**详细规范**：参考 [parameter-definition.md](references/parameter-definition.md)

**关键要点**：
- 为每个参数定义 `name`、`type`、`required`、`io_type` 属性
- **数据类型必须完整**：从原始资料的数据类型列中完整复制所有类型，不允许遗漏
- `io_type` 字段：标识参数的输入输出类型
  - `input`：输入参数（需准备测试数据）
  - `output`：输出参数（无需定义`value_range`和`special_range/special_value`）
- Tensor类型需定义 `format`、`dimensions`、`dtype_with_ranges`
- Scalar/整型需定义 `is_enum`、`dtype_with_values`
- 明确 `value_range` 和 `special_range/special_value`
- 无需定义workspaceSize和executor参数

### 2.1 value_range 定义规范

**必须覆盖的取值范围**：
1. **全量默认值**：按照 `references/parameter-definition.md` 中的默认值定义
2. **算子特定值**：根据算子功能识别的特殊值

**定义步骤**：
1. 复制默认值模板（从 parameter-definition.md）
2. 根据算子功能删除不适用的范围（如归一化操作删除负值范围）
3. 添加算子特定的约束值

**示例**：
- 通用算子：使用全量默认值
- 归一化算子：仅保留 [0, 1] 相关范围
- 矩阵运算：关注数值稳定性范围

**float16 全量默认值示例**：
```yaml
- dtype: "float16"
  value_range: [[0, 0.001], [0.001, 0.01], [0.01, 1], [1, 2], [2, 10], [10, 1000], [-0.001, 0], [-0.01, -0.001], [-1, -0.01], [-2, -1], [-10, -2], [-1000, -10], [-1, 1], [-0.01, 0.01], [-100, 100], [0, 0], [-65504.0, 65504.0], [-0.0078125, 0.0078125], [65504.0, 65504.0], [-65504.0, -65504.0], [-6.103515625e-05, -6.103515625e-05], [6.103515625e-05, 6.103515625e-05], ["inf", "inf"], ["-inf", "-inf"], ["nan", "nan"]]
```

### 3. 使用脚本自动提取测试因子

**推荐方式**：使用 `generate_test_factors.py` 脚本自动提取

```bash
# 在项目根目录下执行
python skills/ascendc-st-design/scripts/generate_test_factors.py <参数定义.yaml> <测试因子.yaml>

# 示例
python skills/ascendc-st-design/scripts/generate_test_factors.py \
    result/{operator_name}/03_参数定义.yaml \
    result/{operator_name}/04_测试因子.yaml
```

**脚本功能**：
- ✅ 自动识别参数类型（aclTensor/aclScalar/int8_t等）
- ✅ 提取所有测试因子（存在性、格式、维度、数据类型、取值范围、特殊值等）
- ✅ 生成规范的YAML格式输出

### 4. 参数依赖关系分析

解析输入资料，理解算子的数学公式及计算逻辑，提取 GetWorkspaceSize 接口中950系列产品支持的参数、参数取值范围和参数依赖关系。

**详细规范**：参考 [dependency-yaml-spec.md](references/dependency-yaml-spec.md)

#### 4.1 约束分析维度（按优先级）

在识别因子依赖关系时，按以下优先级进行分析：

1. **数据类型依赖**
   - 类型推导链、类型等价约束
   - 类型可转换性、类型互斥规则
   - 精度层级关系

2. **形状依赖**
   - 维度数约束、维度大小计算
   - Broadcast方向性和兼容性
   - 指定维度值匹配

3. **数值依赖**
   - 标量取值范围、枚举值定义域
   - 特殊数值语义、数值精度影响

4. **存在性依赖**
   - 可选参数联动
   - 条件参数约束

#### 4.2 约束定义文件规范
**文件结构**：
##### 元信息
metadata:
  operator: "{operator_name}"
  version: "1.0"
  description: "{operator_name}算子约束关系定义"
##### 因子节点定义
factors:
  {param}.{attribute}: {type: {type}, param: {param}, io_type: {io_type}}
  ##### ...
##### 约束定义
constraints:
  - id: "C-XXX-001"
    type: {constraint_type}
    # ...

#### 4.3 因子节点定义

因子节点是约束系统的基本单元，代表一个可测试的属性。

```yaml
# 因子节点标识规范
{id}: "{parameter}.{attribute}"
```

**属性类型**：

| 属性     | 说明     | 示例           |
| -------- | -------- | -------------- |
| `dtype`  | 数据类型 | `self.dtype`   |
| `dimensions`  | 维度   | `self.dimensions`   |
| `shape`  | 张量形状 | `batch1.shape` |
| `value`  | 枚举值   | `cubeMathType.value`   |
| `format` | 数据格式 | `input.format` |
| `exist`  | 存在性   | `bias.exist`   |

#### 4.4 约束类型

| 类型 | 语义 | 表达式示例 | 说明 |
|------|------|------------|------|
| `calculate` | 计算约束 | `target = expr(sources)` | 通用计算，`sources` **不可为空** |
| `broadcast_dim` | 指定维度广播 | 维度兼容性检查 | 单维度广播约束 |
| `broadcast_shape` | 张量广播 | 形状兼容性检查 | 完整形状广播约束，支持单向/双向模式 |
| `conditional` | 条件约束 | if-then-else 分支 | - |
| `match` | 匹配约束 | 维度/属性匹配 | **双向约束**，与 `calculate` 语义不同 |
| `existential` | 存在性约束 | 可选参数联动 | - |
| `convertible` | 可转换约束 | 类型兼容性检查 | - |
| `inferable_filter` | 链式依赖约束 | 多Tensor类型互推导 | - |
| `inferable` | 可推导约束 | 类型推导检查 | - |

**约束类型设计原则**：
 - `calculate` 是通用计算约束，等值场景使用 `expression: "sources[0]"`
 - `match` 是**双向约束**，`calculate` 是**单向计算**，语义不同不应合并
 - `broadcast_dim` 用于单维度广播，`broadcast_shape` 用于完整形状广播
 - 固定取值范围（如 `cubeMathType.enum_values: [0, 1, 2, 3]`）已在 `04_测试因子.yaml` 中定义，无需添加约束

**重要限制**：
- `broadcast_shape` **不支持** `source_shape_expr` 字段（派生形状表达式）
- 如需表达"广播到派生形状"的约束，应使用**两步约束**模式（详见示例）

#### 4.5 约束定义示例

```yaml
# 类型约束：等值（使用 calculate）
- id: "TYPE-002"
  type: calculate
  sources: ["self.dtype"]
  target: "out.dtype"
  expression: "sources[0]"
  description: "输出类型等于self类型"

# 形状约束：计算
- id: "SHAPE-001"
  type: calculate
  sources: ["batch1.shape", "batch2.shape"]
  target: "out.shape"
  expression: "[sources[0][1], sources[1][2]]"
  description: "out.shape = [M, N]"

# 形状约束：指定维度广播
- id: "BCAST-001"
  type: broadcast_dim
  sources: ["batch1.shape"]
  source_index: 0
  target: "batch2.shape"
  target_index: 1
  description: "batch1[0]与batch2[1]需满足广播关系"

# 形状约束：张量广播 - 单向
- id: "BCAST-002"
  type: broadcast_shape
  sources: ["out.shape"]
  target: "bias.shape"
  mode: unidirectional
  description: "bias.shape可广播到out.shape"

# 形状约束：张量广播 - 双向
- id: "BCAST-003"
  type: broadcast_shape
  sources: ["x.shape"]
  target: "y.shape"
  mode: bidirectional
  description: "x.shape和y.shape双向广播"

# 形状约束：广播到派生形状（两步约束模式）
# 场景：self.shape 需广播到 [batch1.shape[1], batch2.shape[2]]
# ❌ 错误写法：broadcast_shape 不支持 source_shape_expr
# - id: "BCAST-004"
#   type: broadcast_shape
#   sources: ["batch1.shape", "batch2.shape"]
#   target: "self.shape"
#   mode: unidirectional
#   source_shape_expr: "[sources[0][1], sources[1][2]]"  # ❌ 不支持

# ✅ 正确写法：使用两步约束
# 步骤1：计算派生形状
- id: "BCAST-004-CALC"
  type: calculate
  sources: ["batch1.shape", "batch2.shape"]
  target: "_broadcast_target.shape"
  expression: "[sources[0][1], sources[1][2]]"
  description: "计算广播目标形状 [M, N]"

# 步骤2：检查广播兼容性
- id: "BCAST-004-BCAST"
  type: broadcast_shape
  sources: ["_broadcast_target.shape"]
  target: "self.shape"
  mode: unidirectional
  description: "self.shape可广播到[M, N]"

# 维度约束：匹配
- id: "MATCH-001"
  type: match
  sources: ["batch1.shape"]
  source_index: 2
  target: "batch2.shape"
  target_index: 1
  description: "K维度匹配: batch1[2] == batch2[1]"

# 类型约束：可转换
- id: "C-TYPE-004"
  type: convertible
  sources: ["self.dtype"]           # 目标类型（转换终点）
  target: "beta.dtype"               # 源类型（需要可转换到目标）
  target_domain: ["float32", "float16", "int32", "int64", "bool"]  # 可选：限制候选域
  description: "beta类型需要可转换成self类型"

# 链式依赖（推荐）：多个Tensor类型需满足互推导关系（如 batch1.dtype, batch2.dtype, self.dtype）
# Step 1: batch1.dtype 作为锚点（Level 0 独立采样）

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

#### 4.6 检查清单
 - 05_约束定义.yaml必须包含metadata、factors、constraints节点
 - constraints节点中的sources不为[]；即：固定取值范围无需定义约束


### 5. 生成通用约束

**推荐方式**：使用 `generate_implicit_constraints.py` 脚本自动生成

```bash
# 在项目根目录下执行
python skills/ascendc-st-design/scripts/generate_implicit_constraints.py <测试因子.yaml> <约束定义.yaml>

# 示例
python skills/ascendc-st-design/scripts/generate_implicit_constraints.py \
    result/{operator_name}/04_测试因子.yaml \
    result/{operator_name}/05_约束定义.yaml \
    --verbose
```

**脚本功能**：
- ✅ 自动识别需要生成隐式依赖的因子
- ✅ 生成三类通用隐式约束
- ✅ 避免重复约束（幂等性保证）
- ✅ 自动备份原有约束文件

**通用依赖规则**：

1. **Tensor 输入**：`{param}.shape` 依赖于 `{param}.dimensions`
   - 根据 dimensions 推导具体的 shape 值

2. **所有输入**：`{param}.value_range` 依赖于 `{param}.dtype`
   - 根据 dtype 选择对应的取值范围（如 float16 对应 [-65504.0, 65504.0]）

3. **非 Tensor 且非枚举类型**：`{param}.value` 依赖于 `{param}.value_range`
   - 根据 value_range 随机生成具体的值

**命令行参数**：
```bash
usage: generate_implicit_constraints.py [-h] [--verbose]
                                      factors_file constraints_file

参数说明:
  factors_file      测试因子YAML文件（04_测试因子.yaml）
  constraints_file  约束定义YAML文件（05_约束定义.yaml）
```

### 6. 构建测试因子依赖图

**推荐方式**：使用 `generate_solver_config.py` 脚本自动生成

```bash
# 在项目根目录下执行
python skills/ascendc-st-design/scripts/generate_solver_config.py <约束定义.yaml> <求解配置.yaml>

# 示例
python skills/ascendc-st-design/scripts/generate_solver_config.py \
    result/{operator_name}/05_约束定义.yaml \
    result/{operator_name}/06_求解配置.yaml
```

**脚本功能**：
- ✅ 解析约束定义，构建因子依赖图
- ✅ 拓扑排序，确定求解层级
- ✅ 识别锚点因子（入度为0）
- ✅ 输出求解配置到YAML文件

**输出示例**：
```yaml
solver:
  strategy: topological
  anchors:
    - batch1.dtype
    - batch1.shape
    - self.dtype
    # ...
  derivation_order:
    level_0: [batch1.dtype, batch1.shape, self.dtype, ...]
    level_1: [alpha.dtype, batch2.shape, beta.dtype, out.dtype]
    level_2: [out.shape]
```

### 7. 约束求解与测试因子值生成

**推荐方式**：使用 `generate_factor_values.py` 脚本自动生成满足约束的测试因子值

```bash
# 在项目根目录下执行
python skills/ascendc-st-design/scripts/generate_factor_values.py <求解配置.yaml> <约束定义.yaml> <测试因子.yaml> <输出.csv> [选项]

# 示例
python skills/ascendc-st-design/scripts/generate_factor_values.py \
    result/{operator_name}/06_求解配置.yaml \
    result/{operator_name}/05_约束定义.yaml \
    result/{operator_name}/04_测试因子.yaml \
    result/{operator_name}/07_因子值.csv \
    --max-cases 10000
```

**脚本功能**：
- ✅ 基于拓扑排序的约束求解
- ✅ 锚点因子独立采样
- ✅ 逐层推导满足依赖关系
- ✅ 自动处理类型推导、形状约束、维度约束等
- ✅ 生成满足所有约束的测试因子组合

**约束求解流程**：
1. 加载求解配置、约束定义和测试因子
2. 识别锚点因子（Level 0），独立随机采样
3. 按拓扑顺序逐层推导：
   - Level 1：根据锚点推导
   - Level 2：根据 Level 1 推导
   - ... 以此类推
4. 验证每个测试用例是否满足所有约束
5. 输出满足约束的测试因子值到 CSV 文件

**命令行参数**：
```bash
usage: generate_factor_values.py [-h] [--max-cases N] [--seed N] [--verbose] solver_config constraints factors output

参数说明:
  solver_config  求解配置YAML文件（06_求解配置.yaml）
  constraints    约束定义YAML文件（05_约束定义.yaml）
  factors        测试因子YAML文件（04_测试因子.yaml）
  output         输出的CSV文件（07_因子值.csv）
  --max-cases    最大用例数（默认10000）
  --seed         随机数种子（用于复现结果）
  --verbose      详细输出模式
```

**输出示例**：
```csv
case_id,batch1.dtype,batch1.shape,batch2.dtype,batch2.shape,self.dtype,self.shape,out.dtype,out.shape
0,float16,"[10, 3, 4]",float16,"[10, 4, 5]",float16,"[3, 5]",float16,"[3, 5]"
1,float32,"[5, 2, 3]",float32,"[5, 3, 4]",float32,"[2, 4]",float32,"[2, 4]"
2,bfloat16,"[1, 8, 16]",bfloat16,"[1, 16, 32]",bfloat16,"[8, 32]",bfloat16,"[8, 32]"
```

### 8. 测试用例生成（L0/L1）

**推荐方式**：使用统一的 `generate_test_cases.py` 脚本

```bash
python scripts/generate_test_cases.py <参数定义.yaml> <测试因子.yaml> <因子值.csv> <输出目录> --level <L0|L1> [选项]
```

**测试用例生成示例（分步执行L0和L1）**：

> **注意**：由于 L0 和 L1 算法复杂度较高，建议分两步执行，并设置 **5分钟超时**。

```bash
# 在项目根目录下执行

# 步骤1: 生成 L0 用例（单因子覆盖，≤200条）
python skills/ascendc-st-design/scripts/generate_test_cases.py \
    result/{operator_name}/03_参数定义.yaml \
    result/{operator_name}/04_测试因子.yaml \
    result/{operator_name}/07_因子值.csv \
    result/{operator_name}/ \
    --level L0 \
    --verbose

# 步骤2: 生成 L1 用例（两两组合覆盖，500~700条）
python skills/ascendc-st-design/scripts/generate_test_cases.py \
    result/{operator_name}/03_参数定义.yaml \
    result/{operator_name}/04_测试因子.yaml \
    result/{operator_name}/07_因子值.csv \
    result/{operator_name}/ \
    --level L1 \
    --target-count 500 \
    --seed 42 \
    --verbose
```

**超时设置**：在调用 Bash 工具时设置 `timeout=300000`（5分钟）

**脚本功能**：
- ✅ 支持L0和L1两种用例级别
- ✅ L0：覆盖所有单个因子值（≤200条）
- ✅ L1：覆盖所有因子值的两两组合（500~700条）
- ✅ 自动生成覆盖度报告
- ✅ 自动转换为标准格式
- ✅ 支持批量生成
- ✅ 严格的参数冲突检测

**命令行参数**：
```bash
必需:
  --level {L0,L1} [...]  用例级别（必填，支持多个）
                        L0: 单因子覆盖（≤200条）
                        L1: 两两组合覆盖（500~700条）

可选:
  --aclnn-name NAME      算子名称（默认从参数定义中提取）
  --target-count N       L1目标用例数量（默认500，仅L1有效）
                        【L0使用此参数会报错退出】
  --seed N               随机数种子（用于复现L1补齐，仅L1有效）
                        【L0使用此参数会报错退出】
  --report-output FILE   覆盖度报告文件名（默认: {level}_coverage_report.yaml）
                        批量生成时自动添加级别前缀（L0_, L1_）
  --case-output FILE     测试用例文件名（默认: {level}_test_cases.csv）
                        批量生成时自动添加级别前缀（L0_, L1_）
  --verbose              详细输出模式
```

**用例生成流程**：

**L0流程**：
1. 加载参数定义、测试因子和因子值
2. 提取所有需要覆盖的因子值
3. 使用贪心算法选择覆盖所有因子值的最小用例集
4. 生成覆盖度报告
5. 转换为标准L0用例格式并输出

**L1流程**：
1. 加载参数定义、测试因子和因子值
2. 提取所有因子值，生成两两组合（笛卡尔积）
3. 使用贪心算法选择覆盖所有两两组合的用例集
4. 如果用例数 < 目标数量，从已选用例中随机补齐
5. 生成两两组合覆盖度报告
6. 转换为标准L1用例格式并输出

**输出文件**：
批量（--level L0 L1）：
- `L0_coverage_report.yaml`, `L0_test_cases.csv`
- `L1_coverage_report.yaml`, `L1_test_cases.csv`

### 9. 测试设计结果总结

在 `result` 目录总结算子测试设计过程与结果：

- 算子参数定义
- 提取测试因子
- 参数依赖关系分析
- 生成隐式约束
- 构建测试因子依赖图
- 约束求解与测试因子值生成
- 测试用例生成（L0/L1）

## 参考文件

### 技能文档

- **[parameter-definition.md](references/parameter-definition.md)** - 参数定义规范和模板
- **[dependency-yaml-spec.md](references/dependency-yaml-spec.md)** - 依赖关系约束表达模型规范

### 自动化工具

- **[scripts/generate_test_factors.py](scripts/generate_test_factors.py)** - 测试因子提取脚本
- **[scripts/generate_implicit_constraints.py](scripts/generate_implicit_constraints.py)** - 隐式约束生成脚本
- **[scripts/generate_solver_config.py](scripts/generate_solver_config.py)** - 求解配置生成脚本
- **[scripts/generate_factor_values.py](scripts/generate_factor_values.py)** - 自动生成满足约束的测试因子值脚本
- **[scripts/generate_test_cases.py](scripts/generate_test_cases.py)** - 测试用例生成脚本（支持L0和L1）

## 工具使用示例

### 完整工作流程（推荐）

#### 步骤1: 定义参数（03_参数定义.yaml）
参考官方文档和算子API说明，编写参数定义文件。

#### 步骤2: 提取测试因子（自动化）
python skills/ascendc-st-design/scripts/generate_test_factors.py \
  result/{operator_name}/03_参数定义.yaml \
  result/{operator_name}/04_测试因子.yaml

#### 步骤3: 生成隐式约束（自动化）
python skills/ascendc-st-design/scripts/generate_implicit_constraints.py \
  result/{operator_name}/04_测试因子.yaml \
  result/{operator_name}/05_约束定义.yaml \
  --verbose

#### 步骤4: 定义约束关系
#### 参考 dependency-yaml-spec.md 文档，生成算子特定的约束
#### 自动生成隐式约束

#### 步骤5: 生成求解配置（自动化）
python skills/ascendc-st-design/scripts/generate_solver_config.py \
  result/{operator_name}/05_约束定义.yaml \
  result/{operator_name}/06_求解配置.yaml

#### 步骤6: 约束求解与测试因子值生成（自动化）

```bash
python skills/ascendc-st-design/scripts/generate_factor_values.py \
  result/{operator_name}/06_求解配置.yaml \
  result/{operator_name}/05_约束定义.yaml \
  result/{operator_name}/04_测试因子.yaml \
  result/{operator_name}/07_因子值.csv \
  --max-cases 10000 \
  --seed 42
```

#### 步骤7: 测试用例生成（分步执行，超时5分钟）

> **注意**：建议分两步执行 L0 和 L1，避免超时

```bash
# 步骤7.1: 生成 L0 用例（超时设置: 5分钟）
python skills/ascendc-st-design/scripts/generate_test_cases.py \
  result/{operator_name}/03_参数定义.yaml \
  result/{operator_name}/04_测试因子.yaml \
  result/{operator_name}/07_因子值.csv \
  result/{operator_name}/ \
  --level L0 \
  --verbose

# 步骤7.2: 生成 L1 用例（超时设置: 5分钟）
python skills/ascendc-st-design/scripts/generate_test_cases.py \
  result/{operator_name}/03_参数定义.yaml \
  result/{operator_name}/04_测试因子.yaml \
  result/{operator_name}/07_因子值.csv \
  result/{operator_name}/ \
  --level L1 \
  --target-count 500 \
  --seed 42 \
  --verbose
```