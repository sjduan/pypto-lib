# UT 生成完整流程

> 本文件由 SKILL.md 引用，包含 Phase UT-1 至 UT-6 的详细操作指南。

**适用场景**：算子 tests/ut/ 目录下不存在 test_*.cpp 文件，需要从零创建 UT。

**目标**：从零创建 UT，覆盖率达到 80%+。

---

## 目录

- [Phase UT-1: 信息收集与自动探索](#phase-ut-1-信息收集与自动探索)
- [Phase UT-2: op_host UT 编写](#phase-ut-2-op_host-ut-编写p0-优先)
- [Phase UT-3: op_api UT 编写](#phase-ut-3-op_api-ut-编写p1-按需)
- [Phase UT-4: op_kernel UT 编写](#phase-ut-4-op_kernel-ut-编写p2-按需)
- [Phase UT-5: 覆盖率验证](#phase-ut-5-覆盖率验证)
- [Phase UT-6: 生成最终报告](#phase-ut-6-生成最终报告)

---

## Phase UT-1: 信息收集与自动探索

**目标**：收集算子基本信息，自动探索层级支持、dtype/format 和编译配置

**输入**：算子完整路径（如 `ops-math/math/add`）

### Step 1.1: 确认算子基本信息

从算子路径提取基本信息：

```bash
# 路径格式：<repo>/<category>/<op>
# 示例：ops-math/math/add

# 提取各部分
REPO=$(echo $OP_PATH | cut -d'/' -f1)      # ops-math
CATEGORY=$(echo $OP_PATH | cut -d'/' -f2)  # math
OP_NAME=$(echo $OP_PATH | cut -d'/' -f3)   # add
```

### Step 1.2: 自动探索层级支持

```bash
# 检测各层级目录是否存在
ls $OP_PATH | grep -E "op_host|op_api|op_kernel|op_kernel_aicpu" 2>/dev/null
```

**输出交付件**：

| 层级名称 | 是否存在 | 优先级 | 是否需要创建 UT |
|----------|----------|--------|-----------------|
| op_host | 是/否 | P0 | 存在则必须创建 |
| op_api | 是/否 | P1 | 存在则按需创建 |
| op_kernel | 是/否 | P2 | 存在则按需创建 |
| op_kernel_aicpu | 是/否 | P2 | 与 op_kernel 互斥 |

**层级优先级**：P0（op_host）→ P1（op_api）→ P2（op_kernel）

**层级互斥关系**：`op_kernel` ⟷ `op_kernel_aicpu`（同一算子只存在一种）

### Step 1.3: 自动探索 dtype/format 支持

```bash
# 从 build.sh 提取支持的 SoC 列表
grep "SUPPORT_COMPUTE_UNIT_SHORT" $REPO/build.sh | sed 's/.*(\(.*\)).*/\1/' | tr -d '"' | tr ',' '\n'

# 从源码提取支持的 dtype
grep -rn "ge::DT_" $OP_PATH/op_host/*.cpp 2>/dev/null | grep -oE "DT_[A-Z0-9]+" | sort -u

# 从源码提取支持的 format
grep -rn "FORMAT_" $OP_PATH/op_host/*.cpp 2>/dev/null | grep -oE "FORMAT_[A-Z0-9]+" | sort -u
```

### Step 1.4: 确认编译命令

**优先通过帮助命令探索**：

```bash
# 查看编译帮助（推荐首先执行）
bash build.sh -h
```

`build.sh -h` 会输出当前算子仓支持的编译参数，包括：
- 支持的 SoC 版本列表
- 编译目标选项（ophost/opapi/opkernel）
- 覆盖率、调试等选项
- 示例命令

**常用编译命令模板**（从帮助信息提取）：

```bash
# 编译 op_host UT
bash build.sh -u --ophost --ops='<op_name>' --soc='<soc_version>'

# 编译 op_api UT
bash build.sh -u --opapi --ops='<op_name>' --soc='<soc_version>'

# 编译 op_kernel UT
bash build.sh -u --opkernel --ops='<op_name>' --soc='<soc_version>'

# 编译并生成覆盖率
bash build.sh -u --ophost --ops='<op_name>' --soc='<soc_version>' --cov
```

### Phase UT-1 完成检查

- [ ] 算子路径和名称已确认
- [ ] 层级支持已探索
- [ ] dtype/format 列表已确认
- [ ] 编译命令已确认

---

## Phase UT-2: op_host UT 编写（P0 优先）

**目标**：完成 op_host 层 Tiling 和 InferShape 测试

**前置条件**：算子存在 op_host/ 目录

> **详细指南**：[op-host-ut-generator.md](op-host-ut-generator.md)

### Step 2.1: 检测 CompileInfo 类型

不同算子的 CompileInfo 类型和命名空间不同，编写 Tiling 测试前必须先检测类型。

**检测方法**：

```bash
# 在 tiling 实现文件中查找 TilingParse<TYPE>
# 注意：文件可能在 arch32/ 或 arch35/ 目录下，根据芯片架构确定
find op_host -name "*_tiling*.cpp" -exec grep -l "TilingParse" {} \;
grep -n "TilingParse" op_host/<arch>/<op>_tiling_<arch>.cpp
```

**典型代码行**：
```cpp
IMPL_OP_OPTILING(Mul).TilingParse<BroadcastCompileInfo>(context, compileInfo);
```

**类型对照表**：

| TYPE | 命名空间 | 头文件 | 示例算子 |
|------|---------|--------|---------|
| `BroadcastCompileInfo` | `Ops::Base::` | `atvoss/broadcast/broadcast_tiling.h` | Mul, Div, Sub |
| `<Op>CompileInfo` | `optiling::` | 算子tiling头文件 | Add, Abs |

### Step 2.2: 分析算子属性

```bash
# 查看算子源码中的 GetAttr 调用
grep -rn "GetAttr\|GetAttrValue" <repo>/<category>/<op>/op_host/
```

**常见算子属性**：

| 算子类型 | 常见属性 | 命名空间 |
|---------|---------|---------|
| MatMul系列 | adj_x1, adj_x2, offset_x, opImplMode | Ops::NN |
| Reduce系列 | axes, keepdims | Ops::Math |
| Conv系列 | pads, strides, dilations, groups | Ops::NN |

### Step 2.3: 编写 Tiling 测试

**目录结构**：
```
<repo>/<category>/<op>/tests/ut/op_host/
├── CMakeLists.txt
├── test_<op>_infershape.cpp
├── test_<op>_tiling.cpp
└── arch35/                     # 芯片架构目录（如需要）
    └── test_<op>_tiling.cpp
```

**测试用例编写顺序**（TDD）：
1. **失败场景**：不支持的 dtype、空 tensor、无效参数
2. **成功场景**：各 dtype 组合、各 format 组合

**关键要点**：
- 必须配置 NodeAttrs：查看源码中的 `GetAttr` 调用确定需要的属性
- 必须配置 NodeInputTd：设置正确的 dtype 和 format
- 详细代码示例见 [op-host-ut-generator.md](op-host-ut-generator.md)

### Step 2.4: 编写 InferShape 测试

```cpp
// InferShape 测试示例
gert::InfershapeContextPara infershapeContextPara("Abs",
    {{{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
    {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
```

### Step 2.5: 编译验证

```bash
cd <repo> && bash build.sh -u --ophost --ops='<op_name>' --soc='<soc_version>'
```

### Phase UT-2 完成检查

- [ ] CompileInfo 类型已检测
- [ ] 算子属性已分析
- [ ] Tiling 测试已编写
- [ ] InferShape 测试已编写
- [ ] 编译通过

---

## Phase UT-3: op_api UT 编写（P1 按需）

**目标**：完成 op_api 层参数校验测试

**前置条件**：算子存在 op_api/ 目录

> **详细指南**：[op-api-ut-generator.md](op-api-ut-generator.md)

### Step 3.1: 确认接口支持

```bash
find <repo> -name "aclnn_<op>*.h" 2>/dev/null
```

### Step 3.2: 编写测试用例

**目录结构**：
```
<repo>/<category>/<op>/tests/ut/op_api/
├── CMakeLists.txt
├── test_aclnn_<op>.cpp
├── test_aclnn_<op>_out.cpp      # out变体（如有）
└── test_aclnn_<op>_inplace.cpp  # inplace变体（如有）
```

**测试用例编写顺序**（TDD）：
1. **异常用例**：nullptr 测试、无效 dtype 测试、shape 不匹配测试、超过 8 维测试
2. **正常用例**：所有支持的 dtype、所有支持的 format
3. **边界用例**：空 tensor、0 维 tensor

**核心组件**：
```cpp
// TensorDesc 构造
auto tensor = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

// OP_API_UT 宏
auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));
```

### Step 3.3: 编译验证

```bash
cd <repo> && bash build.sh -u --opapi --ops='<op_name>' --soc='<soc_version>'
```

### Phase UT-3 完成检查

- [ ] 接口支持已确认
- [ ] nullptr 测试已编写
- [ ] 无效 dtype 测试已编写
- [ ] 正常用例已覆盖所有 dtype
- [ ] 编译通过

---

## Phase UT-4: op_kernel UT 编写（P2 按需）

**目标**：完成 Kernel 层计算逻辑测试

**前置条件**：算子存在 op_kernel/ 或 op_kernel_aicpu/ 目录

> **详细指南**：[op-kernel-ut-generator.md](op-kernel-ut-generator.md)

### Step 4.1: 确认 Kernel 类型

| 目录 | Kernel 类型 | 测试框架 |
|------|------------|---------|
| op_kernel/ | AscendC Kernel | tikicpulib + ICPU_RUN_KF |
| op_kernel_aicpu/ | AICPU Kernel | NodeDefBuilder + RUN_KERNEL |

> **注意**：`op_kernel/` 和 `op_kernel_aicpu/` 是互斥的，同一算子只会存在其中一种。

### Step 4.2: 编写测试用例

**AscendC Kernel 示例**：
```cpp
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void <op>(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

// 分配内存
uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(<Op>TilingData));

// 设置 Tiling 数据
<Op>TilingData* tilingData = reinterpret_cast<<Op>TilingData*>(tiling);
tilingData->usedCoreNum = 1;

// 执行 Kernel
ICPU_SET_TILING_KEY(tilingKey);
ICPU_RUN_KF(<op>, numBlocks, x, y, workspace, tilingData);

// 释放内存
AscendC::GmFree((void*)x);
```

**AICPU Kernel 示例**：
```cpp
#include "node_def_builder.h"

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "OpName", "OpName")               \
        .Input({"input1", data_types[0], shapes[0], datas[0]})       \
        .Output({"output", data_types[1], shapes[1], datas[1]});

RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
```

### Step 4.3: 编译验证

```bash
cd <repo> && bash build.sh -u --opkernel --ops='<op_name>' --soc='<soc_version>'
```

### Phase UT-4 完成检查

- [ ] Kernel 类型已确认
- [ ] 测试用例已编写
- [ ] Tiling 参数已配置（AscendC）
- [ ] 编译通过

---

## Phase UT-5: 覆盖率验证

**目标**：验证 UT 覆盖率是否达标

**前置条件**：所有层级 UT 已编写完成，编译通过

### Step 5.1: 获取覆盖率报告

```bash
cd <repo> && bash build.sh -u --ophost --ops='<op>' --soc='<soc>' --cov
```

### Step 5.2: 判断覆盖率类型并提取

```bash
# 进入覆盖率目录
cd <repo>/build/tests/ut/cov_report/cpp_utest

# 查看覆盖率报告包含的文件路径
lcov --list ops.info_filtered | head -50
```

| 覆盖率类型 | 文件路径特征 | 处理方式 |
|------------|-------------|----------|
| 单算子覆盖率 | 仅包含当前算子路径 | 直接使用 |
| 全局覆盖率 | 包含多个算子路径 | 需要提取 |

**提取单算子覆盖率**：
```bash
lcov --extract ops.info_filtered \
    "*/<category>/<op>/op_api/*" \
    "*/<category>/<op>/op_host/*" \
    "*/<category>/<op>/op_kernel/*" \
    -o /tmp/<op>_cov.info

lcov --summary /tmp/<op>_cov.info
```

> **详细说明**：[覆盖率提取指南](../coverage-enhancement/coverage-extraction-guide.md)

### Step 5.3: 验证覆盖率达标

| 指标 | 要求 | 说明 |
|------|------|------|
| 行覆盖率 | >= 80% | 必须 |
| 函数覆盖率 | >= 80% | 必须 |
| 分支覆盖率 | >= 80% | 推荐 |

### Phase UT-5 完成检查

- [ ] 覆盖率类型已判断
- [ ] 单算子覆盖率已提取（如需要）
- [ ] 行覆盖率 >= 80%
- [ ] 函数覆盖率 >= 80%
- [ ] 覆盖率不达标时已跳转覆盖率增强流程

---

## Phase UT-6: 生成最终报告

**目标**：生成 UT 生成总结报告

**具体操作**：使用 ut-summary-template.md 模板生成报告

**输出交付件**：

| 项目 | 内容 |
|------|------|
| 生成的 UT 文件路径 | 各层级测试文件路径列表 |
| 各层支持情况 | op_host/op_api/op_kernel 支持状态 |
| 覆盖率结果 | 行覆盖率、函数覆盖率、分支覆盖率 |
| 遇到的问题 | 问题描述及解决方案 |

> **模板文件**：[ut-summary-template.md](ut-summary-template.md)

### Phase UT-6 完成检查

- [ ] UT 文件路径已记录
- [ ] 各层支持情况已记录
- [ ] 覆盖率结果已记录
- [ ] 问题及解决方案已记录

---

## 最终验证

- [ ] 编译通过
- [ ] 所有测试用例通过
- [ ] 覆盖率达标（行覆盖率 ≥ 80% 且 函数覆盖率 ≥ 80%）
- [ ] 最终报告生成完成