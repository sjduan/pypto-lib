# NPU 架构代际说明

本文档说明 Ascend NPU 的架构代际划分及其对算子开发的影响。

---

## 目录

1. [架构代际概述](#架构代际概述)
2. [NpuArch 与 SocVersion 对应关系](#npuarch-与-socversion-对应关系)
3. [架构特定代码目录](#架构特定代码目录)
4. [开发指导](#开发指导)
5. [Ascend950 (arch35) 特殊优化](#ascend950-arch35-特殊优化)

---

## 架构代际概述

### 核心概念

| 概念 | 说明 | 特点 |
|-----|------|------|
| **NpuArch** | 芯片架构 | 相对稳定，定义指令集和微架构 |
| **SocVersion** | 片上系统版本 | 变化频繁，软件命名标识 |

### 架构代号别名

在部分文档和代码中，也使用以下简写代号：

| 代号 | 对应芯片 | 说明 |
|-----|---------|------|
| **A2** | Ascend910B 系列 | 910B1, 910B2, 910B3, 910B4 等 |
| **A3** | Ascend910_93 / Ascend910C | 推理芯片（910_93） |
| **A5** | Ascend950DT / Ascend950PR | 新一代芯片（Decode/Prefill） |

**核心关系：一对多**

一个 NpuArch 可以支持多个 SocVersion。例如：
- `DAV_2201` 支持：Ascend910B1、Ascend910B2、Ascend910_93
- 主要差异：核内计算核心数量、核间通信能力

> **注意**：对 NPU 核内算子开发来说，通常不需要感知具体 SocVersion，使用 NpuArch 来区分芯片有利于易用性和可维护性。

---

## NpuArch 与 SocVersion 对应关系

### 架构体系层次

芯片体系包含四个层次：服务器产品系列、芯片架构名、__NPU_ARCH__、架构编译宏（DAV_*）。

### 完整映射表

| 服务器产品 | 芯片架构名 | __NPU_ARCH__ | 架构编译宏 | 目录标识 | 特点 |
|-----------|------------|---------------|------------|---------|------|
| **Atlas A3 推理系列** | Ascend910_93 | 2201 | DAV_2201 | arch32/通用 | 推理芯片 |
| **Atlas A2 训练/推理系列** | Ascend910B | 2201 | DAV_2201 | arch32/通用 | 主流训练芯片 |
| **Atlas 训练系列** | Ascend910 | 1001 | DAV_1001 | arch32/通用 | 初代训练架构 |
| **Atlas 推理系列** | Ascend310P | 2002 | DAV_2002 | arch32/通用 | 推理芯片 |
| **Atlas 200I/500 A2 推理产品** | Ascend310B | 3002 | DAV_3002 | arch32/通用 | 推理芯片 |
| **Atlas 200I/300I A3 推理产品** | Ascend910_93 | 2201 | DAV_2201 | arch32/通用 | 推理芯片 |
| **Atlas 800I A2 推理产品** | Ascend310B | 3002 | DAV_3002 | arch32/通用 | 推理芯片 |
| **Atlas 800 训练系列** | Ascend910B | 2201 | DAV_2201 | arch32/通用 | 训练芯片 |
| **Atlas 900 训练系列** | Ascend910B | 2201 | DAV_2201 | arch32/通用 | 训练芯片 |
| **Atlas A5 训练系列** | Ascend950DT | 3510 | DAV_3510 | arch35 | 新一代（Decode） |
| **Atlas A5 推理系列** | Ascend950PR | 3510 | DAV_3510 | arch35 | 新一代（Prefill） |

**说明**：
- **服务器产品**：面向市场的服务器产品系列，如 Atlas A2、Atlas A3、Atlas A5、Atlas 900 等
- **芯片架构名**：具体的芯片型号，如 Ascend910B、Ascend910_93、Ascend310P、Ascend310B、Ascend950DT、Ascend950PR
- **__NPU_ARCH__**：Device 侧编译宏，用于标识 AI 处理器的架构版本，由四位数字组成
- **架构编译宏（DAV_*）**：Host 侧编译宏，用于条件编译不同架构的代码
- **对应关系**：多个服务器产品可能使用同一芯片架构（如 Atlas A2 和 A3 推理都使用 Ascend910B/Ascend910_93）

**Atlas A5 芯片说明**：
- **Ascend950DT**：950DT = 950 Decode，用于 Decode 场景（如 LLM 生成阶段）
- **Ascend950PR**：950PR = 950 Prefill，用于 Prefill 场景（如 LLM 预填充阶段）

### 架构分组

| 分组 | NpuArch | __NPU_ARCH__ | 目录 | 特点 |
|-----|---------|---------------|------|------|
| **通用架构** | Ascend910/910B/910_93, Ascend310P/310B | 1001, 2002, 2201, 3002 | arch32 / 通用代码 | 通用实现，兼容性好 |
| **Ascend950 架构** | Ascend950DT/910PR | 3510 | arch35 | 特殊优化，性能更优 |

---

## 架构特定代码目录

### 目录结构

```
op_name/
├── op_kernel/
│   ├── *.cpp / *.h          # 通用实现（适用于所有架构）
│   ├── arch32/              # 特定于 arch32 系列的优化（可选）
│   │   └── *_dag.h
│   └── arch35/              # Ascend950 专用优化
│       ├── *_dag.h          # 针对 950 优化的 DAG 定义
│       └── *_struct.h       # 针对 950 的模板参数声明
├── op_host/
│   ├── *tiling*.cpp         # 通用 Tiling 实现
│   └── arch35/              # Ascend950 专用 Tiling
│       └── *_tiling_arch35.cpp
└── examples/
    ├── test_*.cpp           # 通用测试
    └── arch35/              # 950 专用测试
        └── test_*.cpp
```

### 何时使用 arch35 目录

1. **Ascend950 特有硬件特性**：利用 950 新增的指令或硬件单元
2. **内存布局优化**：950 的 UB/L1/L0 大小或布局不同
3. **调度策略差异**：950 的最优分块策略可能不同
4. **性能敏感算子**：需要针对 950 单独优化以达到最佳性能

---

## 开发指导

### 获取当前架构

**Kernel 端（Tiling 阶段）**：

```cpp
#include "platform/soc_spec.h"

auto platformInfo = context->GetPlatformInfo();
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
NpuArch npuArch = ascendcPlatform.GetCurNpuArch();

bool isArch35 = (npuArch == NpuArch::DAV_3510);
bool isArch2201 = (npuArch == NpuArch::DAV_2201);
```

**Host 端（aclnn）**：

```cpp
#include "platform/soc_spec.h"

NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
bool IsArch35 = npuArch == NpuArch::DAV_3510;
```

### 架构条件编译

**推荐方式：模板参数**

```cpp
template <bool isArch35>
__aicore__ inline void ProcessArchSpecific() {
    if constexpr (isArch35) {
        // Ascend950 特定优化
    } else {
        // 通用实现
    }
}
```

**Tiling 中选择实现**：

```cpp
// op_host/tiling 代码
if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
    tilingData.SetTilingKey(ARCH35_TILING_KEY);
} else {
    tilingData.SetTilingKey(GENERAL_TILING_KEY);
}
```

### 编译配置

在 CMakeLists.txt 中指定目标架构：

```cmake
# Ascend910B (arch32)
set(SOC_VERSION "Ascend910B")

# Ascend950 (arch35)
set(SOC_VERSION "Ascend950")
```

---

## Ascend950 (arch35) 特殊优化

Ascend950 架构包含 Ascend950DT（Decode）和 Ascend950PR（Prefill）两种芯片（NpuArch::DAV_3510），具有以下独有特性：

### 950 独有特性

Ascend950 (NpuArch::DAV_3510) 具有以下独有特性，仅在 `arch35/` 目录下使用：

| 特性 | 说明 | 典型算子 | 文件示例 |
|-----|------|---------|---------|
| **Regbase 编程** | 直接操作寄存器，更高性能 | 量化算子 | `quantize_*_regbase.h` |
| **SIMT 编程** | 线程级并行编程模型 | 随机数、排序 | `*_simt.h` |
| **FP8 格式** | 8-bit 浮点格式 | 量化、动态量化 | 见下方说明 |

### 1. Regbase 编程

Regbase 是 Ascend950 独有的高性能编程模式，直接操作寄存器：

```cpp
// 仅在 Ascend950 (arch35) 可用
// 文件位置：arch35/quantize_per_channel_regbase.h
template <typename T, typename T1, typename T2, typename U, ...>
class QuantizePerChannelRegbase : public QuantizeBase<T, T1, T2, U, ...> {
    __aicore__ inline void Init(...) {
        this->SetFloatOverflowModeForRegbase();  // 950 特有
    }
};
```

**使用场景**：
- 量化算子（quantize, dequantize）
- 动态量化（dynamic_quant）
- 需要极致性能的场景

### 2. SIMT 编程

SIMT (Single Instruction Multiple Threads) 是 Ascend950 独有的线程级并行编程模型：

```cpp
// 仅在 Ascend950 (arch35) 可用
// 文件位置：arch35/truncated_normal_v2_simt.h

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void SimtCompute(...) {
    int64_t groupIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    // 线程级并行计算
}

// 调用方式
AscendC::Simt::VF_CALL<SimtCompute<Y_T, OFFSET_T>>(
    AscendC::Simt::Dim3{USED_THREAD}, ...);
```

**SIMT API**：
- `Simt::GetBlockIdx()` - 获取块索引
- `Simt::GetThreadIdx()` - 获取线程索引
- `Simt::GetThreadNum()` - 获取线程数
- `Simt::VF_CALL<Func>()` - 向量函数调用
- `Simt::AtomicAdd()` - 原子加
- `Simt::ThreadBarrier()` - 线程同步

**使用场景**：
- 随机数生成（random_uniform, truncated_normal）
- 排序算法（radix_sort）
- 需要细粒度并行的场景

### 3. FP8/FP4 低精度格式

Ascend950 支持多种 8-bit 浮点格式：

| 格式 | 类型名 | 说明 | 适用场景 |
|-----|-------|------|---------|
| **FP8 E5M2** | `fp8_e5m2_t` | 5位指数，2位尾数 | 训练、梯度 |
| **FP8 E4M3FN** | `fp8_e4m3fn_t` | 4位指数，3位尾数 | 推理、激活 |
| **HiFloat8** | `hifloat8_t` | 华为自定义格式 | 混合精度 |
| **INT4** | `int4b_t` | 4-bit 整数 | 极限量化 |

**代码示例**：

```cpp
// 仅在 Ascend950 (arch35) 可用
if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
    // FP8 E5M2 处理
} else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
    // FP8 E4M3FN 处理
} else if constexpr (IsSameType<U, hifloat8_t>::value) {
    // HiFloat8 处理
}
```

**使用场景**：
- 量化算子（quantize, dequantize）
- 动态量化（dynamic_quant）
- 混合精度训练
- MX 量化（grouped_dynamic_mx_quant）

### 4. 其他架构优化

除了上述独有特性外，arch35 还有以下通用优化：

| 优化类型 | 说明 | 示例算子 |
|---------|------|---------|
| **DAG 定义** | 针对 950 的 UB 布局优化 DAG | abs, add, sin |
| **调度策略** | 不同的分块大小和流水线深度 | reduce, softmax |
| **内存访问** | 利用 950 更大的 UB | matmul, conv |
| **指令优化** | 使用 950 新增的高效指令 | 各类算子 |

---

## arch35 代码示例

**通用 DAG** (适用于 arch32 系列)：

```cpp
// 通用 abs_dag.h
template <typename U, typename T = float>
struct AbsDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, 0>, OpCopyIn0>;
    using OpResult = Bind<Vec::Abs<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, 1>, OpResult>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;
    using OpDag = DAGSch<Elems<OpCopyOut>, void, MemOptCfg<MemLevel::LEVEL_2>>;
};
```

**arch35 优化 DAG**：

```cpp
// arch35/abs_dag.h - 针对 950 优化
template <typename U, typename T = float>
struct AbsDagArch35 {
    // 可能使用不同的内存配置
    using OpDag = DAGSch<Elems<OpCopyOut>, void, MemOptCfg<MemLevel::LEVEL_3>>;
    // 或使用特定的调度策略
};
```

### 文件命名约定

| 文件类型 | 通用 | arch35 专用 |
|---------|------|-------------|
| DAG 定义 | `*_dag.h` | `arch35/*_dag.h` |
| Tiling | `*_tiling.cpp` | `arch35/*_tiling_arch35.cpp` |
| Kernel 入口 | `*_apt.cpp` | 同一文件内通过 TilingKey 区分 |
| 测试 | `test_*.cpp` | `arch35/test_*.cpp` |

---

## 架构兼容性检查清单

开发算子时，请确认：

- [ ] 通用实现在所有目标架构上测试通过
- [ ] 如有 arch35 特殊实现，已单独测试
- [ ] Tiling 逻辑正确识别架构并选择实现
- [ ] 性能在目标架构上达到基线要求

---

## 参考资源

- [NpuArch说明和使用指导](https://gitcode.com/cann/ops-math/wiki/NpuArch%E8%AF%B4%E6%98%8E%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC.md)
- [ops-math 源码](https://gitcode.com/cann/ops-math) - 查看各算子的 arch35 实现
- [ops-nn 源码](https://gitcode.com/cann/ops-nn) - 复杂算子的架构特定优化
