---
name: ascendc-custom-op-to-kernel-launch
description: 当用户想把自定义算子工程中的 kernel 模板改造成 `<<<>>>` kernel 直调形式，或从自定义算子工程中抽取某个 kernel 模板并转换成 `<<<>>>` 直调方式时使用。触发：用户提到"自定义算子转直调"、"从算子工程抽 kernel"、"kernel 模板改 `<<<>>>`"等。不适用于从零开发新算子
---

# AscendC 自定义算子转 `<<<>>>` kernel 直调改造

这个 skill 处理的是两类任务：
1. **把已有 AscendC 自定义算子改造成 `<<<>>>` kernel 直调形态**
2. **从某个现有模板或算子实现中抽取 kernel/tiling/host 关键部分，并转换成 `<<<>>>` 方式**

目标不是“把文件复制过来”，而是识别并拆解现有实现里的 kernel、tiling、host glue 与外部依赖，把它整理成一个 **本地可维护、依赖闭环清晰、kernel 语义不变、可直接 launch** 的版本。

## 默认处理范围

它默认解决的是：
- 从 `op_kernel/archXX` 或类似源树中抽出 header-only / inline kernel 代码
- 删除 `#include "..` 开头的跨目录依赖
- 把真正需要的函数 / 常量 / traits / helper 收口到本地
- 在保持 kernel 计算语义不变的前提下，补齐现有算子转 `<<<>>>` kernel 直调所需的最小必要改写

它默认 **不处理**：
- `aclnn` / `OpDef` / 注册流程
- 完整的新算子设计与开发

它可以处理两类任务：
1. **kernel-only 去耦 / 自包含**
2. **kernel + tiling + `<<<>>>` 直调入口 / standalone sample host glue 适配**

如果用户已经明确告诉你：
- kernel 代码在哪里
- tiling 代码在哪里

就不要再拆成两个技能链路，直接按同一条自定义算子转直调工作流推进。

---

## 先判断任务属于哪一类

优先把任务识别为以下一种：

1. **kernel 本地化搬运**
   - 例如：把 `arch35/` 目录搬到当前路径，删掉相对 include，并把依赖函数搬到本地
2. **kernel 头文件去耦**
   - 例如：不再依赖 `../../rms_norm/...`、`../inc/platform.h`
3. **最小依赖闭包提取**
   - 例如：只搬 `DataCopyImpl` / `ComputeRstd` / `CeilDiv`，不整包复制上游头文件
4. **kernel 入口本地落地**
   - 例如：把 `*_apt.cpp` 一起放到目标目录，并显式指出还缺哪些外部宏/tiling 结构

如果用户明确说：
- “适配到 cann-samples”
- “做 standalone story”
- “要能独立编译运行的 story 工程”

则不要误判成“只做 kernel 去耦”；要按 **kernel + tiling + standalone sample/story host glue** 的完整链路处理。

---

## 默认工作流

### Step 1: 先确认交付边界，再决定走 kernel-only 还是 `<<<>>>` 直调链路
先确认这 3 件事：
- **源目录**：通常是 `op_kernel/` 或 `op_kernel/archXX/`
- **目标目录**：当前路径根、子目录，还是保留源层级
- **交付边界**：只是 kernel 自包含，还是还要让 `*_apt.cpp` / tiling / `<<<>>>` 入口一并可独立编译调用

默认假设：
- 先做 **kernel 代码适配**
- 不主动扩展到 host / graph / 注册文件
- 不重写算法逻辑，只做依赖适配

但如果用户明确要求 standalone story / cann-samples / `<<<>>>` 直调：
- 继续保持 **kernel 语义不变**
- 同时把 tiling / host glue / 直调入口补齐到能在目标工程里独立维护
- 仍然不要主动扩展到 `aclnn` / `OpDef` / 注册链路

### Step 2: 盘点源文件集合
用 `Glob` / `Read` / `Grep` 先确认：
- 目录下有哪些 `.h` / `.cpp`
- 哪个文件是入口文件（例如 `*_apt.cpp`）
- 哪些是 `archXX` 特化头
- 哪些文件已经是“局部本地化后的二次封装”

特别注意：
- 很多 AscendC kernel 是 **header-only** 风格
- 真正要搬运的实现，往往藏在 `common` / `regbase_common` / `base` 头里

### Step 3: 只追踪 `..` 相对 include，并建立“符号级依赖图”
必须先找出所有：
- `#include "../..."`
- `#include "../../..."`

然后对每个相对 include 做两件事：
1. 记录 **当前 kernel 文件真正使用了哪些符号**
2. 记录这些符号的 **真实定义位置**

不要因为某个头被 include 了，就整包复制整个头文件。

输出时至少要列出：
- 相对 include 路径
- 被实际使用的函数 / 常量 / traits / 类型 / 宏
- 是否存在“表面来自 A，实际定义在 B”的转手依赖

### Step 4: 优先选“保留原 kernel 文件 + 新增 local helper 头”的收口方式
默认推荐结构：
- 保留原始 kernel 文件主体
- 新增一个同目录 helper，例如：`xxx_local_deps.h`
- 把外部依赖的最小闭包集中搬到 helper 里

只有在依赖非常少时，才考虑把外部内容直接塞回原头文件。

推荐结构通常是：
- `foo_regbase.h`
- `foo_regbase_common.h`
- `foo_regbase_split_d.h`
- `foo_local_deps.h`
- 可选：`foo_apt.cpp` 或 `foo_apt.h`

这样做的好处：
- 原有 kernel 结构不散
- 外部依赖边界清楚
- 后续复查时容易区分“本地逻辑”和“搬运依赖”

### Step 5: 只搬“最小依赖闭包”
把依赖分成 4 类来搬。

#### 5.1 平台常量
典型例子：
- `platform::GetUbBlockSize`
- `platform::GetVRegSize`

如果只是返回固定常量，优先改成：
- 小型 `constexpr` 包装
- 或直接在本地 helper 中保留最小 `platform` namespace 子集

#### 5.2 基础常量 / traits
典型例子：
- `BUFFER_NUM`
- `DOUBLE_BUFFER_NUM`
- `ONCE_VECTOR_SIZE`
- `CeilDiv`
- `is_same`
- `bfloat16_t` 兼容定义

这类通常来自某个 `*_base.h`，但真正需要的往往只是一小段，不要整包搬。

#### 5.3 reduce/common 层工具
典型例子：
- `V_LENGTH`
- `CeilAlign`
- `ComputeMultiLevelRstd`
- `castTraitB162B32`
- `castTraitB322B16`

这类最容易出现“命名空间错位”。必须核实符号真实归属，而不是照抄 `using`。

#### 5.4 算法 helper / regbase helper
典型例子：
- `DataCopyImpl`
- `ComputeSum`
- `ComputeRstd`
- `ComputeMultiLevelReduce`

只复制 **当前 kernel 真正调用到的函数**，并补齐它们的直接闭包。
不要把整个上游 `regbase_common.h` 原样拖进目标目录。

### Step 6: 如果目标是 `<<<>>>` 直调 / standalone story，再补 tiling / host glue，但保持 kernel 独立
当用户明确要 cann-samples / standalone story / `<<<>>>` 直调时：
- 优先把 tiling struct 改成当前目录可见的 plain struct，或本地等价结构
- 把 `op_host` 的 tiling 逻辑改成本地 helper，而不是继续绑定原工程注册框架
- 让入口按当前 story 的 host dispatch 或 `<<<>>>` launch 方式工作，而不是继续依赖原工程分发宏
- 保持 kernel 实现头尽量独立，host 逻辑只做最小串联

这里的重点是：
- **可以本地化 tiling 与 host glue**
- **要把现有算子形态改造成可直接 launch 的入口**
- **不要把任务扩展成正式 op 注册链路改造**

---

## 决策规则

### 规则 1：优先复用当前 kernel 目录里已经存在的“半本地化”文件
如果源目录中已经有一个 `*_common.h` 明显是从上游算法模板改出来的：
- 优先复用它
- 只把缺失依赖补齐
- 不要回退去重建另一套结构

### 规则 2：对“表面归属”和“真实定义”保持怀疑
典型坑：
- `using RmsNorm::castTraitB162B32;`
- 但真实定义其实在 `NormCommon`

处理原则：
- 以真实定义源为准
- 迁移时顺手修正 `using` 归属
- 不保留错误的中转命名空间引用

### 规则 3：删除死引用，而不是机械保留
如果某个 `using` 或 include 只是在源码里“看起来像要用”，但实际未被引用：
- 直接删
- 不做兼容性保留

典型例子：
- `using RmsNorm::DataCopyCustom;` 只声明不使用

### 规则 4：kernel 本地化完成，不等于入口文件可独立编译或直接 launch
如果用户只要求 kernel 本地化，做到以下即可视为完成：
- kernel 头文件不再依赖 `..` 相对 include
- helper 头已收口最小依赖闭包

但如果入口文件 `*_apt.cpp` 还依赖：
- `DTYPE_X1`
- `GET_TILING_DATA_WITH_STRUCT`
- `TILING_KEY_IS`
- 外部 tiling struct

必须在结果里明确说明：
- **kernel 已本地化**
- **入口仍依赖外部编译环境**
- 若要独立编译或直接 launch，需要继续本地化 tiling / dispatch / dtype 宏

### 规则 5：如果用户要“可独立编译”，优先本地化 tiling struct 或改 host dispatch
常见做法：
- 把 tiling struct 改成 plain struct
- 本地补 `GET_TILING_DATA_WITH_STRUCT` 依赖
- 或改成 Host dispatch / by-value tiling，避免继续绑定原工程宏

### 规则 6：如果 kernel 入口参数保持 `GM_ADDR`，host 侧 device 指针也从一开始就保持 `GM_ADDR`
常见做法：
- host 侧直接声明：`GM_ADDR inputDevice = nullptr;`
- 用 `aclrtMalloc(reinterpret_cast<void**>(&inputDevice), size, ...)` 申请
- kernel launch 时直接传 `inputDevice`
- 用 `aclrtFree(inputDevice)` 释放

不要这样做：
- 先把 device 指针存成 `void*`
- 或包在 `std::unique_ptr<void>` 里
- 然后在 launch 点写 `reinterpret_cast<GM_ADDR>(ptr)`

原因：
- 这不是“风格问题”，而是 **AscendC 编译器会直接拒绝从 `void*` 到 `GM_ADDR` 的这种转换**
- 一旦入口 ABI 决定保留 `GM_ADDR`，host 侧也要沿着这条 ABI 一致到底

### 规则 7：抽取 tiling 数学逻辑时，优先逐分支保真，不要擅自把不同分支“统一写法”
尤其注意：
- `is32BAligned == 1` 的路径
- `is32BAligned == 0` 的路径
- 分母到底是原始 `baseColLen`，还是 `AlignUp(baseColLen, ubMinBlockLen)`

典型例子：
- 32B 对齐时：`baseRowLen = maxTileLen / baseColLen`
- 非 32B 对齐时：`baseRowLen = maxTileLen / AlignUp(baseColLen, ubMinBlockLen)`

不要为了“代码更整齐”把两条公式都改成后一种。

原因：
- 这会改变 tiling 结果
- 轻则性能漂移，重则和原始切块语义不一致

---

## 输出格式要求

完成这类任务时，结果里至少要给出：

1. **目标文件集合**
   - 最终落地了哪些 `.h` / `.cpp`
2. **被删除的相对 include**
   - 精确到文件和 include 语句
3. **本地化的依赖清单**
   - 按来源头文件列出搬了哪些符号
4. **helper 头职责**
   - 为什么新增它、里面装了哪类依赖
5. **剩余外部假设**
   - 哪些宏 / tiling / dtype 仍依赖外部环境
6. **静态验证结论**
   - 是否已没有 `..` include
   - 是否还有错误命名空间 `using`

如果任务明确是 `<<<>>>` 直调 / standalone story / cann-samples，结果中还应补一句：
- 当前版本是 **kernel-only 自包含**，还是已经做到 **kernel + tiling + direct-launch glue 闭环**

---

## 验证清单

快速执行时，直接按 `references/custom-op-to-kernel-launch-checklist.md` 做静态核验。

注意：
- 这个清单对 **自定义算子转直调过程中的 kernel 依赖清理** 最有用
- 如果任务是 standalone story / `<<<>>>` 直调，它仍可用于检查 kernel 闭包是否干净，但不能替代你对 host tiling / direct-launch glue 的单独判断

至少做以下静态检查：

### 1. 相对 include 清理
在目标文件集合中搜索：
- `#include "../`
- `#include "../../`

期望：无匹配。

### 2. 命名空间归属检查
重点搜：
- `using RmsNorm::castTraitB162B32`
- `using RmsNorm::castTraitB322B16`

如果真实定义不在 `RmsNorm`，必须修正。

### 3. 本地闭包检查
确认以下符号要么来自本地 helper，要么来自稳定 SDK 头：
- `CeilDiv`
- `CeilAlign`
- `DataCopyImpl`
- `ComputeSum`
- `ComputeRstd`
- `ComputeMultiLevelReduce`
- `ComputeMultiLevelRstd`

### 4. 死引用检查
确认并删除：
- 未使用的 `using`
- 未使用的 include
- 仅为历史兼容留下的空壳依赖

### 5. 入口文件假设检查
如果同时适配了 `*_apt.cpp` 或 `*_apt.h`，必须明确检查：
- `DTYPE_X1`
- `GET_TILING_DATA_WITH_STRUCT`
- `TILING_KEY_IS`
- tiling struct 是否本地可见

### 6. Host launch ABI 一致性检查
如果最终 direct-launch 入口参数仍然是 `GM_ADDR`，必须明确检查：
- host 侧 device buffer 变量是不是也声明成 `GM_ADDR`
- `aclrtMalloc` 是否按 `reinterpret_cast<void**>(&devicePtr)` 形式写入这个 `GM_ADDR` 变量
- kernel launch 时是否直接传 `devicePtr`
- 是否还残留 `reinterpret_cast<GM_ADDR>(voidPtr)` 这种调用点强转

期望：
- 没有 `void* -> GM_ADDR` 的临门一脚强转
- host / launch / kernel 三侧 ABI 一致

### 7. Tiling 公式逐分支等价性检查
如果把 host tiling 从原工程提取成本地 helper，必须明确检查：
- 对齐分支和非对齐分支是否仍然分别保留
- 是否把原本分支不同的分母、上取整逻辑、上界逻辑误合并成同一套写法
- 关键公式是否逐条和上游比对过，而不是“凭感觉等价”

重点关注：
- `baseRowLen`
- `baseColLen`
- `tileLength`
- `is32BAligned` 相关分支

---

## 常见坑

### 1. 误把整份上游 `*_base.h` / `*_common.h` 原样复制
这会把大量无关实现也拖进来，后续更难维护。默认只搬最小闭包。

### 2. 不核实真实定义源
很多符号是通过中间头间接暴露的。迁移时必须追到真实定义位置。

### 3. 只删 include，不补 helper
这样会让当前目录暂时“看起来干净”，但实际符号解析断掉。

### 4. 把纯 kernel 本地化和 host/工程集成混在一起
用户如果只要 kernel 代码适配，就不要主动扩展到 host / graph / 注册。

### 5. 忽略入口文件仍依赖原工程宏
`*_apt.cpp` 很容易在 include 改完之后，看起来也“在当前目录里了”，但其实仍不能独立编译。必须显式说明。

### 6. host 侧先用 `void*` 持有 device 指针，launch 时再强转成 `GM_ADDR`
这在普通 C++ 里看起来像小问题，但在 AscendC direct-launch 场景里经常会 **直接编译失败**。

正确做法是：
- 如果 kernel 入口参数是 `GM_ADDR`
- 那 host 侧申请 device 内存的变量也从一开始就用 `GM_ADDR`
- `aclrtMalloc(reinterpret_cast<void**>(&devicePtr), ...)`
- launch 直接传 `devicePtr`

### 7. 抽 tiling 时把“对齐分支”和“非对齐分支”机械合并
看起来像是在“清理重复代码”，但这类改法最容易偷偷改掉原始切块语义。

典型危险动作：
- 把只有非 32B 对齐路径才该用的 `AlignUp(baseColLen, ubMinBlockLen)`
- 也套到 32B 对齐路径上

结果往往不是立即编译错，而是：
- tiling 结果漂移
- blockDim / base shape 改掉
- correctness 或性能悄悄偏离上游实现

---

## 参考案例：add_rms_norm arch35

当源目录是：
- `add_rms_norm/arch35/add_rms_norm_regbase.h`
- `add_rms_norm/arch35/add_rms_norm_regbase_common.h`
- `add_rms_norm/arch35/add_rms_norm_regbase_split_d.h`

典型处理方式是：
- 保留这 3 个 kernel 头
- 新增 `add_rms_norm_local_deps.h`
- 删除相对 include：
  - `../inc/platform.h`
  - `../../rms_norm/rms_norm_base.h`
  - `../../rms_norm/arch35/rms_norm_regbase_common.h`
  - `../../norm_common/reduce_common_regbase.h`
- 本地化最小依赖：
  - `GetUbBlockSize` / `GetVRegSize`
  - `BUFFER_NUM` / `DOUBLE_BUFFER_NUM` / `ONCE_VECTOR_SIZE` / `CeilDiv` / `is_same`
  - `V_LENGTH` / `CeilAlign` / `ComputeMultiLevelRstd` / `castTraitB162B32` / `castTraitB322B16`
  - `DataCopyImpl` / `ComputeSum` / `ComputeRstd` / `ComputeMultiLevelReduce`
- 修正命名空间漂移：
  - `castTraitB162B32` / `castTraitB322B16` 应归到 `NormCommon`
- 如同时落地 `add_rms_norm_apt.cpp`，要额外声明：
  - kernel 依赖已本地化
  - 但 `DTYPE_X1` / `GET_TILING_DATA_WITH_STRUCT` 仍来自外部编译环境
- 如果继续把入口做成可直接被其他 kernel/调用方 include 的版本，优先按下面方式改：
  - 把 `*_apt.cpp` 改成 `*_apt.h`
  - 按 tiling key 拆成两个独立入口函数，而不是单一 `add_rms_norm(...)` 分发
  - 不再使用 `GET_TILING_DATA_WITH_STRUCT` / `TILING_KEY_IS`
  - 两种 tiling struct 直接作为函数参数传入
  - 如果用户只要求“增加模板参数”，默认只给入口函数增加 `DTYPE_X1 / DTYPE_X2 / DTYPE_GAMMA / DTYPE_Y / DTYPE_RSTD / DTYPE_X` 这类模板参数，**参数声明仍保持原来的 `GM_ADDR`**，不要擅自改成 `__gm__ T*` 形式
  - 入口头 `*_apt.h` 默认不要写 `using namespace`；优先在入口实现里使用 `::AscendC::...`、`::AddRmsNorm::...` 这类显式限定，避免把命名空间污染扩散给包含方
  - 头文件化后的 kernel 实现默认再用 `#if defined(__NPU_DEVICE__) ... #endif` 包起来，避免 host 侧包含入口头时直接编译 device 实现
  - 但如果用户明确指出宏保护应该放在 `add_rms_norm_local_deps.h / add_rms_norm_regbase_common.h / add_rms_norm_regbase.h / add_rms_norm_regbase_split_d.h` 这类 **kernel 实现头**，就把 `__NPU_DEVICE__` 宏放在这些实现头里，而不是优先放在入口头 `*_apt.h`

---

## 工具偏好

优先使用：
- `Glob`：找文件
- `Grep`：找 include 和符号引用
- `Read`：读源文件和定义文件
- `Edit`：改现有文件
- `Write`：只用于新增 helper 头或新增目标副本

不要用 shell 的 `grep/cat/find` 代替这些专用工具，除非确实没有替代方案。

---

## 一句话原则

这类任务的核心不是“把文件复制过来”，而是：

**把 kernel 代码适配成一个最小依赖闭包明确、语义不变、目标目录内可维护的本地版本。**
