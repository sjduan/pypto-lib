# 自定义算子转 `<<<>>>` kernel 直调检查清单

这个清单用于 **AscendC 自定义算子转 `<<<>>>` kernel 直调改造**。

适用场景：
- 把 `op_kernel/archXX` 或现有算子源树迁到当前目录
- 删除 `..` 相对 include
- 提取最小依赖闭包
- 判断当前结果是否只是“kernel 本地化完成”，还是已经“可直接 launch / 独立编译”

---

## 1. 任务边界先确认

开始前先确认：
- [ ] 这是 **自定义算子转 `<<<>>>` kernel 直调改造**，不是 host / graph / 注册集成
- [ ] 目标目录已明确（当前目录根 / 子目录 / 保留源层级）
- [ ] 是否需要同时处理 `*_apt.cpp` / `*_apt.h`
- [ ] 是否要求“可独立编译 / 可直接 launch”，还是只要求“kernel 本地化”

如果用户要的是 standalone story / cann-samples / `<<<>>>` 直调工程，就继续按本清单推进，不要误退回成纯 kernel 适配。

---

## 2. 源文件盘点

先列出：
- [ ] `archXX/` 下全部 `.h` / `.cpp`
- [ ] 哪个文件是入口（如 `*_apt.cpp`）
- [ ] 哪个文件是真正的依赖汇聚点（常见是 `*_common.h` / `*_regbase_common.h`）
- [ ] 哪些文件已经是从上游模板改造过的“半本地化版本”

判断点：
- [ ] 当前 kernel 是 header-only / inline 风格
- [ ] 依赖实现主要藏在 common / base 头，而不在 `.cpp`

---

## 3. 相对 include 盘点

必须完整列出所有：
- [ ] `#include "../..."`
- [ ] `#include "../../..."`

对每条 include，至少确认：
- [ ] 被哪些目标文件引用
- [ ] 当前目标文件真正用了哪些符号
- [ ] 这些符号的真实定义位置
- [ ] 是否存在“表面来自 A，实际定义在 B”的中转依赖

不要接受“include 了就整包搬”。

---

## 4. 符号分类

把要搬的外部符号分成四类。

### 4.1 平台常量
常见项：
- [ ] `GetUbBlockSize`
- [ ] `GetVRegSize`

判断：
- [ ] 能否直接改成本地 `constexpr`
- [ ] 是否保留最小 `platform` namespace 子集更清晰

### 4.2 基础常量 / traits
常见项：
- [ ] `BUFFER_NUM`
- [ ] `DOUBLE_BUFFER_NUM`
- [ ] `ONCE_VECTOR_SIZE`
- [ ] `CeilDiv`
- [ ] `is_same`
- [ ] `bfloat16_t` 兼容定义

判断：
- [ ] 是否只需一小段基础定义
- [ ] 是否错误地打算整包复制 `*_base.h`

### 4.3 reduce/common 层工具
常见项：
- [ ] `V_LENGTH`
- [ ] `CeilAlign`
- [ ] `ComputeMultiLevelRstd`
- [ ] `castTraitB162B32`
- [ ] `castTraitB322B16`

判断：
- [ ] 命名空间归属是否真实
- [ ] 迁移后 `using` 是否需要修正

### 4.4 算法 helper / regbase helper
常见项：
- [ ] `DataCopyImpl`
- [ ] `ComputeSum`
- [ ] `ComputeRstd`
- [ ] `ComputeMultiLevelReduce`

判断：
- [ ] 是否只搬当前 kernel 真正使用的函数
- [ ] 是否已补齐它们的直接闭包
- [ ] 是否误把整份 `regbase_common.h` 拖了进来

---

## 5. 收口结构决策

默认优先使用：
- [ ] 保留原 kernel 文件主体
- [ ] 新增一个 local helper 头，如 `xxx_local_deps.h`

检查：
- [ ] helper 头只装外部依赖，不混入当前 kernel 主逻辑
- [ ] 原有 `*_common.h` 仍承担当前算子的本地逻辑
- [ ] 文件职责清晰：主逻辑 / 依赖闭包 / 入口分离

仅在依赖极少时，才考虑把外部内容直接塞回原头。

---

## 6. 命名空间与死引用检查

重点检查：
- [ ] `using` 的符号归属是否真实
- [ ] 是否还保留错误中转归属（例如表面从 `RmsNorm` 引，真实在 `NormCommon`）
- [ ] 是否存在未使用 `using`
- [ ] 是否存在未使用 include
- [ ] 是否存在只为历史遗留保留的空壳依赖

典型死引用：
- [ ] `DataCopyCustom` 只声明不使用

---

## 7. 入口文件检查（如果同时处理 `*_apt.cpp` / `*_apt.h`）

必须单独检查：
- [ ] 入口文件最终是保留 `.cpp`，还是改成可被其他代码直接 include 的 `.h`
- [ ] `DTYPE_X1` 是否来自外部环境
- [ ] `GET_TILING_DATA_WITH_STRUCT` 是否来自外部环境
- [ ] `TILING_KEY_IS` 是否来自外部环境
- [ ] tiling struct 是否在当前目录闭环可见
- [ ] 是否需要把单入口按 tiling key 拆成两个独立函数
- [ ] 如果用户要求“加模板参数”，是否只是增加模板形参，而**参数声明仍保持 `GM_ADDR`**
- [ ] 入口头 `*_apt.h` 是否避免了 `using namespace`，并改为 `::AscendC::` / `::AddRmsNorm::` 这类显式限定
- [ ] 头文件化后的 kernel 实现是否已用 `#if defined(__NPU_DEVICE__) ... #endif` 包住
- [ ] `__NPU_DEVICE__` 宏保护应放在入口头，还是放在 `local_deps.h / regbase_common.h / regbase.h / regbase_split_d.h` 这类 kernel 实现头

结论必须二选一写清楚：
- [ ] **kernel 已本地化，但入口仍依赖外部编译环境**
- [ ] **入口也已本地化，可独立编译**

不要混着说。

---

## 8. 静态验证

### 8.1 include 清理
在目标文件集合中搜索：
- [ ] `#include "../`
- [ ] `#include "../../`

目标：无匹配。

### 8.2 错误命名空间引用
搜索：
- [ ] `using RmsNorm::castTraitB162B32`
- [ ] `using RmsNorm::castTraitB322B16`

若真实定义不在 `RmsNorm`，必须修正。

### 8.3 本地闭包完整性
确认以下符号能在目标目录本地解析，或来自稳定 SDK 头：
- [ ] `CeilDiv`
- [ ] `CeilAlign`
- [ ] `DataCopyImpl`
- [ ] `ComputeSum`
- [ ] `ComputeRstd`
- [ ] `ComputeMultiLevelReduce`
- [ ] `ComputeMultiLevelRstd`

### 8.4 helper 头职责
- [ ] helper 头里只有外部依赖闭包
- [ ] 当前算子特有逻辑仍保留在原 kernel 文件中

---

## 9. 结果汇报模板

完成后至少输出：

### 9.1 目标文件集合
- 落地了哪些 `.h` / `.cpp`

### 9.2 删除的相对 include
- 精确到文件和 include 语句

### 9.3 本地化依赖清单
- 按原来源头文件分组列出搬运符号

### 9.4 helper 头职责
- 为什么新增 helper
- helper 里收了哪些依赖

### 9.5 剩余外部假设
- 哪些宏 / tiling / dtype 仍依赖外部环境

### 9.6 静态验证结论
- 是否已没有 `..` include
- 是否还存在错误命名空间 `using`

---

## 10. add_rms_norm arch35 参考判断

如果是类似 `add_rms_norm/arch35` 的任务，通常应满足：
- [ ] 保留 `regbase.h / regbase_common.h / regbase_split_d.h`
- [ ] 新增 `add_rms_norm_local_deps.h`
- [ ] 删除：
  - `../inc/platform.h`
  - `../../rms_norm/rms_norm_base.h`
  - `../../rms_norm/arch35/rms_norm_regbase_common.h`
  - `../../norm_common/reduce_common_regbase.h`
- [ ] 本地化：
  - `GetUbBlockSize` / `GetVRegSize`
  - `BUFFER_NUM` / `DOUBLE_BUFFER_NUM` / `ONCE_VECTOR_SIZE` / `CeilDiv` / `is_same`
  - `V_LENGTH` / `CeilAlign` / `ComputeMultiLevelRstd` / `castTraitB162B32` / `castTraitB322B16`
  - `DataCopyImpl` / `ComputeSum` / `ComputeRstd` / `ComputeMultiLevelReduce`
- [ ] 明确说明 `*_apt.cpp` / `*_apt.h` 是否仍依赖 `DTYPE_X1` / `GET_TILING_DATA_WITH_STRUCT`
- [ ] 如果入口已改造成头文件化封装，明确说明：
  - 是否已经拆成 `fullload` / `splitd` 两个函数
  - 是否由 tiling struct 直接入参
  - 是否只是新增模板参数但仍保留 `GM_ADDR` 形参声明
  - 入口头里是否去掉了 `using namespace`，改用 `::` 显式限定
  - 是否已用 `#if defined(__NPU_DEVICE__) ... #endif` 包住 kernel 实现

---

## 一句话验收标准

**目标目录内的 kernel 代码应当做到：相对依赖清零、最小闭包明确、语义不变、边界说明清楚。**
