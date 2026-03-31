# 算子设计与实施文档模板

> ⚠️ **SubAgent 生成此文档时必须替换以下占位符**：
> - `{operator_name}` → 实际算子名称（如 `softmax0309`）
> - `{分支名}` → 实际分支名称（如 `ar_fullload`）
> 
> **输出路径**：`ops/{operator_name}/docs/design.md`（{operator_name}需替换为实际名称）
> **正确示例**：`ops/softmax0309/docs/design.md`
>
> **改进说明**：将原来的"设计文档"和"计划文档"合并为一个文档，避免分裂和重复。一个文档包含设计方案和实施计划。

---

## 0. 概述

### 0.0 需求类型判断

**判断标准**：
- **特定用例**：用户明确指定了具体的 shape 和 dtype（如："开发 softmax，shape=[1,1024], dtype=float16"）
- **通用**：用户未明确指定 shape 和 dtype，或明确要求支持多种配置

**影响范围**：
- **特定用例模式**：Phase 2/3 测试时只验证用户指定的 shape/dtype，不泛化
- **通用模式**：Phase 2/3 测试时覆盖多种 shape/dtype 组合

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | |
| 算子类别 | Reduction / Elementwise / Broadcast / Conversion / MatMul / Convolution / NN / ... |
| 需求类型 | 特定用例（shape=[...], dtype=...） / 通用 |
| 支持数据类型 | |
| 支持服务器 | A2 / A3 |
| 特殊约束 | |

### 0.2 算子类别识别

- **类别**：{类别名称}
- **判断依据**：{说明为何属于该类别}

### 0.3 成熟方案查阅

- **是否查阅成熟方案**：{是/否}
- **参考文档**：
  - {如有，列出参考的 guide/methodology 文档}
  - {如无，说明原因}

### 0.4 应用关键设计

| 设计项 | 成熟方案 | 应用到当前算子 |
|--------|---------|----------------|
| {设计项1} | {成熟方案} | {具体应用} |
| {设计项2} | {成熟方案} | {具体应用} |

---

## 1. 算子设计

### 1.1 数学公式

```
// 输入输出定义
输入: x - shape, dtype
输出: y - shape, dtype

// 数学公式
y = f(x)
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| {操作1} | {API1} | {参数1=值1} | {布局1} | [链接] |
| {操作2} | {API2} | {参数2=值2} | {布局2} | [链接] |

#### 1.2.1 API 语义验证（⚠️ 强制）

**每个 API 调用前，必须填写验证表**。

**详细方法论**：[API 语义验证方法论](../references/api-semantic-verification.md)

**验证表格**（所有 API 必须填写）：

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| {API1} | {内存排列、连续性、对齐} | {操作类型、维度} | {API签名} | {对齐/类型限制} | {是/否} | [链接] |
| {API2} | {内存排列、连续性、对齐} | {操作类型、维度} | {API签名} | {对齐/类型限制} | {是/否} | [链接] |

**示例**：
| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| ReduceMax | (R, A0) 矩阵，归约 R 维（带stride） | 归约 R 维，输出 A0 个值 | `ReduceMax<T, Pattern::RA>(...)` | srcShape[1] 必须 32 字节对齐 | ✅ | [链接] |
| DataCopyPad | GM→UB，非对齐数据 | 搬运 + 填充 | `DataCopyPad(...)` | 无 | ✅ | [链接] |

**验证清单**（每个 API 必须完成）：
- [ ] 1. 数据布局确认（内存排列、连续性、对齐）
- [ ] 2. 功能需求明确（操作类型、维度、输出格式）
- [ ] 3. 已查阅官方文档（提供链接）
- [ ] 4. 匹配验证（数据布局与 API 能力匹配、限制条件满足）
- [ ] 5. 已记录验证过程

**⚠️ 未完成验证的代码将无法通过审查** ⛔️

 
### 1.3 数据流

 
```
输入 x (Global Tensor)
    ↓ DataCopy
输入 x (Local Tensor)
    ↓ {处理操作}
中间结果
    ↓ {处理操作}
输出 y (Local Tensor)
    ↓ DataCopy
输出 y (Global Tensor)
```
 
### 1.4 核心计算步骤(复杂算子)
 
**⚠️ 重要说明**:
- **本节定位**: 核心计算步骤概览 + 关键设计要点(简化的)
- **详细实现**: 见 2.4 节各分支的核心伪代码(Compute 必须包含，CopyIn/CopyOut 非必须)
- **避免重复**: 不要在本节写完整的 for 循环伪代码
 
**核心计算步骤**:
```
1. {步骤1} - 说明
2. {步骤2} - 说明
3. ...
```
 
**分支差异对比**(如果有多个分支):
 
| 操作 | {分支1} | {分支2} |
|------|---------|---------|
| {操作1} | {方式1} | {方式2} |
| {操作2} | {方式1} | {方式2} |
 
**关键设计要点**:
1. **Buffer 使用**: `xLocal`/`yLocal`(输入/输出), `tmpLocal`(临时)
2. **API 选择**: Level 2 vs Pattern 接口、 Adds/Muls 优化
3. **参数含义**: `rLength`(有效)vs `rLengthAlign`(对齐)
4. **优化技巧**: `Adds(-scalar)`、`src1RepStride=0`
 
**参数使用规则**:
| 参数位置 | 用有效长度 | 用对齐长度 |
|---------|-----------|-----------|
| DataCopyPad blockLen / 计算 API count | ✓ | ✗ |
| UB 数据偏移 / Buffer 大小 | ✗ | ✓ |
 
### 1.5 内存管理(Buffer 规划)
 
| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| inQueueX | 输入数据 | | VECIN |
| outQueueY | 输出数据 | | VECOUT |
| tmpBuf | 临时计算 | | VECCALC |
| ... | | | |
 
**总 UB 使用量**: ___ KB
 
### 1.6 Reduce API 语义验证** ⚠️ **最高优先级(新增)**
 
**对于归约类算子，必须验证以下内容**:
 
#### 1.6.1 数据布局与 API 匹配验证表
 
| 模板 | 数据布局 | Reduce 鎟 | 输出类型 | ✅ 正确 API | ❌ 错误 API | 对齐要求 |
|------|---------|-----------|----------|------------|-------------|-------------|
| **AR** | (A1, R) | R 维（连续） | 标量(1 个值) | `ReduceMax(dst, src, tmp, count)` | Pattern API (过度设计) | 无 |
| **ARA** | (A1, R, A0) | R 维(stride=A0) | 向量(A0 个值) | `ReduceMax<T, Pattern::RA>(dst, src, tmp, srcShape, true)` | `ReduceMax(dst, src, tmp, count)` | `srcShape[1]` 必须 32 字节对齐 |
 
#### 1.6.2 娡板识别与 API 选择决策树
 
```
Reduce 操作?
    │
    ├─ 数据布局？
    │    ├─ AR 模板 (A0=1, 每行 R 个连续元素)
    │    │    └─→ ✅ Level 2 API (推荐)
    │    │             ReduceMax(dst, src, tmp, count)
    │    │             - 无对齐要求
    │    │             - 输出: 标量(1 个值)
    │    │             - 性能最优
    │    │
    │    └─ ARA 模板 (A0>1, R×A0 数据, 归约 R 维需 stride)
    │         └─→ ⚠️ Pattern API (必须)
    │              ReduceMax<T, Pattern::RA>(dst, src, tmp, srcShape, true)
    │              - srcShape[1] 必须 32 字节对齐
    │              - 输出: 向量(A0 个值)
    │              - ⚠️ **禁止使用 Level 2 API**
    │                 （Level 2 无法处理 stride，会导致错误）
    │
    └─ ❌ 禁止凭直觉选择 API
              必须参考此决策树！
```
 
#### 1.6.3 锸️ 强制验证步骤（适用于所有 Reduce API 调用)
 
**必须执行以下验证**:
1. [ ] **确认数据布局**
   - [ ] 数据在内存中如何排列?
   - [ ] 输入输出 Shape 是什么?
   - [ ] 需要执行什么操作?
   - [ ] 确定模板类型 (AR: A0=1 / ARA: A0>1)
   
2. [ ] **查阅官方 API 文档**
   - [ ] 已查阅 `asc-devkit/docs/api/context/ReduceMax.md` (或 ReduceSum.md)
   - [ ] 知道 Level 2 vs Pattern API 的区别
   - [ ] 了解 Pattern::RA vs Pattern::AR 的语义
   
3. [ ] **选择正确的 API** (根据决策树)
   - [ ] AR 模板 → Level 2 API
   - [ ] ARA 模板 → Pattern API (Pattern::RA)
   
4. [ ] **验证对齐要求** (如适用)
   - [ ] Pattern API: srcShape[1] 是否 32 字节对齐?
   - [ ] Level 2 API: 无对齐要求
   
5. [ ] **确认输出类型**
   - [ ] AR 模板: 标量 (1 个值)
   - [ ] ARA 模板: 向量 (A0 个值)
   
6. [ ] **记录验证结果**
   - [ ] 在设计文档中记录 API 选择原因
   - [ ] 记录查阅的官方文档链接
   - [ ] 记录数据布局和 API 参数对应关系
 
**⚠️ 违反此验证要求的代码将无法通过代码审查** ⛔️

 
#### 1.6.4 巠️ 错误示例与正确做法
 
**❌ 错误示例** (ARA 模板中使用 Level 2 API):
 
```cpp
// ❌ 错误: ARA 模板 (R×A0 数据， 归约 R 维
// 数据布局: [r0的全部A0, r1的全部A0, ..., r{R-1}的全部A0]
// 归约需求| 对每个 a0 位置，取 R 个值的 max
// 输出| A0 个最大值
 
// ⚠️ 错误: 使用 Level 2 API (无法处理 stride)
ReduceMax<T>(scalarLocal, xLocal[rowOffset], tmpLocal, rLength, false);
// 问题:
// - Level 2 API 只能处理连续的 count 个元素
// - 无法处理带 stride 的归约 (间隔 A0 个元素)
// - 导致: 数值错误、精度问题、崩溃
```
 
**✅ 正确做法** (ARA 模板使用 Pattern API):
 
```cpp
// ✅ 正确: ARA 模板 (R×A0 数据)
// 数据布局| [r0的全部A0, r1的全部A0, ..., r{R-1}的全部A0]
// 归约需求| 对每个 a0 位置,取 R 个值的 max
// 输出| A0 个最大值
 
// ✅ 使用 Pattern API (可以处理 stride)
uint32_t alignedA0 = ((a0Count * sizeof(T) + 31) / 32) * 32 / sizeof(T);
uint32_t srcShape[] = {rLength, alignedA0};  // ⚠️ srcShape[1] 必须 32 字节对齐
AscendC::LocalTensor<T> maxLocal = maxSumBuf.Get<T>();
AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
 
// ✅ Pattern::RA - 沿第一维(R 维)归约
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
    maxLocal,           // 输出: A0 个最大值
    xLocal,             // 输入: R×A0 数据
    tmpLocal,           // 临时 buffer
    srcShape,            // {rLength, alignedA0}
    true                // srcInnerPad = true (有填充)
);
// 输出: A0 个最大值
```
 
**⚠️ 混淆 Level 2 和 Pattern API 将导致严重错误** ⛔️

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | |
| 单核任务量 | |
| 使用的核数 | ⚠️ **强制动态计算** |
| 负载均衡方式 | |

**核数计算规范** ⚠️：

**核数获取 API 选择**：
| 算子类型 | 使用的 API |
|---------|-----------|
| 纯向量计算 | `ACL_DEV_ATTR_VECTOR_CORE_NUM` |
| 矩阵计算 | `ACL_DEV_ATTR_CUBE_CORE_NUM` |
| 混合计算 | `ACL_DEV_ATTR_AICORE_CORE_NUM` |

```cpp
// Host 侧（强制）- 纯向量算子示例
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// Kernel 侧（强制）
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) return;  // 越界检查
```

**禁止**：❌ 写死核数 `numBlocks = 8;`

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192KB (A2/A3) / 248KB (A5) |
| 单次处理数据量 | |
| 是否需要分 chunk | |
| chunk 大小计算公式 | |

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 | |
| 大 shape | |
| 小 shape | |
| 对齐 | |
| 非对齐 | |
| 边界情况 | |

### 2.4 类别特有设计

> 根据算子类别，查阅对应的 methodology 文档，填写该类别特有的设计要素。
>
> **重要**：对于复杂算子的每个分支，都要在此部分添加对应的核心流程伪代码。
>
> **⚠️ 禁止使用引用**：每个分支的伪代码必须**直接包含**在 2.4 节中，不能使用"见 1.4 节"等引用方式。
>
> **核心伪代码只需要 Compute 核心计算流程，不需要 CopyIn/CopyOut**。
> - ✅ 正确：在 2.4.1/2.4.2 中直接编写完整的伪代码
> - ❌ 错误：**核心流程伪代码**：见 [1.4.1 AR 模板核心流程](#ref)
> - 原因：实现者需要在一个地方看到完整的设计，避免跳转

**算子类别**: ____________

**查阅文档**: `ascendc-tiling-design/references/{类别}-methodology.md`

#### 2.4.1 {分支1名称}

**适用场景**：{条件}

**核心流程伪代码**：

```cpp
// {分支1}的核心计算流程
for (tile = 0; tile < totalTiles; tile++) {
    // ... 核心逻辑
}
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| ... | ... | ... |

#### 2.4.2 {分支2名称}

**适用场景**：{条件}

**核心流程伪代码**：

```cpp
// {分支2}的核心计算流程
for (tile = 0; tile < totalTiles; tile++) {
    // ... 核心逻辑
}
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| ... | ... | ... |

---

## 3. NPU 优化

### 3.1 SIMD

- 使用 API：{矢量 API 名称}
- 数据宽度：{每周期处理元素数}

### 3.2 Tiling 参数计算

| 参数 | 公式 | 说明 |
|------|------|------|
| {参数1} | {公式} | {说明} |
| {参数2} | {公式} | {说明} |

### 3.3 双缓冲

- 是否使用：{是/否}
- 缓冲区大小：{值}
- 重叠策略：{说明}

### 3.4 流水线

- 流水线阶段：{阶段1 → 阶段2 → 阶段3}
- 预期加速比：{值}

---

## 4. 实施计划

### 4.1 文件清单

#### 通用文件

| 序号 | 文件路径 | 说明 |
|------|---------|------|
| 1 | `ops/{operator_name}/docs/design.md` | 设计与实施文档（本文档） |
| 2 | `ops/{operator_name}/CMakeLists.txt` | 构建脚本 |
| 3 | `ops/{operator_name}/gen_golden.py` | Golden 数据生成脚本 |
| 4 | `ops/{operator_name}/run.sh` | 运行脚本 |

#### Kernel 文件（简单算子）

| 序号 | 文件路径 | 说明 |
|------|---------|------|
| 1 | `ops/{operator_name}/{operator_name}.asc` | 主入口 + 所有实现 |

#### Kernel 文件（复杂算子，分支独立）

| 序号 | 文件路径 | 说明 |
|------|---------|------|
| 1 | `ops/{operator_name}/{operator_name}.asc` | Kernel 主入口，分支判断框架 |
| 2 | `ops/{operator_name}/{operator_name}_common.h` | 公共定义（常量、工具函数）|
| 3 | `ops/{operator_name}/{operator_name}_{分支1}.h` | {分支1}（Tiling结构体 + 判断函数 + Kernel实现）|
| 4 | `ops/{operator_name}/{operator_name}_{分支2}.h` | {分支2}（Tiling结构体 + 判断函数 + Kernel实现）|

### 4.2 测试计划

#### 功能测试矩阵

| 维度 | 测试值 | 覆盖场景 |
|------|-------|---------|
| {维度1} | {值} | {场景} |

#### 测试用例

| 序号 | {参数} | 说明 |
|------|-------|------|
| 1 | {值} | {说明} |

#### 精度验证

- **验证方法**：{方法}
- **精度标准**：{标准}
- **验证项**：{项1}, {项2}

#### 边界测试

| 测试项 | 输入 | 预期输出 |
|--------|------|---------|
| {项1} | {输入} | {输出} |

### 4.3 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| {风险1} | {高/中/低} | {说明} | {措施} |

---

## 5. 确认清单

### 5.1 设计确认

- [ ] 多核切分策略已确定
- [ ] UB 切分策略已确定
- [ ] Buffer 规划已完成
- [ ] 分支场景已覆盖
- [ ] 类别特有设计已完成

### 5.2 实施确认

- [ ] 文件清单完整
- [ ] 测试计划完整（功能 + 精度 + 边界）
- [ ] 风险识别充分

---

## 6. 参考资源

- 官方示例: `asc-devkit/examples/xxx/`
- API 文档: `asc-devkit/docs/api/context/xxx.md`
- 类别设计指南: `ascendc-tiling-design/references/xxx.md`
