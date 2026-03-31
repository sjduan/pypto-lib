# 代码审查检查清单

> 本文档用于阶段二（代码审查）的自我审查

---

## 0. Tiling 计算位置 ⚠️ **强制检查**

### 0.1 Tiling 计算位置要求

- [ ] **Tiling 参数计算在 Host 侧完成** ⚠️ **强制**
  - ✅ 正确：在 `xxx_common.h` 中定义 `ComputeXxxTiling()` 函数
  - ✅ 正确：在 `xxx.asc` 的 `main()` 中调用 `ComputeXxxTiling()`
  - ❌ 错误：在 Kernel 的 `Init()` 中计算 Tiling 参数
  - ❌ 错误：在 Kernel 的 `Process()` 中动态计算

**示例**：

```cpp
// ✅ 正确（Host 侧计算)
// common.h
inline void computeSoftmaxTiling(sftmaxTilingData& tiling, ...) {
    tiling.rowsPerLoop = ...;  // 在 Host 侧计算
}

// xxx.asc
int32_t main() {
    sftmaxTilingData tiling;
    computeSoftmaxTiling(tiling, 4, 128);  // Host 侧计算
    KernelCall(..., (uint8_t*)&tiling);
}

// xxx_ar_fullload.h
__aicore__ inline void Init(..., const sftmaxTilingData& tiling) {
    rowsPerLoop = tiling.rowsPerLoop;  // Kernel 直接使用
}

// ❌ 错误（Kernel 中计算)
__aicore__ inline void Init(...) {
    uint32_t ubPerRow = rLengthAlign * sizeof(T);
    rowsPerLoop = availableUB / (2 * ubPerRow);  // ⛔️ 浪费计算资源
}
```

**原因**：
- ✅ 避免 Kernel 中重复计算（每个 core 都要算一次)
- ✅ 提高性能（Host 侧计算一次 vs Kernel 侧计算多次)
- ✅ 易于调试(Host 侧可打印验证)

**违反此要求的代码将无法通过代码审查** ⛔️

---

### 0.2. 动态核数计算 ⚠️ **强制检查**

### 0.2.1 Host 侧核数计算要求

- [ ] **根据算子类型选择正确的 API 获取设备核数** ⚠️ **强制**
- [ ] **根据数据量计算实际使用核数** ⚠️ **强制**
- [ ] **Tiling 参数包含核数信息** ⚠️ **强制**
- [ ] **aclrtGetDeviceInfo 在 aclrtSetDevice 后调用** ⚠️ **强制**

**核数获取 API 选择** ⚠️ **关键**：

| 算子类型 | 使用的 API | 说明 |
|---------|-----------|------|
| **纯向量计算**（Add/Mul/Div/Reduce等） | `ACL_DEV_ATTR_VECTOR_CORE_NUM` | 使用 Vector Core 数量 |
| **矩阵计算**（MatMul/Conv等） | `ACL_DEV_ATTR_CUBE_CORE_NUM` | 使用 Cube Core 数量 |
| **混合计算** | `ACL_DEV_ATTR_AICORE_CORE_NUM` | 使用 AI Core 数量 |

**关键接口**：
```cpp
// ⚠️ 重要：必须先 aclrtSetDevice，再调用 aclrtGetDeviceInfo
aclError ret = aclrtSetDevice(deviceId);  // 先设置设备
if (ret != ACL_SUCCESS) {
    // 错误处理
}

// 获取设备核数（必须在 aclrtSetDevice 之后）
// 纯向量算子示例
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);

// 矩阵算子使用：ACL_DEV_ATTR_CUBE_CORE_NUM
// 混合算子使用：ACL_DEV_ATTR_AICORE_CORE_NUM

// 计算使用核数
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// Tiling 参数
tiling.usedCoreNum = usedNumBlocks;
tiling.rowsPerCore = totalRows / usedNumBlocks;
tiling.rowsTail = totalRows % usedNumBlocks;
```

### 0.2.2 Kernel 侧核数使用要求

- [ ] **通过 GetBlockIdx() 获取当前核 ID** ⚠️ **强制**
- [ ] **越界检查：核 ID 超出使用核数时直接返回** ⚠️ **强制**
- [ ] **正确处理尾核数据** ⚠️ **强制**

**关键代码**：
```cpp
// Kernel 侧越界检查
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) {
    return;  // ⚠️ 强制：超出使用核数直接返回
}

// 尾核处理
uint32_t rowsThisCore = (blockIdx == tiling.usedCoreNum - 1 && tiling.rowsTail != 0) 
                        ? tiling.rowsTail : tiling.rowsPerCore;
```

**常见错误**：

| 错误类型 | 错误示例 | 后果 |
|---------|---------|------|
| **调用顺序错误** | 未调用 `aclrtSetDevice` 就调用 `aclrtGetDeviceInfo` | 获取核数失败或返回错误值 |
| **API 选择错误** | 纯向量算子用 `ACL_DEV_ATTR_AICORE_CORE_NUM` | 未充分利用 Vector Core |
| 写死核数 | `uint32_t numBlocks = 8;` | 不同设备性能不匹配 |
| 越界访问 | 未检查 `blockIdx >= usedCoreNum` | 空转或越界访问内存 |
| 尾核错误 | 尾核未处理余数数据 | 最后一核数据错误 |

**违反此要求的代码将无法通过代码审查** ⛔️

---

### 0.3 多核切分合理性 ⚠️ **强制检查**

#### 0.3.1 核心问题

**算力浪费场景**：多个核处理相同数据，或数据量不足以支撑多核并行，导致：
- ❌ 核空转（部分核没有实际工作）
- ❌ 重复计算（多个核处理相同数据）
- ❌ 资源浪费（分配了多核但只用了一核的计算量）

#### 0.3.2 检查清单

**a. 数据量与核数匹配**
- [ ] **数据量足以支撑多核** ⚠️ **强制**
  - ✅ 正确：数据量 >= 核数时使用多核
  - ❌ 错误：8核处理8个元素（每个核1个元素，7个核浪费）
  - ❌ 错误：8核处理4个元素（4个核空转）

**b. 无重复计算**
- [ ] **各核处理不同数据段** ⚠️ **强制**
  - ✅ 正确：每个核处理 `rowsPerCore` 行（互不重叠）
  - ❌ 错误：每个核都处理全部数据（所有核计算相同结果）

**c. 无算力浪费**
- [ ] **核数 = min(数据量, 可用核数)** ⚠️ **强制**
- [ ] **尾核数据量合理**（不小于单核数据量的 1/2，否则应减少核数）

**d. 单核场景判断**
- [ ] **数据量 <= 阈值时使用单核**（如：数据量 < 核数）
- [ ] **单核模式代码有优化**（无双缓冲开销）

#### 0.3.3 正确示例

```cpp
// ✅ 正确：动态计算核数，避免浪费 ⚠️ 重要：必须在 aclrtSetDevice 接口后调用
// Host 侧 - 纯向量算子示例
uint32_t totalRows = shape[0];
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);

// 关键：数据量与核数匹配
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// 特殊情况：数据量极小时使用单核
if (totalRows <= 8) {
    usedNumBlocks = 1;  // 8个元素用1核处理，避免7核浪费
}

tiling.usedCoreNum = usedNumBlocks;
tiling.rowsPerCore = totalRows / usedNumBlocks;
tiling.rowsTail = totalRows % usedNumBlocks;

// Kernel 侧
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) {
    return;  // 越界检查
}

// 每个核处理不同数据段（startIdx 不同）
uint32_t startIdx = blockIdx * tiling.rowsPerCore;
uint32_t rowsThisCore = (blockIdx == tiling.usedCoreNum - 1 && tiling.rowsTail != 0) 
                        ? tiling.rowsTail : tiling.rowsPerCore;
```

#### 0.3.4 错误示例

| 场景 | 错误代码 | 后果 | 正确做法 |
|------|---------|------|---------|
| **数据量小，多核重复计算** | 8核处理8个元素，每个核都处理全部8个元素 | 7个核的计算浪费 | 1核处理，或每个核处理1个元素 |
| **核数写死** | `numBlocks = 8;` 处理任意数据量 | 小数据量时空转 | `usedNumBlocks = min(totalRows, availableCoreNum)` |
| **尾核数据量过小** | 8核处理9个元素，尾核只有1个元素 | 1核处理1元素效率低 | 使用7核，每核处理2-1-1-1-1-1-2个元素 |
| **未判断单核场景** | 8核处理8个元素 | 每核1元素，7核浪费 | 数据量<=8时使用单核 |

**错误示例代码**：
```cpp
// ❌ 错误：8核处理8个元素，每个核都处理全部数据
uint32_t numBlocks = 8;  // 写死核数
// ...
for (uint32_t i = 0; i < totalRows; i++) {  // 每个核都处理所有行
    // 所有核执行相同计算
}
// 后果：8个核计算相同结果，7个核的计算完全浪费
```

#### 0.3.5 判断逻辑

```
数据量判断：
├─ 数据量 <= 8？
│   └─→ 使用单核（避免多核开销）
│
├─ 数据量 < 可用核数？
│   └─→ usedNumBlocks = 数据量（每个核至少处理1个数据）
│
└─ 数据量 >= 可用核数？
    └─→ usedNumBlocks = 可用核数（充分利用并行）
```

#### 0.3.6 审查要点

**审查时必须回答**：
1. 数据量是多少？（totalRows, totalElements）
2. 可用核数是多少？（availableCoreNum）
3. 实际使用核数是多少？（usedNumBlocks）
4. **usedNumBlocks 是否合理**？
   - ✅ `usedNumBlocks = min(totalRows, availableCoreNum)`
   - ❌ `usedNumBlocks = 8`（写死）
5. **每个核处理的数据是否互不重叠**？
   - ✅ `startIdx = blockIdx * rowsPerCore`（每个核不同）
   - ❌ `startIdx = 0`（所有核相同）
6. **尾核数据量是否合理**？
   - ✅ 尾核数据量 >= rowsPerCore / 2
   - ❌ 尾核数据量 = 1（其他核处理100+个）

**违反此要求的代码将无法通过代码审查** ⛔️

---

### 0.4. CMakeLists.txt 配置正确性 ⚠️ **强制检查**

### 0.5.1 CMake 基础配置（3 分）

- [ ] **使用 `find_package(ASC REQUIRED)`** ⚠️ **强制**
- [ ] **project 包含 ASC 语言**：`project(... LANGUAGES ASC CXX)` ⚠️ **强制**
- [ ] **使用 `add_executable` 而非自定义函数** ⚠️ **强制**
  - ✅ 正确：`add_executable(kernel_name kernel_name.asc)`
  - ❌ 错误：`asc_add_ops_executable(...)`（不存在的函数）

### 0.5.2 库链接正确性（2 分）

- [ ] **链接 tiling_api** ⚠️ **强制**
- [ ] **链接 register** ⚠️ **强制**
- [ ] **链接 platform** ⚠️ **强制**
- [ ] **链接 m（数学库）**
- [ ] **链接 dl（动态链接库）**

**正确示例**：
```cmake
target_link_libraries(kernel_name PRIVATE
    tiling_api
    register
    platform
    m
    dl
)
```

### 0.5.3 编译选项正确性（2 分）

- [ ] **设置 NPU 架构**：`--npu-arch=dav-3101` ⚠️ **强制**
- [ ] **编译选项语法正确**

**正确示例**：
```cmake
target_compile_options(kernel_name PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
)
```

### 0.5.4 CMake 验证

- [ ] **已运行验证脚本**：`python verify_cmake_config.py CMakeLists.txt`
- [ ] **验证脚本返回码为 0**

**验证命令**：
```bash
python <skills_path>/ascendc-kernel-develop-workflow/scripts/verify_cmake_config.py \
    ops/{operator_name}/CMakeLists.txt
```

> `<skills_path>` 需根据实际环境替换，如 `.opencode/skills` 或 `skills`

### 0.5.5 常见错误

| 错误类型 | 错误示例 | 正确做法 |
|---------|---------|---------|
| **语言缺失** | `project(... LANGUAGES CXX)` | `project(... LANGUAGES ASC CXX)` |
| **错误函数** | `asc_add_ops_executable(...)` | `add_executable(...)` |
| **缺少库** | `target_link_libraries(... PRIVATE tiling_api)` | 链接所有必需库：tiling_api, register, platform, m, dl |
| **缺少架构** | 未设置 `--npu-arch` | `target_compile_options(... $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>)` |

**违反 CMake 配置要求的代码将无法通过代码审查** ⛔️

**参考文档**：[CMakeLists-template.md](../templates/CMakeLists-template.md)

---

## 1. API 使用正确性 ✅

### 1.1 数据搬运 API ⚠️ **强制检查 - 违反即不通过**

- [ ] **GM↔UB 数据搬运使用 DataCopyPad** ⚠️ **强制 - 违反即不通过**
  - ✅ 正确：GM → UB 使用 `DataCopyPad`
  - ✅ 正确：UB → GM 使用 `DataCopyPad`
  - ❌ **错误：使用 `DataCopy`（无法处理非对齐数据）** ⛔️ **违反即不通过**
  - ❌ 错误：使用 `GlobalTensor::SetValue` / `GetValue`（效率极低，仅用于调试）
  - ❌ 错误：手动循环搬运（效率低）

**⚠️ 重要：如果在 GM-UB 之间使用了 `DataCopy` 接口，代码审查直接不通过，无需继续审查其他项。**

**示例**：

```cpp
// ✅ 正确（使用 DataCopyPad）
// CopyIn: GM → UB
AscendC::DataCopyPad(xLocal, xGm[tiling.startIdx], 
    {rowsThisLoop, tiling.rLength, tiling.rLengthAlign, 0, 0});

// CopyOut: UB → GM
AscendC::DataCopyPad(yGm[tiling.startIdx], yLocal,
    {rowsThisLoop, tiling.rLength, tiling.rLengthAlign, 0, 0});

// ❌ 错误（使用 DataCopy，无法处理非对齐）⛔️ 违反即不通过
AscendC::DataCopy(xLocal, xGm[...], rowsThisLoop * tiling.rLengthAlign);  // ⛔️ 审查不通过

// ❌ 错误（使用 GlobalTensor::SetValue/GetValue，效率极低）
xGm.SetValue(idx, value);   // ⛔️ 仅用于调试，生产代码禁止使用
T val = xGm.GetValue(idx);  // ⛔️ 仅用于调试，生产代码禁止使用

// ❌ 错误（手动循环，效率低）
for (uint32_t i = 0; i < rowsThisLoop; i++) {
    for (uint32_t j = 0; j < tiling.rLength; j++) {
        xLocal.SetValue(i * tiling.rLengthAlign + j, xGm.GetValue(...));  // ⛔️
    }
}
```

**DataCopyPad 参数说明**：
```cpp
// 单行（AR 模板）
DataCopyPad(dst, src, 
    {rowsThisLoop, rLength, rLengthAlign, 0, 0});
// - rowsThisLoop: 处理的行数
// - rLength: 有效数据长度（非对齐）
// - rLengthAlign: 对齐后长度（UB 存储）

// 多维（ARA 模板）
DataCopyPad(dst, src,
    {rLength, a0TileLen, a0Length, rLengthAlign, a0TileLen});
// - rLength: R 维度有效长度
// - a0TileLen: A0 单次处理长度
// - a0Length: A0 总长度
// - rLengthAlign: R 维度对齐长度（UB 存储）
// - a0TileLen: A0 维度对齐长度（UB 存储）
```

**原因**：
- ✅ 自动处理非对齐数据（32 字节对齐要求）
- ✅ 性能优化（硬件加速）
- ✅ 避免手动搬运错误
- ✅ 支持多维数据布局

**违反此要求的代码将无法通过代码审查** ⛔️

**快速检查方法**：
```bash
# 检查是否使用了 DataCopy（GM-UB 搬运）
grep -n "DataCopy.*Gm\|DataCopy.*Local" *.asc *.cpp

# 如果发现类似以下代码，直接不通过：
# DataCopy(xLocal, xGm, ...)  // ⛔️ 应使用 DataCopyPad
# DataCopy(yGm, yLocal, ...)  // ⛔️ 应使用 DataCopyPad
```

### 1.2 API 选择

- [ ] **已查阅官方 API 文档**（`asc-devkit/docs/api/context/*.md`）
- [ ] **已搜索官方示例**（`asc-devkit/examples/`）
- [ ] **API 适用场景正确**（如：`Duplicate` vs `Broadcast`）
- [ ] **API 参数正确**（类型、对齐要求、取值范围）

### 1.3 API 一致性

- [ ] **相同场景使用相同 API**（所有分支保持一致）
- [ ] **API 使用符合官方推荐**（查阅 best-practices）
- [ ] **无已废弃的 API**（检查 API 文档中的说明）

### 1.4 常见 API 陷阱

| API 类别 | 常见错误 | 检查点 | 严重程度 |
|---------|---------|--------|---------|
| **数据搬运** | 使用 DataCopy 而非 DataCopyPad | GM↔UB 必须用 DataCopyPad | ⛔️ **违反即不通过** |
| **数据搬运** | 使用 GlobalTensor::SetValue/GetValue | 仅用于调试，生产代码禁止 | ⛔️ 违反即不通过 |
| **广播/复制** | `Broadcast` vs `Duplicate` | 标量复制用 `Duplicate` | ⚠️ 需修正 |
| **Reduce** | Pattern 选择 | AR（列方向）vs RA（行方向） | ⚠️ 需修正 |
| **类型转换** | 精度损失 | FP16→FP32→FP16 混合精度 | ⚠️ 需验证 |

### 1.5 通用 API 语义验证 ⚠️ **强制检查（新增）**

**参考文档**：[API 语义验证方法论（通用）](./api-semantic-verification.md)

**强制检查项（适用于所有 API）**：

a. **数据布局验证**
   - [ ] 已在设计文档中明确数据布局
   - [ ] 已确认数据连续性（连续 / 带 stride）
   - [ ] 已确认对齐要求

b. **功能需求验证**
   - [ ] 已明确操作类型（reduce / broadcast / elementwise / copy）
   - [ ] 已明确操作维度
   - [ ] 已明确输出格式

c. **API 能力验证**
   - [ ] 已查阅官方 API 文档（提供链接）
   - [ ] 已确认 API 适用场景
   - [ ] 已确认 API 限制条件

d. **匹配验证** ⚠️ **最高优先级**
   - [ ] **数据布局与 API 能力匹配**（⚠️ 重点关注）
   - [ ] **满足 API 的所有限制条件**
   - [ ] **无更好的 API 选择**（或已说明原因）

e. **验证记录**
   - [ ] 设计文档中有验证记录
   - [ ] 验证结论明确

**常见错误示例**：

| 场景 | ❌ 错误 API | ✅ 正确 API | 错误原因 | 严重程度 |
|------|-----------|-----------|---------|---------|
| **Reduce（带 stride）** | `ReduceMax(dst, src, tmp, count)` | `ReduceMax<T, Pattern::RA>(...)` | Level 2 只能处理连续数据 | ⚠️ 需修正 |
| **GM ↔ UB 搬运** | `DataCopy(dst, src, size)` | `DataCopyPad(dst, src, padParams)` | DataCopy 无法处理非对齐 | ⛔️ **违反即不通过** |
| **GM 单元素访问** | `xGm.SetValue/GetValue(idx, val)` | `DataCopyPad` 批量搬运 | SetValue/GetValue 效率极低 | ⛔️ 违反即不通过 |
| **标量广播** | `Duplicate + Sub` | `Adds(dst, src, -scalar, count)` | 性能低，浪费 buffer | ⚠️ 需优化 |

**违反此要求的代码将无法通过代码审查** ⛔️

---

## 2. 代码一致性 ✅

### 2.1 分支间一致性

- [ ] **所有分支的代码风格一致**
- [ ] **相同功能使用相同实现方式**
- [ ] **变量命名一致**（如：`rLength` vs `r_length`）
- [ ] **Buffer 命名一致**（如：`inQueueX` vs `inputQueue`）

### 2.2 与设计文档一致

- [ ] **文件路径与 design.md §4.1 完全一致**
- [ ] **分支数量与设计一致**
- [ ] **Tiling 参数计算逻辑与设计一致**
- [ ] **无设计文档未列出的额外文件**

---

## 3. 性能优化 ✅

### 3.1 基础优化

- [ ] **使用了 Double Buffer**（depth=2 的队列）
- [ ] **Buffer 大小合理**（不超过 UB 容量）
- [ ] **减少了不必要的数据拷贝**
- [ ] **合理使用 `DataCopyPad`**（非对齐场景）

### 3.2 高级优化

- [ ] **考虑了流水线优化**（CopyIn-Compute-CopyOut）
- [ ] **考虑了基本块优化**（如适用）
- [ ] **Tiling 参数合理**（负载均衡、并行度）

---

## 4. 内存安全 ✅

### 4.1 Buffer 管理

- [ ] **所有 `AllocTensor` 都有对应的 `FreeTensor`**
- [ ] **Queue 的 EnQue/DeQue 配对正确**
- [ ] **Buffer 大小计算正确**（考虑对齐）
- [ ] **无 Buffer 泄漏**（所有路径都释放）

### 4.2 内存访问

- [ ] **GM 访问偏移计算正确**
- [ ] **UB 访问不越界**
- [ ] **处理了非对齐场景**（DataCopyPad）

### 4.3 流水线同步 ⚠️ **强制检查**

- [ ] **DataCopy 后必须使用 EnQue/DeQue 同步** ⚠️ **强制**

**核心问题**：DataCopy 是异步 DMA，立即返回。直接使用搬运后的数据可能读到未完成的旧数据。

| 模式 | 代码 | 评价 |
|------|------|------|
| ❌ 错误 | `DataCopy(x, gm, n); Compute(x);` | 缺少同步，数据未就绪 |
| ✅ 正确 | `DataCopy → EnQue → DeQue → Compute` | 推荐方式 |
| ⚠️ 调试 | `DataCopy → PipeBarrier → Compute` | 仅用于验证同步问题 |

**示例**：
```cpp
// ✅ 正确：EnQue/DeQue 同步
void CopyIn() {
    LocalTensor<T> x = inQueue.AllocTensor<T>();
    DataCopy(x, xGm, count);
    inQueue.EnQue(x);
}
void Compute() {
    LocalTensor<T> x = inQueue.DeQue<T>();
    // ... 计算 ...
    inQueue.FreeTensor(x);
}

// ❌ 错误：缺少同步
LocalTensor<T> x = allocator.Alloc<T, 64>();
DataCopy(x, xGm, count);
Compute(x);  // ⛔️ 数据可能未就绪
```

**诊断方法**：如果输出异常，在 DataCopy 后加 `PipeBarrier<PIPE_ALL>()`，若结果正确则确认是同步问题。

**违反此要求的代码将无法通过代码审查** ⛔️

---

## 5. 精度与数值稳定性 ✅

### 5.1 数值稳定性

- [ ] **使用了稳定的数学公式**（如归一化的数值稳定算法）
- [ ] **处理了数值溢出**（exp、reduce 等）
- [ ] **FP16 场景使用了混合精度**（如需要）

### 5.2 精度要求

- [ ] **精度标准明确**（如：相对误差 < 1e-5）
- [ ] **有精度验证方法**（对比 golden）
- [ ] **边界值处理正确**（max、min、特殊值）

---

## 6. 编码规范 ✅

### 6.1 命名规范

- [ ] **类名：大驼峰**（`Kernel{Operator}{Branch}`）
- [ ] **函数名：小驼峰**（`copyIn`, `compute`）
- [ ] **变量名：下划线**（`row_idx`, `block_length`）
- [ ] **常量：全大写**（`UB_SIZE`, `BLOCK_SIZE`）

### 6.2 代码风格

- [ ] **缩进一致**（4空格或1tab）
- [ ] **花括号风格一致**（K&R 或 Allman）
- [ ] **适当的空行**（逻辑分组）
- [ ] **无多余的注释**（代码应该自解释）

### 6.3 API 注释规范 ⚠️ **强制检查**

- [ ] **每个 API 调用都有参数注释** ⚠️ **强制**
  - ✅ 包含 API 名称
  - ✅ 包含功能说明
  - ✅ 包含参数说明（逐个解释）
  - ✅ 包含关键标注（有效长度 vs 对齐长度等）

**示例**：
```cpp
// ✅ 正确：完整的 API 注释
// API: ReduceMax
// 功能: 求当前行的最大值
// 参数:
//   - dst: scalarLocal - 输出最大值（单个标量）
//   - src: xLocal[rowOffset] - 输入数据（当前行）
//   - tmp: reduceTmp - 临时计算 buffer
//   - count: rLength - 有效数据个数（非对齐长度）
//   - calIndex: false - 不计算索引
AscendC::ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmp, 
    static_cast<int32_t>(rLength), false);

// ❌ 错误：缺少注释
AscendC::ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmp, 
    static_cast<int32_t>(rLength), false);  // ⛔️ 无参数说明

// ❌ 错误：注释不完整
// 求 max
AscendC::ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmp, 
    static_cast<int32_t>(rLength), false);  // ⛔️ 缺少参数说明
```

**原因**：
- ✅ 方便代码审查（快速理解参数含义）
- ✅ 提高代码可维护性
- ✅ 减少 API 使用错误
- ✅ 便于后续优化和调试

**违反此要求的代码将无法通过代码审查** ⛔️

### 6.4 禁止事项

- [ ] **无 `std::` 命名空间函数**（kernel 中禁止）
- [ ] **无动态内存分配**（禁止 `new`/`malloc`）
- [ ] **无递归调用**
- [ ] **无未初始化的变量**

---

## 7. 错误处理 ✅

### 7.1 边界检查

- [ ] **处理了空 tensor**（shape 包含 0）
- [ ] **处理了单元素 tensor**（R=1 或 A0=1）
- [ ] **处理了非对齐场景**
- [ ] **处理了极大/极小值**

### 7.2 异常情况

- [ ] **多核切分的尾核处理正确**
- [ ] **分载模式的 last chunk 处理正确**
- [ ] **无除零风险**
- [ ] **无数组越界风险**

---

## 8. 文档完整性 ✅

### 8.1 代码注释

- [ ] **复杂逻辑有注释说明**
- [ ] **关键算法有公式注释**
- [ ] **重要常量有说明**

### 8.2 配套文档

- [ ] **design.md 完整且已更新**
- [ ] **README.md 包含使用说明**
- [ ] **测试用例文档完整**

---

## 9. 可维护性 ✅

### 9.1 代码组织

- [ ] **职责划分清晰**（CopyIn/Compute/CopyOut）
- [ ] **函数长度适中**（< 100 行）
- [ ] **无重复代码**（抽取公共函数）

### 9.2 可扩展性

- [ ] **易于添加新分支**
- [ ] **易于支持新数据类型**
- [ ] **易于调整 Tiling 参数**

---

## 10. 自我审查清单 ✅

### 10.1 开发过程回顾

- [ ] **每个 API 选择都查阅了官方文档**
- [ ] **参考了官方示例代码**
- [ ] **考虑了至少 2 种实现方案并对比**
- [ ] **遇到问题先探索，不凭直觉**

### 10.2 同行审查模拟

- [ ] **如果别人看这段代码，能理解吗？**
- [ ] **代码中有让我犹豫的地方吗？**
- [ ] **有没有"先这样写，以后再改"的代码？**
- [ ] **有没有未经验证的假设？**

---

## 审查通过标准

**必须满足**：
- ✅ 所有 **⚠️ 强制项** 全部勾选
- ✅ 总体勾选率 ≥ 90%
- ✅ 无明显的性能问题
- ✅ 无潜在的内存安全问题
- ✅ 代码一致性良好

**审查未通过**：
- ❌ 任何一项标记为"不确定"或"未完成"
- ❌ 存在已知的 bug 或性能问题
- ❌ 与设计文档不一致

---

## 审查结果

**审查日期**：___________  
**审查人员**：___________  
**审查结果**：⬜ 通过 / ⬜ 不通过  
**总体评分**：_______ / 100  

**改进建议**：
```
（列出需要改进的地方）
```

---

## 参考资源

- **API 文档**：`asc-devkit/docs/api/context/*.md`
- **官方示例**：`asc-devkit/examples/`
- **编码规范**：`ascendc-coding-standards` 技能
- **最佳实践**：`ascendc-api-best-practices` 技能
