# Ascend C 编码规范

> 本文档定义 Ascend C 算子开发的编码规范，是所有阶段的基础参考。

---

## 三条黄金法则

### 1. 理解官方示例原理后实现

- 代码结构：Kernel 函数定义 → KernelCall 函数 → main 函数
- **禁止使用前向声明**：Kernel 入口函数必须定义在调用之前
- 遵循官方命名规范：`{功能}_custom`（如 `reduce_custom`、`elementwise_custom`）
- 理解双缓冲、事件同步等优化原理后自行实现

### 2. 硬件适配性法则 ⚠️

- ❌ **禁止**：写死核数、UB大小、TILE_LENGTH 等硬件相关参数
- ✅ **必须**：动态获取硬件资源（AI Core 数量、UB 容量）
- ✅ **必须**：Tiling 切分大小基于实际 UB 容量计算

**错误示例**：
```cpp
uint32_t blockDim = 8;  // ❌ 写死核数
constexpr uint32_t TILE_LENGTH = 4096;  // ❌ 写死分块大小
```

**正确示例**：
```cpp
// Kernel 内部获取启动的块数（官方推荐）
uint32_t blockDim = AscendC::GetBlockNum();

// Host 侧获取核数（用于 tiling 配置）
auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
uint32_t blockDim = platform->GetCoreNum();

// TILE_LENGTH 基于实际 UB 容量计算
uint32_t tileLength = CalculateTileSize(ubSize);  // ✅
```

**注意**：
- `AscendC::GetBlockNum()` 在 **Kernel 代码内部**调用
- `aclrtGetDeviceInfo()` 在 **Host 侧**调用，且必须在`aclrtSetDevice`接口后调用

#### 2.3 动态核数计算 ⚠️ **强制要求**

**核心规范**：
1. Host 侧根据算子类型选择正确的 API 获取设备核数
2. Host 侧根据数据量计算：`usedNumBlocks = min(totalRows, availableCoreNum)`
3. Tiling 必须包含：`totalRows`, `rowsPerCore`, `rowsTail`, `usedCoreNum`
4. Kernel 侧必须检查：`if (blockIdx >= usedCoreNum) return;`

**核数获取 API 选择** ⚠️ **关键**：

| 算子类型 | 使用的 API | 说明 |
|---------|-----------|------|
| **纯向量计算**（Add/Mul/Div/Reduce等） | `ACL_DEV_ATTR_VECTOR_CORE_NUM` | 使用 Vector Core 数量 |
| **矩阵计算**（MatMul/Conv等） | `ACL_DEV_ATTR_CUBE_CORE_NUM` | 使用 Cube Core 数量 |
| **混合计算** | `ACL_DEV_ATTR_AICORE_CORE_NUM` | 使用 AI Core 数量 |

**关键接口**：
```cpp
// Host 侧 - 纯向量算子示例
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// Host 侧 - 矩阵算子示例
int64_t availableCoreNum = 8;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &availableCoreNum);

// Tiling 结构体
struct CustomTiling {
    uint32_t totalRows;       // 总行数
    uint32_t rowsPerCore;     // 每核处理行数
    uint32_t rowsTail;        // 尾核行数
    uint32_t usedCoreNum;     // 实际使用核数
};

// Kernel 侧
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) return;  // 越界检查
```

**常见错误**：
| 错误 | 正确做法 |
|-----|---------|
| `uint32_t numBlocks = 8;` | 动态获取：`aclrtGetDeviceInfo()` |
| 纯向量算子用 `ACL_DEV_ATTR_AICORE_CORE_NUM` | 使用 `ACL_DEV_ATTR_VECTOR_CORE_NUM` |
| 未检查 `blockIdx >= usedCoreNum` | 添加越界检查 |
| 尾核未处理 `rowsTail` | 最后一个核处理余数 |

### 3. 遇问题处理流程

- 第一步：调用 `/ascendc-precision-debug` 进行问题定位
- 第二步：查阅本地文档和示例（Read、Grep、Glob）
- 第三步：定位问题后修复，禁止简化代码或推翻重写

---

## 命名规范

| 类型 | 规范 | 示例 |
|-----|------|------|
| 类名 | 大驼峰 | `Kernel{Operator}{Branch}` |
| 函数名 | 小驼峰 | `copyIn`, `compute` |
| 变量名 | 下划线 | `row_idx`, `block_length` |
| 常量 | 全大写 | `UB_SIZE`, `BLOCK_SIZE` |

---

## 代码结构

```cpp
// 1. Kernel 类定义
class Kernel{Operator} {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);
private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueueY;
};

// 2. Kernel 入口函数（必须在调用之前定义）
__global__ __aicore__ void {operator_name}_custom(GM_ADDR x, GM_ADDR y) {
    Kernel{Operator} op;
    op.Process(x, y);
}

// 3. Host 侧调用
extern "C" void {operator_name}_custom_do(...) {
    Kernel{Operator}<<<blockDim, l2ctrl>>>();
}
```

---

## 常见错误

| 错误类型 | 根本原因 | 解决方法 |
|---------|---------|---------|
| `ambiguous` 调用错误 | Kernel 函数定义顺序错误 | 先定义后调用，不用前向声明 |
| 命名冲突 | 使用了标准库函数名 | 改用其他名称 |
| 编译失败 | 不满足对齐要求 | 检查 32/64/512 字节对齐规范 |
| 精度错误 | 数据类型不匹配 | 确认 API 支持的类型列表 |
| 硬件参数写死 | 使用固定值 | 动态获取硬件参数 |

---

## ⛔️ API 黑名单

### 数据搬运 API

**禁止使用 DataCopy 进行 GM ↔ UB 数据搬运**

| API | 状态 | 原因 | 替代方案 |
|-----|------|------|---------|
| `DataCopy(GM, UB)` | ❌ **禁止** | 不支持非对齐数据，易导致隐蔽 bug | `DataCopyPad` |
| `DataCopy(UB, GM)` | ❌ **禁止** | 不支持非对齐数据，易导致隐蔽 bug | `DataCopyPad` |

**错误示例**：
```cpp
// ❌ 错误：当数据长度不是 32 字节的倍数时会出错
AscendC::DataCopy(xLocal, xGm, dataLength);  // 危险！
AscendC::DataCopy(yGm, yLocal, dataLength);   // 危险！
```

**正确示例**：
```cpp
// ✅ 正确：统一使用 DataCopyPad
AscendC::DataCopyPadParams padParams;
AscendC::DataCopyPad(xLocal, xGm, 
    {1, static_cast<uint16_t>(dataBytes), 0, 0}, padParams);
AscendC::DataCopyPad(yGm, yLocal, 
    {1, static_cast<uint16_t>(dataBytes), 0, 0});
```

**注意**：
- `DataCopyPad` 可以正确处理对齐和非对齐数据
- 生产环境强制使用 `DataCopyPad`，无例外
- 仅在调试时可临时使用 `DataCopy`

---

## 禁止事项

- ❌ **不要在 kernel 中使用 `std::` 命名空间函数**
- ❌ **不要使用动态内存分配**（`new`/`malloc`）
- ❌ **不要使用递归调用**
- ❌ **不要使用未初始化的变量**
- ❌ **不要使用高阶封装 API**（算子级封装，如归一化类算子）
- ❌ **不要写死硬件参数**（核数、UB大小、TILE_LENGTH）
