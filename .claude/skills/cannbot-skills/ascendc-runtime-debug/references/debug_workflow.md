# 运行时错误调试工作流程

## 快速决策树

```
运行时错误
    │
    ├─ 返回码非0？
    │   ├─ 是 → 获取错误信息 (aclGetRecentErrMsg())
    │   │   │
    │   │   ├─ 错误信息清楚 → 针对性处理
    │   │   │
    │   │   └─ 错误信息不清楚 → 按错误码类型排查
    │   │       ├─ 161xxx（参数错误）→ 检查参数
    │   │       ├─ 361xxx（Runtime）→ 检查环境/设备
    │   │       └─ 561xxx（内部错误）→ 按具体错误码处理
    │   │
    │   └─ 否（程序崩溃）→ Coredump 调试
    │       └─ GDB 分析 coredump 文件
    │
    ├─ 程序卡死？
    │   └─ Kernel挂起调试 → 查看plog → 定位卡死位置
    │
    └─ 复杂场景 → 开启日志调试
```

## 详细流程

### 流程1：错误码处理

#### Step 1: 获取错误信息

```cpp
// 在 aclnn 调用后立即检查
aclnnStatus status = aclnnXxxGetWorkspaceSize(...);
if (status != ACLNN_SUCCESS) {
    const char* error_msg = aclGetRecentErrMsg();
    printf("Error: %s\n", error_msg);
    // 根据错误码进入对应处理流程
}
```

#### Step 2: 按错误码分类处理

##### 161xxx - 参数错误

```
ACLNN_ERR_PARAM_NULLPTR (161001)
    └─ 参数校验错误，参数中存在非法的nullptr

ACLNN_ERR_PARAM_INVALID (161002)
    └─ 参数校验错误
        │
        ├─ 检查输入/输出tensor的dtype是否匹配
        │   └─ 如：输入的两个数据类型不满足输入类型推导关系
        ├─ 检查输入/输出tensor的shape是否满足要求
        ├─ 检查属性值是否在合法范围内
        └─ 检查tensor format是否支持
```

#### UT调试方案 - 参数校验错误（161xxx）

**优势**：
- 无需 NPU 设备，在 Host 层快速迭代
- 可自动化批量测试各种参数组合
- 隔离环境因素，聚焦参数逻辑问题

**调试流程**：
1. 根据错误码判断问题参数
2. 使用 `ascendc-ut-generator` 技能生成参数校验测试
3. 运行 UT 定位具体参数问题

**详细 UT 开发与运行**：参考 `ascendc-ut-generator` 技能

##### 361xxx - Runtime错误

```
ACLNN_ERR_RUNTIME_ERROR (361001)
    │
    └─ API内部调用npu runtime接口异常
        ├─ 使用 aclGetRecentErrMsg() 获取详细错误信息
        └─ 根据报错提示排查
```

##### 561002 - Tiling错误

```
ACLNN_ERR_INNER_TILING_ERROR
    └─ Tiling 处理异常
        │
        ├─ 检查 TilingFunc 函数实现
        │   ├─ 确保所有分支都设置了 TilingKey
        │   └─ 确保没有除零、数组越界等异常
        │
        └─ 检查输入参数
            ├─ shape 元素个数是否超过限制
            └─ 参数组合是否在支持范围内
```

#### UT调试方案 - Tiling错误（561002）

**优势**：
- 快速验证 Tiling 逻辑，无需 NPU 设备
- 可测试所有 TilingKey 分支覆盖情况
- 快速定位是哪个输入组合导致 Tiling 失败
- 隔离 Tiling 逻辑问题与 Kernel 执行问题

**调试流程**：
1. 记录 ST 失败的输入参数（shape/dtype/属性）
2. 使用 `ascendc-ut-generator` 技能生成 Tiling 逻辑测试
3. 运行 UT 验证 Tiling 逻辑是否正确

**快速诊断**：
- UT 通过 → Tiling 逻辑正确，检查 ST 环境/Kernel
- UT 失败 → Tiling 逻辑有问题，修复 Host 代码

**详细 UT 开发与运行**：参考 `ascendc-ut-generator` 技能

##### 561003 - Kernel查找失败

```
ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    │
    └─ Kernel 查找失败
        │
        ├─ 检查算子是否已安装
        │   ├─ ls $ASCEND_OPP_PATH/vendors/<vendor_name>/op_impl/ai_core/tbe/op_api/lib/
        │   └─ 确认有对应的 .so 文件
        │
        ├─ 检查 vendor_name
        │   ├─ 编译时 --vendor_name 参数
        │   └─ 安装后的目录名是否一致
        │
        ├─ 检查 SOC 版本
        │   ├─ 编译时 --soc 参数
        │   └─ 运行时设备 SOC 是否匹配
        │
        ├─ 检查算子二进制编译
        │   └─ 算子二进制编译失败 → 检查编译日志
        │
        ├─ 检查输入类型匹配
        │   └─ 输入类型和信息库不匹配 → 检查 dtype/shape 是否在{op_name}_def.cpp原型库支持范围内
        │
        └─ 检查环境变量
            ├─ export ASCEND_OPP_PATH=/path/to/opp
            └─ export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/<vendor_name>/op_api/lib/:$LD_LIBRARY_PATH
```

##### 561107 - 环境变量缺失

详细配置步骤及常见错误见 [ascendc-env-check skill](skill:ascendc-env-check)

##### 561112 - 算子二进制包未加载

```
ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND
    │
    └─ 算子二进制包未加载
        │
        └─ 安装算子包
```

### 流程2：Kernel挂起调试

#### Step 1: 查看plog日志

```bash
# plog 默认路径
ls $HOME/ascend/log/debug/plog/plog-pid_*.log

# 或开启日志打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

#### Step 2: 分析plog内容

**核心超时**

```
症状：日志中出现 "timeout" 或程序长时间无响应

可能原因：
    ├─ Buffer 未释放 → 检查 AllocTensor/FreeTensor 配对
    ├─ 死锁 → 检查 EnQue/DeQue 配对
    ├─ 无限循环 → 检查循环终止条件
    └─ 阻塞操作 → 检查同步点
```

**内存访问越界**

```
症状：aic error 或 程序长时间无响应

可能原因：
    ├─ DataCopy 长度错误 → 检查 size 参数
    ├─ GM 地址错误 → 检查 offset 计算
    ├─ UB 访问越界 → 检查 buffer 大小
    └─ 非对齐访问 → 检查 32 字节对齐
```

#### Step 3: Kernel调试方法

**方法1：Printf调试**

```cpp
// 在 Kernel 中打印关键变量
AscendC::PRINTF("blockLength=%llu, tileNum=%llu\n", blockLength_, tileNum_);
```

**方法2：DumpTensor调试**

```cpp
// 打印 tensor 内容
AscendC::LocalTensor<T> xLocal = inQueue.DeQue<T>();
DumpTensor(xLocal, 0, 128);  // 打印前128个元素
```

**方法3：单步调试（msDebug）**

```bash
# 使用 msDebug 工具进行单步调试
# 参考：https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug
```

### 流程3：Coredump 调试（程序崩溃）

**适用场景**：程序崩溃、Segmentation Fault、Abort

#### Step 1: 启用 coredump

```bash
ulimit -c unlimited  # 启用 coredump
```

#### Step 2: 生成并分析 coredump

```bash
# 运行程序（崩溃时生成 core 文件）
./your_executable

# 使用 GDB 分析 coredump
gdb <executable> <core_file>

# GDB 常用命令 bt              # 查看调用栈 bt full         # 查看完整调用栈（包含局部变量） frame N         # 切换到第 N 层栈帧 info locals     # 查看局部变量 p variable      # 打印变量值
```

#### Step 3: 定位问题

常见崩溃原因：
- **空指针解引用**：检查 tensor 是否为 nullptr
- **内存越界**：检查 DataCopy 长度、GM/UB 访问范围
- **栈溢出**：检查递归深度或大数组

### 流程4：环境检查

使用 [ascendc-env-check skill](skill:ascendc-env-check) 进行环境检查

## 调试工具速查

| 工具/方法 | 用途 | 使用场景 | 适用错误 |
|----------|------|---------|---------|
| `aclGetRecentErrMsg()` | 获取错误详情 | 返回码非0时 | 所有错误 |
| `plog日志` | 查看运行时日志 | 所有错误 | 所有错误 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 日志打屏 | 需要实时查看日志 | 所有错误 |
| `AscendC::PRINTF` | Kernel内打印 | Kernel逻辑调试 | Kernel问题 |
| `DumpTensor` | 打印tensor内容 | 数据验证 | 精度问题 |
| `msDebug` | 单步调试 | 复杂问题（卡死、越界） | Kernel挂起 |
| `ulimit -c unlimited` | 启用 coredump | 程序崩溃前设置 | 崩溃问题 |
| `gdb <exe> <core>` | 分析 coredump | 程序崩溃时优先使用 | Segmentation Fault |
| **UT 测试** | **Host 层调试** | **参数校验/Tiling逻辑** | **161xxx/561002** |

## 未知错误码处理（兜底方案）

遇到速查表中未列出的错误码时：

### Step 1: 获取详细错误信息

```cpp
if (status != ACLNN_SUCCESS) {
    const char* error_msg = aclGetRecentErrMsg();
    printf("Error code: %d, Message: %s\n", status, error_msg);
}
```

查看 plog 日志：
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 或
ls $HOME/ascend/log/debug/plog/
```

### Step 2: 搜索官方文档

使用 `ascendc-docs-search` 技能搜索：
- 关键词：错误码数值 + 错误信息关键字
- 示例：搜索 "561118" 或 "workspace invalid"

### Step 3: 搜索社区资源

- [CANN 社区 Issue](https://gitee.com/ascend/cann/issues)
- [昇腾论坛](https://www.hiascend.com/forum)
- 搜索关键词：`错误码 + 错误信息`
