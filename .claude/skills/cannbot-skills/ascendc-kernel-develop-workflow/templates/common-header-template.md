# 公共头文件模板

**文件名**：`{operator_name}_common.h`

## 标准结构

```cpp
#ifndef {OPERATOR}_COMMON_H
#define {OPERATOR}_COMMON_H

#include "kernel_operator.h"

// Tiling 结构体
struct {Operator}TilingData {
    uint32_t totalLength;
    uint32_t tileLength;
    // ... 其他参数
};

// 分支枚举（如有多分支）
enum class BranchType {
    BRANCH_1,
    BRANCH_2,
    // ...
};

// Host 侧分支判断函数
inline BranchType DetermineBranch(const {Operator}TilingData& tiling) {
    // ...
}

// Host 侧 Tiling 计算函数
inline void ComputeTiling({Operator}TilingData& tiling, /* 参数 */) {
    // ...
}

#endif // {OPERATOR}_COMMON_H
```

## 必须包含

1. **Tiling 结构体定义**
2. **分支枚举和判断函数**（如果有多个分支）
3. **公共常量和工具函数**
