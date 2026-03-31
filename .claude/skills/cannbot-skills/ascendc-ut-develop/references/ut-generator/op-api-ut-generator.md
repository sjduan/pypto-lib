# OP_API UT 详细指南

## 概述

测试 ACLNN 算子接口的正确性：
- GetWorkspaceSize 返回值
- 输入参数校验（dtype、shape、format、nullptr）
- 输出结果正确性

### 目录结构

```
<repo>/<category>/<op>/tests/ut/op_api/
├── CMakeLists.txt
├── test_aclnn_<op>.cpp          # 主接口测试
├── test_aclnn_<op>_out.cpp      # out变体测试（如有）
└── test_aclnn_<op>_inplace.cpp  # inplace变体测试（如有）
```

### 核心组件

#### TensorDesc

```cpp
// 基本构造
auto tensor = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

// 链式调用
auto tensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
              .ValueRange(-2.0, 2.0)
              .Precision(0.0001, 0.0001);

// 非连续内存
auto tensor = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
```

#### ScalarDesc

```cpp
auto alpha = ScalarDesc(1.0f);           // float
auto value = ScalarDesc(42);             // int
auto flag = ScalarDesc(true);            // bool
```

#### OP_API_UT 宏

```cpp
// 单输入单输出
auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));

// 多输入带Scalar
auto ut = OP_API_UT(aclnnAdd, INPUT(self, other, alpha), OUTPUT(out));

// nullptr测试
auto ut = OP_API_UT(aclnnAbs, INPUT((aclTensor*)nullptr), OUTPUT(out));
```

### 测试用例示例

```cpp
// 异常用例（最先编写）
TEST_F(l2_abs_test, case_anullptr_input) {
    auto out = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAbs, INPUT((aclTensor*)nullptr), OUTPUT(out));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

// 正常用例
TEST_F(l2_abs_test, case_abs_for_float_type) {
    auto self = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out = TensorDesc(self).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
```

### CMakeLists.txt

```cmake
if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
```

### 常见返回值

| 返回值 | 说明 |
|--------|------|
| `ACL_SUCCESS` | 成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 参数为空指针 |
| `ACLNN_ERR_PARAM_INVALID` | 参数无效 |

