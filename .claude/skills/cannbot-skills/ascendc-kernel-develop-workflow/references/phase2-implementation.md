# Phase 2: 算子实现

## 核心原则 ⚠️ **第一眼必须看到**

### ⚠️ 关键说明：使用 Task 工具调度通用 Agent

**只有主 Agent 才能调用 task() 工具，通用 Agent（通用 Agent）没有这个权限。**

### 流程说明

通用 Agent（实现者）**独立完成**步骤 0-8（包括审视代码），返回后主 Agent 启动另一个通用 Agent 进行验收审查。

### 验收失败处理

- **评分 >= 8.5 + 测试通过**：✅ 通过
- **评分 < 8.5 或测试失败**：启动【新通用 Agent】修复
  - 新 Agent 必须先阅读设计文档
  - 针对具体问题修复代码
  - 编译测试验证

---

### 职责分离

```
主 Agent（调度者）
   │
   ├─ Task 工具调度 → 通用 Agent（实现者+审视代码）
   │                    └─ 步骤 0-8: 实现+审视代码+编译+测试 → 返回报告
   │
   ├─ Task 工具调度 → 通用 Agent（验收者）← 独立调用
   │                    └─ 审查代码 + 复跑测试 → 返回验收报告（评分）
   │
   └─ 根据评分决定：
        ├─ 评分 >= 8.5 + 测试通过：✅ 通过
        └─ 评分 < 8.5 或测试失败：启动【新通用 Agent】修复
                                 └─ 遵循设计文档，针对问题修复
                                 └─ 步骤 0-8 + 编译测试
```

| 角色 | 职责 | 工具权限 |
|-----|------|---------|
| **主 Agent** | 调度通用 Agent（实现+验收）<br>验收失败时启动新 Agent 修复<br>根据评分决定后续操作 | ✅ **有 task() 权限** |
| **通用 Agent(实现者)** | 代码实现（步骤 0-5）<br>**审视代码（步骤 6）** ⚠️ 必须执行<br>编译测试（步骤 7-8）<br>返回完整报告 | ❌ **无 task() 权限** |
| **通用 Agent(验收者)** | 代码审查<br>复跑测试验证<br>返回评分报告 | ❌ **无 task() 权限** |
| **新通用 Agent** | 根据审查问题修复代码<br>遵循设计文档实现<br>编译测试验证 | ❌ **无 task() 权限** |

---

## 流程图

```
主 Agent
   │
   ├─ 1. 决定分支优先级
   │   └─ 根据设计文档确定分支顺序（简单 → 复杂）
   │
   ├─ 2. Task 工具调度 通用 Agent（实现者+自审）
   │    └─ 步骤 0-5: 读取环境、设计文档、实现代码
   │    └─ 步骤 6: 审视代码 ⚠️ 必须执行
   │    └─ 步骤 7: 编译验证
   │    └─ 步骤 8: 测试验证（3 个基础用例）
   │    └─ 步骤 9: 返回完整报告
   │
   ├─ 3. Task 工具调度 通用 Agent（验收者）← 独立调用
   │    └─ 代码审查
   │    └─ 复跑通用 Agent(实现者) 的测试用例
   │    └─ 返回验收报告（评分）
   │
   ├─ 4. 检查验收结果
   │   ├─ 评分 >= 8.5 + 测试通过 → ✅ 通过
   │   └─ 评分 < 8.5 或测试失败 → 启动【新通用 Agent】修复
   │                           └─ 读取设计文档 + 理解问题
   │                           └─ 针对问题修复代码
   │                           └─ 编译 + 测试验证
   │                           └─ 循环步骤 3-4（最多 3 次）
   │
   └─ 5. 重复步骤 2-4 直到所有分支完成
```

**关键点**：
- **通用 Agent A 审视代码**：步骤 6 由 通用 Agent A 自己完成 ⚠️ 必须执行
- **通用 Agent B 验收**：主 Agent 调度独立的 通用 Agent B 进行最终验收
- **复跑测试**：通用 Agent B 需要复跑 通用 Agent A 的测试用例验证结果
- **新 通用 Agent 修复**：验收失败时，启动全新的 通用 Agent，遵循设计文档针对问题修复

---

## 步骤 6：审视代码 ⚠️ **通用 Agent A 必须执行**

> 审视代码是 通用 Agent A 在实现代码后、编译验证前的**强制步骤**，目的是提前发现并修复代码问题，避免编译失败或运行时错误。

### 6.1 审视内容清单

| 序号 | 审视项 | 检查要点 | 参考文档 |
|-----|--------|---------|---------|
| 1 | **Tiling 计算位置** ⚠️ **强制** | Tiling 参数必须在 Host 侧计算，不能在 Kernel 中动态计算 | code-review-checklist.md §0.1 |
| 2 | **Host侧API调用顺序** ⚠️ **强制** | aclrtGetDeviceInfo 必须在 aclrtSetDevice 之后调用，禁止在 aclInit 之前调用 | code-review-checklist.md §0.2.1, api-host-runtime.md §1.1 |
| 3 | **动态核数计算** ⚠️ **强制** | 根据算子类型选择正确API，向量算子用ACL_DEV_ATTR_VECTOR_CORE_NUM，禁止写死核数 | code-review-checklist.md §0.2.1 |
| 4 | **多核切分合理性** | 数据量与核数匹配、无重复计算、无算力浪费、尾核处理正确 | code-review-checklist.md §0.3 |
| 5 | **CMakeLists.txt 配置** | find_package(ASC)、add_executable、链接库、npu-arch | code-review-checklist.md |
| 6 | **Kernel侧API使用** | API 参数类型、返回值、对齐要求 | ascendc-api-best-practices |
| 7 | **数据搬运 API** ⚠️ **强制** | GM↔UB 必须使用 DataCopyPad，禁止使用 DataCopy | code-review-checklist.md §1.1 |
| 8 | **数据类型匹配** | FP32/FP16 类型一致性、Cast 转换 | 设计文档 |
| 9 | **Buffer 大小** | UB 容量检查、tmpBuf 大小计算 | 设计文档 |
| 10 | **编码规范** | 命名规范、注释、代码风格 | coding-standards.md |

**⚠️ 强制检查项（第1、2、3、7项）：如果违反，代码审查直接不通过，无需继续审查其他项。**

### 6.2 审视执行步骤

```bash
# 1. ⚠️ 强制检查：Tiling 计算位置
grep -n "Compute.*Tiling" *.h  # 确认 Tiling 在 common.h 或 Host 侧计算
# 如果在 Kernel 的 Init() 或 Process() 中发现计算逻辑 → 直接不通过

# 2. ⚠️ 强制检查：Host侧API调用顺序
# 查找 aclrtGetDeviceInfo 的行号
grep -n "aclrtGetDeviceInfo" *.asc
# 查找 aclrtSetDevice 的行号
grep -n "aclrtSetDevice" *.asc
# 查找 aclInit 的行号
grep -n "aclInit" *.asc
# 验证顺序：aclInit → aclrtSetDevice → aclrtGetDeviceInfo
# 如果 aclrtGetDeviceInfo 在 aclrtSetDevice 之前 → 直接不通过

# 3. ⚠️ 强制检查：动态核数计算
grep -n "numBlocks.*=.*[0-9]" *.asc *.cpp  # 检查是否写死核数
grep -n "usedCoreNum.*=.*[0-9]" *.asc *.cpp
# 如果发现 numBlocks=8 或 usedCoreNum=40 等写死的值 → 直接不通过
# 正确：usedCoreNum 通过 aclrtGetDeviceInfo 获取

# 4. 检查多核切分合理性
# - 数据量与核数是否匹配
# - 尾核处理是否正确（检查 rowsTail/elementsTail 逻辑）

# 5. 检查 CMakeLists.txt
cat CMakeLists.txt | grep -E "find_package|add_executable|target_link"

# 6. ⚠️ 强制检查：GM-UB 数据搬运 API
grep -n "DataCopy.*Gm\|DataCopy.*Local" *.asc *.cpp
# 如果发现 DataCopy(xLocal, xGm, ...) 或 DataCopy(yGm, yLocal, ...)
# 直接不通过，必须使用 DataCopyPad

# 7. 检查 Kernel 侧 API 使用
# - ReduceMax/ReduceSum 参数是否正确
# - Exp/Sub/Div count 参数是否匹配
# - Cast RoundMode 是否正确

# 8. 检查数据类型
# - FP16 输入是否使用 FP32 中间计算
# - Cast 转换方向是否正确

# 9. 运行验证脚本（如有）
python verify_cmake_config.py CMakeLists.txt
```

### 6.3 审视输出

审视完成后，通用 Agent A 应输出：

```python
{
    "审视阶段": "步骤6：审视代码",
    "审视结果": {
        "Tiling计算位置": "✅ 正确 - 在 Host 侧计算 / ❌ 错误 - 在 Kernel 中计算",
        "Host侧API调用顺序": "✅ 正确 - aclrtSetDevice 在 aclrtGetDeviceInfo 之前 / ❌ 错误 - 调用顺序错误",
        "动态核数计算": "✅ 正确 - 根据算子类型动态获取核数 / ❌ 错误 - 写死核数",
        "多核切分合理性": "✅ 正确 - 数据量与核数匹配 / ❌ 错误 - 核数浪费或重复计算",
        "CMake配置": "✅ 正确",
        "数据搬运API": "✅ 正确 - 使用 DataCopyPad / ❌ 错误 - 使用 DataCopy",
        "Kernel侧API使用": "✅ 正确",
        "数据类型": "✅ 正确",
        "Buffer大小": "✅ 正确"
    },
    "发现问题": ["问题1：xxx", "问题2：xxx"],
    "修复动作": ["修复1：xxx", "修复2：xxx"],
    "状态": "审视通过，可进入编译阶段"
}
```

**⚠️ 如果发现 GM-UB 之间使用了 DataCopy，应输出**：
```python
{
    "审视阶段": "步骤6：审视代码",
    "审视结果": {
        "数据搬运API": "❌ 错误 - 使用了 DataCopy，应使用 DataCopyPad"
    },
    "发现问题": ["发现 DataCopy(xLocal, xGm, ...) - 违反 GM-UB 数据搬运规范"],
    "修复动作": ["将 DataCopy 替换为 DataCopyPad"],
    "状态": "❌ 审视不通过 - 必须修复后重新审视"
}
```

### 6.4 审视未通过处理

- **发现问题**：记录问题 → 修复代码 → 重新审视 → 直至通过
- **审视通过后**：进入步骤 7（编译验证）

---

## 核心规则 ⚠️ **强制**

| 规则 | 强制 | 禁止 |
|-----|------|------|
| **职责分离** | ✅ 主 Agent 调度 通用 Agent A 和 B | ❌ 主 Agent 实现代码 |
| **通用 Agent A 审视代码** | ✅ 步骤 6 审视代码必须执行 | ❌ 跳过步骤6直接编译 |
| **通用 Agent B 验收** | ✅ 主 Agent 调度独立 通用 Agent B 验收 | ❌ 通用 Agent A 内部完成验收 |
| **复跑测试** | ✅ 通用 Agent B 必须复跑测试用例 | ❌ 只看报告不验证 |
| **新 通用 Agent 修复** | ✅ 验收失败时，启动新 通用 Agent 遵循设计文档修复 | ❌ 恢复旧 通用 Agent 上下文 |
| **修复规范** | ✅ 新 通用 Agent 先读设计文档，针对问题修复 | ❌ 盲目修改、跳过设计文档 |
| **实际验证** | ✅ 主 Agent 实际检查文件和命令 | ❌ 相信报告不验证 |
| **代码质量** | ✅ 验收评分 >= 8.5 + 测试通过 | ❌ 评分 < 8.5 或测试失败就继续 |

---

## ⚠️ 常见错误

### ❌ 错误示例 1：通用 Agent A 不审视代码

```python
# ❌ 错误：通用 Agent A 跳过步骤 6 审视代码
# 问题：代码质量问题未提前发现，编译失败率高

impl_result = task(
    description="实现代码",
    prompt="实现步骤 0-5 → 直接编译测试 → 返回"
)
# 缺少步骤6审视代码
```

### ❌ 错误示例 2：通用 Agent B 不复跑测试

```python
# ❌ 错误：通用 Agent B 只看报告不复跑测试
# 问题：无法验证测试结果真实性

review_result = task(
    description="代码审查",
    prompt="审查代码质量 → 给出评分"
)
# 没有复跑测试用例验证
```

### ✅ 正确做法：审视代码 + 验收复跑 + 新 通用 Agent 修复

```python
# ✅ 正确：通用 Agent A 审视代码 + 通用 Agent B 复跑验收 + 新 通用 Agent 修复

# 步骤 1: 通用 Agent A 完成所有步骤（0-8）
impl_result = task(
    description="实现代码",
    prompt="实现步骤 0-8（实现+审视代码+编译+测试）→ 返回完整报告"
)

# 步骤 2: 通用 Agent B 验收（审查+复跑测试）
review_result = task(
    description="代码验收",
    prompt="审查代码质量 + 复跑测试用例验证结果 → 返回验收报告"
)

# 步骤 3: 根据评分决定后续
if review_result['总分'] >= 8.5 and review_result['测试通过']:
    # 验收通过
    pass
else:
    # 验收失败：启动【新 通用 Agent】修复（不恢复旧 通用 Agent）
    impl_result = task(
        description="修复代码",
        prompt=f"""
你是一位 Ascend C 算子开发专家。请根据验收反馈修复代码。

**⚠️ 重要**：
- 先阅读设计文档：ops/{operator_name}/docs/design.md
- 针对具体问题进行修复，不要盲目修改

**验收评分**：{review_result['总分']}/10

**必须修改的问题**：
{review_result.get('问题列表', '无')}

**任务**：
1. 步骤 0: 读取环境报告
2. 步骤 1: 读取设计文档
3. 针对问题修改代码
4. 步骤 7: 编译验证
5. 步骤 8: 测试验证

**返回内容**：
1. 修改说明
2. 编译结果
3. 测试结果
"""
    )
```

---

## 详细文档导航

| 需要了解... | 查看文档 | 关键内容 |
|-----------|---------|---------|
| **主 Agent 如何调度?** | [phase2-detailed-guide.md](phase2-detailed-guide.md) | Task 工具调用示例<br>新 通用 Agent 修复流程<br>循环审查-修复流程 |
| **通用 Agent 如何实现?** | [phase2-detailed-guide.md](phase2-detailed-guide.md) | 步骤 0-9 详细说明<br>返回内容格式 |
| **准出检查清单?** | [phase2-detailed-guide.md](phase2-detailed-guide.md) | 主 Agent 验证项<br>完整性检查项 |
| **通用 Agent Prompt 模板?** | [../templates/branch-implementation-prompt.md](../templates/branch-implementation-prompt.md) | 步骤 0-9 模板<br>返回内容格式 |

> **⚠️ 使用原则**：主文档提供概览，详细文档提供具体实现。按需加载。

---

## 准出条件

| 检查项 | 要求 | 验证方法 |
|-------|------|---------|
| **通用 Agent A 审视代码** | ✅ 步骤 6 由 通用 Agent A 完成 | 检查报告中审视代码结果 |
| **通用 Agent B 验收** | ✅ 主 Agent 调用独立的 通用 Agent B | 检查验收报告来源 |
| **复跑测试** | ✅ 通用 Agent B 复跑测试用例 | 检查复跑的测试结果 |
| **验收评分** | ✅ >= 8.5 | 检查评分字段 |
| **测试通过** | ✅ 3 个基础用例通过 | 检查测试报告 |
| **新 通用 Agent 修复** | ✅ 验收失败时启动新 通用 Agent | 检查是否为新 task |
| **实际验证** | ✅ 主 Agent 实际检查 | 执行验证命令 |
