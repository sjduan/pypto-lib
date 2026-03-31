---
name: ascendc-kernel-develop-workflow
description: Ascend C 算子开发标准流程（Kernel 直调）。包含7个阶段：环境准备 → 需求分析与方案设计 → 算子实现 → 全面测试 → 问题处理 → 开发结果总结 → 编写算子文档。触发：用户提出算子开发需求。**不适用于**算子仓（ops-*）开发。
---

# Ascend C 算子开发工作流程

## ⚠️ Kernel 直调强制要求

> **本 skill 仅适用于 Kernel 直调模式**，最终产物必须是**可独立运行的可执行程序**，不是算子库（.so）。

**阶段2 准出前必须满足**：
- ✅ 有 `main` 函数
- ✅ 有 ACL runtime（aclInit → aclrtSetDevice → aclrtCreateStream → ... → aclFinalize）
- ✅ 有 kernel 直调 `<<<blocks, nullptr, stream>>>`
- ✅ 在 NPU 上实际运行（不是仅编译通过）
- ✅ NPU 输出与 CPU golden 对比验证

**完整示例**：`asc-devkit/examples/00_introduction/01_add/basic_api_memory_allocator_add/add.asc`

---

> **强制工作流**：阶段0 → 阶段1 → 阶段2 → 阶段3 → 阶段4-6
> 
> **禁止**：直接写代码、跳过设计、跳过验收审查

---

## 流程完整性检查表

> 主 Agent 完成每阶段后，对照此表检查。**跳过任何步骤都会导致问题遗漏**。

| 阶段 | 必须的 Task | 次数 | 为什么重要 |
|-----|------------|------|-----------|
| 阶段0 | 初始化项目 + 验证环境 | 1 次 | 确保环境可用、目录结构完整 |
| 阶段1 | 设计 + 评估 | 2 次 | 评估能发现设计缺陷，避免返工 |
| 阶段2 | 实现+审视 + **验收** | 2 次/分支 | 验收能发现代码问题（如核数写死、API误用） |
| 阶段3 | 测试 | 按需 | 确保功能正确 |

**常见错误**（实际发生过）：
- 阶段0 跳过初始化脚本 → 目录结构不完整、environment.json 未保存
- 阶段2 只调用 1 个 Task → 核数写死问题未被发现
- 看到编译通过就结束 → 未验收，代码规范问题遗漏

---

## 快速检查清单

- [ ] 阶段0：项目目录 + environment.json 已保存？（CP-0）
- [ ] 阶段1：设计文档 + 评分 >= 8.5？（CP-1）
- [ ] 阶段2：**验收审查报告** + 编译 + 测试通过？（CP-2）
- [ ] 阶段3：测试通过记录？（CP-3）

> **主 Agent 启动自检**：
> - 我没有直接写代码
> - 我不会跳过阶段0（初始化项目和验证环境是基础）
> - 我不会跳过验收审查（这是发现代码问题的关键步骤）
> - 我将按流程执行：环境准备 → 设计评估 → 实现+审视 → 验收审查 → 测试

---

## 流程概览

```
阶段0: 环境准备 [1 次 Task]
    ↓
阶段1: 需求分析与方案设计  [2 次 Task]
    ↓
阶段2: 算子实现             [2 次 Task/分支]
    ↓
阶段3: 全面测试
    ↓
阶段4-6: 问题处理与交付
```

**详细指南**：[references/phase0-environment.md](references/phase0-environment.md)

---

## 强制检查点

| 检查点 | 时机 | 检查内容 | 通过标准 |
|--------|-----|---------|---------|
| **CP-0** | **阶段0后** | **项目目录 + environment.json** | **目录存在 + 文件完整** |
| CP-1 | 阶段1后 | design.md + 评分 | 2 个 Task 记录 + 评分 >= 8.5 |
| CP-2 | 阶段2每分支后 | 验收报告 + 编译 + 测试 | 2 个 Task 记录 + 评分 >= 8.5 |
| CP-3 | 阶段3后 | 测试记录 | 测试通过 |

---

## 阶段0：环境准备 ⚠️ 强制

> **禁止**：跳过阶段0直接进入设计阶段

### Step 1: 初始化算子项目

```bash
bash <skills_path>/ascendc-kernel-develop-workflow/scripts/init_operator_project.sh {operator_name}
```

> `<skills_path>` 需根据实际环境替换，如 `.opencode/skills` 或 `skills`

**创建的目录结构**：
```
ops/{operator_name}/
├── docs/           # 文档目录
├── build/          # 编译输出
├── test/           # 测试数据
└── README.md       # 项目说明
```

### Step 2: 验证环境并保存结果

```bash
bash <skills_path>/ascendc-kernel-develop-workflow/scripts/verify_environment.sh {operator_name}
```

> `<skills_path>` 需根据实际环境替换，如 `.opencode/skills` 或 `skills`

**输出**：`ops/{operator_name}/docs/environment.json`

### CP-0 准出条件

- [ ] 项目目录已创建（`ops/{operator_name}/`）
- [ ] 子目录已创建（docs/, build/, test/）
- [ ] **environment.json 已保存**（包含 NPU 型号、CANN 版本等）

**详细指南**：[references/phase0-environment.md](references/phase0-environment.md)

---

## 阶段1：需求分析与方案设计

> **前置条件**：阶段0已完成，项目目录和 environment.json 已存在

```
主 Agent
 ├─ Task: 设计 → 步骤1-7 → 设计文档
 └─ Task: 评估 → 评分 >= 8.5 准出
```

**详细指南**：[references/phase1-design.md](references/phase1-design.md)

---

## 阶段2：算子实现

**测试范围**（由阶段1的"需求类型"决定）：
- **特定用例模式**：只测试用户指定的 shape/dtype（至少 1 个）
- **通用模式**：测试多种 shape/dtype（至少 3 个）

### 执行流程

```
for each branch:
    ┌─ Task 1: 实现+审视
    │   → 实现 Kernel 代码
    │   → 自我审视（检查 Tiling、API、规范）
    │   → 编译验证
    │   → 测试验证
    │   → 返回报告（必须含"审视代码结果"字段）
    │
    └─ Task 2: 验收审查
        → 验证审视代码结果字段存在
        → 代码审查（强制检查项）
        → 复跑测试用例
        → 评分（10分制）
        
    评分 >= 8.5? → 下一分支 : 修复后重新验收
```

### ⚠️ 阶段2 准出条件（必须全部满足）

| 条件 | 检查方式 | 不满足时 |
|-----|---------|---------|
| **1. 执行了 2 个 Task** | 检查 Task 调用记录 | ❌ 必须补执行验收 Task |
| **2. 验收 Agent 返回了评分** | 检查返回的"总分"字段 | ❌ 重新执行验收 Task |
| **3. 评分 >= 8.5** | 检查评分值 | ❌ 修复问题后重新验收 |
| **4. 编译成功** | 检查 build/ 目录产物 | ❌ 修复编译错误 |
| **5. 测试通过** | 复跑测试用例 | ❌ 修复测试失败 |

**验收审查重点检查项**（详见 code-review-checklist.md）：
1. Tiling 计算位置（必须在 Host 侧）
2. 动态核数计算（根据算子类型选择正确 API，禁止写死）
3. 数据搬运 API（必须使用 DataCopyPad）
4. 设计文档一致性

**详细指南**：
- 概览：[references/phase2-implementation.md](references/phase2-implementation.md)
- 详细：[references/phase2-detailed-guide.md](references/phase2-detailed-guide.md)
- 审视清单：[references/code-review-checklist.md](references/code-review-checklist.md)

---

## 阶段3-6：测试与交付

| 阶段 | 内容 |
|-----|------|
| 阶段3 | 全面测试（Level 0-4，特定用例模式除外） |
| 阶段4 | 问题处理（按需） |
| 阶段5 | 开发结果总结 |
| 阶段6 | 编写算子文档 |

**详细指南**：[references/phase3-testing.md](references/phase3-testing.md)

---

## 快速索引

### 模板
- 设计文档：[templates/design-template.md](templates/design-template.md)
- CMakeLists：[templates/CMakeLists-template.md](templates/CMakeLists-template.md)
- 分支实现：[templates/branch-implementation-prompt.md](templates/branch-implementation-prompt.md)

### 脚本
- `init_operator_project.sh` - 初始化算子目录
- `verify_environment.sh` - 环境验证
- `verify_cmake_config.py` - CMake 配置验证

### 相关技能
- `ascendc-tiling-design` - Tiling 设计指南
- `ascendc-api-best-practices` - API 最佳实践
- `ascendc-precision-debug` - 精度调试
- `ascendc-runtime-debug` - 运行时调试
- `ascendc-npu-arch` - NPU 架构
