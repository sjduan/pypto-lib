---
name: ascendc-ut-develop
description: Ascend C 算子 UT 开发与覆盖率增强，支持 ops-math/ops-nn/ops-transformer/ops-cv。**主动触发场景**：用户提及"UT"、"单元测试"、"覆盖率"、"测试用例"、"未覆盖代码"、"测试补充"、"提升覆盖率"等关键词时自动触发。当需要生成 UT、补充测试用例、提升覆盖率、分析未覆盖代码时使用。注意：ST 测试不适用。
---

# AscendC 算子 UT 开发

本技能提供 Ascend C 算子 UT 开发的完整工作流程，支持从零创建 UT 和覆盖率增强两种模式。

## 核心理念

UT 生成遵循四步节奏：**理解 → 规划 → 实现 → 验证**。

1. **自动探索信息** — 给定算子路径后，自动探索层级支持、dtype/format、编译命令
2. **遵循 TDD 原则** — 异常用例先行，正常用例跟进
3. **分层独立测试** — op_host、op_api、op_kernel 各层互不依赖
4. **覆盖率目标** — UT 生成模式目标 80%+；覆盖率增强模式目标 90%+

## 使用时机

**适用**：算子开发完成后生成 UT、补充测试用例、提升覆盖率、验证算子正确性

**不适用**：算子功能尚未实现、编译错误未解决、环境配置问题

---

## 第一步：模式选择

> **进入主流程前，必须先判断走哪条路径。这是整个技能的核心决策点。**

### 检测方法

```bash
# 检查 tests/ut 目录下是否有测试文件
find <repo>/<category>/<op>/tests/ut -name "test_*.cpp" 2>/dev/null | head -5
```

### 决策逻辑

```
[UT 存在性检测] tests/ut/ 目录下是否有 test_*.cpp 测试文件？
    │
    ├─ 是 → 【覆盖率增强模式】
    │   │
    │   └─ UT 已存在，跳转到"覆盖率增强流程"
    │       目标：提升现有覆盖率至 90%+
    │
    └─ 否 → 【UT 生成模式】
        │
        └─ UT 不存在，继续"UT 生成流程"
            目标：从零创建 UT，覆盖率 80%+
```

### 模式说明

| 模式 | 触发条件 | 目标 | 流程 |
|------|----------|------|------|
| **覆盖率增强模式** | tests/ut/ 下存在 test_*.cpp 文件 | 覆盖率 90%+ | 跳转到 [覆盖率增强流程](#覆盖率增强流程) |
| **UT 生成模式** | tests/ut/ 下不存在 test_*.cpp 文件 | 覆盖率 80%+ | 继续 [UT 生成完整流程](#ut-生成完整流程) |

---

## UT 生成完整流程

**适用场景**：算子 tests/ut/ 目录下不存在 test_*.cpp 文件，需要从零创建 UT。

**完整流程**（Phase UT-1 至 UT-6）详见 [ut-generator-workflow.md](references/ut-generator/ut-generator-workflow.md)

### 核心步骤概览

| Phase | 任务 | 输出 |
|-------|------|------|
| UT-1 | 信息收集与自动探索 | 层级支持、SoC、dtype/format、编译命令 |
| UT-2 | op_host UT 编写（P0） | Tiling 测试、InferShape 测试 |
| UT-3 | op_api UT 编写（P1） | 参数校验测试 |
| UT-4 | op_kernel UT 编写（P2） | Kernel 计算逻辑测试 |
| UT-5 | 覆盖率验证 | 覆盖率报告、达标状态 |
| UT-6 | 生成最终报告 | UT 生成总结报告 |

### 详细指南

- **op_host UT**：[op-host-ut-generator.md](references/ut-generator/op-host-ut-generator.md)
- **op_api UT**：[op-api-ut-generator.md](references/ut-generator/op-api-ut-generator.md)
- **op_kernel UT**：[op-kernel-ut-generator.md](references/ut-generator/op-kernel-ut-generator.md)

---

## 覆盖率增强流程

**适用场景**：算子 UT 代码已存在，需要提升覆盖率。

**完整流程**（Phase CE-1 至 CE-6）详见 [coverage-enhancement-workflow.md](references/coverage-enhancement/coverage-enhancement-workflow.md)

### 核心步骤概览

| Phase | 任务 | 输出 |
|-------|------|------|
| CE-1 | 收集算子基本信息 | 层级、算子名、输出路径 |
| CE-2 | 获取初始覆盖率 | 覆盖率基线报告 |
| CE-3 | 分析与实现测例 | 新增测试用例 |
| CE-4 | 持续迭代优化 | 覆盖率提升报告 |
| CE-5 | 生成最终报告 | 可覆盖/无法覆盖代码清单 |
| CE-6 | 输出进入条件清单 | 参数配置指南 |

### 子智能体调用
覆盖率增强流程通过 `Task` 工具调用子智能体执行（因迭代分析需要较大上下文空间）：
- **基线覆盖率**：[baseline-coverage-agent.md](references/coverage-enhancement/baseline-coverage-agent.md)
- **测试实现**：[test-implementation-agent.md](references/coverage-enhancement/test-implementation-agent.md)

---

## Agent 使用指南

### UT生成计数规则

```
尝试次数 = 0
每层编译失败 → 查看错误日志 → 尝试修复 → 尝试次数+1
```

### 覆盖率验证规则

编译产物的覆盖率可能是全局的（包含多个算子），需要先判断类型再决定是否提取：

```
1. 编译后先判断覆盖率类型：
   - 查看覆盖率报告中的文件路径
   - 如果包含多个算子路径 → 全局覆盖率，需提取
   - 如果仅包含当前算子路径 → 单算子覆盖率，可直接使用

2. 全局覆盖率处理（如需要）：
   - 使用 lcov --extract 提取单算子覆盖率
   - 详见 [coverage-extraction-guide.md](references/coverage-enhancement/coverage-extraction-guide.md)

3. 验证标准：行覆盖率 >= 80% 且函数覆盖率 >= 80%（分支覆盖率 >= 80% 推荐但非必须）
```

---

## 参考资料

### 覆盖率增强流程
- [coverage-enhancement-workflow.md](references/coverage-enhancement/coverage-enhancement-workflow.md) - 覆盖率增强完整流程
- [coverage-extraction-guide.md](references/coverage-enhancement/coverage-extraction-guide.md) - 覆盖率提取指南
- [baseline-coverage-agent.md](references/coverage-enhancement/baseline-coverage-agent.md) - 基线覆盖率获取智能体
- [test-implementation-agent.md](references/coverage-enhancement/test-implementation-agent.md) - 测试用例实现智能体

### UT 生成流程
- [ut-generator-workflow.md](references/ut-generator/ut-generator-workflow.md) - UT 生成完整流程
- [op-api-ut-generator.md](references/ut-generator/op-api-ut-generator.md) - op_api UT详细指南
- [op-host-ut-generator.md](references/ut-generator/op-host-ut-generator.md) - op_host UT详细指南（Tiling + InferShape）
- [op-kernel-ut-generator.md](references/ut-generator/op-kernel-ut-generator.md) - op_kernel UT详细指南
- [ut-summary-template.md](references/ut-generator/ut-summary-template.md) - UT生成总结模板
