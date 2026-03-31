---
name: ascendc-whitebox-design
description: Ascend C 算子白盒测试用例生成系统。分析算子源码提取参数维度，自动枚举参数组合，生成可执行的白盒测试用例。支持 low/medium/high 三档覆盖级别。触发场景：(1) "为 X 算子生成白盒测试用例" (2) "算子白盒用例生成" (3) "generate whitebox test cases for operator"。
---

## 当用户触发此 skill 时

直接展示以下内容，不要展示内部步骤细节：

```
Ascend C 算子白盒测试用例生成系统。

请提供以下信息：
- 算子名称（operator_name）（必选）
- 目标平台（默认 ascend950）
- 覆盖档位（默认 medium）：
  · low —— 常见网络 shape，~10-30 个，快速冒烟
  · medium —— Pairwise 组合 + 常见网络 shape，~100-300 个，简单自测
  · high —— 全笛卡尔积 + 常见网络 shape，~1000+ 个，全量覆盖
```

详细执行流程参考：[`references/workflow.md`](references/workflow.md)

## 输入

缺失项主动询问，一次性问完。

| 输入 | 必选/可选 | 默认值 | 说明 |
|------|----------|--------|------|
| 算子名称（operator_name） | 必选 | 无 | |
| 目标平台 | 可选 | ascend950 | |
| 覆盖档位 | 可选 | medium | **low**：常见网络 shape（~10-30 个）；**medium**：Pairwise + 常见网络 shape（~100-300 个）；**high**：全笛卡尔积 + 常见网络 shape（~1000+ 个） |
| 输出目录 | 可选 | `{operator_name}/tests/whitebox/` | |

用户输入后，展示即将执行的完整参数并等待确认：

> "即将为 {op_name} 生成白盒测试用例：
> - 平台：ascend950（64 核, 240KB UB）
> - 覆盖档位：medium（Pairwise + 常见网络 shape，~100-300 个）
> - 输出目录：{operator_name}/tests/whitebox/
>
> 如需修改核数、UB 大小，或有额外特殊条件需添加，请告知。确认后开始分析。"

## 最终产物

```
{operator_name}/tests/whitebox/
├── path_list.json           # Step 2 (Agent A + Agent D)
├── param_def.json          # Step 2
├── test_design.md           # Step 2 + Step 3 验证结论（用户检视文档）
├── verification_report.json # Step 3
├── cases.json              # Step 4
├── review_report.json       # Step 5
└── test_{op_name}.py        # Step 6
```

## 资源文件

| 文件 | 何时加载 | 用途 |
|------|----------|------|
| `references/workflow.md` | 首次使用 | 详细执行流程 |
| `references/prompts/code-analyzer.md` | Step 2 Phase 1 | 源码分析提示词 (Agent A) |
| `references/prompts/param-derivation.md` | Step 2 Phase 2 | 路径分析 + 参数推导提示词 (Agent D) |
| `references/prompts/test-design-template.md` | Step 2 | test_design.md 模板 |
| `references/prompts/test-design-checker.md` | Step 3 | 验证提示词 |
| `references/prompts/result-checker.md` | Step 5 | 审查提示词 |
| `references/prompts/pytest-generator.md` | Step 6 | pytest 生成提示词 |
| `scripts/run.py` | Step 4 | CLI 入口 |

## 运行前提

- Python 3.7+
- 算子源码（tiling 代码 + kernel 代码，或 torch 接口）
- 无额外 pip 依赖（仅 stdlib）
