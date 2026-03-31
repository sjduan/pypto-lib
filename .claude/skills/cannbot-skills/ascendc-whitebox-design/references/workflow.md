# White-box Pytest Test Generation Workflow

## 流程全景

```
Step 1  用户输入算子路径、平台、覆盖档位、输出目录
           │
Step 2  Phase 1: 子 agent 并行分析源码（A: tiling+kernel, B: 接口, C: 网络 shape）
           │
        ──── 如有 disputed 路径，向用户提问（一次性问完）────
           │
        Phase 2: 子 agent 串行推导（D: 路径分析 + 参数推导，此时输入已无争议）
           ├─→ path_list.json     （路径清单 + 源码约束表）
           ├─→ param_def.json    （参数定义 + constraints + low_configs）
           └─→ test_design.md     （人可读设计文档）
           │
Step 3  test-design-checker 子 agent 交叉验证
           └─→ verification_report.json
           │    fail → 回 Step 2 修正，最多 3 轮
           │
        ──── 向用户展示 test_design.md 摘要，确认后继续 ────
           │
Step 4  run.py 调用 enumerator（消费 param_def + constraints + low_configs）
           └─→ cases.json
           │
Step 5  result-checker 子 agent 审查 cases.json
           └─→ review_report.json
           │    fail → 回 Step 2 或 Step 4 修正
           │
Step 6  主 Claude 生成 pytest（验证 py_compile + collect-only）
           └─→ test_{op_name}.py
```

---

## Step 1：输入收集

从用户消息中提取以下信息。缺失项**主动询问**（一次性问完）：

- **算子源码路径**（必选）
- **目标平台**（默认 ascend950）
- **覆盖档位**（默认 medium——Pairwise + 常见网络 shape，~100-300 个）
  - low——常见网络 shape，~10-30 个，快速冒烟
  - medium——Pairwise + 常见网络 shape，~100-300 个，简单自测
  - high——全笛卡尔积 + 常见网络 shape，~1000+ 个，全量覆盖
- **输出目录**（默认 `{算子源码路径}/tests/whitebox/`）

平台参数默认值：

| 平台 | 核数 | UB 大小 |
|------|------|---------|
| ascend910b | 48 | 192KB |
| ascend950 | 64 | 240KB |

确认平台后声明：

> "目标平台 {platform}，将使用 {核数} 核、{UB}KB UB。结果输出到 {算子源码路径}/tests/whitebox/。如需修改核数、UB 大小，或有额外特殊条件需添加，请告知。确认后开始分析。"

```
IF 用户消息中包含算子路径 + 平台 + 档位:
    直接进入 Step 2
ELSE:
    询问缺失项（一次性问完）
    等用户回复后再进入 Step 2
```

---

## Step 2：并行分析源码

**Phase 1**：派 3 个子 agent 并行分析（用 Agent tool，subagent_type="general-purpose"）：

- **Agent A（tiling + kernel）**：读 op_host/*_tiling*.cpp + op_kernel/arch35/*.h → 分支树 + 路径清单 + 源码约束表
- **Agent B（接口）**：读 _def.cpp 或 torch_ops_extension/csrc/ + proto.h → 合法输入空间 + 平台限制
- **Agent C（网络搜索）**：用 WebSearch 搜索该算子常见网络 shape → low_configs。搜不到则由主 Claude 推断

Agent A 的路径清单持久化为 `path_list.json`。

参考提示词（Agent A）：`references/prompts/code-analyzer.md`

**Phase 1 完成后：检查 disputed 路径**

主 Claude 检查 Agent A 的路径清单，如果存在接口层声明不支持但 kernel 有实现的路径（disputed），向用户提问（一次性问完）：

> "源码分析 Phase 1 完成，发现 {N} 条代码路径。
>
> 以下路径在接口层声明不支持，但 kernel 中有完整实现，需要您确认是否纳入测试：
> 1. {路径名}——{描述}——建议：{处理方式}
> 2. {路径名}——{描述}——建议：{处理方式}
>
> 请逐条回复'包含'或'排除'，或回复'全部接受建议'继续。"

用户确认后，将 disputed 路径标记为 reachable 或排除（排除的记入 test_design.md）。

**Phase 2**：派 1 个子 agent 串行推导（接收 Agent A/B 的文本输出 + 用户确认结果，不读源码）：

- **Agent D（路径分析）**：路径清单（disputed 已解决）+ 接口合法输入空间 → 可达路径 + 路径分组 + 参数定义 + 约束 + 一致性检查

参考提示词（Agent D）：`references/prompts/param-derivation.md`

**主 Claude 合并后输出**：`path_list.json` + `param_def.json` + `test_design.md`

参考提示词（test_design.md 模板）：`references/prompts/test-design-template.md`

---

## Step 3：交叉验证

**执行方式**：派 1 个独立子 agent（不复用 Step 2 的 agent，确保独立视角）。

输入：param_def.json + test_design.md + path_list.json + 源码路径
输出：`verification_report.json`

**处理验证结果**：
1. **fail 项**：必须修正，回 Step 2 重新分析，最多 3 轮
2. **warn 项**：逐条判断是否需要修正，将结论写入 test_design.md 的"验证结论"节
3. **pass 项**：无需处理

参考提示词：`references/prompts/test-design-checker.md`

### Step 2+3 完成后：停下来等用户确认

将验证结论更新到 test_design.md 后，**必须停下来**展示摘要并等待用户确认：

> "源码分析和交叉验证已完成，请检视 test_design.md：
> - {N} 个测试 group，预估 ~{M} 个参数组合
> - 覆盖档位：{level}（{描述}）
> - {K} 个未确认项需要您决定
> - 验证状态：{status}
>
> 确认后继续生成。"

**用户确认后才能进入 Step 4。** 如果用户要调整，修改后重新展示。

---

## Step 4：枚举参数组合

**执行方式**：主 Claude 直接调用 Bash 运行脚本。

```bash
python scripts/run.py \
  --param-def <output_dir>/param_def.json \
  --output_dir <output_dir>/ \
  --coverage {用户选择的档位} --seed 42
```

引擎读取 param_def + constraints + low_configs，展开后自动过滤不合法组合。

输出：`cases.json` + `coverage_report.json`（单因子和 pairwise 覆盖率报告）

---

## Step 5：审查参数组合

**执行方式**：派 1 个独立子 agent。

输入：cases.json + coverage_report.json + param_def.json + path_list.json + 源码路径
输出：`review_report.json`

检查：约束合规性、覆盖完整性、low_configs 包含、源码交叉验证。

fail 时检查 constraints 是否遗漏，回 Step 2 或 Step 4 修正。

参考提示词：`references/prompts/result-checker.md`

---

## Step 6：生成 pytest

**执行方式**：主 Claude 直接执行（需要完整上下文来写代码）。

输入：cases.json + reference 实现 + 接口信息
输出：`test_{op_name}.py`

生成后验证：
1. `python -m py_compile test_{op_name}.py` — 语法检查
2. `pytest --collect-only test_{op_name}.py` — 收集检查（区分语法错误 vs 环境缺依赖）

参考提示词：`references/prompts/pytest-generator.md`

---

## 最终产物

```
{算子源码路径}/tests/whitebox/
├── path_list.json           # Step 2 (Agent A + Agent D)
├── param_def.json          # Step 2
├── test_design.md           # Step 2 + Step 3 验证结论
├── verification_report.json # Step 3
├── cases.json               # Step 4
├── coverage_report.json     # Step 4（覆盖率报告）
├── review_report.json       # Step 5
└── test_{op_name}.py        # Step 6
```

## 参考提示词索引

| Step | 提示词 | 执行方式 |
|------|--------|---------|
| 2 Phase 1 | `references/prompts/code-analyzer.md` | 子 agent 并行 (Agent A) |
| 2 Phase 2 | `references/prompts/param-derivation.md` | 子 agent 串行 (Agent D) |
| 2 模板 | `references/prompts/test-design-template.md` | 主 Claude 合并输出 |
| 3 | `references/prompts/test-design-checker.md` | 独立子 agent |
| 5 | `references/prompts/result-checker.md` | 独立子 agent |
| 6 | `references/prompts/pytest-generator.md` | 主 Claude |
