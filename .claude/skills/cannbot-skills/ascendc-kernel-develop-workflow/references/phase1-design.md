# 阶段一：需求分析与方案设计 - 主 Agent 指导

> ⚠️ **强制要求**：
> 1. **必须**使用 Task 工具调度通用 Agent（不是直接自己写代码）
> 2. **必须**先调用相关 skill 获取指导（npu-arch、kernel-design、api-best-practices）
> 3. **必须**先设计评估通过后才能进入实现阶段
> 
> **禁止**：直接开始写代码、凭经验实现、跳过设计评估流程

> **主 Agent 自检**：
> - [ ] 我准备使用 Task 工具调度通用 Agent（设计阶段）
> - [ ] 我已经/准备调用相关 skill 获取指导
> - [ ] 我不会直接开始写代码

---

## 步骤0：前置验证 ⚠️ 强制

> 在进入设计阶段前，必须完成算子类型识别和 Shape 验证。

**详细指南**：[phase0-verification.md](phase0-verification.md)

### 执行要点

| 步骤 | 内容 | 输出 |
|------|------|------|
| 0.1 | 识别算子类型 | Broadcast / 矩阵乘法 / 归约 / 卷积 / ... |
| 0.2 | 推导输出 Shape | 按对应类型的规则推导 |
| 0.3 | 验证正确性 | numpy 或手动验证 |

### 准出条件

- [ ] 算子类型已识别
- [ ] 输出 Shape 已推导并验证通过

> ⚠️ **注意**：此步骤由通用 Agent 在设计阶段开始时执行，不是独立阶段。

---

## 主 Agent 工作流

```
主 Agent
 │
 ├─► 使用 Task 工具启动通用 Agent（设计阶段）
 │     subagent_type: "general"
 │     prompt: 包含步骤1-7的完整任务描述
 │     输出：design.md
 │
 └─► 使用 Task 工具启动通用 Agent（评估阶段）
       subagent_type: "general"
       prompt: 包含评估模板的任务描述
       输出：评估结果（评分 >= 8.5 准出）
```

---

## Task 工具调用示例

### 设计阶段（步骤1-7）

```cpp
task(
    description = "执行 softmax0312 算子设计",
    prompt = """
你是 Ascend C 算子设计专家。请为 {operator_name} 执行以下步骤：

**详细步骤指导**：见 [phase1-design-subagent.md](phase1-design-subagent.md)

## 步骤概览
1. 需求检查（算子名称、数学公式、输入输出规格、精度要求）
2. 查询 NPU 架构（调用 /ascendc-npu-arch）
3. 获取设计指导（调用 /ascendc-tiling-design）
4. 确认 API 用法（调用 /ascendc-api-best-practices）
5. API 可行性验证（查阅官方文档、填写映射表）
6. 分支规划（判断简单/复杂算子、规划分支场景）
7. 输出设计文档（输出到 {operator_name}/docs/design.md）

输出要求：完整的设计文档，包含每个分支的核心伪代码（2.4节）
""",
    subagent_type = "general"
)
```

### 评估阶段（步骤8）

```cpp
task(
    description = "评估 softmax0312 设计文档",
    prompt = """
你是 Ascend C 算子设计文档评审专家。请评估 {operator_name} 的设计文档。

**设计文档**：`ops/{operator_name}/docs/design.md`

**详细评估标准**：见 [phase1-evaluation-subagent.md](phase1-evaluation-subagent.md)

**准出条件**：总分 >= 8.5
""",
    subagent_type = "general"
)
```

---

## 文件导航

| 文件 | 用途 | 目标读者 |
|------|------|---------|
| [phase1-design.md](phase1-design.md) | 主 Agent 指导（本文档） | 主 Agent |
| [phase1-design-subagent.md](phase1-design-subagent.md) | 通用 Agent 详细步骤（1-7） | 通用 Agent |
| [phase1-evaluation-subagent.md](phase1-evaluation-subagent.md) | 评估 Prompt 模板（8） | 通用 Agent（评估） |

---

## 准出条件

- [ ] 设计文档已生成（design.md）
- [ ] 设计文档评分 >= 8.5
- [ ] 所有维度评分合理（特别是多核切分策略、分支伪代码完整性）
