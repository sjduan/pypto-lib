# Phase 2 详细指南

> ⚠️ **主 Agent 必须替换以下占位符后再调度 SubAgent**：
> - `{operator_name}` 或 `{operator_name}` → 实际算子名称（如 `softmax0309`）
> - `{branch}` 或 `{分支名}` → 实际分支名称（如 `ar_fullload`）
> 
> **错误示例**：`ops/{operator_name}/docs/design.md`（未替换）
> **正确示例**：`ops/softmax0309/docs/design.md`（已替换）

> **文档说明**：本文档说明主 Agent 如何调度 通用 Agent A 和 通用 Agent B 完成算子实现

---

## 目录

1. [主 Agent 工作流](#1-主-agent-工作流)
2. [通用 Agent 步骤 0-9](#2-subagent-步骤-0-9)
3. [常见问题](#3-常见问题)

---

# 1. 主 Agent 工作流

## 1.1 核心问题与解决方案

### 问题：通用 Agent 无法调用 task() 工具

**现象**：通用 Agent A 无法在内部调度 通用 Agent B 进行代码审查

**原因**：只有主 Agent 才能调用 task() 工具，通用 Agent 没有这个权限

**解决方案**：主 Agent 负责调度两个独立的 通用 Agent

### 问题：审查未通过时如何继续修改？

**方案**：启动全新的 通用 Agent 遵循设计文档针对问题修复

```
主 Agent
    ├─ task() → 通用 Agent A（实现）→ 返回报告
    ├─ task() → 通用 Agent B（审查）→ 返回评分 < 8.5
    └─ task() → 新 通用 Agent（修复）→ 遵循设计文档 + 针对问题修复
```

---

## 1.2 完整工作流

```python
branches = ["ar_fullload", "ar_colsplit", "ara_fullload", "ara_rowsplit"]
completed = []
max_iterations = 3
operator_name = "softmax0309"

for branch in branches:
    # ===== 步骤 1: 调度 通用 Agent A 实现代码 =====
    impl_result = task(
        description=f"实现 {branch}",
        prompt=f"""
你是一位 Ascend C 算子开发专家。请实现 {branch} 分支。

**⚠️ 重要**：
- **只负责实现代码，不负责代码审查**
- 代码审查将由主 Agent 调度独立的 通用 Agent B 完成

**任务**（包含步骤 6 审视代码）：
1. 步骤 0: 读取环境报告
2. 步骤 1: 读取设计文档
3. 步骤 2: 查阅参考文档
4. 步骤 3-4: 创建配置文件
5. 步骤 5: 实现代码
6. 步骤 6: 审视代码（自审查，参考 code-review-checklist.md）⚠️ **强制执行**
7. 步骤 7: 编译验证
8. 步骤 8: 测试验证

**返回内容（⚠️ 必须包含以下所有字段，否则验收直接失败）**：

0. **审视代码结果** ⛔️ **必须有，缺少此项将导致验收失败**
   ```json
   {
     "审视阶段": "步骤6：审视代码",
     "审视结果": {
       "Tiling计算位置": "✅/❌",
       "CMake配置": "✅/❌",
       "API使用": "✅/❌",
       "数据类型": "✅/❌",
       "编码规范": "✅/❌"
     },
     "发现问题": ["问题1", "问题2"],
     "修复动作": ["修复1", "修复2"],
     "状态": "审视通过/未通过"
   }
   ```

1. 环境信息
2. 编译结果
3. 测试结果
4. 文件清单
5. 验证命令
6. 实现总结

**准出条件**：
- [ ] 步骤 6 审视代码已执行（报告必须包含"审视代码结果"）
- [ ] 编译成功
- [ ] 测试通过（根据需求类型）：
  - **特定用例模式**：用户指定的 shape/dtype 测试通过（至少 1 个用例）
  - **通用模式**：3 条基础用例通过（如果 NPU 可用）
 """,
        subagent_type="general"
    )
    
    # ===== 步骤 2: 循环审查 → 修复 =====
    for iteration in range(max_iterations):
        # 调度 通用 Agent B（代码审查 + 验证报告，独立调用）
        review_result = task(
            description=f"代码审查+验证报告：{branch}",
            prompt=f"""
你是 Ascend C 代码审查专家。请完成以下任务：

## 任务 0：验证审视代码是否执行 ⛔️ **强制（第一优先级）**

通用 Agent A 返回的报告必须包含"审视代码结果"字段。

**检查清单**：
- [ ] 报告包含"审视代码结果"字段
- [ ] 审视代码结果包含所有检查项（Tiling计算位置/CMake配置/API使用/数据类型）
- [ ] 审视代码结果包含"发现问题"和"修复动作"字段

**判定规则**：
- ⛔️ 如果缺少"审视代码结果"字段 → **直接判定验收失败，评分 0/10**
- ⛔️ 如果审视代码结果不完整 → **验收失败，评分 0/10**
- ✅ 审视代码结果完整 → 继续执行任务 1-2

## 任务 1：验证 通用 Agent A 的报告

通用 Agent A 返回的报告位于：ops/{operator_name}/docs/impl_report_{branch}.json

请验证以下内容：
1. **编译结果**：编译是否成功？
2. **测试用例结果**：3个基础用例是否都通过？复跑这3个用例验证
3. **报告完整性**：是否包含所有必要信息？

## 任务 2：代码审查（重点：强制检查项 + 设计文档一致性）

审查文件：
- 代码：ops/{operator_name}/{operator_name}_{branch}.h
- 设计：ops/{operator_name}/docs/design.md

**审查维度（10分制）**：

| 维度 | 分值 | 说明 |
|-----|------|------|
| 1. **强制检查项** ⚠️ **必须全部通过** | **4分** | Tiling计算位置（§0.1）、API调用顺序（§0.2.1）、动态核数（§0.2.1）、DataCopyPad（§1.1） |
| 2. 设计文档一致性 | **3分** | 代码是否按照设计文档实现，Buffer规划、Tiling参数、分支逻辑是否一致 |
| 3. 多核切分合理性 | **2分** | 数据量与核数是否匹配、无重复计算、无算力浪费 |
| 4. 性能优化 | 1分 | Double Buffer、流水线优化、Buffer使用效率 |

**⚠️ 强制检查项详细要求**：

1. **Tiling计算位置**（code-review-checklist.md §0.1）：
   - [ ] Tiling参数在Host侧计算（common.h或main函数）
   - [ ] Kernel的Init()中只读取Tiling参数，不进行计算

2. **Host侧API调用顺序**（code-review-checklist.md §0.2.1, api-host-runtime.md §1.1）：
    - [ ] aclrtGetDeviceInfo在aclrtSetDevice之后调用
    - [ ] 调用顺序：aclInit → aclrtSetDevice → aclrtGetDeviceInfo
    - [ ] 禁止在aclInit之前调用aclrtGetDeviceInfo

3. **动态核数计算**（code-review-checklist.md §0.2.1）：
    - [ ] 根据算子类型选择正确的API（向量算子用ACL_DEV_ATTR_VECTOR_CORE_NUM）
    - [ ] 使用aclrtGetDeviceInfo动态获取核数
    - [ ] 禁止写死核数（如numBlocks=8, usedCoreNum=40）
    - [ ] Kernel侧有越界检查：if (blockIdx >= usedCoreNum) return;

4. **数据搬运API**（code-review-checklist.md §1.1）：
   - [ ] GM↔UB使用DataCopyPad
   - [ ] 禁止在GM-UB之间使用DataCopy

**判定规则**：
- ⛔️ 如果强制检查项有1项不通过 → **验收失败，评分 0/10**
- ✅ 强制检查项全部通过 → 继续评分维度2-4

## 准出条件
- 编译成功
- 3个测试用例全部通过
- **总分 >= 8.5**

## 输出格式
```
## 验证报告

### 任务0：审视代码执行验证
- 审视代码结果字段存在：✅/❌
- 审视代码结果完整：✅/❌
- 判定：✅ 继续 / ❌ 验收失败（评分 0/10）

### 任务1：报告验证
- 编译结果：✅/❌
- 测试用例1：✅/❌
- 测试用例2：✅/❌
- 测试用例3：✅/❌

### 任务2：强制检查项验证 ⚠️ **必须全部通过**
1. Tiling计算位置（§0.1）：✅/❌ - {说明}
2. Host侧API调用顺序（§0.2.1）：✅/❌ - {说明}
3. 动态核数计算（§0.2.1）：✅/❌ - {说明}
4. 数据搬运API（§1.1）：✅/❌ - {说明}

**判定**：✅ 全部通过，继续评分 / ❌ 有不通过项，验收失败（评分 0/10）

### 任务3：代码审查评分
1. 设计文档一致性：X/3
2. 多核切分合理性：X/2
3. 性能优化：X/1

总分：X/10

### 多核切分合理性检查
- 数据量与核数匹配：✅/❌
- 无重复计算：✅/❌
- 无算力浪费：✅/❌
- 说明：{具体问题或通过说明}

### 问题列表
必须修改：...
建议改进：...

### 准出判定
[ ] 通过 / [ ] 不通过
```
""",
            subagent_type="general"
        )
        
        score = review_result.get('总分', 0)
        
        # 检查评分
        if score >= 8.5:
            print(f"✅ {branch} 审查通过（{score}/10）")
            completed.append(branch)
            break
        else:
            print(f"⚠️ {branch} 审查未通过（{score}/10），第 {iteration+1} 次修复...")
            
            # 启动【新 通用 Agent】修复（不恢复旧 通用 Agent）
            impl_result = task(
                description=f"修复 {branch}",
                prompt=f"""
你是一位 Ascend C 算子开发专家。请根据验收反馈修复代码。

**⚠️ 重要**：
- 先阅读设计文档：ops/{operator_name}/docs/design.md
- 针对具体问题进行修复，不要盲目修改

**审查评分**：{score}/10

**必须修改的问题**：
{review_result.get('问题列表', '无')}

**任务**：
1. 步骤 0: 读取环境报告
2. 步骤 1: 读取设计文档（⚠️ 必须先读）
3. 针对问题修改代码
4. 步骤 7: 编译验证
5. 步骤 8: 测试验证

**返回内容**：
1. 修改说明
2. 编译结果
3. 测试结果
""",
                subagent_type="general"
            )
            
            if iteration == max_iterations - 1:
                print(f"❌ {branch} 达到最大修复次数（{max_iterations}），跳过")

# 完整性检查
if len(completed) == len(branches):
    print("✅ Phase 2 完成")
else:
    print(f"⚠️ Phase 2 部分完成：{completed}")
```

---

## 1.3 准出条件检查清单

```python
def verify_report(result, branch_name):
    """验证 通用 Agent A 的实现报告"""
    
    # 检查 0: ⚠️ 审视代码结果（强制，缺少则直接失败）
    if "审视代码结果" not in result:
        print(f"❌ 报告缺少：审视代码结果，步骤 6 未执行")
        print(f"❌ 验收直接失败，评分：0/10")
        return False
    
    review = result["审视代码结果"]
    required_review_fields = ["审视结果", "发现问题", "修复动作"]
    for field in required_review_fields:
        if field not in review:
            print(f"❌ 审视代码结果缺少：{field}")
            return False
    
    # 检查审视结果的完整性
    if "审视结果" in review:
        required_checks = ["Tiling计算位置", "CMake配置", "API使用", "数据类型"]
        for check in required_checks:
            if check not in review["审视结果"]:
                print(f"❌ 审视代码结果不完整，缺少检查项：{check}")
                return False
    
    print(f"✅ 步骤 6 审视代码已执行且完整")
    
    # 检查 1: 报告完整性（5 个部分）
    required_parts = ['环境信息', '编译结果', 
                      '测试结果', '文件清单', '验证命令']
    for part in required_parts:
        if part not in result:
            print(f"❌ 报告缺少：{part}")
            return False
    
    # 检查 2: 文件存在
    branch_file = f"ops/{operator_name}/{operator_name}_{branch_name}.h"
    if not os.path.exists(branch_file):
        print(f"❌ 文件不存在：{branch_file}")
        return False
    
    # 检查 3: 编译产物
    binary = f"ops/{operator_name}/build/{operator_name}_custom"
    if not os.path.exists(binary):
        print(f"❌ 编译产物不存在")
        return False
    
    # 检查 4: 所有验证命令可执行成功
    verify_cmd = result.get('验证命令', {})
    for key, cmd in verify_cmd.items():
        exit_code = bash(cmd).returncode
        if exit_code != 0:
            print(f"❌ 验证命令失败：{key}")
            return False
    
    print(f"✅ {branch_name} 实现报告验证通过")
    return True
```

---

# 2. 通用 Agent 步骤 0-8（审视代码 + 编译测试）

> **⚠️ 重要修改**：
> - **步骤 6（审视代码）由 通用 Agent A 自己完成 ⚠️ 必须执行**
> - **通用 Agent A 完成步骤 0-8 后返回完整报告**
> - **主 Agent 调度 通用 Agent B 进行验收（复跑测试）**

## 步骤 0-8：实现 + 审视代码 + 编译测试

通用 Agent A 依次完成：
1. 步骤 0-5：实现代码
2. **步骤 6：审视代码** ⚠️ 必须执行
3. 步骤 7：编译验证
4. 步骤 8：测试验证

完成后返回完整报告。

## 验收阶段

主 Agent 调度 通用 Agent B 进行验收审查：
- 代码审查
- 复跑测试用例

### 验收通过

分支实现完成。

### 验收未通过

主 Agent 启动【新 通用 Agent】修复：
- 新 通用 Agent 先读取设计文档
- 针对审查问题修复代码
- 编译测试验证
- 重新验收

---

## 步骤 0：读取环境报告

```bash
cat ops/{operator_name}/docs/environment.json
```

## 步骤 1：读取设计文档

```bash
cat ops/{operator_name}/docs/design.md
```

## 步骤 2：查阅参考文档

- CMake 模板：`../templates/CMakeLists-template.md`
- 编码规范：`coding-standards.md`
- 公共头文件模板：`../templates/common-header-template.md`
- 官方示例：`asc-devkit/examples/`

## 步骤 3-4：检查/创建配置文件

创建 CMakeLists.txt 和 {operator_name}_common.h

## 步骤 5：实现分支代码

创建 {operator_name}_{分支名}.h

## 步骤 6：审视代码 ⚠️ 必须执行

> 审视代码是 通用 Agent A 在实现代码后、编译验证前的**强制步骤**

### 审视内容

通用 Agent A 进行自我代码审查：
- [ ] 检查代码是否符合设计文档
- [ ] 检查 API 使用是否正确
- [ ] 检查编码规范是否符合要求
- [ ] **检查 Tiling 计算位置（必须在 Host 侧）**
- [ ] 检查 CMakeLists.txt 配置
- [ ] 检查数据类型一致性（FP32/FP16）

### 审视执行命令

```bash
# 1. 检查 Tiling 计算位置
grep -n "Compute.*Tiling" *.h

# 2. 检查 CMakeLists.txt 配置
cat CMakeLists.txt | grep -E "find_package|add_executable|target_link"

# 3. 检查 API 使用（根据具体算子）
# - ReduceMax/ReduceSum 参数
# - Exp/Sub/Div count 参数
# - Cast RoundMode

# 4. 运行验证脚本
python verify_cmake_config.py CMakeLists.txt
```

### 审视输出

```python
{
    "审视阶段": "步骤6：审视代码",
    "审视结果": {
        "Tiling计算位置": "✅/❌",
        "CMake配置": "✅/❌",
        "API使用": "✅/❌",
        "数据类型": "✅/❌"
    },
    "发现问题": [],
    "修复动作": [],
    "状态": "审视通过/未通过"
}
```

## 步骤 7：编译验证

```bash
cd ops/{operator_name}
mkdir -p build && cd build
cmake .. && make
```

## 步骤 8：测试验证

### 8.1 测试策略判断 ⚠️ 强制

**根据设计文档 `design.md` 中的"需求类型"字段决定测试范围**：

#### 特定用例模式

如果需求类型为"特定用例"（用户明确指定了 shape 和 dtype）：

**测试范围**：
- ✅ **只测试用户明确指定的 shape 和 dtype**
- ✅ 至少需要 1 个测试用例（用户指定的配置）
- ❌ **不需要**测试其他 shape/dtype 组合
- ❌ **不需要**泛化到其他配置

**示例**：
```
用户需求："开发 softmax 算子，shape=[1,1024], dtype=float16"

✅ 只测试：
  - shape=[1,1024], dtype=float16

❌ 不测试：
  - shape=[2,2048], dtype=float16
  - shape=[1,1024], dtype=float32
  - 其他任何配置
```

**生成测试用例**：
```bash
# 只生成用户指定的配置
python3 gen_golden.py "1,1024" -1 float16 test_specific_case
```

#### 通用模式

如果需求类型为"通用"（用户未明确指定或要求支持多种配置）：

**测试范围**：
- ✅ Phase 2 至少需要 **3 个基础用例**（覆盖不同 shape/dtype）
- ✅ Phase 3 执行 Level 0-4 全量测试
- ✅ 需要泛化到多种 shape/dtype 组合

**生成测试用例**：
```bash
# 生成 3 个基础用例（不同 shape/dtype）
python3 gen_golden.py "8" -1 float32 test_case_1
python3 gen_golden.py "1024" -1 float16 test_case_2
python3 gen_golden.py "64,128" -1 float32 test_case_3
```

### 8.2 执行测试

根据判断结果执行对应范围的测试

## 步骤 9：返回标准化报告（完整报告）

返回包含审视代码+编译+测试的完整报告：

```python
{
    "阶段": "完整实现",
    "状态": "审视代码+编译+测试完成",
    "审视代码结果": {...},
    "编译结果": {...},
    "测试结果": [...],
    "文件清单": [...],
    "验证命令": {...},
    "实现总结": "..."
}
```

### 第二阶段返回（审查通过后步骤 7-9 后）

```python
{
    "阶段": "第二阶段",
    "状态": "编译测试完成",
    "编译结果": {...},
    "测试结果": [...],
    "文件清单": [...],
    "验证命令": {...},
    "实现总结": "..."
}
```

---

# 3. 常见问题

## Q1: 通用 Agent A 为什么不能调用 task()？

**A**: 只有主 Agent 才能调用 task() 工具，通用 Agent 没有这个权限。

## Q2: 通用 Agent B 验收和 通用 Agent A 审视代码的区别？

**A**: 
- **审视代码**：通用 Agent A 自己检查代码质量（是否符合设计、API是否正确、Tiling位置等）
- **验收**：通用 Agent B 独立验证（代码审查 + 复跑测试用例）

## Q3: 验收未通过时如何继续修改？

**A**: 主 Agent 启动全新的 通用 Agent（不是恢复旧的），新 通用 Agent 必须：
1. 先阅读设计文档
2. 针对审查问题修复代码
3. 编译测试验证
4. 返回结果供重新验收

## Q4: 为什么启动新 通用 Agent 而不是恢复旧的？

**A**: 
- 新 通用 Agent 从全新视角审视问题，避免思维定势
- 新 通用 Agent 必须先读设计文档，确保修复方案符合设计
- 每次修复都是独立的任务，问题更清晰

## Q5: 如何保证步骤 6（审视代码）一定被执行？

**A**: 采用三层保障机制：

### 第 1 层：Agent A Prompt 强制返回字段

```python
**返回内容（⚠️ 必须包含以下所有字段，否则验收直接失败）**：

0. **审视代码结果** ⛔️ **必须有，缺少此项将导致验收失败**
```

### 第 2 层：Agent B 验证层

```python
## 任务 0：验证审视代码是否执行 ⛔️ **强制（第一优先级）**

**判定规则**：
- ⛔️ 如果缺少"审视代码结果"字段 → **直接判定验收失败，评分 0/10**
```

### 第 3 层：准出检查层

```python
def verify_report(result, branch_name):
    # 检查 0: ⚠️ 审视代码结果（强制，缺少则直接失败）
    if "审视代码结果" not in result:
        print(f"❌ 报告缺少：审视代码结果，步骤 6 未执行")
        return False
```

**三层保障总结**：

| 层级 | 执行者 | 机制 | 失败后果 |
|-----|-------|------|---------|
| Prompt 层 | Agent A | 强制返回字段要求 | Prompt 明确说明 |
| 验收层 | Agent B | 任务 0 优先验证 | 评分 0/10，验收失败 |
| 准出层 | 主 Agent | verify_report() 检查 | 返回 False，阻止通过 |

