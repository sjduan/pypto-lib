# 阶段二分支实现 Prompt 模板

> ⚠️ **主 Agent 必须替换以下占位符后再调度 SubAgent**：
> - `{operator_name}` → 实际算子名称（如 `softmax0309`）
> - `{分支名}` → 实际分支名称（如 `ar_fullload`）
> 
> **错误示例**：`cat ops/{operator_name}/docs/environment.json`（未替换）
> **正确示例**：`cat ops/softmax0309/docs/environment.json`（已替换）

**使用时机**：主 Agent 调度 通用 Agent A 实现每个分支时使用此模板。

**⚠️ 重要修改**：
- 通用 Agent A 独立完成步骤 0-8（包含审视代码）
- 通用 Agent A 返回完整报告（实现+审视代码+编译+测试）
- 通用 Agent B 进行验收审查（复跑测试）

---

## 通用 Agent 执行流程（步骤 0-8）

```
步骤 0：读取环境检查报告 ⚠️ 强制首先执行
    ↓
步骤 1：读取设计文档
    ↓
步骤 2：查阅参考文档（CMake、编码规范、官方示例）
    ↓
步骤 3：检查/创建 CMakeLists.txt
    ↓
步骤 4：检查/创建公共头文件（如需要）
    ↓
步骤 5：实现分支代码
    ↓
步骤 6：审视代码
    ↓
步骤 7：编译验证
    ↓
步骤 8：测试验证（至少 3 个用例）
    ↓
步骤 9：返回标准化报告
```

---

### 步骤 0：读取环境检查报告 ⚠️ **强制首先执行**

**通用 Agent 必须首先读取环境检查报告**，避免错误假设。

**步骤**：
```bash
cat ops/{operator_name}/docs/environment.json
```

**根据环境信息决定测试策略**：
- ✅ **NPU 可用**：运行算子测试并验证精度
- ⚠️ **NPU 不可用**：只做编译验证，跳过运行测试

---

### 步骤 1：读取设计文档

**必须读取**：`ops/{operator_name}/docs/design.md`

**提取信息**：
- 当前分支的核心流程伪代码
- Buffer 规划
- API 映射

---

### 步骤 2：查阅参考文档

**必须查阅**：
1. CMake 模板：`./CMakeLists-template.md`
2. 公共头文件模板：`./common-header-template.md`
3. 编码规范：`../references/coding-standards.md`
4. 官方示例（至少 1 个）

---

### 步骤 3：检查/创建 CMakeLists.txt

**如果是第一个分支**，创建 CMakeLists.txt（参考 `./CMakeLists-template.md`）

**如果不是第一个分支**，检查是否存在

---

### 步骤 4：检查/创建公共头文件

**如果是第一个分支**，创建 {operator_name}_common.h（参考 `./common-header-template.md`）

---

### 步骤 5：实现分支代码

创建 {operator_name}_{分支名}.h

**强制要求**：
1. 严格遵循设计文档的伪代码
2. Tiling 计算在 Host 侧
3. Buffer 规划正确

---

### ~~步骤 6：代码审查~~ 

**⚠️ 由 通用 Agent A 自己完成

在完成代码实现后，通用 Agent A 需要进行自我代码审查。

**详细检查清单**：请参考 [code-review-checklist.md](../references/code-review-checklist.md)

**必须检查的项目**：

| 序号 | 检查项 | 说明 |
|-----|--------|------|
| 1 | Tiling 计算位置 | Tiling 参数必须在 Host 侧计算 |
| 2 | CMakeLists.txt 配置 | find_package、add_executable、链接库 |
| 3 | API 使用正确性 | 参数类型、对齐要求 |
| 4 | 数据类型匹配 | FP32/FP16 类型一致性 |
| 5 | Buffer 大小 | UB 容量检查 |
| 6 | 编码规范 | 命名、注释、代码风格 |

**审视输出格式**：
```python
{
    "审视阶段": "步骤6：审视代码",
    "审视结果": {
        "Tiling计算位置": "✅/❌",
        "CMake配置": "✅/❌",
        "API使用": "✅/❌",
        "数据类型": "✅/❌",
        "Buffer大小": "✅/❌",
        "编码规范": "✅/❌"
    },
    "发现问题": [],
    "修复动作": [],
    "状态": "审视通过/未通过"
}
```

**审视未通过**：发现问题 → 修复代码 → 重新审视 → 直至通过
**审视通过后**：进入步骤 7（编译验证）

---

### 步骤 7：编译验证

```bash
cd ops/{operator_name}
mkdir -p build && cd build
cmake .. && make
```

**如果失败**：检查错误 → 修复 → 重新编译

---

### 步骤 8：测试验证

**如果 NPU 可用**，运行至少 3 个基础用例：
1. 小 shape（如 8 元素）
2. 中 shape（如 128 元素）
3. 2D shape（如 (4, 128)）

---

### 步骤 9：返回标准化报告

**⚠️ task_id 由系统自动生成**，主 Agent 从 task() 结果中提取。

**返回格式**（完整报告）：

```python
{
    # 完整报告（包含审视代码+编译+测试）
    "环境信息": {
        "CANN 版本": "9.0.0",
        "NPU 可用": true,
        "芯片型号": "dav-2201"
    },
    
    "审视代码结果": {
        "设计一致性": "通过/有问题",
        "API 正确性": "通过/有问题",
        "编码规范": "通过/有问题",
        "问题列表": [...]
    },
    
    "编译结果": {
        "命令": "cmake .. && make",
        "输出": "成功/失败",
        "错误信息": ""
    },
    
    "测试结果": [
        {
            "用例": "1",
            "Shape": "(4, 128)",
            "Axis": -1,
            "Dtype": "FP32",
            "结果": "通过/失败"
        }
    ],
    
    "文件清单": [
        "ops/{operator_name}/CMakeLists.txt",
        "ops/{operator_name}/{operator_name}_common.h",
        "ops/{operator_name}/{operator_name}_{分支名}.h"
    ],
    
    "验证命令": {
        "编译": "cd ops/{operator_name}/build && cmake .. && make",
        "运行": "./build/{operator_name}_custom",
        "精度验证": "python verify_precision.py"
    },
    
    "实现总结": "..."
}
```

---

## ⚠️ 强制要求

1. **必须完成审视代码**：步骤 6 由 通用 Agent A 自己完成
2. **必须完成编译测试**：步骤 7-8 必须执行
3. **task_id 由系统生成**：主 Agent 提取，通用 Agent 不需要关心
4. **必须返回完整报告**（包含审视代码+编译+测试）

---

## 准出条件

- [ ] 步骤 0-5 完成（代码实现）
- [ ] 步骤 6 完成（审视代码）
- [ ] 编译成功
- [ ] 3 条基础用例通过（如果 NPU 可用）
- [ ] 返回完整报告（包含审视代码+编译+测试）
