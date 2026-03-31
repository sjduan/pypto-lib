# 阶段零：环境准备 ⚠️ 强制

## 前置步骤：算子项目初始化

**⚠️ 在执行环境验证之前，必须先初始化算子项目目录。**

### 初始化命令

```bash
# 初始化算子项目
bash <skills_path>/ascendc-kernel-develop-workflow/scripts/init_operator_project.sh {operator_name}
```

> `<skills_path>` 需根据实际环境替换，如 `.opencode/skills` 或 `skills`

**示例**：
```bash
bash <skills_path>/ascendc-kernel-develop-workflow/scripts/init_operator_project.sh softmax0309
```

### 创建的目录结构

```
ops/{operator_name}/
├── docs/           # 文档目录（环境检查、设计文档等）
├── build/          # 编译输出目录
├── test/           # 测试数据目录
└── README.md       # 项目说明文件
```

### 初始化完成标志

- [ ] 项目目录已创建
- [ ] docs/, build/, test/ 子目录已创建
- [ ] README.md 已生成

---

## 核心要点

| 错误类型 | ❌ 错误 | ✅ 正确 |
|---------|--------|--------|
| 环境变量 | `ASCEND_HOME` | `ASCEND_HOME_PATH` |
| 编译器 | `tikcc`, `bangc` | `bisheng` |
| 构建方式 | 编写 `build.sh` | `cmake && make` |

---

## 环境变量设置

```bash
# ✅ 正确
export ASCEND_HOME_PATH=/home/developer/Ascend/cann

# ❌ 错误
export ASCEND_HOME=/home/developer/Ascend/cann
```

**优先级**：`$ENV{ASCEND_HOME_PATH}` > `$ENV{HOME}/Ascend/cann` > `/usr/local/Ascend/cann`

---

## 验证步骤

**⚠️ 前置要求**：必须先完成"算子项目初始化"步骤。

### 自动验证（推荐）

```bash
# 验证环境并保存结果
bash <skills_path>/ascendc-kernel-develop-workflow/scripts/verify_environment.sh {operator_name}
```

> `<skills_path>` 需根据实际环境替换，如 `.opencode/skills` 或 `skills`

**参数说明**：
- `{operator_name}`：算子名称（必需）

**示例**：
```bash
# 初始化项目（首次必须）
bash <skills_path>/ascendc-kernel-develop-workflow/scripts/init_operator_project.sh softmax0309

# 验证环境并保存结果到 softmax0309/docs/environment.json
bash <skills_path>/ascendc-kernel-develop-workflow/scripts/verify_environment.sh softmax0309
```

### 手动验证

```bash
# 1. 检查环境变量
echo $ASCEND_HOME_PATH

# 2. 检查编译器
ls $ASCEND_HOME_PATH/aarch64-linux/ccec_compiler/bin/bisheng

# 3. 检查头文件
ls $ASCEND_HOME_PATH/include/kernel_operator.h

# 4. 检查库文件
ls $ASCEND_HOME_PATH/lib64/libregister.so

# 5. 检查 asc-devkit
ls asc-devkit/

# 6. 检查 NPU 设备（如果有）
npu-smi info
```

---

## 环境检查结果文件

### 文件位置

```
ops/{operator_name}/docs/environment.json
```

### 文件格式

详见模板文件：[../templates/environment.json](../templates/environment.json)

### 使用场景

1. **算子实现前**：SubAgent 读取 `environment.json`，了解环境配置
2. **代码生成时**：根据 CANN 版本和架构生成对应代码
3. **测试验证时**：确认 NPU 设备可用，选择合适的测试策略

---

## 准出条件

- [ ] `ASCEND_HOME_PATH` 已设置
- [ ] CANN 编译器存在（bisheng）
- [ ] 头文件存在（kernel_operator.h）
- [ ] 库文件存在（libregister.so）
- [ ] asc-devkit 目录存在
- [ ] **环境检查结果已保存**（`ops/{operator_name}/docs/environment.json`）

---

## 常见问题

| 问题 | 解决方法 |
|-----|---------|
| `ASCEND_HOME_PATH` 未设置 | `export ASCEND_HOME_PATH=/usr/local/Ascend/cann` |
| `bisheng` 命令不存在 | 检查 CANN 安装路径，添加到 PATH |
| asc-devkit 不存在 | `git clone https://gitcode.com/cann/asc-devkit` |
| NPU 不可用 | 检查驱动安装，运行 `npu-smi info` |
| 环境检查结果未保存 | 添加 `--operator {operator_name} --save` 参数 |

---

## 详细参考

- [environment-setup.md](environment-setup.md) - 环境配置详细说明
- [CMakeLists-template.md](../templates/CMakeLists-template.md) - CMakeLists.txt 完整配置模板（阶段二使用）
