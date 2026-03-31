# 基线覆盖率获取智能体

## 任务描述

准备UT测试环境并获取初始覆盖率信息，为后续测试用例设计提供基线。

## 输入参数

| 参数 | 说明 | 示例 |
|------|------|------|
| operator_repo | 算子所在算子仓路径 | /mnt/workspace/ops-transformer |
| operator_name | 算子名称 | moe_distribute_combine_add_rms_norm |
| soc_version | SOC版本列表 | Ascend910b,Ascend950 |
| output_dir | 输出目录 | /mnt/workspace/ut_gen/output/{operator_name} |

## 执行步骤

### 1. 配置覆盖率收集（可选）

如果算子仓使用 `test_config.yaml` 配置覆盖率，根据算子类型修改以下配置：

```yaml
# 启用覆盖率收集
coverage:
  enabled: true
  output_path: "{output_dir}/cov_result"

# 日志配置
logging:
  level: debug
  output: "{output_dir}/log"
```

> **注意**：部分算子仓通过编译参数 `--cov` 启用覆盖率，无需修改配置文件。

### 2. 运行UT获取覆盖率

执行编译测试命令：

```bash
cd {PROJECT_DIR} && bash build.sh -u --ophost --ops='{operator_name}' --soc='{soc_version}' --cov 2>&1 | tee {output_dir}/log/round_0.log
```

**示例**：
```bash
cd /mnt/workspace/ops-transformer && bash build.sh -u --ophost --ops='moe_distribute_combine_add_rms_norm' --soc='Ascend910b','Ascend950' --cov 2>&1 | tee /mnt/workspace/ut_gen/output/moe_distribute_combine_add_rms_norm/log/round_0.log
```

**建议**：--soc 后加入所有支持的SOC版本，减少编译次数。

### 3. 分析覆盖率报告

使用 lcov 工具分析覆盖率：

```bash
# 覆盖率报告路径（编译后在 build 目录生成）
COV_PATH={PROJECT_DIR}/build/tests/ut/cov_report/cpp_utest

# 查看覆盖率摘要
lcov --summary $COV_PATH/ops.info_filtered

# 查找未覆盖的代码行
lcov --list $COV_PATH/ops.info_filtered | grep ":0"
```

**输出要求**：
1. 生成覆盖率摘要（总覆盖率百分比）
2. 提取未覆盖代码清单（文件名:行号）
3. 识别可覆盖 vs 无法覆盖的代码

### 4. 输出结果

输出以下信息供主智能体使用：

```markdown
# 初始覆盖率报告

## 覆盖率摘要
- 总覆盖率：XX.XX%
- 已覆盖行数：XXX
- 未覆盖行数：XXX

## 未覆盖代码清单
| 文件 | 行号 | 代码类型 |
|------|------|----------|
| xxx_tiling.cpp | 123-145 | 条件分支 |
| xxx_tiling.cpp | 200-210 | 死代码？ |

## 基线信息
- 可覆盖代码行数：XXX
- 疑似无法覆盖代码行数：XXX
```

## 注意事项

- 确保 build.sh 路径正确
- 处理可能的构建错误，记录错误信息
- 保留原始覆盖率信息用于后续对比
- 保存编译日志到指定路径