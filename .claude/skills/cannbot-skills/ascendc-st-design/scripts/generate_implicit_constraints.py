#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隐式约束生成脚本

功能：
1. 从04_测试因子.yaml中识别需要生成隐式依赖的因子
2. 生成隐式约束并追加到05_约束定义.yaml
3. 保持约束ID的唯一性（避免重复）
4. 保持原始文件格式不变

隐式依赖规则：
1. Tensor输入：{param}.shape 依赖于 {param}.dimensions
   - 当参数有 dimensions 因子时，生成 shape 约束
2. 所有输入：{param}.value_range 依赖于 {param}.dtype
   - 当参数有 dtype 和 value_range 因子时，生成 value_range 约束
3. 非Tensor且非枚举类型输入：{param}.value 依赖于 {param}.value_range
   - 当参数有 value_range 因子且非 Tensor 非枚举时，生成 value 约束

使用方法:
    python scripts/generate_implicit_constraints.py <测试因子.yaml> <约束定义.yaml>
    
示例:
    python scripts/generate_implicit_constraints.py \
        result/04_测试因子.yaml \
        result/05_约束定义.yaml
"""

import sys
import yaml
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import OrderedDict
from datetime import datetime


class ImplicitConstraintGenerator:
    """隐式约束生成器"""

    def __init__(self, factors_path: str, constraints_path: str):
        """
        初始化隐式约束生成器
        
        Args:
            factors_path: 测试因子YAML文件路径（04_测试因子.yaml）
            constraints_path: 约束定义YAML文件路径（05_约束定义.yaml）
        """
        self.factors_path = factors_path
        self.constraints_path = constraints_path
        self.factors_data = {}
        self.constraints_data = {}
        self.original_content = ""
        self.existing_constraint_ids = set()
        self.new_constraints = []

        # 加载测试因子定义
        print(f"正在加载测试因子: {factors_path}")
        try:
            with open(self.factors_path, "r", encoding="utf-8") as f:
                self.factors_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"测试因子文件不存在: {factors_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"测试因子文件格式错误: {e}")

        # 加载现有约束定义
        print(f"正在加载约束定义: {constraints_path}")
        try:
            with open(self.constraints_path, "r", encoding="utf-8") as f:
                self.original_content = f.read()
                f.seek(0)
                self.constraints_data = yaml.safe_load(f)
                if self.constraints_data is None:
                    self.constraints_data = {}
        except FileNotFoundError:
            print(f"约束定义文件不存在，将创建新文件: {constraints_path}")
            self.constraints_data = {}
            self.original_content = ""
        except yaml.YAMLError as e:
            raise ValueError(f"约束定义文件格式错误: {e}")

        # 初始化约束数据结构
        if "constraints" not in self.constraints_data:
            self.constraints_data["constraints"] = []

        # 提取现有约束ID
        self.existing_constraint_ids = set(
            c.get("id") for c in self.constraints_data.get("constraints", []) if c.get("id")
        )
        print(f"已存在 {len(self.existing_constraint_ids)} 个约束")

    def _get_param_type(self, param_name: str) -> str:
        """
        获取参数类型
        
        Args:
            param_name: 参数名
            
        Returns:
            参数类型 (aclTensor, aclScalar, 或其他)
        """
        if param_name in self.factors_data:
            return self.factors_data[param_name].get("type", "unknown")
        return "unknown"

    def _is_tensor_param(self, param_name: str) -> bool:
        """判断是否为 Tensor 类型参数"""
        param_type = self._get_param_type(param_name)
        return param_type == "aclTensor"

    def _is_enum_param(self, param_name: str, factors: Dict) -> bool:
        """
        判断是否为枚举类型参数
        
        枚举类型的特征：有 value 因子但没有 value_range 因子
        """
        has_value = f"{param_name}.value" in factors
        has_value_range = any(key.startswith(f"{param_name}.value_range") for key in factors.keys())
        return has_value and not has_value_range

    def _identify_implicit_dependencies(self) -> List[Dict]:
        """
        识别需要生成隐式依赖的因子
        
        Returns:
            需要生成隐式约束的因子列表
        """
        dependencies = []

        if not self.factors_data:
            print("警告: 测试因子数据为空")
            return dependencies
        print('++self.factors_data: ', self.factors_data)

        for param_name, param_data in self.factors_data.items():
            if not isinstance(param_data, dict):
                continue

            factors = param_data.get("factors", {})
            if not factors:
                continue

            param_type = self._get_param_type(param_name)

            # 规则1: Tensor输入 - shape 依赖于 dimensions
            # 只要 Tensor 有 dimensions 因子，就应该生成 shape 约束
            if self._is_tensor_param(param_name):
                has_dimensions = f"{param_name}.dimensions" in factors
                if has_dimensions:
                    constraint_id = f"IMPLICIT-SHAPE-{param_name}"
                    if constraint_id not in self.existing_constraint_ids:
                        dependencies.append(
                            {
                                "rule": "shape_from_dimensions",
                                "param_name": param_name,
                                "constraint_id": constraint_id,
                            }
                        )
                        print(f"  发现隐式依赖: {param_name}.shape <- {param_name}.dimensions")

            # 规则2: 所有输入 - value_range 依赖于 dtype
            # 当参数有 dtype 和 value_range 因子时
            if self._should_generate_value_range_constraint(param_name, factors):
                constraint_id = f"IMPLICIT-RANGE-{param_name}"
                if constraint_id not in self.existing_constraint_ids:
                    dependencies.append(
                        {
                            "rule": "value_range_from_dtype",
                            "param_name": param_name,
                            "constraint_id": constraint_id,
                        }
                    )
                    print(f"  发现隐式依赖: {param_name}.value_range <- {param_name}.dtype")

            # 规则3: 非Tensor且非枚举类型 - value 依赖于 value_range
            # 只要非 Tensor 有 value_range 因子，就应该生成 value 约束
            if not self._is_tensor_param(param_name) and not self._is_enum_param(param_name, factors):
                has_value_range = any(key.startswith(f"{param_name}.value_range") for key in factors.keys())
                if has_value_range:
                    constraint_id = f"IMPLICIT-VALUE-{param_name}"
                    if constraint_id not in self.existing_constraint_ids:
                        dependencies.append(
                            {
                                "rule": "value_from_value_range",
                                "param_name": param_name,
                                "constraint_id": constraint_id,
                            }
                        )
                        print(f"  发现隐式依赖: {param_name}.value <- {param_name}.value_range")

        return dependencies

    def _should_generate_value_range_constraint(self, param_name: str, factors: Dict) -> bool:
        """
        判断是否需要生成 value_range 依赖于 dtype 的约束
        
        Args:
            param_name: 参数名
            factors: 因子字典
            
        Returns:
            是否需要生成约束
        """
        has_dtype = f"{param_name}.dtype" in factors
        has_value_range = any(key.startswith(f"{param_name}.value_range") for key in factors.keys())
        return has_dtype and has_value_range

    def _generate_shape_constraint(self, param_name: str) -> Dict:
        """
        生成 shape 依赖于 dimensions 的约束
        
        Args:
            param_name: 参数名
            
        Returns:
            约束字典
        """
        return OrderedDict([
            ("id", f"IMPLICIT-SHAPE-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.dimensions"]),
            ("target", f"{param_name}.shape"),
            ("expression", "derive_shape_from_dimensions(sources[0])"),
            ("description", f"根据dimensions生成{param_name}.shape"),
            ("implicit", True),
        ])

    def _generate_value_range_constraint(self, param_name: str) -> Dict:
        """
        生成 value_range 依赖于 dtype 的约束
        
        Args:
            param_name: 参数名
            
        Returns:
            约束字典
        """
        return OrderedDict([
            ("id", f"IMPLICIT-RANGE-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.dtype"]),
            ("target", f"{param_name}.value_range"),
            ("expression", "derive_value_range_from_dtype(sources[0])"),
            ("description", f"根据dtype选择{param_name}.value_range"),
            ("implicit", True),
        ])

    def _generate_value_constraint(self, param_name: str) -> Dict:
        """
        生成 value 依赖于 value_range 的约束
        
        Args:
            param_name: 参数名
            
        Returns:
            约束字典
        """
        return OrderedDict([
            ("id", f"IMPLICIT-VALUE-{param_name}"),
            ("type", "calculate"),
            ("sources", [f"{param_name}.value_range"]),
            ("target", f"{param_name}.value"),
            ("expression", "derive_value_from_range(sources[0])"),
            ("description", f"根据value_range生成{param_name}.value"),
            ("implicit", True),
        ])

    def generate(self) -> List[Dict]:
        """
        生成所有隐式约束
        
        Returns:
            生成的隐式约束列表
        """
        print("\n开始识别隐式依赖...")
        dependencies = self._identify_implicit_dependencies()

        if not dependencies:
            print("未发现需要生成的隐式约束")
            return []

        print(f"\n生成 {len(dependencies)} 个隐式约束...")
        for dep in dependencies:
            rule = dep["rule"]
            param_name = dep["param_name"]

            if rule == "shape_from_dimensions":
                constraint = self._generate_shape_constraint(param_name)
            elif rule == "value_range_from_dtype":
                constraint = self._generate_value_range_constraint(param_name)
            elif rule == "value_from_value_range":
                constraint = self._generate_value_constraint(param_name)
            else:
                print(f"警告: 未知的规则类型 {rule}")
                continue

            self.new_constraints.append(constraint)

        return self.new_constraints

    def _format_constraint_yaml(self, constraint: Dict) -> str:
        """
        将约束格式化为 YAML 字符串，保持一致的缩进和格式
        
        Args:
            constraint: 约束字典
            
        Returns:
            格式化后的 YAML 字符串
        """
        lines = [f"  - id: \"{constraint['id']}\""]
        lines.append(f"    type: {constraint['type']}")
        lines.append(f"    sources: {yaml.dump(constraint['sources'], default_flow_style=True).strip()}")
        lines.append(f"    target: \"{constraint['target']}\"")
        lines.append(f"    expression: \"{constraint['expression']}\"")
        lines.append(f"    description: \"{constraint['description']}\"")
        lines.append(f"    implicit: {constraint['implicit']}")
        return "\n".join(lines)

    def save(self):
        """
        保存约束到文件，保持原始文件格式
        
        - 合并到现有约束定义
        - 创建备份
        - 保持原始格式追加新约束
        """
        if not self.new_constraints:
            print("无需保存，没有新生成的约束")
            return

        # 创建备份
        backup_path = Path(self.constraints_path).with_suffix(".yaml.backup")
        backup_created = False
        try:
            with open(self.constraints_path, "r", encoding="utf-8") as f:
                original_content = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            print(f"✅ 备份已保存: {backup_path}")
            backup_created = True
        except FileNotFoundError:
            print("原文件不存在，跳过备份")
            original_content = ""

        # 如果原文件为空，创建新的结构
        if not original_content.strip():
            content = self._create_new_constraints_file()
        else:
            # 在原文件基础上追加约束
            content = self._append_constraints_to_file(original_content)

        # 写入文件
        try:
            with open(self.constraints_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"✅ 隐式约束已追加到: {self.constraints_path}")
            print(f"  新增 {len(self.new_constraints)} 个约束")
            print(f"  总约束数: {len(self.constraints_data['constraints']) + len(self.new_constraints)}")

            # 成功后删除备份文件
            if backup_created and backup_path.exists():
                backup_path.unlink()
                print(f"✅ 备份文件已删除: {backup_path}")

        except Exception as e:
            print(f"❌ 保存失败，备份文件保留: {backup_path}")
            print(f"   错误: {e}")
            raise

    def _create_new_constraints_file(self) -> str:
        """创建新的约束定义文件"""
        lines = [
            "# 隐式约束定义",
            f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "constraints:",
        ]
        
        for constraint in self.new_constraints:
            lines.append(self._format_constraint_yaml(constraint))
        
        lines.append("")  # 末尾空行
        return "\n".join(lines)

    def _append_constraints_to_file(self, original_content: str) -> str:
        """
        在原文件基础上追加约束，保持原始格式
        
        隐式约束插入到 constraints: 后面的最前面位置
        
        Args:
            original_content: 原始文件内容
            
        Returns:
            追加后的内容
        """
        lines = original_content.split("\n")
        
        # 查找 constraints: 的位置
        constraints_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "constraints:":
                constraints_index = i
                break
        
        if constraints_index == -1:
            # 如果没有 constraints 部分，在文件末尾添加
            result = original_content.rstrip() + "\n\nconstraints:\n"
        else:
            # 构建结果列表
            result_lines = []
            
            # 1. 保留 constraints: 之前的所有内容（包括 constraints: 这一行）
            result_lines.extend(lines[:constraints_index + 1])
            
            # 2. 添加空行分隔
            result_lines.append("")
            
            # 3. 添加注释说明这是隐式约束
            result_lines.append("  # ========================================")
            result_lines.append("  # 隐式约束（自动生成）")
            result_lines.append("  # ========================================")
            result_lines.append("")
            
            # 4. 添加新的隐式约束（放在 constraints: 后的最前面）← 关键修改
            for constraint in self.new_constraints:
                constraint_yaml = self._format_constraint_yaml(constraint)
                result_lines.append(constraint_yaml)
            
            # 5. 添加分隔空行
            result_lines.append("")
            
            # 6. 保留所有原始约束（constraints: 之后的所有内容）
            # 跳过 constraints: 行后面的空白行（如果有）
            i = constraints_index + 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            
            # 添加原始约束
            while i < len(lines):
                result_lines.append(lines[i])
                i += 1
            
            # 7. 确保文件末尾有一个空行
            if result_lines and result_lines[-1].strip() != "":
                result_lines.append("")
            
            result = "\n".join(result_lines)
        
        return result

    # def _append_constraints_to_file(self, original_content: str) -> str:
    #     """
    #     在原文件基础上追加约束，保持原始格式
        
    #     Args:
    #         original_content: 原始文件内容
            
    #     Returns:
    #         追加后的内容
    #     """
    #     lines = original_content.split("\n")
        
    #     # 查找 constraints: 的位置
    #     constraints_index = -1
    #     for i, line in enumerate(lines):
    #         if line.strip() == "constraints:":
    #             constraints_index = i
    #             break
        
    #     if constraints_index == -1:
    #         # 如果没有 constraints 部分，在文件末尾添加
    #         result = original_content.rstrip() + "\n\nconstraints:\n"
    #     else:
    #         # 保留 constraints: 之前的所有内容（包括 constraints: 这一行）
    #         result_lines = lines[:constraints_index + 1]
            
    #         # 保留所有原始约束（constraints: 之后的所有内容）
    #         i = constraints_index + 1
    #         while i < len(lines):
    #             result_lines.append(lines[i])
    #             i += 1
            
    #         # 确保最后一行不是空行，以便添加分隔
    #         while result_lines and result_lines[-1].strip() == "":
    #             result_lines.pop()
            
    #         # 添加空行分隔
    #         result_lines.append("")
            
    #         # 添加注释说明这是隐式约束
    #         result_lines.append("  # ========================================")
    #         result_lines.append("  # 隐式约束（自动生成）")
    #         result_lines.append("  # ========================================")
    #         result_lines.append("")
            
    #         # 添加新约束
    #         for constraint in self.new_constraints:
    #             constraint_yaml = self._format_constraint_yaml(constraint)
    #             result_lines.append(constraint_yaml)
            
    #         # 添加末尾空行
    #         result_lines.append("")
            
    #         result = "\n".join(result_lines)
        
    #     return result


def main():
    parser = argparse.ArgumentParser(
        description="从测试因子定义生成隐式约束并追加到约束定义文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python generate_implicit_constraints.py result/04_测试因子.yaml result/05_约束定义.yaml
    python generate_implicit_constraints.py factors.yaml constraints.yaml --verbose
        """,
    )
    parser.add_argument("factors_file", help="测试因子YAML文件（04_测试因子.yaml）")
    parser.add_argument("constraints_file", help="约束定义YAML文件（05_约束定义.yaml）")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出模式")

    args = parser.parse_args()

    # 验证文件存在性
    factors_path = Path(args.factors_file)
    if not factors_path.exists():
        print(f"错误: 测试因子文件不存在: {factors_path}")
        sys.exit(1)

    constraints_path = Path(args.constraints_file)

    # 生成隐式约束
    try:
        generator = ImplicitConstraintGenerator(str(factors_path), str(constraints_path))
        implicit_constraints = generator.generate()

        if args.verbose and implicit_constraints:
            print("\n生成的隐式约束详情:")
            for constraint in implicit_constraints:
                print(f"  - {constraint['id']}: {constraint['type']}")
                print(f"    {constraint['sources']} -> {constraint['target']}")
                print(f"    {constraint['description']}")

        # 保存到文件
        if implicit_constraints:
            print(f"\n正在保存 {len(implicit_constraints)} 个隐式约束...")
            generator.save()
            print("\n✅ 完成")
        else:
            print("\n无需生成新的隐式约束")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
