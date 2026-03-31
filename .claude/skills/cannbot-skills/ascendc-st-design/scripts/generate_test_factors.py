#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试因子提取脚本

功能：从参数定义.yaml中提取测试因子，输出测试因子.yaml
输入：参数定义.yaml
输出：测试因子.yaml

使用方法：
    python generate_test_factors.py input.yaml output.yaml
    python generate_test_factors.py input.yaml  # 输出到标准输出
"""

import yaml
import sys
import argparse
import re
from typing import Dict, List, Any
from collections import OrderedDict

SPECIAL_STRINGS = {"inf", "-inf", "nan", "+inf"}

# 默认 value_range 定义
DEFAULT_VALUE_RANGES = {
    "float16": [[-65504.0, 65504.0]],
    "float32": [[-3.4028235e38, 3.4028235e38]],
    "float64": [[-1.7976931348623157e308, 1.7976931348623157e308]],
    "bfloat16": [[-3.3895313892515355e38, 3.3895313892515355e38]],
    "int8": [[-128, 127]],
    "uint8": [[0, 255]],
    "int16": [[-32768, 32767]],
    "uint16": [[0, 65535]],
    "int32": [[-2147483648, 2147483647]],
    "uint32": [[0, 4294967295]],
    "int64": [[-9223372036854775808, 9223372036854775807]],
    "uint64": [[0, 18446744073709551615]],
    "bool": [[0, 1]],
}


def get_default_value_range(dtype: str) -> List[List]:
    """
    获取 dtype 的默认 value_range
    
    Args:
        dtype: 数据类型字符串
    
    Returns:
        默认 value_range 列表
    """
    normalized = normalize_dtype(dtype) if 'normalize_dtype' in dir() else dtype.lower()
    return DEFAULT_VALUE_RANGES.get(normalized, DEFAULT_VALUE_RANGES.get(dtype, [[0, 100]]))


def normalize_dtype(dtype_str: str) -> str:
    """
    标准化 dtype 字符串
    
    Args:
        dtype_str: dtype 字符串
    
    Returns:
        标准化后的 dtype
    """
    dtype_map = {
        "float16": "float16", "fp16": "float16", "half": "float16",
        "float32": "float32", "float": "float32", "fp32": "float32",
        "float64": "float64", "double": "float64", "fp64": "float64",
        "bfloat16": "bfloat16", "bf16": "bfloat16",
        "int8": "int8", "s8": "int8",
        "uint8": "uint8", "u8": "uint8",
        "int16": "int16", "s16": "int16",
        "uint16": "uint16", "u16": "uint16",
        "int32": "int32", "s32": "int32", "int": "int32",
        "uint32": "uint32", "u32": "uint32",
        "int64": "int64", "s64": "int64", "long": "int64",
        "uint64": "uint64", "u64": "uint64",
        "bool": "bool", "boolean": "bool",
    }
    return dtype_map.get(dtype_str.lower(), dtype_str.lower())


def convert_special_value(value):
    """
    转换特殊值：只有 inf/-inf/nan 保持字符串，其余转为数值

    Args:
        value: 输入值

    Returns:
        转换后的值
    """
    if isinstance(value, str):
        if value.lower() in SPECIAL_STRINGS or value in SPECIAL_STRINGS:
            return value
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    return value


def convert_range_list(range_list):
    """
    转换范围列表中的值

    Args:
        range_list: 范围列表 [[min, max], ...]

    Returns:
        转换后的范围列表
    """
    result = []
    for item in range_list:
        if isinstance(item, list):
            result.append([convert_special_value(v) for v in item])
        else:
            result.append(convert_special_value(item))
    return result


def represent_str(dumper, data):
    """自定义字符串表示器，特殊字符串保持引号"""
    if data in SPECIAL_STRINGS:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def represent_list(dumper, data):
    """自定义列表表示器，处理嵌套列表"""
    if not data:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    if all(isinstance(x, (int, float, str)) for x in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


def setup_yaml_ordered_dict():
    """配置YAML使用OrderedDict保持顺序"""

    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)
    yaml.add_representer(str, represent_str)


def load_yaml(filepath: str) -> Dict:
    """加载YAML文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, filepath: str = None):
    """保存YAML数据到文件或标准输出"""
    setup_yaml_ordered_dict()

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                data, f, allow_unicode=True, default_flow_style=False, sort_keys=False
            )
        print(f"✅ 测试因子已保存到: {filepath}")
    else:
        print(
            yaml.dump(
                data, allow_unicode=True, default_flow_style=False, sort_keys=False
            )
        )


def extract_tensor_factors(param: Dict) -> Dict[str, List]:
    """
    提取Tensor类型参数的测试因子

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    factors = OrderedDict()
    name = param["name"]
    io_type = param.get("io_type", "input")

    # 1. 存在性因子
    factors[f"{name}.exist"] = [True]
    if not param.get("required", True):
        factors[f"{name}.exist"] = [True, False]

    # 2. 数据格式因子
    if "format" in param:
        format_value = param["format"]
        if isinstance(format_value, list):
            factors[f"{name}.format"] = format_value
        else:
            factors[f"{name}.format"] = [format_value]

    # 3. 维度因子
    if "dimensions" in param:
        dims = param["dimensions"]
        if isinstance(dims, list):
            factors[f"{name}.dimensions"] = dims
        else:
            factors[f"{name}.dimensions"] = [dims]

    # 4. 数据类型因子
    if "dtype_with_ranges" in param:
        dtypes = [item["dtype"] for item in param["dtype_with_ranges"]]
        factors[f"{name}.dtype"] = dtypes

        # 5. 取值范围因子（仅输入参数需要定义，输出参数不需要）
        if io_type == "input":
            for dtype_item in param["dtype_with_ranges"]:
                dtype = dtype_item["dtype"]

                # 获取 value_range，如果未定义则使用默认值
                if "value_range" in dtype_item:
                    value_range = dtype_item["value_range"]
                else:
                    value_range = get_default_value_range(dtype)

                # 转换并存储
                converted_range = convert_range_list(value_range)
                factors[f"{name}.value_range_{normalize_dtype(dtype)}"] = converted_range

    return factors


def extract_scalar_factors(param: Dict) -> Dict[str, List]:
    """
    提取Scalar类型参数的测试因子

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    factors = OrderedDict()
    name = param["name"]
    is_enum = param.get("is_enum", False)

    # 1. 存在性因子
    factors[f"{name}.exist"] = [True]
    if not param.get("required", True):
        factors[f"{name}.exist"] = [True, False]

    # 2. 数据类型因子
    if "dtype_with_values" in param:
        dtypes = [item["dtype"] for item in param["dtype_with_values"]]
        factors[f"{name}.dtype"] = dtypes

        # 3. 枚举值因子（如果是枚举类型）
        if is_enum:
            for dtype_item in param["dtype_with_values"]:
                if "value" in dtype_item:  # 从 special_value 改为 value
                    factors[f"{name}.value"] = dtype_item["value"]
                    break  # 枚举值通常在第一个dtype中定义
        else:
            # 4. 取值范围因子（非枚举类型）
            for dtype_item in param["dtype_with_values"]:
                dtype = dtype_item["dtype"]

                # 获取 value_range，如果未定义则使用默认值
                if "value_range" in dtype_item:
                    value_range = dtype_item["value_range"]
                else:
                    value_range = get_default_value_range(dtype)

                # 转换并存储（统一使用 value_range）
                converted_range = convert_range_list(value_range)
                factors[f"{name}.value_range_{normalize_dtype(dtype)}"] = converted_range

    return factors

def extract_int_factors(param: Dict) -> Dict[str, List]:
    """
    提取整型参数的测试因子

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    factors = OrderedDict()
    name = param["name"]
    is_enum = param.get("is_enum", False)

    # 1. 存在性因子
    factors[f"{name}.exist"] = [True]
    if not param.get("required", True):
        factors[f"{name}.exist"] = [True, False]

    # 2. 数据类型因子
    if "dtype_with_values" in param:
        dtypes = [item["dtype"] for item in param["dtype_with_values"]]
        factors[f"{name}.dtype"] = dtypes

        # 3. 枚举值因子（如果是枚举类型）
        if is_enum:
            for dtype_item in param["dtype_with_values"]:
                # 修复：检查 "value" 字段而不是 "special_value"
                if "value" in dtype_item:
                    converted_values = [
                        convert_special_value(v) for v in dtype_item["value"]
                    ]
                    # 修复：因子名称应该是 {name}.value 而不是 {name}.enum_values
                    factors[f"{name}.value"] = converted_values
                    break
        else:
            # 4. 取值范围因子（非枚举类型）
            for dtype_item in param["dtype_with_values"]:
                dtype = dtype_item["dtype"]

                # 获取 value_range，如果未定义则使用默认值
                if "value_range" in dtype_item:
                    value_range = dtype_item["value_range"]
                else:
                    value_range = get_default_value_range(dtype)

                # 转换并存储
                converted_range = convert_range_list(value_range)
                factors[f"{name}.value_range_{dtype}"] = converted_range

    return factors


def extract_factors_from_param(param: Dict) -> Dict[str, List]:
    """
    根据参数类型提取测试因子

    Args:
        param: 参数定义字典

    Returns:
        测试因子字典
    """
    param_type = param.get("type", "")

    if param_type == "aclTensor":
        return extract_tensor_factors(param)
    elif param_type == "aclScalar":
        return extract_scalar_factors(param)
    elif param_type in [
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
    ]:
        return extract_int_factors(param)
    else:
        print(f"⚠️  未知的参数类型: {param_type}")
        return OrderedDict()


def extract_all_factors(params: List[Dict]) -> Dict:
    """
    从所有参数中提取测试因子

    Args:
        params: 参数定义列表

    Returns:
        完整的测试因子字典（仅包含 test_factors）
    """
    factors_section = OrderedDict()

    for param in params:
        param_name = param["name"]
        param_factors = extract_factors_from_param(param)

        if param_factors:
            factors_section[param_name] = OrderedDict(
                [("type", param.get("type", "")), ("factors", param_factors)]
            )

    return factors_section


def generate_factor_summary(factors: Dict) -> Dict:
    """
    生成测试因子摘要

    Args:
        factors: 测试因子字典

    Returns:
        摘要信息
    """
    total_params = len(factors)
    total_factors = sum(len(param_data["factors"]) for param_data in factors.values())

    summary = OrderedDict(
        [
            ("total_parameters", total_params),
            ("total_factors", total_factors),
            ("by_type", OrderedDict()),
            ("by_category", OrderedDict()),
        ]
    )

    # 按参数类型统计
    for param_name, param_data in factors.items():
        param_type = param_data["type"]
        if param_type not in summary["by_type"]:
            summary["by_type"][param_type] = {"count": 0, "factors": 0}
        summary["by_type"][param_type]["count"] += 1
        summary["by_type"][param_type]["factors"] += len(param_data["factors"])

    # 按因子类别统计
    category_count = OrderedDict()
    for param_name, param_data in factors.items():
        for factor_name in param_data["factors"].keys():
            # 提取因子类别（如exist, format, dtype等）
            # 因子名格式: {param}.{attribute} 或 {param}.{attribute}_{dtype}
            if "." in factor_name:
                attribute_part = factor_name.split(".", 1)[1]
                # 提取主属性（去掉dtype后缀）
                if "_" in attribute_part:
                    category = attribute_part.rsplit("_", 1)[0]
                else:
                    category = attribute_part
                category_count[category] = category_count.get(category, 0) + 1

    summary["by_category"] = category_count

    return summary


def print_factor_summary(factors: Dict):
    """打印测试因子摘要"""
    summary = generate_factor_summary(factors)

    print("\n" + "=" * 60)
    print("📊 测试因子提取摘要")
    print("=" * 60)
    print(f"参数总数: {summary['total_parameters']}")
    print(f"因子总数: {summary['total_factors']}")
    print()

    print("按参数类型统计:")
    for ptype, stats in summary["by_type"].items():
        print(f"  {ptype}: {stats['count']}个参数, {stats['factors']}个因子")
    print()

    print("按因子类别统计:")
    for category, count in summary["by_category"].items():
        print(f"  {category}: {count}个")
    print("=" * 60 + "\n")


def validate_yaml_structure(data: Dict) -> bool:
    """验证YAML结构是否正确"""
    if "parameters" not in data:
        print("❌ 错误: YAML文件缺少 'parameters' 字段")
        return False

    params = data["parameters"]
    if not isinstance(params, list):
        print("❌ 错误: 'parameters' 必须是列表")
        return False

    for i, param in enumerate(params):
        if "name" not in param:
            print(f"❌ 错误: 参数{i + 1}缺少 'name' 字段")
            return False
        if "type" not in param:
            print(
                f"❌ 错误: 参数{i + 1} ({param.get('name', 'unknown')}) 缺少 'type' 字段"
            )
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="从参数定义.yaml中提取测试因子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取测试因子并保存到文件
  python generate_test_factors.py params.yaml factors.yaml
  
  # 提取测试因子并输出到标准输出
  python generate_test_factors.py params.yaml
  
  # 查看帮助
  python generate_test_factors.py --help
        """,
    )

    parser.add_argument("input", help="输入的参数定义YAML文件")
    parser.add_argument(
        "output", nargs="?", help="输出的测试因子YAML文件（可选，默认输出到标准输出）"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="静默模式，不打印摘要"
    )

    args = parser.parse_args()

    try:
        # 加载参数定义
        print(f"📖 正在读取: {args.input}")
        data = load_yaml(args.input)

        # 验证结构
        if not validate_yaml_structure(data):
            sys.exit(1)

        # 提取测试因子
        print("🔍 正在提取测试因子...")
        factors = extract_all_factors(data["parameters"])

        # 打印摘要
        if not args.quiet:
            print_factor_summary(factors)

        # 保存结果
        save_yaml(factors, args.output)

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{args.input}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ YAML解析错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
