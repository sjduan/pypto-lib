#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用例生成脚本
支持 L0（单因子覆盖）和 L1（两两组合覆盖）
"""

import argparse
import sys
import re
import pandas as pd
import yaml
import random
import json
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Any, Optional
from ast import literal_eval


def main():
    """主函数"""
    # 1. 解析参数
    args = parse_arguments()
    
    # 2. 解析级别
    levels = parse_levels(args.level)
    
    # 3. 参数校验（冲突检测）
    validate_parameters(args, levels)
    
    # 4. 加载数据（只加载一次，批量生成时共享）
    param_def, factors, df = load_data(args)
    
    # 5. 提取因子值（只提取一次，批量生成时共享）
    all_factor_values = extract_all_factor_values(factors, df)
    
    if args.verbose:
        print(f"[INFO] 将生成级别: {', '.join(levels)}\n")
    
    # 6. 按级别生成用例
    for level in levels:
        if args.verbose:
            print(f"[INFO] {'='*10} 开始生成 {level} 用例 {'='*10}")
        
        # 根据级别选择算法
        if level == 'L0':
            selected_indices, coverage_info = select_cases_L0(df, all_factor_values, args.verbose)
            report = generate_L0_report(all_factor_values, coverage_info)
        else:  # L1
            # 生成两两组合
            pairwise_combinations = generate_pairwise_combinations(all_factor_values)
            
            if args.verbose:
                print(f"[INFO] 两两组合数: {len(pairwise_combinations)}")
            
            # 筛选用例
            selected_indices, coverage_info = select_cases_L1(
                df, pairwise_combinations, all_factor_values, args.verbose, args.sample_size
            )
            
            # 补齐（如果需要）
            if len(selected_indices) < args.target_count:
                if args.verbose:
                    print(f"[INFO] 补齐用例: {len(selected_indices)} -> {args.target_count}")
                selected_indices = pad_cases(
                    selected_indices, args.target_count, args.seed
                )
            
            report = generate_L1_report(
                all_factor_values, pairwise_combinations, coverage_info, args.target_count
            )
        
        # 7. 确定输出文件名
        report_file, case_file = get_output_filenames(
            args.report_output, args.case_output, level, len(levels) > 1
        )
        
        # 8. 转换格式
        aclnn_name = extract_aclnn_name(args)
        selected_df = df.iloc[selected_indices].reset_index(drop=True)
        case_df = convert_to_standard_format(selected_df, param_def, aclnn_name, level)
        
        # 9. 保存结果
        # L0: 保存报告和用例; L1: 只保存用例
        if level == 'L0':
            save_results(report, case_df, args.output_dir, report_file, case_file, args.verbose)
        else:  # L1
            # 只保存用例CSV，不生成报告
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            case_path = output_dir / case_file
            case_df.to_csv(case_path, index=False)
            if args.verbose:
                print(f"[INFO] 生成用例: {case_path} ({len(case_df)}条)")
        
        if args.verbose:
            print(f"[INFO] {'='*10} {level} 用例生成完成 {'='*10}\n")
    
    if args.verbose:
        print(f"[INFO] 完成! 共生成 {len(levels)} 个级别的用例")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='测试用例生成脚本（支持L0和L1）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成L0用例
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L0
  
  # 生成L1用例（500条）
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L1 --target-count 500
  
  # 批量生成L0和L1
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L0 L1
  
  # 自定义输出文件名
  python generate_test_cases.py param.yaml factors.yaml values.csv output/ --level L0 --report-output my_report.yaml
        """
    )
    
    parser.add_argument('param_def', help='参数定义YAML文件（03_参数定义.yaml）')
    parser.add_argument('factors', help='测试因子YAML文件（04_测试因子.yaml）')
    parser.add_argument('values', help='因子值CSV文件（07_因子值.csv）')
    parser.add_argument('output_dir', help='输出目录')
    
    parser.add_argument('--level', nargs='+', required=True,
                       help='用例级别（必填，支持多个）: L0=单因子覆盖（≤200条），L1=两两组合覆盖（500~700条）')
    parser.add_argument('--aclnn-name', help='算子名称（默认从文件路径中提取）')
    parser.add_argument('--target-count', type=int, default=500,
                       help='L1目标用例数量（默认500，仅L1有效）')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='L1每次迭代采样的候选用例数量（默认1000，仅L1有效，用于加速大数据集）')
    parser.add_argument('--seed', type=int, help='随机数种子（用于复现L1补齐，仅L1有效）')
    parser.add_argument('--report-output', help='覆盖度报告文件名（默认: {level}_coverage_report.yaml）')
    parser.add_argument('--case-output', help='测试用例文件名（默认: {level}_test_cases.csv）')
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    
    return parser.parse_args()


def parse_levels(level_arg):
    """解析 --level 参数"""
    levels = []
    
    for item in level_arg:
        # 支持逗号分隔
        if ',' in item:
            levels.extend([l.strip() for l in item.split(',')])
        else:
            levels.append(item.strip())
    
    # 去重并排序（L0在前）
    levels = sorted(set(levels), key=lambda x: (x != 'L0', x))
    
    # 验证
    valid_levels = {'L0', 'L1'}
    invalid = set(levels) - valid_levels
    if invalid:
        print(f"[ERROR] 无效的级别: {invalid}，仅支持 L0 和 L1")
        sys.exit(1)
    
    return levels


def validate_parameters(args, levels):
    """验证参数冲突"""
    errors = []
    
    # 只生成L0时，检查L1专用参数
    if 'L0' in levels and len(levels) == 1:
        if args.target_count != 500:  # 非默认值
            errors.append({
                'param': '--target-count',
                'value': args.target_count,
                'reason': 'L0 不支持 --target-count 参数',
                'detail': 'L0 的用例数量由算法自动确定（覆盖所有因子值的最小集合）',
                'solution': [
                    '移除 --target-count 参数',
                    '或使用 --level L1 生成 L1 用例'
                ]
            })
        
        if args.seed is not None:
            errors.append({
                'param': '--seed',
                'value': args.seed,
                'reason': 'L0 不支持 --seed 参数',
                'detail': 'L0 不需要补齐，因此无需随机数种子',
                'solution': [
                    '移除 --seed 参数',
                    '或使用 --level L1 生成 L1 用例'
                ]
            })
    
    if errors:
        print("\n" + "="*80)
        print("参数冲突错误")
        print("="*80 + "\n")
        
        for error in errors:
            print(f"[ERROR] {error['param']}={error['value']} 对 L0 无效")
            print(f"        {error['reason']}")
            print(f"        {error['detail']}")
            print(f"        解决方法：")
            for i, sol in enumerate(error['solution'], 1):
                print(f"        {i}. {sol}")
            print()
        
        print("="*80)
        print("脚本已退出（错误码：1）")
        print("="*80 + "\n")
        sys.exit(1)


def load_data(args):
    """加载数据文件"""
    param_def = load_yaml(args.param_def)
    factors = load_yaml(args.factors)
    df = pd.read_csv(args.values)
    
    if args.verbose:
        print(f"[INFO] 加载参数定义: {args.param_def}")
        print(f"[INFO] 加载测试因子: {args.factors}")
        print(f"[INFO] 加载因子值: {args.values} ({len(df)}个用例)\n")
    
    return param_def, factors, df


# ============ L0 相关函数 ============

def select_cases_L0(df, all_factor_values, verbose=False):
    """
    L0: 使用贪心算法选择覆盖所有单个因子值的最小用例集
    
    Args:
        df: 因子值 DataFrame
        all_factor_values: 所有因子及其值
        verbose: 是否输出详细信息
    
    Returns:
        Tuple[List[int], Dict]: (选中的用例索引列表, 覆盖度信息)
    """
    if verbose:
        print(f"[INFO] 提取因子: {len(all_factor_values)}个因子, {sum(len(v) for v in all_factor_values.values())}个因子值")
    
    selected_indices = []
    uncovered = deepcopy(all_factor_values)
    covered = {k: set() for k in all_factor_values.keys()}
    
    iteration = 0
    while any(uncovered.values()):
        iteration += 1
        best_idx = None
        best_new_coverage = 0
        best_covered = {}
        
        for idx, row in df.iterrows():
            if idx in selected_indices:
                continue
            
            new_coverage = 0
            current_covered = {}
            
            for factor_name, required_values in uncovered.items():
                if factor_name not in row.index:
                    continue
                
                value = row[factor_name]
                value_key = make_hashable(value)
                
                if value_key in required_values:
                    new_coverage += 1
                    current_covered[factor_name] = value_key
            
            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_idx = idx
                best_covered = current_covered
        
        if best_idx is None or best_new_coverage == 0:
            break
        
        selected_indices.append(best_idx)
        
        for factor_name, value in best_covered.items():
            covered[factor_name].add(value)
            uncovered[factor_name].discard(value)
        
        if verbose and iteration % 5 == 0:
            total_uncovered = sum(len(v) for v in uncovered.values())
            print(f"[INFO] L0迭代{iteration}: 选择用例{best_idx}, 新覆盖{best_new_coverage}个因子值, 剩余{total_uncovered}")
    
    if verbose:
        total_values = sum(len(v) for v in all_factor_values.values())
        covered_count = sum(len(v) for v in covered.values())
        print(f"[INFO] 完成用例选择: {len(selected_indices)}个用例")
        print(f"[INFO] 因子值覆盖: {covered_count}/{total_values} ({covered_count/total_values*100:.2f}%)")
    
    coverage_info = {
        'covered_values': covered,
        'uncovered_values': uncovered,
        'total_factors': len(all_factor_values),
        'total_values': sum(len(v) for v in all_factor_values.values()),
        'covered_count': sum(len(v) for v in covered.values())
    }
    
    return selected_indices, coverage_info


def generate_L0_report(all_factor_values, coverage_info):
    """生成L0覆盖度报告"""
    total_values = sum(len(v) for v in all_factor_values.values())
    covered_count = coverage_info['covered_count']
    uncovered_count = total_values - covered_count
    
    report = {
        'summary': {
            'level': 'L0',
            'strategy': 'single_factor_coverage',
            'total_factors': coverage_info['total_factors'],
            'total_factor_values': total_values,
            'covered_factor_values': covered_count,
            'uncovered_factor_values': uncovered_count,
            'coverage_rate': f"{covered_count/total_values*100:.2f}%" if total_values > 0 else "N/A",
            'minimal_case_count': coverage_info['covered_count']
        },
        'details': {}
    }
    
    for factor_name, required_values in all_factor_values.items():
        covered = coverage_info['covered_values'].get(factor_name, set())
        uncovered = coverage_info['uncovered_values'].get(factor_name, set())
        
        report['details'][factor_name] = {
            'target_values': sorted([str(v) for v in required_values]),
            'covered_values': sorted([str(v) for v in covered]),
            'uncovered_values': sorted([str(v) for v in uncovered]),
            'coverage_rate': f"{len(covered)/len(required_values)*100:.2f}%" if required_values else "N/A"
        }
    
    return report


# ============ L1 相关函数 ============

def generate_pairwise_combinations(all_factor_values):
    """
    生成所有因子的两两组合（优化版：只考虑离散因子）
    
    Args:
        all_factor_values: Dict[str, List[Any]] - 因子名 -> 因子值列表
    
    Returns:
        Set[Tuple]: 所有两两组合集合
    
    优化说明：
        1. 只保留离散因子（dtype, dimensions, exist, format, 枚举值等）
        2. 过滤派生因子（.value, .shape）和连续值因子（.value_range）
        3. 过滤固定值因子（只有1个值的因子）
        
    性能提升：
        - 原始：192,810个组合（包含value_range）
        - 优化后：~500个组合（只有离散因子）
        - 性能提升：~400倍
    """
    # 定义离散因子模式
    discrete_patterns = [
        '.dtype',        # 数据类型（离散）
        '.dimensions',   # 维度数（离散）
        '.exist',        # 存在性（离散）
        '.format',       # 数据格式（离散）
        'cubeMathType.value',  # 枚举值（离散）
    ]
    
    def is_discrete_factor(factor_name):
        """判断是否为离散因子"""
        # 排除派生因子和连续值因子
        if factor_name.endswith('.value') and not factor_name == 'cubeMathType.value':
            return False
        if factor_name.endswith('.shape'):
            return False
        if factor_name.endswith('.value_range'):
            return False
        
        # 保留离散因子
        for pattern in discrete_patterns:
            if pattern in factor_name:
                return True
        
        return False
    
    # 过滤离散因子
    discrete_factors = {
        name: values 
        for name, values in all_factor_values.items()
        if is_discrete_factor(name)
    }
    
    # 过滤只有1个值的因子
    multi_value_factors = {
        name: values 
        for name, values in discrete_factors.items() 
        if len(values) > 1
    }
    
    fixed_value_factors = {
        name: values 
        for name, values in discrete_factors.items() 
        if len(values) == 1
    }
    
    # 统计过滤的因子
    filtered_continuous = {
        name: values 
        for name, values in all_factor_values.items()
        if name.endswith('.value_range') or (name.endswith('.value') and name != 'cubeMathType.value') or name.endswith('.shape')
    }
    
    if fixed_value_factors:
        print(f"[INFO] 过滤固定值因子: {len(fixed_value_factors)}个")
    if filtered_continuous:
        print(f"[INFO] 过滤连续值/派生因子: {len(filtered_continuous)}个")
    print(f"[INFO] 保留离散多值因子: {len(multi_value_factors)}个")
    
    factor_names = sorted(multi_value_factors.keys())
    pairwise_combinations = set()
    
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            factor1_name = factor_names[i]
            factor2_name = factor_names[j]
            
            values1 = multi_value_factors[factor1_name]
            values2 = multi_value_factors[factor2_name]
            
            for v1 in values1:
                for v2 in values2:
                    v1_key = make_hashable(v1)
                    v2_key = make_hashable(v2)
                    
                    pair = (
                        (factor1_name, v1_key),
                        (factor2_name, v2_key)
                    )
                    pairwise_combinations.add(pair)
    
    return pairwise_combinations


def select_cases_L1(df, pairwise_combinations, all_factor_values, verbose=False, sample_size=None):
    """
    L1: 使用贪心算法选择覆盖所有两两组合的用例集（优化版：只考虑离散因子）
    
    Args:
        df: 因子值 DataFrame
        pairwise_combinations: 所有两两组合
        all_factor_values: 所有因子及其值
        verbose: 是否输出详细信息
        sample_size: 每次迭代采样的候选用例数量（None表示全部考虑）
    
    Returns:
        Tuple[List[int], Dict]: (选中的用例索引列表, 覆盖度信息)
    
    优化说明：
        1. 只考虑离散因子（dtype, dimensions, exist, format, 枚举值等）
        2. 过滤连续值因子（value_range）和派生因子（value, shape）
    """
    # 定义离散因子模式
    discrete_patterns = [
        '.dtype',        # 数据类型（离散）
        '.dimensions',   # 维度数（离散）
        '.exist',        # 存在性（离散）
        '.format',       # 数据格式（离散）
        'cubeMathType.value',  # 枚举值（离散）
    ]
    
    def is_discrete_factor(factor_name):
        """判断是否为离散因子"""
        # 排除派生因子和连续值因子
        if factor_name.endswith('.value') and not factor_name == 'cubeMathType.value':
            return False
        if factor_name.endswith('.shape'):
            return False
        if factor_name.endswith('.value_range'):
            return False
        
        # 保留离散因子
        for pattern in discrete_patterns:
            if pattern in factor_name:
                return True
        
        return False
    
    # 过滤离散因子
    discrete_factors = {
        name: values 
        for name, values in all_factor_values.items()
        if is_discrete_factor(name)
    }
    
    # 过滤只有1个值的因子
    multi_value_factors = {
        name: values 
        for name, values in discrete_factors.items() 
        if len(values) > 1
    }
    
    if verbose:
        print(f"[INFO] 提取离散因子: {len(multi_value_factors)}个（已过滤连续值和派生因子）")
        print(f"[INFO] 因子值总数: {sum(len(v) for v in multi_value_factors.values())}")
    
    selected_indices = []
    uncovered = deepcopy(pairwise_combinations)
    covered = set()
    
    factor_names = [col for col in df.columns if col in multi_value_factors]
    all_indices = set(df.index.tolist())
    
    iteration = 0
    while uncovered:
        iteration += 1
        best_idx = None
        best_new_coverage = 0
        best_covered_pairs = set()
        
        # 采样候选用例以加速
        candidate_indices = list(all_indices - set(selected_indices))
        if sample_size and len(candidate_indices) > sample_size:
            candidate_indices = random.sample(candidate_indices, sample_size)
        
        for idx in candidate_indices:
            row = df.loc[idx]
            
            covered_pairs = set()
            
            for i in range(len(factor_names)):
                for j in range(i + 1, len(factor_names)):
                    factor1_name = factor_names[i]
                    factor2_name = factor_names[j]
                    
                    v1 = row[factor1_name]
                    v2 = row[factor2_name]
                    
                    v1_key = make_hashable(v1)
                    v2_key = make_hashable(v2)
                    
                    pair = (
                        (factor1_name, v1_key),
                        (factor2_name, v2_key)
                    )
                    
                    if pair in uncovered:
                        covered_pairs.add(pair)
            
            if len(covered_pairs) > best_new_coverage:
                best_new_coverage = len(covered_pairs)
                best_idx = idx
                best_covered_pairs = covered_pairs
        
        if best_idx is None or best_new_coverage == 0:
            break
        
        selected_indices.append(best_idx)
        covered.update(best_covered_pairs)
        uncovered -= best_covered_pairs
        
        if verbose and iteration % 10 == 0:
            print(f"[INFO] L1迭代{iteration}: 选择用例{best_idx}, 新覆盖{best_new_coverage}个两两组合, 剩余{len(uncovered)}")
    
    if verbose:
        print(f"[INFO] 完成用例选择: {len(selected_indices)}个用例")
        print(f"[INFO] 两两组合覆盖: {len(covered)}/{len(pairwise_combinations)} ({len(covered)/len(pairwise_combinations)*100:.2f}%)")
    
    coverage_info = {
        'total_pairwise': len(pairwise_combinations),
        'covered_pairwise': len(covered),
        'uncovered_pairwise': len(uncovered),
        'coverage_rate': len(covered) / len(pairwise_combinations) * 100 if pairwise_combinations else 0,
        'selected_count': len(selected_indices)
    }
    
    return selected_indices, coverage_info


def pad_cases(selected_indices, target_count, seed=None):
    """
    补齐用例到目标数量（L1专用）
    
    Args:
        selected_indices: 已选中的用例索引列表
        target_count: 目标用例数量
        seed: 随机数种子
    
    Returns:
        List[int]: 补齐后的用例索引列表
    """
    if len(selected_indices) >= target_count:
        return selected_indices[:target_count]
    
    if seed is not None:
        random.seed(seed)
    
    need_pad = target_count - len(selected_indices)
    padded_indices = selected_indices.copy()
    
    for _ in range(need_pad):
        random_idx = random.choice(selected_indices)
        padded_indices.append(random_idx)
    
    return padded_indices


def generate_L1_report(all_factor_values, pairwise_combinations, coverage_info, target_count):
    """生成L1覆盖度报告"""
    selected_count = coverage_info['selected_count']
    padded_count = max(0, target_count - selected_count)
    
    report = {
        'summary': {
            'level': 'L1',
            'strategy': 'pairwise_coverage',
            'total_factors': len(all_factor_values),
            'total_factor_values': sum(len(v) for v in all_factor_values.values()),
            'total_pairwise_combinations': coverage_info['total_pairwise'],
            'covered_pairwise_combinations': coverage_info['covered_pairwise'],
            'uncovered_pairwise_combinations': coverage_info['uncovered_pairwise'],
            'coverage_rate': f"{coverage_info['coverage_rate']:.2f}%",
            'selected_case_count': selected_count,
            'target_case_count': target_count,
            'padded_case_count': padded_count
        },
        'factor_statistics': {}
    }
    
    for factor_name, values in all_factor_values.items():
        report['factor_statistics'][factor_name] = {
            'value_count': len(values),
            'values': sorted([str(v) for v in values])
        }
    
    return report


# ============ 共享工具函数 ============

def load_yaml(file_path):
    """加载YAML文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_all_factor_values(factors_dict, df=None):
    """
    提取所有需要覆盖的因子值
    
    策略：
    1. 从 CSV DataFrame 中提取每个因子的实际值（如果提供了 df）
    2. 过滤 .shape 因子（由 dimensions 动态生成）
    3. 处理 value_range_{dtype} 模式：
       - YAML 中定义的是按 dtype 分组的可选值（如 value_range_float16）
       - CSV 中的列名是统一的 value_range
       - 应该从 CSV 中提取 value_range 列的实际值，而不是合并 YAML 中的所有可能值
    
    Args:
        factors_dict: YAML 中的因子定义
        df: CSV DataFrame（如果提供，从中提取实际值）
    
    Returns:
        Dict[str, Set]: 因子名 -> 值集合
    """
    all_factor_values = {}
    value_range_dtype_pattern = re.compile(r'^(.+)\.value_range_([a-z0-9]+)$')
    
    # 收集所有因子名（过滤 shape，处理 value_range_{dtype}）
    factor_names_from_yaml = set()
    for param_name, param_info in factors_dict.items():
        factors = param_info.get('factors', {})
        
        for factor_name in factors.keys():
            # 过滤 shape 因子
            if '.shape' in factor_name:
                continue
            
            # 检测 value_range_{dtype} 模式
            match = value_range_dtype_pattern.match(factor_name)
            if match:
                # 转换为统一的 value_range 名称
                base_factor_name = f"{match.group(1)}.value_range"
                factor_names_from_yaml.add(base_factor_name)
            else:
                factor_names_from_yaml.add(factor_name)
    
    # 如果提供了 DataFrame，从中提取实际值
    if df is not None:
        for factor_name in factor_names_from_yaml:
            if factor_name in df.columns:
                # 从 CSV 列中提取唯一值
                unique_values = df[factor_name].unique()
                all_factor_values[factor_name] = set(
                    make_hashable(v) for v in unique_values if not pd.isna(v)
                )
    else:
        # 从 YAML 定义中提取（兼容旧行为）
        for param_name, param_info in factors_dict.items():
            factors = param_info.get('factors', {})
            
            for factor_name, factor_values in factors.items():
                if '.shape' in factor_name:
                    continue
                
                match = value_range_dtype_pattern.match(factor_name)
                if match:
                    base_factor_name = f"{match.group(1)}.value_range"
                    if base_factor_name not in all_factor_values:
                        all_factor_values[base_factor_name] = set()
                    all_factor_values[base_factor_name].update(
                        make_hashable(v) for v in factor_values
                    )
                else:
                    all_factor_values[factor_name] = set(
                        make_hashable(v) for v in factor_values
                    )
    
    return all_factor_values


def make_hashable(value):
    """将值转换为可哈希的类型"""
    if isinstance(value, list):
        return tuple(value)
    elif isinstance(value, dict):
        return tuple(sorted(value.items()))
    else:
        return value


def extract_aclnn_name(args):
    """从参数定义中提取算子名称"""
    # 1. 优先使用命令行指定的名称
    if args.aclnn_name:
        return args.aclnn_name
    
    # 2. 从参数定义中获取
    param_def_path = Path(args.param_def)
    try:
        with open(param_def_path, 'r', encoding='utf-8') as f:
            param_def = yaml.safe_load(f)
        if param_def.get('aclnn_name'):
            return param_def.get('aclnn_name')
    except Exception:
        pass
    
    # 3. 从文件路径中提取（查找包含 aclnn 的目录名）
    for part in param_def_path.parts:
        if part.startswith('aclnn'):
            return part
    
    return 'UnknownOperator'


def get_output_filenames(report_output, case_output, level, is_batch):
    """
    确定输出文件名
    
    Args:
        report_output: 用户指定的报告文件名（可能为None）
        case_output: 用户指定的用例文件名（可能为None）
        level: 当前级别 (L0/L1)
        is_batch: 是否批量生成
    
    Returns:
        Tuple[str, str]: (报告文件名, 用例文件名)
    """
    # 报告文件名
    if report_output:
        report_name = report_output
        if is_batch:
            report_name = f"{level}_{report_name}"
    else:
        if is_batch:
            report_name = f"{level}_coverage_report.yaml"
        else:
            report_name = f"{level.lower()}_coverage_report.yaml"
    
    # 用例文件名
    if case_output:
        case_name = case_output
        if is_batch:
            case_name = f"{level}_{case_name}"
    else:
        if is_batch:
            case_name = f"{level}_test_cases.csv"
        else:
            case_name = f"{level.lower()}_test_cases.csv"
    
    return report_name, case_name


def convert_to_standard_format(df, param_def, aclnn_name, case_level):
    """
    转换为标准用例格式
    
    Args:
        df: 因子值 DataFrame
        param_def: 参数定义
        aclnn_name: 算子名称
        case_level: 用例级别（L0/L1）
    
    Returns:
        DataFrame: 标准格式的用例表
    """
    cases = []
    
    # 转换参数定义格式：列表 -> 字典（以 name 为 key）
    if isinstance(param_def.get('parameters'), list):
        params_dict = {}
        for param in param_def['parameters']:
            param_name = param.get('name')
            if param_name:
                params_dict[param_name] = param
        param_def = params_dict
    
    # 获取参数顺序（用于确定input_tensor_index）
    param_names = list(param_def.keys())
    
    for idx, row in df.iterrows():
        case = {
            'aclnn_name': aclnn_name,
            'case_name': f"OP-{aclnn_name}-{case_level}-{idx+1:03d}",
            'bin_dir': '',
            'genetic': '',
            'precision_mode': '1',
            'precision_tolerance': '((0.001,0.001,999999,0.001,0),)',
            'red_range': ''
        }
        
        # 处理输入Tensor
        input_tensors = []
        input_indices = []
        
        for param_name, param_info in param_def.items():
            if param_info.get('io_type') == 'input':
                param_type = param_info.get('type', '')
                
                if param_type == 'aclTensor':
                    exist_col = f"{param_name}.exist"
                    if exist_col in row.index and row[exist_col] == False:
                        continue
                    
                    input_tensors.append({
                        'shape': parse_list_value(row.get(f"{param_name}.shape", '[]')),
                        'range': parse_list_value(row.get(f"{param_name}.value_range", '[]')),
                        'dtype': row.get(f"{param_name}.dtype", 'float32'),
                        'format': row.get(f"{param_name}.format", 'ND'),
                        'type': 'tensor'
                    })
                    input_indices.append(param_names.index(param_name))
        
        if input_tensors:
            case['input_tensor_shape'] = format_list([t['shape'] for t in input_tensors])
            case['input_tensor_range'] = format_list([t['range'] for t in input_tensors])
            case['input_tensor_dtype'] = format_quoted_list([convert_dtype_format(t['dtype']) for t in input_tensors])
            case['input_tensor_format'] = format_quoted_list([t['format'] for t in input_tensors])
            case['input_tensor_type'] = format_quoted_list([t['type'] for t in input_tensors])
            case['input_tensor_index'] = str(input_indices)
        
        # 处理输出Tensor
        output_tensors = []
        
        for param_name, param_info in param_def.items():
            if param_info.get('io_type') == 'output':
                param_type = param_info.get('type', '')
                
                if param_type == 'aclTensor':
                    output_tensors.append({
                        'shape': parse_list_value(row.get(f"{param_name}.shape", '[]')),
                        'dtype': row.get(f"{param_name}.dtype", 'float32'),
                        'format': row.get(f"{param_name}.format", 'ND'),
                        'type': 'tensor'
                    })
        
        if output_tensors:
            case['output_tensor_shape'] = format_list([t['shape'] for t in output_tensors])
            case['output_tensor_range'] = ''
            case['output_tensor_dtype'] = format_quoted_list([convert_dtype_format(t['dtype']) for t in output_tensors])
            case['output_tensor_format'] = format_quoted_list([t['format'] for t in output_tensors])
            case['output_tensor_type'] = format_quoted_list([t['type'] for t in output_tensors])
        
        # 处理属性参数（非 tensor/tensorlist 的输入参数，以及既不是 input 也不是 output 的参数）
        attr_idx = 0
        for param_name, param_info in param_def.items():
            param_type = param_info.get('type', '')
            io_type = param_info.get('io_type', '')
            
            # 判断是否为属性：
            # 1. io_type 不是 input/output 的参数
            # 2. io_type 是 input 但类型不是 aclTensor/aclTensorList 的参数
            is_attr = False
            if io_type not in ['input', 'output']:
                is_attr = True
            elif io_type == 'input' and param_type not in ['aclTensor', 'aclTensorList']:
                is_attr = True
            
            if is_attr:
                exist_col = f"{param_name}.exist"
                if exist_col in row.index and row[exist_col] == False:
                    continue
                
                attr_prefix = '' if attr_idx == 0 else f'.{attr_idx}'
                param_type = param_info.get('type', '')
                
                case[f'attr_name{attr_prefix}'] = param_name
                case[f'attr_type{attr_prefix}'] = get_attr_type(param_type)
                case[f'attr_dtype{attr_prefix}'] = get_attr_dtype(param_type, row.get(f"{param_name}.dtype", ''))
                case[f'attr_value{attr_prefix}'] = format_attr_value(row.get(f"{param_name}.value", ''))
                
                attr_idx += 1
        
        cases.append(case)
    
    # 定义固定列的顺序
    fixed_columns = [
        'aclnn_name', 'case_name', 'bin_dir', 'genetic', 'precision_mode', 
        'precision_tolerance', 'red_range'
    ]
    
    # 创建 DataFrame 并重新排列列顺序
    df_result = pd.DataFrame(cases)
    
    # 获取所有列，固定列在前，其他列按出现顺序排列
    other_columns = [col for col in df_result.columns if col not in fixed_columns]
    all_columns = fixed_columns + other_columns
    
    return df_result[all_columns]


def parse_list_value(value):
    """解析列表值"""
    if isinstance(value, str):
        try:
            return literal_eval(value)
        except:
            return value
    return value


def format_list(items):
    """格式化列表为字符串"""
    formatted = []
    for item in items:
        if isinstance(item, (list, tuple)):
            formatted.append(str(list(item)))
        else:
            formatted.append(str(item))
    return f'[{",".join(formatted)}]'


def format_quoted_list(items):
    """格式化带引号的列表"""
    quoted = [f"'{item}'" for item in items]
    return f"[{','.join(quoted)}]"


def convert_dtype_format(dtype_str):
    """将数据类型从 float32/float16/bfloat16/float64 转换为 fp32/fp16/bf16/fp64"""
    dtype_mapping = {
        'float32': 'fp32',
        'float16': 'fp16',
        'bfloat16': 'bf16',
        'float64': 'fp64'
    }
    return dtype_mapping.get(dtype_str, dtype_str)


def get_attr_type(param_type):
    """获取属性类型"""
    type_mapping = {
        'aclScalar': 'scalar',
        'int8_t': 'buildins',
        'int16_t': 'buildins',
        'int32_t': 'buildins',
        'int64_t': 'buildins',
        'uint8_t': 'buildins',
        'uint16_t': 'buildins',
        'uint32_t': 'buildins',
        'uint64_t': 'buildins',
        'float': 'buildins',
        'double': 'buildins',
        'bool': 'buildins'
    }
    return type_mapping.get(param_type, 'buildins')


def get_attr_dtype(param_type, dtype_value):
    """获取属性dtype"""
    if param_type == 'aclScalar':
        dtype = dtype_value if dtype_value else 'float32'
    else:
        dtype = param_type
    return convert_dtype_format(dtype)


def format_attr_value(value):
    """格式化属性值"""
    if pd.isna(value):
        return ''
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return str(value)


def save_results(report, case_df, output_dir, report_file, case_file, verbose):
    """保存结果文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存覆盖度报告
    report_path = output_dir / report_file
    with open(report_path, 'w', encoding='utf-8') as f:
        yaml.dump(report, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    # 保存用例CSV
    case_path = output_dir / case_file
    case_df.to_csv(case_path, index=False)
    
    if verbose:
        print(f"[INFO] 生成覆盖度报告: {report_path}")
        print(f"[INFO] 生成用例: {case_path} ({len(case_df)}条)")


if __name__ == '__main__':
    main()
