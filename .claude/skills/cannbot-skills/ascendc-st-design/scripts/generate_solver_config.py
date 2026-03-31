#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
求解配置生成器

功能：
1. 解析约束定义 YAML 文件
2. 构建因子依赖图
3. 拓扑排序确定求解层级
4. 识别锚点因子（入度为0）
5. 输出求解配置到 YAML 文件

使用方法：
    python generate_solver_config.py <约束定义.yaml> [输出配置.yaml]

示例：
    python generate_solver_config.py 05_约束定义.yaml 07_求解配置.yaml
"""

import sys
import yaml
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
from pathlib import Path


class DependencyGraph:
    """因子依赖图"""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.out_degree: Dict[str, int] = defaultdict(int)
        self.constraint_map: Dict[str, List[str]] = defaultdict(list)
        self.bidirectional_groups: List[Tuple[str, str, str]] = []
        
    def add_node(self, node: str):
        if node not in self.nodes:
            self.nodes.add(node)
            self.in_degree[node] = 0
            self.out_degree[node] = 0
    
    def add_edge(self, source: str, target: str, constraint_id: str = ""):
        self.add_node(source)
        self.add_node(target)
        
        if source == target:
            if constraint_id and constraint_id not in self.constraint_map[target]:
                self.constraint_map[target].append(constraint_id)
            return
        
        if target not in self.edges[source]:
            self.edges[source].add(target)
            self.reverse_edges[target].add(source)
            self.in_degree[target] += 1
            self.out_degree[source] += 1
            
        if constraint_id and constraint_id not in self.constraint_map[target]:
            self.constraint_map[target].append(constraint_id)
    
    def add_bidirectional(self, factor1: str, factor2: str, constraint_id: str):
        self.add_node(factor1)
        self.add_node(factor2)
        self.bidirectional_groups.append((factor1, factor2, constraint_id))
        
        if constraint_id:
            if constraint_id not in self.constraint_map[factor1]:
                self.constraint_map[factor1].append(constraint_id)
            if constraint_id not in self.constraint_map[factor2]:
                self.constraint_map[factor2].append(constraint_id)
    
    def get_anchors(self) -> List[str]:
        return sorted([n for n in self.nodes if self.in_degree[n] == 0])
    
    def get_sinks(self) -> List[str]:
        return sorted([n for n in self.nodes if self.out_degree[n] == 0])
    
    def topological_sort(self) -> List[List[str]]:
        in_deg = {n: self.in_degree[n] for n in self.nodes}
        levels = []
        remaining = set(self.nodes)
        
        while remaining:
            current_level = [n for n in remaining if in_deg[n] == 0]
            
            if not current_level:
                current_level = list(remaining)
                print(f"警告：检测到循环依赖，剩余节点: {current_level}")
            
            current_level.sort()
            levels.append(current_level)
            
            for node in current_level:
                remaining.remove(node)
                for target in self.edges[node]:
                    if target in remaining:
                        in_deg[target] -= 1
        
        return levels
    
    def get_dependencies(self, node: str) -> Set[str]:
        return self.reverse_edges.get(node, set())
    
    def get_dependents(self, node: str) -> Set[str]:
        return self.edges.get(node, set())
    
    def print_graph(self):
        print("\n" + "=" * 60)
        print("依赖图结构")
        print("=" * 60)
        
        print(f"\n节点总数: {len(self.nodes)}")
        print(f"边总数: {sum(len(t) for t in self.edges.values())}")
        
        print("\n锚点因子（入度=0）:")
        for a in self.get_anchors():
            print(f"  - {a}")
        
        print("\n双向约束组:")
        for f1, f2, cid in self.bidirectional_groups:
            print(f"  - {f1} <-> {f2} ({cid})")
        
        print("\n拓扑层级:")
        for i, level in enumerate(self.topological_sort()):
            print(f"  level_{i}: {level}")


class ConstraintParser:
    """约束解析器"""
    
    UNIDIRECTIONAL_CONSTRAINTS = {'calculate', 'derive', 'range', 'inferable_filter'}
    BIDIRECTIONAL_CONSTRAINTS = {'inferable', 'match', 'broadcast_dim'}
    CONVERTIBLE_CONSTRAINTS = {'convertible'}
    BROADCAST_CONSTRAINTS = {'broadcast_shape'}
    
    def __init__(self, yaml_data: Dict[str, Any]):
        self.data = yaml_data
        self.metadata = yaml_data.get('metadata', {})
        self.factors = yaml_data.get('factors', {})
        self.constraints = yaml_data.get('constraints', [])
        self.graph = DependencyGraph()
        
    def _is_valid_factor(self, factor_name: str) -> bool:
        return factor_name in self.factors
    
    def _infer_factor_info(self, factor_name: str) -> Dict[str, Any]:
        """根据因子名称推断因子信息"""
        parts = factor_name.rsplit('.', 1)
        if len(parts) != 2:
            return {}
        
        param_name, factor_type = parts
        
        # 查找同参数的其他因子以获取 io_type
        io_type = 'input'
        for existing_factor, info in self.factors.items():
            if existing_factor.startswith(f"{param_name}."):
                io_type = info.get('io_type', 'input')
                break
        
        return {
            'type': factor_type,
            'param': param_name,
            'io_type': io_type
        }
    
    def _add_implicit_factors(self):
        """从约束中推断并添加缺失的因子定义（包括中间因子）"""
        for constraint in self.constraints:
            target = constraint.get('target', '')
            sources = constraint.get('sources', [])
            
            # 添加 target 因子（如果未定义）
            if target and target not in self.factors:
                factor_info = self._infer_factor_info(target)
                if factor_info:
                    self.factors[target] = factor_info
                    print(f"  自动添加因子: {target}")
            
            # 添加 source 因子（如果未定义）
            for source in sources:
                if source and source not in self.factors:
                    factor_info = self._infer_factor_info(source)
                    if factor_info:
                        self.factors[source] = factor_info
                        print(f"  自动添加因子: {source}")
    
    def parse(self) -> DependencyGraph:
        # 先添加隐式约束中引用的因子
        self._add_implicit_factors()
        
        for factor_name in self.factors:
            self.graph.add_node(factor_name)
        
        for constraint in self.constraints:
            self._parse_constraint(constraint)
        
        return self.graph
    
    def _parse_constraint(self, constraint: Dict[str, Any]):
        constraint_type = constraint.get('type', '')
        
        if constraint_type in self.UNIDIRECTIONAL_CONSTRAINTS:
            self._parse_unidirectional(constraint)
        elif constraint_type in self.BIDIRECTIONAL_CONSTRAINTS:
            self._parse_bidirectional(constraint)
        elif constraint_type in self.CONVERTIBLE_CONSTRAINTS:
            self._parse_convertible(constraint)
        elif constraint_type in self.BROADCAST_CONSTRAINTS:
            self._parse_broadcast_shape(constraint)
        elif constraint_type == 'conditional':
            self._parse_conditional(constraint)
    
    def _parse_unidirectional(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        
        if not target or not self._is_valid_factor(target):
            return
        
        if constraint_id and constraint_id not in self.graph.constraint_map[target]:
            self.graph.constraint_map[target].append(constraint_id)
        
        for source in sources:
            if self._is_valid_factor(source):
                self.graph.add_edge(source, target, constraint_id)
    
    def _parse_bidirectional(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        constraint_type = constraint.get('type', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        
        if constraint_type == 'inferable':
            valid_sources = [s for s in sources if self._is_valid_factor(s)]
            
            if len(valid_sources) >= 2:
                for i in range(len(valid_sources) - 1):
                    self.graph.add_bidirectional(
                        valid_sources[i], valid_sources[i + 1], constraint_id
                    )
                
                anchor = valid_sources[0]
                for factor in valid_sources[1:]:
                    self.graph.add_edge(anchor, factor, constraint_id)
        
        elif constraint_type in ('match', 'broadcast_dim'):
            if sources and target and self._is_valid_factor(target):
                source = sources[0]
                if self._is_valid_factor(source):
                    self.graph.add_bidirectional(source, target, constraint_id)
                    self.graph.add_edge(source, target, constraint_id)
    
    def _parse_convertible(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        
        if not target or not self._is_valid_factor(target):
            return
        
        for source in sources:
            if self._is_valid_factor(source):
                self.graph.add_edge(source, target, constraint_id)
    
    def _parse_broadcast_shape(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        sources = constraint.get('sources', [])
        target = constraint.get('target', '')
        mode = constraint.get('mode', 'unidirectional')
        
        if not target or not self._is_valid_factor(target):
            return
        
        if mode == 'unidirectional':
            for source in sources:
                if self._is_valid_factor(source):
                    self.graph.add_edge(source, target, constraint_id)
        else:
            for source in sources:
                if self._is_valid_factor(source):
                    self.graph.add_bidirectional(source, target, constraint_id)
                    self.graph.add_edge(target, source, constraint_id)
    
    def _parse_conditional(self, constraint: Dict[str, Any]):
        constraint_id = constraint.get('id', '')
        condition = constraint.get('condition', {})
        condition_factor = condition.get('factor', '')
        
        then_clause = constraint.get('then', {})
        else_clause = constraint.get('else', {})
        
        then_target = then_clause.get('target', '')
        else_target = else_clause.get('target', '')
        
        if condition_factor and self._is_valid_factor(condition_factor):
            if then_target and self._is_valid_factor(then_target):
                self.graph.add_edge(condition_factor, then_target, constraint_id)
            if else_target and self._is_valid_factor(else_target):
                self.graph.add_edge(condition_factor, else_target, constraint_id)


class SolverConfigGenerator:
    """求解配置生成器"""
    
    def __init__(self, graph: DependencyGraph, parser: ConstraintParser):
        self.graph = graph
        self.parser = parser
        
    def generate(self) -> str:
        levels = self.graph.topological_sort()
        anchors = self.graph.get_anchors()
        
        lines = []
        
        lines.append("# 求解配置")
        lines.append("# 基于约束定义自动生成")
        lines.append("# ")
        lines.append("# 依赖关系分析：")
        
        for constraint in self.parser.constraints:
            cid = constraint.get('id', '')
            ctype = constraint.get('type', '')
            sources = constraint.get('sources', [])
            target = constraint.get('target', '')
            
            if ctype == 'inferable':
                sources_str = ', '.join(sources)
                lines.append(f"# - {cid}: {sources_str} (互推导)")
            elif ctype == 'match':
                si = constraint.get('source_index', '')
                ti = constraint.get('target_index', '')
                if sources and target:
                    lines.append(f"# - {cid}: {sources[0]}[{si}] <-> {target}[{ti}] (匹配)")
            elif ctype == 'broadcast_dim':
                si = constraint.get('source_index', '')
                ti = constraint.get('target_index', '')
                if sources and target:
                    lines.append(f"# - {cid}: {sources[0]}[{si}] <-> {target}[{ti}] (广播)")
            elif ctype == 'broadcast_shape':
                mode = constraint.get('mode', '')
                if sources and target:
                    lines.append(f"# - {cid}: {sources[0]} -> {target} ({mode} 广播)")
            elif ctype == 'calculate':
                if sources and target:
                    lines.append(f"# - {cid}: {target} <- {', '.join(sources)}")
            elif ctype == 'convertible':
                if sources and target:
                    lines.append(f"# - {cid}: {target} <- {', '.join(sources)} (可转换)")
        
        lines.append("")
        lines.append("solver:")
        lines.append("  strategy: topological")
        lines.append("  ")
        lines.append("  # ========== 锚点因子 ==========")
        lines.append("  # 定义：入度为0的因子，无依赖，可独立随机采样")
        lines.append("  anchors:")
        
        dtype_anchors = []
        shape_anchors = []
        other_anchors = []
        
        for a in anchors:
            if '.dtype' in a or '.value' in a:
                dtype_anchors.append(a)
            elif '.shape' in a:
                shape_anchors.append(a)
            else:
                other_anchors.append(a)
        
        if dtype_anchors:
            lines.append("    # 类型锚点")
            for a in dtype_anchors:
                lines.append(f"    - {a}")
        
        if shape_anchors:
            lines.append("    # 形状锚点")
            for a in shape_anchors:
                lines.append(f"    - {a}")
        
        if other_anchors:
            lines.append("    # 固定值因子（无依赖）")
            for a in other_anchors:
                lines.append(f"    - {a}")
        
        lines.append("  ")
        lines.append("  # ========== 推导顺序 ==========")
        lines.append("  derivation_order:")
        
        for i, level in enumerate(levels):
            if i == 0:
                lines.append(f"    # Level 0: 锚点因子（无依赖，可独立采样）")
            else:
                deps_info = []
                for factor in level:
                    deps = self.graph.get_dependencies(factor)
                    if deps:
                        deps_list = sorted(list(deps))
                        deps_info.append(f"{factor} <- [{', '.join(deps_list)}]")
                
                if deps_info:
                    lines.append(f"    # Level {i}: 从前面层级推导")
                    for info in deps_info[:5]:
                        lines.append(f"    #   - {info}")
                    if len(deps_info) > 5:
                        lines.append(f"    #   ... (+{len(deps_info) - 5} more)")
                else:
                    lines.append(f"    # Level {i}:")
            
            lines.append(f"    level_{i}:")
            for factor in level:
                lines.append(f"      - {factor}")
        
        lines.append("")
        lines.append("# ========== 依赖图说明 ==========")
        lines.append("# ")
        lines.append("# 类型依赖图：")
        
        for constraint in self.parser.constraints:
            if constraint.get('type') in ['inferable', 'calculate', 'convertible']:
                cid = constraint.get('id', '')
                sources = constraint.get('sources', [])
                target = constraint.get('target', '')
                ctype = constraint.get('type', '')
                
                if ctype == 'inferable' and sources:
                    sources_str = ' <-> '.join(sources)
                    lines.append(f"#   {sources_str}  (互推导，{cid})")
                elif ctype == 'calculate' and target and sources:
                    lines.append(f"#   {target} <- {', '.join(sources)}  ({cid})")
                elif ctype == 'convertible' and target and sources:
                    lines.append(f"#   {target} <- {', '.join(sources)}  (可转换，{cid})")
        
        lines.append("# ")
        lines.append("# 形状依赖图：")
        
        for constraint in self.parser.constraints:
            if constraint.get('type') in ['match', 'broadcast_dim', 'broadcast_shape', 'calculate']:
                sources = constraint.get('sources', [])
                target = constraint.get('target', '')
                ctype = constraint.get('type', '')
                cid = constraint.get('id', '')
                
                has_shape = any('.shape' in s for s in sources) if sources else False
                if has_shape or (target and '.shape' in target):
                    if ctype == 'match':
                        si = constraint.get('source_index', '')
                        ti = constraint.get('target_index', '')
                        if sources:
                            lines.append(f"#   {sources[0]}[{si}] <-> {target}[{ti}]  (匹配，{cid})")
                    elif ctype == 'broadcast_dim':
                        si = constraint.get('source_index', '')
                        ti = constraint.get('target_index', '')
                        if sources:
                            lines.append(f"#   {sources[0]}[{si}] <-> {target}[{ti}]  (广播，{cid})")
                    elif ctype == 'broadcast_shape' and sources:
                        lines.append(f"#   {sources[0]} <- {target}  (广播约束，{cid})")
                    elif ctype == 'calculate' and target and sources:
                        lines.append(f"#   {target} <- {', '.join(sources)}  ({cid})")
        
        return '\n'.join(lines)


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_text(content: str, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def print_summary(graph: DependencyGraph, levels: List[List[str]]):
    print("\n" + "=" * 70)
    print("求解配置生成摘要")
    print("=" * 70)
    
    print(f"\n因子总数: {len(graph.nodes)}")
    print(f"层级总数: {len(levels)}")
    
    anchors = graph.get_anchors()
    print(f"\n锚点因子（入度=0）: {len(anchors)}个")
    for a in anchors:
        print(f"  - {a}")
    
    print(f"\n双向约束组: {len(graph.bidirectional_groups)}组")
    for f1, f2, cid in graph.bidirectional_groups:
        print(f"  - {f1} <-> {f2} ({cid})")
    
    print(f"\n推导层级:")
    for i, level in enumerate(levels):
        level_preview = ', '.join(level[:5])
        suffix = '...' if len(level) > 5 else ''
        print(f"  level_{i} ({len(level)}个): [{level_preview}{suffix}]")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python generate_solver_config.py <约束定义.yaml> [输出配置.yaml]")
        print("示例: python generate_solver_config.py 05_约束定义.yaml 07_求解配置.yaml")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    print(f"正在读取: {input_file}")
    
    yaml_data = load_yaml(input_file)
    
    print("正在解析约束关系...")
    parser = ConstraintParser(yaml_data)
    graph = parser.parse()
    
    if '--verbose' in sys.argv or '-v' in sys.argv:
        graph.print_graph()
    
    print("正在生成求解配置...")
    generator = SolverConfigGenerator(graph, parser)
    config_text = generator.generate()
    
    levels = graph.topological_sort()
    print_summary(graph, levels)
    
    if output_file:
        save_text(config_text, output_file)
        print(f"\n求解配置已保存到: {output_file}")
    else:
        print("\n" + "=" * 70)
        print("生成的配置:")
        print("=" * 70)
        print(config_text)


if __name__ == "__main__":
    main()
