#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plog 日志解析脚本
用途：解析 Ascend plog 日志，提取关键错误信息

使用方法：
    python3 parse_plog.py <plog_file_path>
    python3 parse_plog.py  # 使用最新日志
"""

import os
import sys
import re
import glob
from datetime import datetime
from typing import List, Dict, Tuple

class PlogParser:
    """plog 日志解析器"""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.errors = []
        self.warnings = []
        self.timeouts = []
        self.crashes = []
        
    def parse(self) -> Dict:
        """解析日志文件"""
        if not os.path.exists(self.log_path):
            return {"error": f"日志文件不存在: {self.log_path}"}
        
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            self._parse_line(line.strip(), line_num)
        
        return {
            "log_file": self.log_path,
            "total_lines": len(lines),
            "errors": self.errors,
            "warnings": self.warnings,
            "timeouts": self.timeouts,
            "crashes": self.crashes,
            "summary": self._generate_summary()
        }
    
    def _parse_line(self, line: str, line_num: int):
        """解析单行日志"""
        # 错误级别
        if re.search(r'\[ERROR\]', line, re.IGNORECASE):
            self.errors.append({
                "line": line_num,
                "content": line,
                "type": self._classify_error(line)
            })
        
        # 警告级别
        elif re.search(r'\[WARN\]', line, re.IGNORECASE):
            self.warnings.append({
                "line": line_num,
                "content": line
            })
        
        # 超时
        if re.search(r'timeout|hang|stuck', line, re.IGNORECASE):
            self.timeouts.append({
                "line": line_num,
                "content": line
            })
        
        # 崩溃
        if re.search(r'segment fault|core dump|crash|SIGSEGV', line, re.IGNORECASE):
            self.crashes.append({
                "line": line_num,
                "content": line
            })
    
    def _classify_error(self, line: str) -> str:
        """错误分类"""
        if re.search(r'ACLNN_ERR_PARAM', line):
            return "参数错误"
        elif re.search(r'ACLNN_ERR_RUNTIME', line):
            return "Runtime错误"
        elif re.search(r'ACLNN_ERR_INNER_TILING', line):
            return "Tiling错误"
        elif re.search(r'ACLNN_ERR_INNER_FIND_KERNEL', line):
            return "Kernel查找错误"
        elif re.search(r'ACLNN_ERR_INNER_OPP', line):
            return "环境配置错误"
        else:
            return "其他错误"
    
    def _generate_summary(self) -> str:
        """生成摘要"""
        summary_lines = []
        summary_lines.append(f"错误总数: {len(self.errors)}")
        summary_lines.append(f"警告总数: {len(self.warnings)}")
        summary_lines.append(f"超时次数: {len(self.timeouts)}")
        summary_lines.append(f"崩溃次数: {len(self.crashes)}")
        
        if self.errors:
            error_types = {}
            for err in self.errors:
                error_type = err["type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            summary_lines.append("\n错误类型分布:")
            for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
                summary_lines.append(f"  - {etype}: {count}")
        
        return "\n".join(summary_lines)


def find_latest_plog() -> str | None:
    """查找最新的 plog 文件"""
    log_dir = os.path.expanduser("~/ascend/log/debug/plog")
    if not os.path.exists(log_dir):
        return None
    
    log_files = glob.glob(os.path.join(log_dir, "plog-pid_*.log"))
    if not log_files:
        return None
    
    # 按修改时间排序，返回最新的
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]


def print_report(result: Dict):
    """打印解析报告"""
    print("=" * 60)
    print("plog 日志解析报告")
    print("=" * 60)
    print(f"日志文件: {result['log_file']}")
    print(f"总行数: {result['total_lines']}")
    print()
    print(result['summary'])
    print()
    
    # 打印详细错误
    if result['errors']:
        print("=" * 60)
        print("错误详情 (前10条)")
        print("=" * 60)
        for err in result['errors'][:10]:
            print(f"[Line {err['line']}] [{err['type']}]")
            print(f"  {err['content'][:200]}")
            print()
    
    # 打印超时信息
    if result['timeouts']:
        print("=" * 60)
        print("超时信息")
        print("=" * 60)
        for timeout in result['timeouts']:
            print(f"[Line {timeout['line']}]")
            print(f"  {timeout['content'][:200]}")
            print()
    
    # 打印崩溃信息
    if result['crashes']:
        print("=" * 60)
        print("崩溃信息")
        print("=" * 60)
        for crash in result['crashes']:
            print(f"[Line {crash['line']}]")
            print(f"  {crash['content'][:200]}")
            print()


def main():
    """主函数"""
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_plog()
        if not log_path:
            print("错误: 未找到 plog 日志文件")
            print("用法: python3 parse_plog.py <plog_file_path>")
            sys.exit(1)
        print(f"使用最新日志: {log_path}")
    
    parser = PlogParser(log_path)
    result = parser.parse()
    
    if "error" in result:
        print(f"错误: {result['error']}")
        sys.exit(1)
    
    print_report(result)
    
    # 返回码
    if result['crashes']:
        sys.exit(2)  # 崩溃
    elif result['timeouts']:
        sys.exit(3)  # 超时
    elif result['errors']:
        sys.exit(1)  # 错误
    else:
        sys.exit(0)  # 正常


if __name__ == "__main__":
    main()
