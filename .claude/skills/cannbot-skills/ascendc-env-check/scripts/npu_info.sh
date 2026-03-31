#!/bin/bash
# NPU 设备信息查询脚本
# 用途：查询 NPU 设备列表、状态、资源使用情况

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================"
echo "NPU 设备信息查询"
echo "================================"
echo ""

WARNINGS=0

# 检查 npu-smi 是否可用
echo -e "${BLUE}[1/3] 检查 npu-smi 工具...${NC}"
if command -v npu-smi &> /dev/null; then
    npu_smi_version=$(npu-smi -v 2>/dev/null | head -1 || echo "未知")
    echo -e "${GREEN}✓ npu-smi 可用 (版本: $npu_smi_version)${NC}"
else
    echo -e "${RED}✗ npu-smi 不可用${NC}"
    echo "  可能原因："
    echo "    1. 未安装 CANN"
    echo "    2. 未 source CANN 环境变量"
    echo "    3. 当前环境不支持 NPU（模拟环境）"
    exit 1
fi

echo ""

# 设备信息（包含列表和资源）
echo -e "${BLUE}[2/3] 设备信息...${NC}"
echo "----------------------------------------"
npu-smi info
echo "----------------------------------------"

echo ""

# 资源监控
echo -e "${BLUE}[3/3] 资源使用情况...${NC}"
echo "----------------------------------------"
if command -v npu-smi &> /dev/null; then
    echo -e "${YELLOW}进程信息：${NC}"
    npu-smi info 2>/dev/null | grep -A 10 "Process" || echo "  无运行中的进程"
fi
echo "----------------------------------------"

echo ""
echo "================================"
echo "查询完成"
echo "================================"