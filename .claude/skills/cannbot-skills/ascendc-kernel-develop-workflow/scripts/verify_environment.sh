#!/bin/bash
# verify_environment.sh - 验证 Ascend C 开发环境并保存结果（JSON 格式）
# 使用：bash verify_environment.sh <operator_name>
# 示例：bash verify_environment.sh softmax0309

set -e

# 获取脚本所在目录（兼容不同平台）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW_DIR="$(dirname "$SCRIPT_DIR")"

# 解析参数
OPERATOR_NAME=${1:-""}
if [ -z "$OPERATOR_NAME" ]; then
    echo "用法: $0 <operator_name>"
    echo "示例: $0 softmax0309"
    exit 1
fi

# 设置保存路径
SAVE_DIR="ops/${OPERATOR_NAME}/docs"
SAVE_FILE="${SAVE_DIR}/environment.json"

# 检查项目是否已初始化
if [ ! -d "$SAVE_DIR" ]; then
    echo "❌ 错误：项目目录不存在"
    echo ""
    echo "请先运行项目初始化："
    echo "  bash ${SCRIPT_DIR}/init_operator_project.sh ${OPERATOR_NAME}"
    exit 1
fi

echo "================================================================"
echo "Ascend C 开发环境验证"
echo "================================================================"
echo ""
echo "算子名称: ${OPERATOR_NAME}"
echo "保存路径: ${SAVE_FILE}"
echo ""

ERRORS=0
WARNINGS=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 用数组收集 JSON 数据
declare -A ENV_DATA

# 辅助函数
error() {
    echo -e "${RED}❌ $1${NC}"
    ERRORS=$((ERRORS + 1))
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    WARNINGS=$((WARNINGS + 1))
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# JSON 转义函数
json_escape() {
    local str="$1"
    str="${str//\\/\\\\}"  # 反斜杠
    str="${str//\"/\\\"}"  # 双引号
    str="${str//$'\n'/\\n}" # 换行
    str="${str//$'\r'/\\r}" # 回车
    str="${str//$'\t'/\\t}" # 制表符
    echo "$str"
}

# 收集环境信息
collect_env_info() {
    # 1. 检查环境变量
    echo "[1/7] 检查环境变量..."
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -z "$ASCEND_HOME_PATH" ]; then
        error "ASCEND_HOME_PATH 未设置"
        ENV_DATA[ascend_home_path]=""
        ENV_DATA[ascend_home_path_valid]="false"
        
        echo ""
        echo "  解决方法："
        echo "  export ASCEND_HOME_PATH=/home/developer/Ascend/cann"
        echo "  或"
        echo "  source /home/developer/Ascend/ascend-toolkit/set_env.sh"
        echo ""
    else
        success "ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
        ENV_DATA[ascend_home_path]="$(json_escape "$ASCEND_HOME_PATH")"
        ENV_DATA[ascend_home_path_valid]="true"
    fi
    
    # 2. 检查 CANN 安装
    echo ""
    echo "[2/7] 检查 CANN 安装..."
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -d "$ASCEND_HOME_PATH" ]; then
        success "CANN 目录存在"
        ENV_DATA[cann_dir_exists]="true"
        
        # 提取版本号
        CANN_VERSION=$(basename "$ASCEND_HOME_PATH" | sed 's/cann-//' | sed 's/-beta//')
        ENV_DATA[cann_version]="$(json_escape "$CANN_VERSION")"
    else
        error "CANN 目录不存在: $ASCEND_HOME_PATH"
        ENV_DATA[cann_dir_exists]="false"
    fi
    
    if [ -d "$ASCEND_HOME_PATH/aarch64-linux" ]; then
        success "aarch64-linux 目录存在"
        ENV_DATA[aarch64_dir_exists]="true"
        ENV_DATA[arch_dir]="aarch64-linux"
    elif [ -d "$ASCEND_HOME_PATH/x86_64-linux" ]; then
        success "x86_64-linux 目录存在"
        ENV_DATA[aarch64_dir_exists]="true"
        ENV_DATA[arch_dir]="x86_64-linux"
    else
        error "架构目录不存在"
        ENV_DATA[aarch64_dir_exists]="false"
    fi
    
    # 3. 检查编译器
    echo ""
    echo "[3/7] 检查 Ascend C 编译器..."
    echo "────────────────────────────────────────────────────────────────"
    
    ARCH_DIR="${ENV_DATA[arch_dir]:-aarch64-linux}"
    COMPILER_PATH="$ASCEND_HOME_PATH/$ARCH_DIR/ccec_compiler/bin/bisheng"
    
    if [ -f "$COMPILER_PATH" ]; then
        success "编译器存在: bisheng"
        ENV_DATA[bisheng_path]="$(json_escape "$COMPILER_PATH")"
        ENV_DATA[bisheng_exists]="true"
        
        if [ -x "$COMPILER_PATH" ]; then
            success "编译器可执行"
            ENV_DATA[bisheng_executable]="true"
        else
            error "编译器不可执行"
            ENV_DATA[bisheng_executable]="false"
        fi
    else
        error "编译器不存在: bisheng"
        ENV_DATA[bisheng_path]=""
        ENV_DATA[bisheng_exists]="false"
    fi
    
    # 4. 检查头文件
    echo ""
    echo "[4/7] 检查头文件..."
    echo "────────────────────────────────────────────────────────────────"
    
    HEADER_PATHS=(
        "$ASCEND_HOME_PATH/$ARCH_DIR/ascendc/include/basic_api/kernel_operator.h"
        "$ASCEND_HOME_PATH/$ARCH_DIR/asc/include/kernel_operator.h"
        "$ASCEND_HOME_PATH/$ARCH_DIR/include/ascendc/basic_api/kernel_operator.h"
        "$ASCEND_HOME_PATH/include/kernel_operator.h"
    )
    
    HEADER_FOUND=false
    HEADER_PATH=""
    
    for path in "${HEADER_PATHS[@]}"; do
        if [ -f "$path" ]; then
            HEADER_PATH="$path"
            HEADER_FOUND=true
            break
        fi
    done
    
    if [ "$HEADER_FOUND" = true ]; then
        success "头文件存在: kernel_operator.h"
        ENV_DATA[kernel_operator_h]="$(json_escape "$HEADER_PATH")"
        ENV_DATA[header_exists]="true"
    else
        error "头文件不存在: kernel_operator.h"
        ENV_DATA[kernel_operator_h]=""
        ENV_DATA[header_exists]="false"
    fi
    
    # 5. 检查库文件
    echo ""
    echo "[5/7] 检查库文件..."
    echo "────────────────────────────────────────────────────────────────"
    
    LIB_REGISTER="$ASCEND_HOME_PATH/lib64/libregister.so"
    LIB_ACL="$ASCEND_HOME_PATH/lib64/libascendcl.so"
    LIBS_OK=true
    
    if [ -f "$LIB_REGISTER" ]; then
        success "libregister.so 存在"
        ENV_DATA[libregister_so]="$(json_escape "$LIB_REGISTER")"
    else
        error "libregister.so 不存在"
        ENV_DATA[libregister_so]=""
        LIBS_OK=false
    fi
    
    if [ -f "$LIB_ACL" ]; then
        success "libascendcl.so 存在"
        ENV_DATA[libascendcl_so]="$(json_escape "$LIB_ACL")"
    else
        error "libascendcl.so 不存在"
        ENV_DATA[libascendcl_so]=""
        LIBS_OK=false
    fi
    
    ENV_DATA[all_libs_exist]="$LIBS_OK"
    
    # 6. 检查 asc-devkit
    echo ""
    echo "[6/7] 检查 asc-devkit..."
    echo "────────────────────────────────────────────────────────────────"
    
    ASC_DEVKIT_PATH="asc-devkit"
    if [ -d "$ASC_DEVKIT_PATH" ]; then
        success "asc-devkit 目录存在"
        ENV_DATA[asc_devkit_path]="$(json_escape "$ASC_DEVKIT_PATH")"
        ENV_DATA[asc_devkit_exists]="true"
        
        # 检查 API 文档
        if [ -d "$ASC_DEVKIT_PATH/docs/api" ]; then
            success "API 文档目录存在"
            ENV_DATA[api_docs_exist]="true"
        else
            warning "API 文档目录不存在"
            ENV_DATA[api_docs_exist]="false"
        fi
        
        # 检查 CMake 配置
        if [ -d "$ASC_DEVKIT_PATH/cmake" ]; then
            success "CMake 配置目录存在"
            ENV_DATA[cmake_config_exists]="true"
        else
            warning "CMake 配置目录不存在"
            ENV_DATA[cmake_config_exists]="false"
        fi
        
        # 统计示例数量
        EXAMPLES_COUNT=$(find "$ASC_DEVKIT_PATH/examples" -type f -name "*.asc" 2>/dev/null | wc -l)
        if [ "$EXAMPLES_COUNT" -gt 0 ]; then
            success "找到 $EXAMPLES_COUNT 个示例"
            ENV_DATA[examples_count]="$EXAMPLES_COUNT"
        else
            warning "未找到示例文件"
            ENV_DATA[examples_count]="0"
        fi
    else
        error "asc-devkit 目录不存在"
        ENV_DATA[asc_devkit_path]=""
        ENV_DATA[asc_devkit_exists]="false"
    fi
    
    # 7. 检查 NPU 设备（可选）
    echo ""
    echo "[7/7] 检查 NPU 设备..."
    echo "────────────────────────────────────────────────────────────────"
    
    if command -v npu-smi &> /dev/null; then
        if npu-smi info &> /dev/null; then
            success "NPU 设备可用"
            ENV_DATA[npu_available]="true"
            ENV_DATA[npu_device_count]="1"
        else
            warning "NPU 设备不可用（可继续开发，但无法运行测试）"
            ENV_DATA[npu_available]="false"
            ENV_DATA[npu_device_count]="0"
        fi
    else
        warning "npu-smi 命令不存在（可继续开发，但无法运行测试）"
        ENV_DATA[npu_available]="false"
        ENV_DATA[npu_device_count]="0"
    fi
}

# 生成 JSON 文件
generate_json() {
    cat > "$SAVE_FILE" << EOF
{
  "check_time": "$(date -Iseconds)",
  "operator": "${OPERATOR_NAME}",
  "environment": {
    "ascend_home_path": "${ENV_DATA[ascend_home_path]}",
    "ascend_home_path_valid": ${ENV_DATA[ascend_home_path_valid]},
    "cann_dir_exists": ${ENV_DATA[cann_dir_exists]},
    "cann_version": "${ENV_DATA[cann_version]}",
    "arch_dir": "${ENV_DATA[arch_dir]}",
    "aarch64_dir_exists": ${ENV_DATA[aarch64_dir_exists]},
    "bisheng_path": "${ENV_DATA[bisheng_path]}",
    "bisheng_exists": ${ENV_DATA[bisheng_exists]},
    "bisheng_executable": ${ENV_DATA[bisheng_executable]},
    "kernel_operator_h": "${ENV_DATA[kernel_operator_h]}",
    "header_exists": ${ENV_DATA[header_exists]},
    "libregister_so": "${ENV_DATA[libregister_so]}",
    "libascendcl_so": "${ENV_DATA[libascendcl_so]}",
    "all_libs_exist": ${ENV_DATA[all_libs_exist]},
    "asc_devkit_path": "${ENV_DATA[asc_devkit_path]}",
    "asc_devkit_exists": ${ENV_DATA[asc_devkit_exists]},
    "api_docs_exist": ${ENV_DATA[api_docs_exist]},
    "cmake_config_exists": ${ENV_DATA[cmake_config_exists]},
    "examples_count": ${ENV_DATA[examples_count]},
    "npu_available": ${ENV_DATA[npu_available]},
    "npu_device_count": ${ENV_DATA[npu_device_count]}
  },
  "validation": {
    "all_passed": $([ $ERRORS -eq 0 ] && echo "true" || echo "false"),
    "error_count": $ERRORS,
    "warning_count": $WARNINGS
  }
}
EOF
}

# 主流程
collect_env_info
generate_json

# 输出总结
echo ""
echo "================================================================"
echo "验证结果"
echo "================================================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 环境验证通过！${NC}"
    echo ""
    echo "环境检查结果已保存到："
    echo "  ${SAVE_FILE}"
    echo ""
    echo "后续步骤："
    echo "  1. 开始 Phase 1：需求分析与方案设计"
    echo "  2. 生成设计文档：docs/design.md"
else
    echo -e "${RED}✗ 环境验证失败${NC}"
    echo ""
    echo "错误数量：$ERRORS"
    echo "警告数量：$WARNINGS"
    echo ""
    echo "请根据上述错误信息修复环境配置后重试。"
    exit 1
fi
