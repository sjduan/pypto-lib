# Ascend C 开发环境配置

## ⛔️ 关键环境变量（必须正确设置）

### 1. 环境变量名称

**正确名称**：`ASCEND_HOME_PATH`（不是 ASCEND_HOME）

```bash
# ✅ 正确
export ASCEND_HOME_PATH=/home/developer/Ascend/cann

# ❌ 错误
export ASCEND_HOME=/home/developer/Ascend/cann
```

### 2. 优先级顺序

```cmake
1. $ENV{ASCEND_HOME_PATH}    (环境变量)  ← 标准方式
2. $ENV{HOME}/Ascend/cann     (默认路径)
3. /usr/local/Ascend/cann     (备用默认)
```

---

## 环境检查清单

### Step 1: 检查 CANN 安装位置

```bash
# 常见安装位置
ls -la /home/developer/Ascend/cann          # 用户目录安装
ls -la $HOME/Ascend/latest                    # 普通用户符号链接
ls -la $HOME/Ascend/cann                    # 普通用户
ls -la /usr/local/Ascend/cann               # 系统目录安装
ls -la /usr/local/Ascend/latest               # 系统目录安装符号链接
```

### Step 2: 验证 CANN 工具链

```bash
# 检查编译器
CANN_PATH=/home/developer/Ascend/cann
ls $CANN_PATH/aarch64-linux/ccec_compiler/bin/bisheng  # Ascend C 编译器

# 检查头文件
ls $CANN_PATH/include/kernel_operator.h

# 检查库文件
ls $CANN_PATH/lib64/libregister.so
ls $CANN_PATH/lib64/libgraph_base.so
```

### Step 3: 设置环境变量

```bash
# 方式 1：手动设置
export ASCEND_HOME_PATH=/home/developer/Ascend/cann

# 方式 2：使用官方脚本
source $ASCEND_HOME_PATH/bin/set_env.sh
```

### Step 4: 验证环境变量

```bash
echo $ASCEND_HOME_PATH
# 应该输出：/home/developer/Ascend/cann
```

---

## 常见错误

### 错误 1：使用错误的环境变量名

```bash
# ❌ 错误
export ASCEND_HOME=/home/developer/Ascend/cann

# CMake 会报错：
# CMake Error: Could not find ASCEND_HOME_PATH
```

### 错误 2：未设置环境变量

```bash
# cmake 会使用默认路径，但可能不正确
# 建议显式设置
```

### 错误 3：路径错误

```bash
# ❌ 错误：指向 ascend-toolkit 而非 cann
export ASCEND_HOME_PATH=/home/developer/Ascend/ascend-toolkit

# ✅ 正确：指向 cann 目录
export ASCEND_HOME_PATH=/home/developer/Ascend/cann
```

---

## 编译器说明

### Ascend C 编译器

**编译器名称**：`bisheng`（不是 tikcc、bangc）

**位置**：`$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler/bin/bisheng`

**检查方式**：
```bash
bisheng --version
```

### 编译器发现机制

CMake 通过 `find_package(ASC)` 自动发现编译器，无需手动配置。

---

## 构建系统说明

### 正确的 CMakeLists.txt 结构

参考 `asc-devkit/examples/03_libraries/05_reduce/reducemax/CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.16)

# ✅ 使用 find_package 自动发现 CANN
find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    reducemax.asc
)

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

# ✅ 指定 NPU 架构
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>  # A2 服务器
)
```

### 错误的构建方式

```cmake
# ❌ 错误：手动指定编译器路径
set(ASCEND_HOME $ENV{ASCEND_HOME})  # 错误的环境变量名
add_custom_target(kernel
    COMMAND ${ASCEND_HOME}/compiler/tikcpp/tikcc ...  # 错误的编译器
    COMMAND ${ASCEND_HOME}/compiler/tikcpp/bangc ...  # 错误的编译器
)
```

### 正确的构建流程

```bash
# ✅ 标准 cmake 流程（无 build.sh）
mkdir build && cd build
cmake ..
make
```

---

## 完整环境验证脚本

```bash
#!/bin/bash
# verify_environment.sh - 验证 Ascend C 开发环境

set -e

echo "=== Ascend C 开发环境验证 ==="

# 1. 检查环境变量
echo "[1] 检查环境变量..."
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "❌ ASCEND_HOME_PATH 未设置"
    echo "   请执行: export ASCEND_HOME_PATH=/path/to/cann"
    exit 1
else
    echo "✓ ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
fi

# 2. 检查 CANN 目录
echo "[2] 检查 CANN 安装..."
if [ ! -d "$ASCEND_HOME_PATH" ]; then
    echo "❌ CANN 目录不存在: $ASCEND_HOME_PATH"
    exit 1
else
    echo "✓ CANN 目录存在"
fi

# 3. 检查编译器
echo "[3] 检查编译器..."
COMPILER="$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler/bin/bisheng"
if [ ! -f "$COMPILER" ]; then
    echo "❌ 编译器不存在: $COMPILER"
    exit 1
else
    echo "✓ 编译器存在: $COMPILER"
fi

# 4. 检查头文件
echo "[4] 检查头文件..."
HEADER="$ASCEND_HOME_PATH/include/kernel_operator.h"
if [ ! -f "$HEADER" ]; then
    echo "❌ 头文件不存在: $HEADER"
    exit 1
else
    echo "✓ 头文件存在"
fi

# 5. 检查库文件
echo "[5] 检查库文件..."
LIBS=(
    "$ASCEND_HOME_PATH/lib64/libregister.so"
    "$ASCEND_HOME_PATH/lib64/libgraph_base.so"
)
for LIB in "${LIBS[@]}"; do
    if [ ! -f "$LIB" ]; then
        echo "❌ 库文件不存在: $LIB"
        exit 1
    fi
done
echo "✓ 库文件存在"

# 6. 检查 asc-devkit
echo "[6] 检查 asc-devkit..."
if [ ! -d "asc-devkit" ]; then
    echo "⚠ asc-devkit 目录不存在"
    echo "  建议执行: git clone https://gitcode.com/cann/asc-devkit"
else
    echo "✓ asc-devkit 目录存在"
fi

echo ""
echo "=== 环境验证通过 ✓ ==="
echo ""
echo "后续步骤:"
echo "1. 参考 asc-devkit/examples/ 中的示例"
echo "2. 使用 find_package(ASC) 配置 CMakeLists.txt"
echo "3. 使用 cmake && make 构建"
