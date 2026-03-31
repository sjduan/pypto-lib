# ArgMax/ArgMin 算子详细设计

## 场景1：最后一轴（axis=-1, A0=1）

### 特征
- 输入：`(A1, R)`，数据**连续**
- 输出：`(A1,)`，每个元素是**标量索引**
- API：ReduceMax(calIndex=true)

### API用法

```cpp
AscendC::LocalTensor<float> dstVal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> sharedTmpBuffer = tmpQueue.AllocTensor<float>();

AscendC::ReduceMax<float>(dstVal, srcVal, sharedTmpBuffer, count, true);
// calIndex=true 同时返回值和索引

// 输出格式：dst[0]=最大值, dst[1]=索引
float maxVal = dstVal.GetValue(0);
float idxRaw = dstVal.GetValue(1);
uint32_t maxIdx = *reinterpret_cast<uint32_t*>(&idxRaw);  // 类型转换！
```

### tmpBuffer计算（calIndex=true）

需要多轮迭代空间，使用API计算：

```cpp
// 推荐使用 GetReduceMaxMinTmpSize API
uint32_t tmpSize = AscendC::GetReduceMaxMinTmpSize<T>(count, true);

// 或手动计算（复杂）
int typeSize = sizeof(T);
int elementsPerBlock = 32 / typeSize;
int elementsPerRepeat = 256 / typeSize;

int firstMaxRepeat = (count + elementsPerRepeat - 1) / elementsPerRepeat;
int iter1OutputCount = firstMaxRepeat * 2;
int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock;
int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;
int iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock;
int iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;
int iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;

int tmpBufferSize = (iter2AlignStart + iter3AlignStart + iter3AlignEnd) * typeSize;
```

### Buffer规划

| Buffer | 大小 | 用途 |
|--------|------|------|
| srcQueue | count×sizeof(T) | 输入数据 |
| dstQueue | 2×sizeof(T) | 输出值+索引 |
| tmpQueue | tmpBufSize×sizeof(T) | 中间计算 |

### 约束
- ✅ 只支持**连续数据**
- ✅ 索引范围：half ≤ 65535, float ≤ 2^32
- ✅ 多个最大值返回**第一个**

---

## 场景2：非最后一轴（A0>1）

### 特征
- 输入：`(A1, R, A0)`，数据**不连续**（stride=A0）
- 输出：`(A1, A0)`，每个位置是**向量索引**
- API：Compare + Select

### 实现方案

```cpp
// 设计要素与 Reduction ARA模板完全相同
// 参考 references/reduction/ara-fullload.md

class KernelArgMax {
public:
    __aicore__ inline void Process() {
        // 1. 初始化：第一行作为初始最大值，索引为0
        AscendC::DataCopy(maxLocal, srcLocal[0], alignedCols);
        AscendC::Duplicate(idxLocal, (int32_t)0, alignedCols);
        
        // 2. 逐行比较更新（r从1到R-1）
        for (uint32_t r = 1; r < R; r++) {
            uint32_t rowOffset = r * tiling.alignedCols;
            
            // 比较当前行 > 已知最大值
            AscendC::Compare(cmpLocal, srcLocal[rowOffset], maxLocal, 
                             AscendC::CMPMODE::GT, a0Count);
            
            // 生成当前行的索引值
            AscendC::Duplicate(idxTempLocal, (int32_t)r, alignedCols);
            
            // 根据mask选择更新（关键：值和索引同步！）
            AscendC::Select(maxLocal, cmpLocal, srcLocal[rowOffset], maxLocal, 
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, a0Count);
            AscendC::Select(idxLocal, cmpLocal, idxTempLocal, idxLocal, 
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, a0Count);
        }
        
        // 3. 输出索引
        AscendC::DataCopy(dstLocal, idxLocal, a0Count);
    }
};
```

### Buffer规划（7个）

| Buffer | 大小 | 用途 |
|--------|------|------|
| inQueueX | R×alignedCols×sizeof(T) | 输入数据（Double Buffer=2） |
| outQueueY | alignedCols×sizeof(int32) | 输出索引（Double Buffer=2） |
| maxBuf | alignedCols×sizeof(T) | 当前最大值 |
| idxBuf | alignedCols×sizeof(int32) | 当前索引 |
| idxTempBuf | alignedCols×sizeof(int32) | 临时索引 |
| cmpBuf | alignedCols/8 | 比较结果mask |
| selectTmpBuf | 8192 | Select内部临时空间 |

### UB计算

```
UB = 2×(R×alignedCols×sizeof(T))     // inQueueX (depth=2)
   + 2×(alignedCols×sizeof(int32))   // outQueueY (depth=2)
   + alignedCols×sizeof(T)           // maxBuf
   + alignedCols×sizeof(int32)       // idxBuf
   + alignedCols×sizeof(int32)       // idxTempBuf
   + alignedCols/8                    // cmpBuf
   + 8192                             // selectTmpBuf
```

**示例**（FP32, R=127, A0=2479, alignedCols=128）：
```
UB = 2×(127×128×4) + 2×(128×4) + 128×4 + 128×4 + 128×4 + 128/8 + 8192
   = 129024 + 1024 + 512 + 512 + 512 + 16 + 8192
   = 140,792 Bytes ≈ 137KB < 192KB ✓
```

### 关键点

1. **值和索引同步更新**：使用相同的 cmpLocal
2. **mask类型**：Compare输出是 uint8_t
3. **数据对齐**：alignedCols 需32字节对齐
4. **API count**：用 a0Count（有效长度），不用 alignedCols

### Compare API 256字节对齐约束

**问题**：Compare API 要求 `count` 个元素所占空间必须 **256 字节对齐**。

**解决方案**：数据对齐 Padding 策略

```cpp
// 1. 计算对齐后的元素数（float 类型：向上取整到 64 的倍数）
constexpr uint32_t A0 = 32;                           // 原始大小
constexpr uint32_t A0_ALIGN = (A0 + 63) / 64 * 64;    // = 64（256字节对齐）

// 2. UB Buffer 使用对齐大小
pipe.InitBuffer(inQueueX, 1, R * A0_ALIGN * sizeof(float));

// 3. CopyIn 时用极值填充 padding 区域
AscendC::Duplicate(xLocal, -FLT_MAX, R * A0_ALIGN);  // ArgMax用极小值
// 然后拷贝实际数据到前 A0 个位置

// 4. Compare/Select 使用对齐大小
AscendC::Compare(cmpLocal, srcLocal[rowOffset], maxLocal, 
                 AscendC::CMPMODE::GT, A0_ALIGN);  // 使用 A0_ALIGN

// 5. CopyOut 只输出有效数据
AscendC::DataCopy(dstLocal, idxLocal, A0);  // 只输出 A0 个
```

**极值选择**：

| 算子 | Padding 极值 | 原因 |
|------|-------------|------|
| ArgMax | `-FLT_MAX` 或 `-INFINITY` | padding 区域永不大于实际数据 |
| ArgMin | `FLT_MAX` 或 `INFINITY` | padding 区域永不小于实际数据 |

**Buffer 大小变化**：
- 原：`R × A0 × sizeof(T)`
- 新：`R × A0_ALIGN × sizeof(T)`
- 增加：`R × (A0_ALIGN - A0) × sizeof(T)`

**适用场景**：
- A0 × sizeof(T) 不满足 256 字节对齐时**必须使用**
- 例：A0=32, float → 128字节 ❌，需对齐到 256字节（A0_ALIGN=64）

---

## ArgMin 实现

与 ArgMax 唯一区别：比较模式 GT → LT

```cpp
// ReduceMax → ReduceMin
AscendC::ReduceMin<float>(dstVal, srcVal, sharedTmpBuffer, count, true);

// Compare GT → LT
AscendC::Compare(cmpLocal, srcLocal[rowOffset], minLocal, 
                 AscendC::CMPMODE::LT, a0Count);  // GT → LT
```

---

## 约束汇总

| 约束 | 说明 |
|------|------|
| 索引范围 | half ≤ 65535, float/int32 ≤ 2^32-1 |
| 多个最大值 | 返回**第一个**最大值的索引 |
| 索引类型转换 | ReduceMax返回的索引按dst类型存储 |
| **Compare 256字节对齐** | Compare API 要求 count 个元素占 256 字节对齐 |
| Buffer数量 | 最后一轴3个，非最后一轴7个 |

---

## 设计检查清单

### 场景识别
- [ ] 3D抽象计算 A1, R, A0
- [ ] 判断模板类型（AR vs ARA）
- [ ] 判断载入模式（全载 vs 分载）

### API选择
- [ ] 最后一轴：ReduceMax(calIndex=true)
- [ ] 非最后一轴：Compare+Select
- [ ] 值和索引同步更新（Compare+Select）

### Buffer规划
- [ ] 最后一轴：3个buffer
- [ ] 非最后一轴：7个buffer
- [ ] tmpBuffer空间足够（calIndex=true）

### 实现细节
- [ ] 索引类型转换
- [ ] 数据对齐处理
- [ ] API count用有效长度
