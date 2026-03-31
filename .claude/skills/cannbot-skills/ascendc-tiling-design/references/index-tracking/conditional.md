# 条件查找算子详细设计

## 算子分类

| 算子 | 功能 | 输出 | 典型应用 |
|------|------|------|---------|
| **Where** | 返回满足条件的索引 | indices(N,) | 稀疏索引 |
| **NonZero** | 返回非零元素索引 | indices(N,) | 数据过滤 |
| **MaskedSelect** | 返回满足条件的值 | values(N,) | 数据提取 |
| **MaskedFill** | 填充满足条件的值 | tensor(原shape) | 数据清洗 |

---

## 核心API

| API | 功能 | 输出类型 |
|-----|------|---------|
| **Compare** | 生成条件mask | uint8_t (mask) |
| **Select** | 条件选择 | T (值) |
| **ReduceSum** | 统计满足条件数量 | T (count) |

**关键**：AscendC **没有** 压缩API，需要手动实现。

---

## 场景1：MaskedFill（条件填充）

**功能**：将满足条件的值填充为指定值

**实现**：
```cpp
// 生成mask
AscendC::Compare(maskLocal, dataLocal, thresholdLocal, 
                 AscendC::CMPMODE::GT, count);

// 生成填充值
AscendC::Duplicate(fillLocal, fillValue, count);

// 选择：mask=1用fillValue，mask=0用原值
AscendC::Select(dstLocal, maskLocal, fillLocal, dataLocal, 
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
```

**Buffer规划**：5个
- dataLocal（输入）
- maskLocal（比较结果）
- fillLocal（填充值）
- dstLocal（输出）
- selectTmpBuf（8192固定）

---

## 场景2：MaskedSelect（条件选择值）

**功能**：返回满足条件的值（不压缩）

**实现**：
```cpp
// 生成mask
AscendC::Compare(maskLocal, dataLocal, thresholdLocal, 
                 AscendC::CMPMODE::GT, count);

// 选择：mask=1用data，mask=0用0
AscendC::Duplicate(zeroLocal, (T)0, count);
AscendC::Select(dstLocal, maskLocal, dataLocal, zeroLocal, 
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

// 注意：dstLocal 仍有count个元素，只是不满足条件的为0
// 需要压缩时，需手动实现
```

---

## 场景3：Where/NonZero（返回压缩索引）

**功能**：返回满足条件的**压缩后**索引

**⚠️ 难点**：AscendC **没有压缩API**，需手动实现

### 方法1：迭代实现（简单但慢）

```cpp
// 生成mask
AscendC::Compare(maskLocal, dataLocal, thresholdLocal, 
                 AscendC::CMPMODE::GT, count);

// 迭代扫描（性能差）
uint32_t idx = 0;
for (uint32_t i = 0; i < count; i++) {
    uint8_t maskVal = maskLocal.GetValue(i);
    if (maskVal == 1) {
        indicesLocal.SetValue(idx, i);
        idx++;
    }
}
matchCount = idx;
```

**性能**：O(n)，但标量操作，慢

### 方法2：向量化实现（复杂但快）

**核心思想**：Prefix Sum + 压缩

```cpp
// 1. 生成mask（int32类型）
AscendC::Compare(maskU8Local, dataLocal, thresholdLocal, 
                 AscendC::CMPMODE::GT, count);
AscendC::Cast(maskI32Local, maskU8Local, AscendC::RoundMode::CAST_NONE, count);

// 2. 计算前缀和（无专用API，需手动实现或用ReduceSum分段）
// 伪代码：
// prefixSum[0] = mask[0]
// prefixSum[i] = prefixSum[i-1] + mask[i]

// 3. 根据prefixSum压缩索引
// 伪代码：
// for (i = 0; i < count; i++) {
//     if (mask[i] == 1) {
//         indices[prefixSum[i] - 1] = i;
//     }
// }
```

**挑战**：
- PrefixSum无专用API
- 需要多次迭代或复杂逻辑
- 实现复杂度高

### 方法3：Host侧压缩（推荐，如果允许）

```cpp
// NPU侧：输出mask + 原始索引序列
AscendC::Compare(maskLocal, dataLocal, thresholdLocal, CMPMODE::GT, count);
AscendC::ArithmeticProgression(idxSeqLocal, 0, 1, count);  // 生成0,1,2,...

// GM输出
DataCopy(maskGm, maskLocal, count);
DataCopy(idxSeqGm, idxSeqLocal, count);

// Host侧压缩
// Python: indices = idxSeq[mask == 1]
```

---

## 场景4：统计满足条件元素数

**功能**：返回满足条件的元素数量

**实现**：
```cpp
// 生成mask
AscendC::Compare(maskLocal, dataLocal, thresholdLocal, 
                 AscendC::CMPMODE::GT, count);

// Cast to int32
AscendC::Cast(maskI32Local, maskLocal, AscendC::RoundMode::CAST_NONE, count);

// ReduceSum
AscendC::ReduceSum<int32_t>(countLocal, maskI32Local, tmpBuffer, count, false);

int32_t matchCount = countLocal.GetValue(0);
```

---

## Buffer规划对比

| 场景 | Buffer数量 | 复杂度 |
|------|-----------|--------|
| MaskedFill | 5 | ★★ |
| MaskedSelect（不压缩） | 5 | ★★ |
| Where（迭代） | 3 | ★★★（性能差） |
| Where（向量化） | 7-10 | ★★★★★ |
| Where（Host压缩） | 4 | ★★（推荐） |

---

## API 限制

| API | 限制 |
|-----|------|
| Compare | mask是uint8_t，每个元素1字节 |
| Select | 不压缩输出 |
| ReduceSum | 只返回标量，不返回索引 |

**关键结论**：AscendC **没有** 原生的压缩API，Where/NonZero 需要变通实现。

---

## 设计检查清单

- [ ] 确定是否需要压缩输出
- [ ] 选择实现方案（迭代/向量化/Host压缩）
- [ ] mask类型是uint8_t
- [ ] Buffer规划（5-10个）
- [ ] 性能权衡（简单实现慢，快实现复杂）
