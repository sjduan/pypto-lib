# Top-K 算子详细设计

## 算子分类

| 算子 | 功能 | 典型API | 复杂度 |
|------|------|---------|--------|
| **TopK** | 前K个值+索引 | TopK API | ★★★ |
| **TopKMask** | 极小K（≤128） | TopKMask API | ★★ |
| **多次ArgMax** | 小K（≤10） | Compare+Select循环 | ★★★ |
| **Sort+切片** | 大K（>4096） | Sort + 取前K | ★★★ |

---

## API 选择决策树

```
需要 TopK（值+索引）？
    ↓
   是
    ↓
K ≤ 128？
    ↓
   是 ────────────────────┐
    │                      ↓
    │            ✅ TopKMask API
    │            ├─ 极小K专用
    │            └─ 性能最优
    │
   否
    ↓
K ≤ 4096 且 inner ≤ 4096？
    ↓
   是 ────────────────────┐
    │                      ↓
    │            ✅ TopK API
    │            ├─ Normal模式
    │            ├─ inner限制4096
    │            └─ K无独立上限
    │
   否
    ↓
K 很小（≤10）？
    ↓
   是 ────────────────────┐
    │                      ↓
    │            ⚠️ 多次ArgMax
    │            ├─ K次循环
    │            ├─ 每次找最大后标记
    │            └─ 性能较差
    │
   否 ────────────────────┐
                           ↓
                ⚠️ Sort + 切片
                ├─ 全排序
                ├─ 取前K个
                └─ K大时更高效
```

---

## 场景1：TopK API（推荐，K≤4096, inner≤4096）

### API原型

```cpp
template <typename T, bool isInitIndex = false, bool isReuseSource = false>
__aicore__ inline void TopK(
    const LocalTensor<T>& dstVal,      // 输出值
    const LocalTensor<T>& dstIdx,      // 输出索引
    const LocalTensor<T>& src,         // 输入
    const LocalTensor<T>& tmpBuffer,   // 临时空间
    const TopKParams& params           // 参数
);

struct TopKParams {
    uint32_t totalLength;  // 输入长度
    uint32_t K;            // Top-K
    bool islargest;        // true=TopK大值, false=TopK小值
};
```

### 使用示例

```cpp
AscendC::LocalTensor<float> srcLocal = inQueue.AllocTensor<float>();
AscendC::LocalTensor<float> valLocal = outQueueVal.AllocTensor<float>();
AscendC::LocalTensor<float> idxLocal = outQueueIdx.AllocTensor<float>();
AscendC::LocalTensor<float> tmpBuffer = tmpQueue.AllocTensor<float>();

AscendC::TopKParams params;
params.totalLength = inner;  // 4096
params.K = 10;
params.islargest = true;

AscendC::TopK<float>(valLocal, idxLocal, srcLocal, tmpBuffer, params);

// 输出格式：
// valLocal[0..K-1] = 最大的K个值（降序）
// idxLocal[0..K-1] = 对应索引（需类型转换）
for (int i = 0; i < K; i++) {
    float val = valLocal.GetValue(i);
    float idxRaw = idxLocal.GetValue(i);
    uint32_t idx = *reinterpret_cast<uint32_t*>(&idxRaw);
}
```

### 约束

| 参数 | 限制 |
|------|------|
| inner | ≤ 4096，32字节对齐 |
| n | 1 ≤ n ≤ inner |
| K | 1 ≤ K ≤ inner |
| 算法 | RADIX_SELECT（推荐）或 MERGE_SORT |

---

## 场景2：TopKMask API（极小K≤128）

### API原型

```cpp
template <typename T>
__aicore__ inline void TopKMask(
    const LocalTensor<T>& dstVal,
    const LocalTensor<T>& dstIdx,
    const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer,
    const uint32_t k,
    const uint32_t mask,
    const uint32_t repeatTimes
);
```

### 使用场景
- K ≤ 128
- 性能优于通用TopK

---

## 场景3：多次ArgMax（K很小≤10）

### 实现方案

```cpp
// 找Top-K：循环K次，每次找最大值
for (uint32_t k = 0; k < K; k++) {
    // 找当前最大值和索引
    AscendC::ReduceMax<float>(dstLocal, srcLocal, tmpBuffer, count, true);
    
    float maxVal = dstLocal.GetValue(0);
    uint32_t maxIdx = *reinterpret_cast<uint32_t*>(dstLocal.GetValue(1));
    
    // 保存结果
    valLocal.SetValue(k, maxVal);
    idxLocal.SetValue(k, maxIdx);
    
    // 标记已选元素为最小值（排除）
    srcLocal.SetValue(maxIdx, -FLT_MAX);
}
```

### 约束
- K很小（K≤10），否则性能差
- 需要修改源数据或维护mask

---

## 场景4：Sort + 切片（大K或inner>4096）

### 实现方案

```cpp
// 完整排序
AscendC::SortParams sortParams;
sortParams.totalLength = inner;
sortParams.sortAxis = -1;

AscendC::Sort<float, true>(valLocal, idxLocal, srcLocal, tmpBuffer, sortParams);

// 切片取前K个
// valLocal[0..K-1] 就是 Top-K 的值（升序）
// idxLocal[0..K-1] 就是对应的索引

// 降序：取反→排序→取反，或反转结果
```

---

## Buffer规划（TopK API）

| Buffer | 用途 | 大小计算 |
|--------|------|---------|
| inQueueValue | 输入值 | inner×sizeof(T) |
| outQueueValue | 输出值 | kPad×outter×sizeof(T) |
| outQueueIndex | 输出索引 | kPadIndex×outter×4 |
| tmpLocalBuf | 临时空间 | GetTopKMaxMinTmpSize() |

**kPad计算**：
- FP32: kPad = (K + 7) / 8 × 8
- FP16: kPad = (K + 15) / 16 × 16

---

## 性能对比

| 方案 | 时间复杂度 | 适用场景 | 性能 |
|------|-----------|---------|------|
| TopKMask | O(n) | K≤128 | ★★★★★ |
| TopK (RADIX) | O(n) | K≤4096, inner≤4096 | ★★★★ |
| TopK (MERGE) | O(n log n) | K≤4096, inner≤4096 | ★★★ |
| 多次ArgMax | O(K×n) | K≤10 | ★★ |
| Sort+切片 | O(n log n) | 大K或inner>4096 | ★★★ |

---

## 设计检查清单

- [ ] 识别K值大小
- [ ] 识别inner长度（是否≤4096）
- [ ] 选择正确的API
- [ ] 计算tmpBuffer空间
- [ ] 索引类型转换
- [ ] 输出顺序（升序/降序）
