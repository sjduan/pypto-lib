# Sort 算子详细设计

## API 选择

| API | 支持索引 | 排序方向 | 长度限制 | 适用场景 |
|-----|---------|---------|---------|---------|
| **Sort** | ✅ | 升序 | 通用 | 推荐使用 |
| **BitonicSort** | ✅ | 升序 | 2^n | 性能更优 |
| **RadixSort** | ❌ | 升序 | 整数 | 仅值排序 |

---

## Sort API

### API原型

```cpp
template <typename T, bool withIndices = true>
__aicore__ inline void Sort(
    const LocalTensor<T>& dstVal,      // 排序后的值
    const LocalTensor<T>& dstIdx,      // 排序后的索引
    const LocalTensor<T>& src,         // 输入
    const LocalTensor<T>& tmpBuffer,   // 临时空间
    const SortParams& params
);

struct SortParams {
    uint32_t totalLength;  // 排序长度
    int32_t sortAxis;      // 排序轴（-1=最后一轴）
};
```

### 使用示例

```cpp
AscendC::LocalTensor<float> srcLocal = inQueue.AllocTensor<float>();
AscendC::LocalTensor<float> valLocal = outQueueVal.AllocTensor<float>();
AscendC::LocalTensor<float> idxLocal = outQueueIdx.AllocTensor<float>();
AscendC::LocalTensor<float> tmpBuffer = tmpQueue.AllocTensor<float>();

AscendC::SortParams params;
params.totalLength = count;
params.sortAxis = -1;

// withIndices=true 同时返回值和索引
AscendC::Sort<float, true>(valLocal, idxLocal, srcLocal, tmpBuffer, params);

// 输出（升序）：
// valLocal[0..count-1] = 排序后的值（升序）
// idxLocal[0..count-1] = 对应的原始索引（需类型转换）
for (int i = 0; i < count; i++) {
    float val = valLocal.GetValue(i);
    float idxRaw = idxLocal.GetValue(i);
    uint32_t idx = *reinterpret_cast<uint32_t*>(&idxRaw);
}
```

---

## BitonicSort API（2^n长度）

### 使用场景
- 长度 = 2^n（如 1024, 2048, 4096）
- 性能优于通用Sort

### API原型

```cpp
template <typename T, bool withIndices = true>
__aicore__ inline void BitonicSort(
    const LocalTensor<T>& dstVal,
    const LocalTensor<T>& dstIdx,
    const LocalTensor<T>& src,
    const LocalTensor<T>& tmpBuffer,
    const BitonicSortParams& params
);

struct BitonicSortParams {
    uint32_t totalLength;  // 必须是 2^n
    bool isAscending;      // true=升序
};
```

---

## 降序排序

Sort只支持升序，降序需要：

### 方法1：取反→排序→取反

```cpp
// 取反
AscendC::Muls(negLocal, srcLocal, -1.0f, count);

// 排序（升序）
AscendC::Sort<float, true>(valLocal, idxLocal, negLocal, tmpBuffer, params);

// 再取反（恢复）
AscendC::Muls(valLocal, valLocal, -1.0f, count);
```

### 方法2：排序后反转

```cpp
AscendC::Sort<float, true>(valLocal, idxLocal, srcLocal, tmpBuffer, params);

// 反转数组（手动实现或循环）
for (int i = 0; i < count / 2; i++) {
    // swap valLocal[i] and valLocal[count-1-i]
    // swap idxLocal[i] and idxLocal[count-1-i]
}
```

---

## Argsort（只返回索引）

如果只需要索引，不需要值：

```cpp
// 方法1：使用Sort但丢弃valLocal
AscendC::Sort<float, true>(valLocal, idxLocal, srcLocal, tmpBuffer, params);
// 只使用 idxLocal

// 方法2：Sort<false>（如果API支持）
AscendC::Sort<float, false>(valLocal, srcLocal, tmpBuffer, params);
// 注意：withIndices=false 可能不支持，需查文档
```

---

## Buffer规划

| Buffer | 用途 | 大小 |
|--------|------|------|
| inQueue | 输入 | count×sizeof(T) |
| outQueueVal | 排序值 | count×sizeof(T) |
| outQueueIdx | 排序索引 | count×sizeof(T) |
| tmpQueue | 临时空间 | GetSortTmpSize() |

**tmpBuffer计算**：
```cpp
uint32_t tmpSize = AscendC::GetSortTmpSize<T>(count, withIndices);
```

---

## 性能对比

| API | 时间复杂度 | 空间复杂度 | 适用长度 |
|-----|-----------|-----------|---------|
| Sort | O(n log n) | O(n) | 通用 |
| BitonicSort | O(n log² n) | O(log n) | 2^n |
| RadixSort | O(n × k) | O(n) | 整数 |

---

## 约束

| 约束 | 说明 |
|------|------|
| 排序方向 | 只支持升序，降序需转换 |
| 数据类型 | half/float/int32 |
| 索引类型 | 与dst相同，需类型转换 |
| 稳定性 | 不稳定（相等元素顺序不确定） |

---

## 设计检查清单

- [ ] 确定排序方向（升序/降序）
- [ ] 长度是否为2^n（选择BitonicSort）
- [ ] 是否需要索引（withIndices参数）
- [ ] tmpBuffer空间足够
- [ ] 索引类型转换
