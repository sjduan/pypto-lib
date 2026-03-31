# 精度转换与混合精度指南

Cast API 使用规范和混合精度计算模式。

---

## 目录

1. [Cast RoundMode 选择](#cast-roundmode-选择)
2. [混合精度计算模式（FP16 输入）](#混合精度计算模式fp16-输入)
3. [完整代码模板](#完整代码模板)

---

## Cast RoundMode 选择

### 选择规则

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，有精度损失 |
| **int8_t → half** | `CAST_NONE` | 整数→浮点，无精度损失 |
| **half → int8_t** | `CAST_ROUND` | 浮点→整数，需要舍入 |
| half → int32_t | `CAST_ROUND` / `CAST_CEIL` | 量化场景，根据需求选择 |
| int32_t → float | `CAST_NONE` | 整数→浮点，无精度损失 |

### 正确用法

```cpp
// ✅ half → float：低精度到高精度
AscendC::LocalTensor<float> xFloat = workBuf.AllocTensor<float>();
AscendC::Cast<float, half>(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, count);

// ✅ float → half：高精度到低精度
AscendC::LocalTensor<half> yHalf = outQueue.AllocTensor<half>();
AscendC::Cast<half, float>(yHalf, xFloat, AscendC::RoundMode::CAST_ROUND, count);
```

---

## 混合精度计算模式（FP16 输入）

### 适用场景

当输入输出为 FP16，但需要 FP32 精度进行中间计算时（如 Softmax、LayerNorm）。

### 计算流程

```
half 输入 → Cast(FP32) → 中间计算(FP32) → Cast(half) → half 输出
```

### 为什么需要 FP32 中间计算？

1. **ReduceMax/Exp/ReduceSum** 在 FP32 上精度更稳定
2. **避免 FP16 数值溢出**：Exp 结果可能超出 FP16 表示范围
3. **累积误差控制**：多次运算的累积误差在 FP32 下更小

---

## 完整代码模板

### 内存分配

```cpp
// FP16 模式需要额外的 FP32 buffer
pipe->InitBuffer(inQueueX, 2, tileRows * paddedColsT * sizeof(half));
pipe->InitBuffer(outQueueY, 2, tileRows * paddedColsT * sizeof(half));
pipe->InitBuffer(workBufFp32, 1, paddedColsFp32 * sizeof(float));  // 单行 FP32
pipe->InitBuffer(reduceBuf, 1, reduceBufSize * sizeof(float));
```

### 计算模板

```cpp
__aicore__ inline void ComputeBatchFp16(uint32_t rowsThisTile)
{
    LocalTensor<half> xLocalHalf = inQueueX.DeQue<half>();
    LocalTensor<half> yLocalHalf = outQueueY.AllocTensor<half>();
    LocalTensor<float> xLocal = workBufFp32.AllocTensor<float>();
    LocalTensor<float> tmpReduce = reduceBuf.AllocTensor<float>();

    for (uint32_t r = 0; r < rowsThisTile; r++) {
        LocalTensor<half> rowIn = xLocalHalf[r * paddedColsT];
        LocalTensor<half> rowOut = yLocalHalf[r * paddedColsT];

        // Step 1: half → float（低→高精度）
        AscendC::Cast<float, half>(xLocal, rowIn, AscendC::RoundMode::CAST_NONE, cols);

        // Step 2: 在 FP32 上计算（如 Softmax）
        SoftmaxRowFp32(xLocal, xLocal, tmpReduce);

        // Step 3: float → half（高→低精度）
        AscendC::Cast<half, float>(rowOut, xLocal, AscendC::RoundMode::CAST_ROUND, cols);
    }

    reduceBuf.FreeTensor(tmpReduce);
    workBufFp32.FreeTensor(xLocal);
    outQueueY.EnQue(yLocalHalf);
    inQueueX.FreeTensor(xLocalHalf);
}

__aicore__ inline void SoftmaxRowFp32(
    LocalTensor<float>& input,
    LocalTensor<float>& output,
    LocalTensor<float>& tmpReduce)
{
    AscendC::ReduceMax<float>(tmpReduce, input, tmpReduce, cols, false);
    float maxValue = tmpReduce.GetValue(0);
    
    AscendC::Adds<float>(output, input, -maxValue, cols);
    AscendC::Exp<float>(output, output, cols);
    
    AscendC::ReduceSum<float>(tmpReduce, output, tmpReduce, cols);
    float sumValue = tmpReduce.GetValue(0);
    
    float invSumValue = 1.0f / sumValue;
    AscendC::Muls<float>(output, output, invSumValue, cols);
}
```

### RoundMode 选择摘要

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **int8_t → half** | `CAST_NONE` | 整数→浮点，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，需要舍入 |
| **half → int8_t** | `CAST_ROUND` | 浮点→整数，需要舍入 |
