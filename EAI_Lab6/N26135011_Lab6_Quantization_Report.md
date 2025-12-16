# Lab 6 - Transformer Quantization Report

## 實驗結果

### Benchmark

| Method | Accuracy | Latency | 
|--------|----------|---------|
| FP32 | 98.33% | 0.0058s |
| W8A8 SmoothQuant | 98.26% | 0.0108s |

Accuracy 只掉了 0.07%，記憶體從 82.66 MB 降到 20.67 MB（4倍壓縮）。

### Activation 變化

| | Before | After |
|--|--------|-------|
| Max 值 | 6.59 | 1.33 |

---

## 問題回答

### 1. SmoothQuant 怎麼做的

SmoothQuant 的想法是把 activation 的 outlier 轉移到 weight 上面。

原本 activation 有些 channel 數值特別大（outlier），這樣量化會很不準。但 weight 通常比較平均，好量化。所以用一個 smoothing factor `s` 來調整：

```
Y = X × W  →  Y = (X/s) × (W×s)
```

結果一樣，但 activation 變平了，weight 變尖了。因為 weight 是固定的可以事先處理，activation 每次都不同，所以這樣做對量化比較有利。

實作就是：
1. 跑 calibration data 收集 activation 的 max
2. 算 smoothing factor
3. LayerNorm 除以 s，Linear weight 乘以 s
4. 最後做 INT8 量化

### 2. 為什麼速度反而變慢

這個是正常的。

因為這次做的是 fake quantization，實際上還是用 FP32 算，只是多了 quantize/dequantize 的步驟，所以反而慢。

要真的加速要用 TensorRT 之類的框架部署到支援 INT8 的硬體上才行。

### 3. 3D 圖代表什麼

Weight 那張圖：
- 原本比較平
- smooth 之後變尖了（因為乘了 s）

Activation 那張圖：
- 原本有很多尖的 outlier
- smooth 之後壓平了（因為除了 s）

這就是 SmoothQuant 的效果，把 activation 的問題轉到 weight 上。

### 4. 為什麼只 smooth QKV 和 FC1

因為 SmoothQuant 要把 smooth 操作融合到前面的 LayerNorm：
- norm1 → qkv：可以 smooth
- norm2 → fc1：可以 smooth  
- fc1 → GELU → fc2：fc2 前面沒有 LayerNorm，沒辦法融合

所以 fc2 不做。

### 5. 精度分析

FP32: 98.33%  
W8A8: 98.26%  
掉了 0.07%

這個結果很好，因為 SmoothQuant 把 activation max 從 6.59 壓到 1.33，量化誤差變小很多。
