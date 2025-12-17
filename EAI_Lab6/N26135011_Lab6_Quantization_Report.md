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

### 1. What's the difference between SmoothQuant and the method in Lab3? 10%

Lab3 做的是 PTQ 和 QAT，SmoothQuant 是一種改進版的 PTQ。

**Lab3 的做法：**
- PTQ：訓練完直接量化，用 calibration data 收集統計值算 scale/zero_point
- QAT：訓練時插入 FakeQuantize 讓模型學習適應量化誤差
- 量化對象是 CNN (ResNet-50)，activation 分布比較平均

**SmoothQuant 的做法：**
- 也是 PTQ，不需要重新訓練
- 但多了一步 smoothing：把 activation 的 outlier 轉移到 weight
- 公式：`Y = (X/s) × (W×s)`，結果不變但分布改變了

**為什麼需要 SmoothQuant：**

Transformer 的 activation 有嚴重的 outlier 問題，某些 channel 的值可能比其他大 100 倍。如果直接用 Lab3 的 PTQ，per-token 量化會因為這些 outlier 導致 scale 太大，其他正常值的精度就爆掉了。

SmoothQuant 透過數學等價轉換，把 activation 壓平（除以 s），讓 outlier 轉移到 weight（乘以 s）。Weight 是靜態的可以用 per-channel 量化處理，這樣兩邊都好量化了。

### 2. When applying SmoothQuant, where do activation values get divided by the smooth factor? 10%

在 LayerNorm 層。

做法是修改 LayerNorm 的參數：`ln.weight /= s`，`ln.bias /= s`。這樣 LayerNorm 的輸出就等於原本的輸出除以 s，不用在 runtime 額外做除法。

具體來說是 smooth 這兩組：
- norm1 → qkv（attention 的輸入）
- norm2 → fc1（FFN 的輸入）

fc2 前面沒有 LayerNorm 所以沒辦法融合，就不做。

### 3. How is the smooth factor being calculated? 10%

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

- $X_j$：第 j 個 input channel 的 activation max（跑 calibration data 收集）
- $W_j$：第 j 個 input channel 的 weight max
- $\alpha$：平衡參數，預設 0.5

α = 0.5 代表 activation 和 weight 各分擔一半的量化難度。如果 activation outlier 很嚴重可以調高到 0.75。

### 4. What's the difference between ViT-S and CNN models when doing quantization? 10%

**Activation 分布不同：**
- CNN：activation 分布相對均勻，各 channel 差異不大
- ViT：attention 計算會產生 outlier，某些 channel 數值特別大（可能大 100 倍）

**量化難度：**
- CNN：直接用 Lab3 的 per-tensor 或 per-channel 量化就夠了
- ViT：需要 SmoothQuant 這類方法先處理 outlier，不然精度掉很多

**原因分析：**

ViT 的 self-attention 會計算 softmax(QK^T/√d)，這個過程會放大某些 channel 的數值。而且 LayerNorm 的輸出特性是 channel 間差異大、channel 內差異小，這跟 CNN 的 BatchNorm 不一樣。

所以 ViT 量化的關鍵是處理 activation outlier，SmoothQuant 就是針對這個問題設計的。

### 5. What's your observation on the visualization of weight and activation values distribution? 10%

從 3D 圖可以看到：

**Weight：**
- 原本比較平
- smooth 之後變尖了（因為乘了 s）

**Activation：**
- 原本有很多尖的 outlier（max 6.59）
- smooth 之後壓平了（max 1.33），降了快 5 倍

這就是 SmoothQuant 的效果：把 activation 的 outlier 問題轉移到 weight 上。Activation 變好量化了，weight 雖然變尖但因為是 per-channel 量化所以還是可以處理。

最後精度只掉 0.07%，證明這個方法有效。
