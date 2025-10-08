# 2025EAI_Project - Lab1

這個 repository 包含了 2025 年深度學習課程的 Lab1 作業，主要實現 ResNet18 模型在 CIFAR-10 數據集上的圖像分類任務。

## 項目結構

```
2025EAI_Project/
├── Lab1 Material/
│   ├── Lab1-Task1.ipynb           # Task1: 基礎 ResNet18 實現
│   ├── Lab1-Task1-GPU.ipynb       # Task1 GPU 版本
│   ├── Lab1-Task1-Report.md       # Task1 報告
│   ├── Lab1-Task2.ipynb           # Task2: 優化版 ResNet18
│   ├── Lab1-Task2-Report.md       # Task2 報告
│   └── best_resnet18*.pth         # 訓練好的模型文件（本地）
├── Lab1 Material_backup/
│   ├── Lab1-Task1.ipynb           # 原始模板文件
│   └── Lab1-Task2.ipynb           # 原始模板文件
└── README.md
```

## 主要特點

### Task1: 基礎實現
- 完整的 ResNet18 架構實現
- CIFAR-10 數據集分類
- 基礎的訓練和驗證流程

### Task2: 優化實現
- **數據增強**：隨機水平翻轉、旋轉、顏色調整、隨機裁切、隨機平移、隨機遮蔽
- **權重初始化**：Kaiming Normal 初始化，適合 ReLU 激活函數
- **混合精度訓練**：使用 Automatic Mixed Precision (AMP) 提升訓練效率
- **學習率調度**：Warmup + Cosine Annealing 策略
- **早停機制**：防止過擬合
- **正則化**：Label Smoothing 和 Weight Decay

## 實驗結果

### Task2 優化版本性能
- **測試準確率**: 94.83%
- **最佳驗證準確率**: 91.78%
- **相比基準線提升**: 7.80%
- **基準線準確率**: 83.98%

### 模型參數
- **參數量**: ~11.2M
- **FLOPs**: ~557M
- **訓練時間**: 約100個epoch（包含早停）

## 技術細節

### 權重初始化
```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
```

### 學習率調度
- **Warmup**: 前5個epoch線性增加學習率
- **Cosine Annealing**: 餘下epoch使用餘弦退火

### 數據增強策略
- 隨機水平翻轉 (p=0.5)
- 隨機旋轉 (±15度)
- 顏色抖動 (亮度、對比度、飽和度、色調)
- 隨機裁切 (32x32, padding=4)
- 隨機平移 (±10%)
- 隨機遮蔽 (p=0.3)

## 環境要求

- Python 3.9+
- PyTorch 2.5.1+
- CUDA 支援（可選，用於GPU加速）
- 其他依賴：
  - torchvision
  - numpy
  - matplotlib
  - thop (用於計算FLOPs)
  - torchsummary

## 使用方法

1. 克隆 repository
2. 安裝依賴包
3. 運行 Jupyter notebook
4. 執行訓練和測試

```bash
pip install torch torchvision thop torchsummary
jupyter notebook
```

## 注意事項

- 模型文件 (*.pth) 由於大小限制未上傳到 GitHub
- 數據集文件夾 (data/) 由於大小限制未上傳到 GitHub
- 如需完整文件，請聯繫作者

## 作者

livejiaquan

## 許可證

此項目僅用於學術目的。
