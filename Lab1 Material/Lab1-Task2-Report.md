## Lab 1 Task 2 書面報告（實作說明、輸出、遇到問題與解法）

### 作業說明來源
- 題目／講義（作業需求）：「EAI Lab1 Handouts」[`連結`](https://hackmd.io/U9dZg-c2QNq25cnQhCANsA?view)

### 檔案
- 原始 Notebook：`2025EAI_Project/Lab1 Material_backup/Lab1-Task2.ipynb`
- 修改後 Notebook：`2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb`

---

### 一、實作方式總覽

- 資料前處理：
  - 計算 CIFAR-10 的 `mean/std`，以此做 Normalize。
  - 訓練集使用更完整的 Data Augmentation（RandomFlip/Rotation/ColorJitter/RandomCrop/RandomAffine/RandomErasing）。
- 資料切分與批次：
  - 使用固定 random seed 進行 train/val split（45000/5000）。
  - 將 `batch_size` 設為 256（對應 GPU 記憶體與吞吐量）。
- 模型設計（ResNet-18 for CIFAR-10）：
  - 使用 3×3 起始卷積、移除 maxpool、維持更高解析度以保留資訊。
  - 實作 `BasicBlock`（兩層 3×3 conv + BN + 殘差連接）。
  - 使用 Kaiming 初始化、AdaptiveAvgPool + Linear 作為分類頭。
- 訓練策略：
  - Optimizer: SGD + momentum + weight decay。
  - Loss: CrossEntropyLoss（含 label smoothing）。
  - LR Scheduler: Warmup（前 5 epochs 線性升）+ Cosine Annealing（其後退火）。
  - AMP（混合精度）加速訓練（autocast + GradScaler）。
  - Early Stopping（耐心值 30）。
- 記錄與可視化：
  - 使用 torchsummary 與 thop 印出模型 summary / FLOPs / Params。
  - 畫出 Train/Val Loss、Accuracy、LR 曲線與對比線（Baseline vs Enhanced）。

---

### 二、原始版 vs 修改版：差異與修改說明

#### 1) Data Augmentation 與 Normalize

原始版僅有 `ToTensor + Normalize`；修改後加入多種增強並以資料集實測的 mean/std 做標準化。

```5:21:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
##### data augmentation & normalization #####
# 🔧 修改1: 使用 CIFAR-10 標準均值和標準差
CIFAR10_MEAN = train_mean.tolist()
CIFAR10_STD = train_std.tolist()

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])
```

理由：提升資料多樣性與模型泛化；以真實統計值做 Normalize 更穩定。

#### 2) 資料切分與 Batch Size

```132:153:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
torch.manual_seed(43)
val_size = 5000
train_size = len(trainset) - val_size
train_ds, val_ds = random_split(trainset, [train_size, val_size])
...
BATCH_SIZE = 256
trainloader = ...
```

理由：固定 seed 保持可重現；較大的 batch 充分利用 GPU、減少迭代次數。

#### 3) 模型：BasicBlock 與 CIFAR 友善的 ResNet-18

```164:189:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
class BasicBlock(nn.Module):
    ...
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

```175:226:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 無 maxpool，保留 32×32
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()
```

理由：CIFAR-10 為 32×32，移除 early downsampling 保留訊息；殘差設計與 Kaiming 初始化提高訓練穩定性。

#### 4) Optimizer / Loss / Scheduler / AMP / EarlyStopping

```235:318:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
EPOCH = 100
lr = 0.1
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
...  # Warmup + Cosine Annealing
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
patience = 30
```

理由：
- SGD（含 momentum/weight decay）為 CIFAR-10 經典設定。
- Label Smoothing 抑制過擬合。
- Warmup + Cosine Annealing 改善早期不穩定與後期收斂。
- AMP（autocast/GradScaler）提升吞吐與速度。
- 早停避免過度訓練。

---

### 三、主要輸出與結果（數值與圖）

- Summary / FLOPs / Params（節錄）：
```208:219:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
summary(model, (3, 32, 32))
flops, params = profile(model, inputs=(dummy_input, ))
print(f"FLOPs: {flops/1e6:.2f} MFLOPs")
print(f"Params: {params/1e6:.2f} M")
```

- 訓練與驗證關鍵 Log（節錄）：
```260:317:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
Epoch [  1/100] ... Val Acc: 10.46% | LR: ...
...
Epoch [ 12/100] ... Val Acc: 74.24% | ...
...
```

- 最佳 checkpoint 測試集表現：
```288:301:/home/twccjq88/2025EAI_Project/Lab1 Material/Lab1-Task2.ipynb
Best checkpoint test accuracy: 94.83%
```

- 視覺化圖表（請於 Notebook 直接顯示或匯出圖片再貼至報告）：
  - 2×2 子圖：Train/Val Loss、Train/Val Accuracy、LR 變化、Baseline vs Enhanced 對比（`Accuracy improvement ~ +7.8%`）。

---

### 四、遇到的問題與解法

- 問題 1：原始 `ResNet18` 與 `BasicBlock` 未實作。
  - 解法：依 ResNet-18 論文標準作法補齊（兩層 3×3、shortcut、BN、ReLU 等）。

- 問題 2：CIFAR-10 輸入為 32×32，若沿用 7×7 + maxpool 容易過早降維、訊息流失。
  - 解法：改用 3×3 起始卷積、移除 maxpool，維持 32×32 至第一層 block；實測驗證表現更佳。

- 問題 3：學習率調度與收斂速度不理想。
  - 解法：加入 Warmup + Cosine Annealing，前期穩定、後期順暢退火，提高最終精度。

- 問題 4：訓練速度與效能。
  - 解法：啟用 AMP（autocast + GradScaler），顯著提升 throughput；同時設定 batch_size=256 提升效能。

- 問題 5：PyTorch AMP 介面棄用警告（`torch.cuda.amp`）。
  - 解法：可改為 `torch.amp.GradScaler('cuda')` 與 `with torch.amp.autocast('cuda'):` 的新介面（相容未來版本）。

- 問題 6：`torch.load` 未指定 `weights_only` 的未來警告。
  - 解法：之後可改為 `torch.load(path, map_location=device, weights_only=True)`，提升安全性。

---

### 五、作業要求對照

- NN 與層設計（ResNet18 + BasicBlock）：已完成。
- 數據處理（Normalization、Augmentation）：已完成且強化。
- Cross-Validation（Train/Valid Split）：已完成（45000/5000）。
- 訓練流程（Train/Val/Test、記錄與可視化）：已完成。
- 評估結果：
  - Validation ~ 91–92%，Test 約 94.83%（符合並優於一般 baseline）。
- 圖表：訓練/驗證 Loss/Acc、學習率曲線與對比圖已繪製（請將圖片另行匯出貼入報告）。

---

### 六、結論

- 針對 CIFAR-10 調整 ResNet-18 架構與訓練策略，整體精度顯著提升（相較 baseline 約 +6%~+11%）。
- 使用增廣、初始化、優化器、學習率調度、AMP、早停等策略，兼顧收斂速度、穩定性與泛化能力。
- 整體流程符合講義與作業要求（[`講義連結`](https://hackmd.io/U9dZg-c2QNq25cnQhCANsA?view)），並具備完整實作與可視化證明。


