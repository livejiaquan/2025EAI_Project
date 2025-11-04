### 檔案：`sparsity_train.ipynb` 作業報告

- **目的**：依 Network Slimming 進行稀疏訓練，透過對各層 BatchNorm 的縮放因子 γ 加上 L1 正則化，讓後續剪枝可依據 γ 的大小決定通道保留與否。

### 實作重點
- **資料與模型設定**：CIFAR-10、`ResNet50(num_classes=10)`，多卡 `DataParallel`；資料增強含 `Pad(4)`、`RandomCrop`、`RandomHorizontalFlip`、標準化。
- **超參數（題目指定 λ）**：`LAMBDA = 1e-4`，`epochs=40`，`Adam`；支援額外對比 `λ=1e-5` 與 `λ=0`（Baseline）。
- **稀疏正則化關鍵**：在反傳之後、更新參數之前，對每個 `nn.BatchNorm2d` 的 `weight(gamma)` 施加
  `m.weight.grad.data.add_(LAMBDA * torch.sign(m.weight.data))`，跳過 `affine=False` 的層。
- **訓練/測試流程**：每個 epoch 回傳並記錄 `train_acc/train_loss/test_acc/test_loss`，以便後續作圖與比對。
- **學習率排程**：於 50% 與 75% 訓練過程各降 10x。

### 與評分項對應
- **Plot 原始模型（稀疏訓練, λ=1e-4）之訓練/測試準確率曲線（5%）**
  - 產出圖檔：`figures/sparsity_training_accuracy.png`。
  - 記錄（執行結果）：Best Test Acc ≈ 91.15%，Final Test Acc ≈ 90.96%。

- **不同 λ 的 scaling factor 分布（5%）**
  - 產出圖檔：
    - `figures/scaling_factor_distribution_lambda_1e-4.png`
    - `figures/scaling_factor_distribution_lambda_1e-5.png`
    - `figures/scaling_factor_distribution_lambda_0.png`
  - 說明：λ 越大，γ 趨近 0 的比例越高，分布更「尖銳」，利於後續通道篩選。

### 關鍵設計說明（摘要）
- 稀疏訓練不改變模型結構，只改變 BN 的 γ 分布，為剪枝提供排序依據。
- 多卡訓練時以 `DataParallel` 包覆，不影響稀疏正則化邏輯。
- 保留完整歷史（list）以生成圖與儲存最佳權重（含 `best_test_acc`）。

### 問題與解法（與題目要求呼應）
- 問題：多卡訓練下 I/O 容易成為瓶頸。
  - 解法：提升 `num_workers`、啟用 `pin_memory`、調整 `prefetch_factor`。
- 問題：稀疏正則化學習率敏感。
  - 解法：保守學習率與階段性衰減，確保穩定收斂。


---

### 更細節：具體修改了哪些程式、為什麼這樣做、如何操作

- 修改點 1：`updateBN()` 新增 L1 稀疏正則化（對 BN 的 γ）
  - 位置：`sparsity_train.ipynb` → 定義 `updateBN()` 的 Cell。
  - 變更：對每個 `nn.BatchNorm2d`，在 `loss.backward()` 之後、`optimizer.step()` 之前，執行
    `m.weight.grad.data.add_(LAMBDA * torch.sign(m.weight.data))`。
  - 原因：Network Slimming 方法核心，讓 BN γ 變小，便於後續通道剪枝依 γ 大小排序。
  - 風險/對策：僅在 `m.weight`/`m.weight.grad` 非空時才加，且跳過 `affine=False` 的層，避免異常。

- 修改點 2：記錄訓練歷史並繪圖（acc/loss）
  - 位置：`train()`/`test()` 迴圈外追加 `train_acc_history/test_acc_history` 等 list 的 append 與保存。
  - 原因：滿足題目要求的曲線圖與最佳表現追蹤（評分依據）。

- 修改點 3：多 GPU 與資料載入優化
  - 位置：載入模型後以 `nn.DataParallel(model)` 包裝；`DataLoader` 設 `num_workers/pin_memory/prefetch_factor`。
  - 原因：充分利用雙 V100，降低 input pipeline 瓶頸，縮短訓練時間。

### 遇到的實際問題與解法（本檔案）
- 問題：λ 選太大會導致訓練卡頓或發散。
  - 解法：遵循題目 λ=1e-4，另外做 1e-5/0 的對照，並採階段降學習率。
- 問題：多 GPU 下偶發資料讀取變慢。
  - 解法：適度提高 `num_workers`（8）與 `prefetch_factor`（2），配合 `pin_memory=True`。


