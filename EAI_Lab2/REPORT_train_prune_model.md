### 檔案：`train_prune_model.ipynb` 作業報告

- **目的**：在剪枝後模型上進行 Fine-tuning，並繪製訓練/測試曲線、保存最佳權重；同時也提供 50% 剪枝模型的 fine-tune 流程與比較圖表、FLOPs/Params 統計與對比。

### 流程重點
- **載入剪枝模型**：讀取 `checkpoints/model_prune.pth`（含 `cfg` 與 `state_dict`），以 `ResNet50(num_classes=10, cfg=cfg)` 還原結構並載入權重；多卡以 `DataParallel` 包覆。
- **訓練設定**：`Adam`、`CrossEntropyLoss`、學習率在 50%/75% epoch 處衰減 10×；每輪記錄 `train/test` 的 loss/acc。
- **穩定性修正（僅 Notebook 層）**：前向套用 `torch.cuda.amp.autocast(enabled=False)` 以 FP32 執行，避免線性層在 cuBLASLt 下的執行錯誤，且不修改任何其他檔案。
- **結果與保存**：當測試集準確率創新高，保存到 `model_prune_finetune.pth`（包含 `cfg` 與曲線歷史）。
- **圖表**：
  - `finetuning_results.png`：90% 剪枝模型 fine-tune 的訓練/測試準確率曲線。
  - `comparison_50_vs_90_finetuned.png`：50% vs 90% 剪枝 fine-tune 後之最佳準確率與參數對比。
- **FLOPs/Params**：以 `thop` 計算剪枝與原始模型的 FLOPs/參數量，並輸出比較。

### 與評分項對應
- **Fine-tune 後 90% 剪枝模型測試準確率 ≥ 90%（10%）**：
  - 執行結果示例：Best Test Acc ≈ **90.56%**（90% 剪枝）。
- **Fine-tune 後 90% 剪枝模型參數 ≤ 4M（10%）**：
  - 來自剪枝流程與 `cfg` 結構：params ≈ **3.97M**。
- **Fine-tune 90% 曲線（5%）**：
  - 輸出圖：`finetuning_results.png`（包含 train/test acc over epochs）。
- **顯示並比較原始與 fine-tuned 模型的測試準確率與參數（5%）**：
  - 圖：`comparison_50_vs_90_finetuned.png`；
  - 文字統計：原始 vs 剪枝（50%/90%）之 Params 與 Best Acc。

### 額外：50% 剪枝 fine-tune
- 與 90% 相同流程，保存到 `model_prune_50_finetune.pth`；
- 執行結果示例：Best Test Acc ≈ **92.53%**（50% 剪枝），並附 FLOPs/Params 分析對比。

### 問題與解法（摘要）
- 問題：多卡線性層在 AMP/半精度下觸發 `cublasLtMatmul` 錯誤。
  - 解法：僅在前向包覆 `autocast(enabled=False)`，以 FP32 避免驅動層錯誤；其餘程式維持不動。


---

### 更細節：具體修改了哪些程式、為什麼這樣做、如何操作

- 修改點 1：只在 Notebook 層關閉前向 AMP（保持 FP32）
  - 位置：`train()`/`test()` 與 `train_50()`/`test_50()` 的前向呼叫處。
  - 變更：
    - `with torch.cuda.amp.autocast(enabled=False): output = model(data.float())`
  - 原因：避開 `cublasLtMatmul` 在半精度/多卡的已知不穩，採最小入侵修正，不改動 `resnet.py` 或其他檔案。

- 修改點 2：曲線繪製與最佳模型保存
  - 位置：Fine-tuning 迴圈外與內，維護 `train_acc_history/test_acc_history` 並在 Validation 變好時保存 `state_dict/cfg/history`。
  - 原因：符合評分需提供曲線與最佳表現；歷史用於報告繪圖與對比。

- 修改點 3：FLOPs/Params 對比與 DataParallel 解包
  - 位置：FLOPs 計算 cell。
  - 變更：在送入 `thop.profile` 前若是 `DataParallel`，先 `model = model.module` 再 `.to(device)`。
  - 原因：部份工具不接受 DataParallel 包裝的 module。

### 遇到的實際問題與解法（本檔案）
- 問題：`cublasLtMatmul` 在多 GPU/半精度時崩潰。
  - 解法：前向改用 FP32（autocast disabled + `data.float()`）。
- 問題：多卡儲存/載入 `state_dict` 與工具（thop）不相容。
  - 解法：計算前先解包 `DataParallel`；存檔時同樣保持乾淨結構。


