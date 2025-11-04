### 檔案：`modelprune.ipynb` 作業報告

- **目的**：依稀疏訓練後 BN γ 的大小做結構化剪枝（channel pruning），建立 `cfg` 與 `cfg_mask`，構建剪枝後模型，並將原始模型權重對應複製到剪枝模型中，最後測試與存檔。

### 剪枝流程
1. **蒐集閾值**：
   - 遍歷所有 `nn.BatchNorm2d`（略過 `affine=False` 的 downsample BN），把 `|gamma|` 排序並依 `PRUNE_PERCENT` 取閾值。
2. **建立 `cfg` 與 `cfg_mask`**：
   - 依據閾值為每層 BN 產生 0/1 mask、統計保留通道數寫入 `cfg`。
   - 為避免破壞殘差維度，強制每個 bottleneck 的第三個 BN（BN3）輸出維度固定回到原始：`[256, 512, 1024, 2048]`，對應層數 `[3,4,6,3]`。
3. **構建新模型**：以 `ResNet50(cfg=cfg)` 建立剪枝模型並計算參數量。
   - 例：`PRUNE_PERCENT=0.90` 時，參數量約 `3.97M`（≤ 4M，符合題目要求）。
4. **權重複製（關鍵）**：
   - BN：依 `cfg_mask` 取 `weight/bias/running_mean/running_var` 的對應 index。
   - Conv：以前一層 `start_mask` 與當前層 `end_mask` 的非零 index 切片 `out×in` 權重。
   - Downsample 分支：輸入用 block 開頭的 `block_input_mask`，輸出用該 block 的 `block_output_mask`（即 BN3 的 mask）。
   - Linear：依最後一層特徵向量的通道 mask 選取全連接權重的輸入維度，bias 直拷。
5. **測試與儲存**：以 CIFAR-10 測試集評估剪枝後（未 fine-tune）模型，並存成 `checkpoints/model_prune.pth`；亦提供 `PRUNE_PERCENT=0.5` 的流程與存檔 `model_prune_50.pth`。

### 與評分項對應
- **90% 剪枝後參數 ≤ 4M（10%）**：`~3.97M`（輸出即時印出）。
- **剪枝後測試準確率（50% 與 90%）（各 5%）**：在 Notebook 末段呼叫 `test(model)` 直接印出 Test Acc（執行時顯示）。
- **「如何複製原始權重到剪枝模型」說明（5%）**：如上「權重複製（關鍵）」段落所述，對 BN/Conv/Downsample/Linear 逐層以 mask 對齊。

### 重要細節
- 為避免殘差維度錯配，BN3 輸出維度固定到原版，每個 bottleneck block 的輸出通道與同層其他 block 一致，使 identity 加法合法。
- Downsample 的 BN 設為 `affine=False`，因此在複製時只需處理其 `running_mean/var`（依 block 輸出 mask）。


---

### 更細節：具體修改了哪些程式、為什麼這樣做、如何操作

- 修改點 1：建立 `cfg`/`cfg_mask` 的流程
  - 作法：
    1) 走訪有 `weight` 的 `BatchNorm2d`，取 `|gamma|` 並排序，依 `PRUNE_PERCENT` 取得 `threshold`。
    2) 以 `mask = (|gamma| > threshold)` 建立 0/1 遮罩；若全 0，保底保留 top-k（最多 3 個）。
    3) 將遮罩之和寫入 `cfg`，遮罩本體存入 `cfg_mask`。
  - 原因：以 γ 的稀疏性決定通道存留，並避免極端情況（全 0）造成拓撲斷裂。

- 修改點 2：固定每個 bottleneck 的 BN3 輸出通道 = 原始通道（Layer1~4: 256/512/1024/2048）
  - 作法：按 `[3,4,6,3]` block 結構計算 BN3 在 `cfg` 的索引，強制覆寫數值，並將對應 `cfg_mask` 設為 all-ones。
  - 原因：確保同一層內各 block 的輸出維度一致，殘差加法合法；同時維持後續層輸入通道一致。

- 修改點 3：權重對齊複製（BN/Conv/Downsample/Linear）
  - BN：依 `cfg_mask` 的非零 index 對 `weight/bias/running_mean/running_var` 進行 `index_select`。
  - Conv：用 `start_mask` 與 `end_mask` 的非零 index 對 `out×in` 權重做二次 `index_select`（先 in 後 out）。
  - Downsample：
    - 輸入 index 用 `block_input_mask`（block 開頭的通道），輸出 index 用 `block_output_mask`（對應 BN3）。
    - 其 BN 為 `affine=False`，僅複製 `running_mean/var` 的對應索引。
  - Linear：依最後一層 `start_mask` 的非零 index 選取 `weight` 的輸入維度，`bias` 直接拷貝。
  - 原因：確保拓撲與張量形狀完全對齊，避免 runtime 形狀錯誤。

### 遇到的實際問題與解法（本檔案）
- 問題：某些 BN 可能全部被裁掉（mask 全 0）。
  - 解法：保底保留 top-k（最多 3 個）以維持資訊流通。
- 問題：殘差維度對不齊導致相加報錯。
  - 解法：固定 BN3 輸出通道為原始數量，並於 `_make_layer` 內側重同層一致（對應 `resnet.py` 的 assert）。
- 問題：`DataParallel` 產生 `state_dict` 多了 `module.` prefix。
  - 解法：儲存/載入前以 `OrderedDict` 移除 `module.` 前綴，對齊鍵名。


