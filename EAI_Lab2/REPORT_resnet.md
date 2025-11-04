### 檔案：`models/resnet.py` 作業報告

- **目的**：實作可支援通道剪枝設定（`cfg`）的 ResNet（Bottleneck 版），讓通道數可依剪枝結果動態生成，同時維持殘差維度一致。

### 結構設計
- **`Bottleneck`**：`1×1 → 3×3 → 1×1`，以 `expansion=4`。每層帶 `BatchNorm2d` 與 `ReLU(inplace=True)`。
- **Stem**：`conv1(3 → cfg[0])`、`bn1`、`relu`、`maxpool`。
- **Layer1~4**：由 `_make_layer` 動態構建；每個 layer 內各 block 的三段卷積通道皆從 `cfg` 讀取。
- **下採樣（shortcut）**：如 `stride!=1` 或輸入通道改變，用 `1×1 conv` + `BatchNorm2d(affine=False)` 對齊維度。
- **Classifier**：`AdaptiveAvgPool2d(1,1)` + `Linear(self.inplanes → num_classes)`。

### `cfg` 與維度一致性
- `self.inplanes` 追蹤當前通道數；每個 layer 第一個 block 可能 stride/降維，最後 `self.inplanes` 更新為該 layer 的輸出通道。
- 透過 `layer_out_channels = out_channels[2]` 強制該 layer 內所有 block 的最終輸出一致（assert 檢查），確保殘差加法合法。
- 下採樣分支的 BN 設為 `affine=False`，避免引入額外可學參數，有助於與題目參數量對齊（原始約 23,513,162）。

### Forward 流程
`conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → flatten → fc`。

### 與評分項對應
- **說明如何修改 `resnet.py`（5%）**：
  - 以 `cfg` 驅動各層通道數，`_make_layer` 內以斷言保證同層 block 輸出一致；
  - 下採樣分支使用 `affine=False` 的 BN 以匹配參數量；
  - `AdaptiveAvgPool2d` + `Linear` 形成分類頭；
  - `self.inplanes` 動態更新確保後續層輸入維度正確。

### 為何需「固定各 bottleneck 的輸入/輸出通道為剪枝前的通道數」？（5%）
- **原因**：殘差相加要求主分支與捷徑分支張量形狀完全一致。若各 block 的最終輸出（BN3）通道彼此不一致，將導致：
  - 形狀不匹配無法相加（runtime error），或
  - 被迫插入額外投影/填充，破壞原模型設計與參數對齊。
- **作法**：固定 BN3 為原始通道（Layer1/2/3/4 分別為 256/512/1024/2048），確保：
  - 同層所有 block 的輸出維度一致；
  - 下游層與 downsample 的維度對齊；
  - 參數量可精準計算與對比。


---

### 更細節：具體修改了哪些程式、為什麼這樣做、如何操作

- 修改點 1：以 `cfg` 驅動通道數
  - 位置：`__init__` 與 `_make_layer`。
  - 變更：`conv1` 的輸出通道用 `cfg[0]`；每個 bottleneck 的三段卷積通道分別取 `cfg` 連續三項；`self.inplanes` 在每個 layer 的第一個 block 後更新為該 layer 輸出通道。
  - 原因：支援剪枝後的變寬/變窄結構，保持 forward 流程不變。

- 修改點 2：殘差維度一致性的斷言
  - 位置：`_make_layer` 內 for 迴圈
  - 變更：對 layer 內每個 block 的 `out_channels[2]` 斷言必須等於 `layer_out_channels`。
  - 原因：避免同層 block 之間輸出通道不一致導致殘差相加錯誤，提早在建構期檢出錯誤 `cfg`。

- 修改點 3：Downsample 的 BN 設為 `affine=False`
  - 位置：`downsample = nn.Sequential(Conv1x1, BatchNorm2d(affine=False))`
  - 原因：題目參數量對齊需求（原始 ~23,513,162），避免在捷徑 BN 上引入可學參數數量偏差。

### 遇到的實際問題與解法（本檔案）
- 問題：使用不一致的 `cfg` 造成 forward 維度錯誤。
  - 解法：在 `_make_layer` 中加入 assert；剪枝端（`modelprune.ipynb`）固定 BN3 輸出通道，雙重確保。
- 問題：首層 `conv1` 的通道數與後續 `layer1` 匹配。
  - 解法：用 `cfg[0]` 建立 `conv1`，並以 `self.inplanes = cfg[0]` 作為後續層的起點。


