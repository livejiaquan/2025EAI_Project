# EAI Lab5 作業說明 - 壓縮模型的網頁部署

## 作業概述

本作業的目標是將 PyTorch 訓練好的 ResNet18 模型轉換為 ONNX 格式，進行 INT8 靜態量化，並使用 Gradio 建立網頁介面進行推論展示。

模型資訊：
- 訓練資料集：CIFAR-10
- 輸入大小：32×32 RGB 圖片
- 輸出類別（10 類）：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 現有檔案說明

```
EAI_Lab5/
├── EAI_lab5.ipynb      # 作業模板（需要填寫 TODO 部分）
├── best_model.pth      # 老師提供的預訓練權重
└── resnet18.py         # ResNet18 完整架構（參考用）
```

---

## 環境設定（重要）

**請使用 conda 建立獨立環境，不要直接在本機安裝套件。**

```bash
# 建立新環境
conda create -n eai_lab5 python=3.10 -y
conda activate eai_lab5

# 安裝必要套件
pip install torch torchvision torchaudio
pip install onnx onnxscript onnxruntime onnxruntime-tools
pip install gradio
pip install pillow numpy
```

---

## 程式碼撰寫原則（重要）

**請遵守以下原則：**

1. **保留原本結構**：`EAI_lab5.ipynb` 是老師提供的模板，只需要填寫 TODO 部分，不要刪除原本的程式碼、註解或 cell 結構

2. **保留原有註解**：老師寫的註解（如 `# TODO`、`# 提醒`）都要保留，不要刪除

3. **適當加入自己的註解**：在填寫的程式碼中加入適當的中文註解，說明每段程式碼的用途，展現理解程度

4. **只完善程式碼**：把 TODO 的部分補完即可，不要重構或大幅改動原本的架構

5. **不要自動執行**：完成程式碼後讓使用者自己手動執行，不要自動 run notebook

---

## 繳交檔案結構

最終需繳交以下檔案（學號請自行替換）：

```
學號_lab5.zip
├── 學號.ipynb           # 完成的主程式 notebook
├── 學號_FP32.onnx       # FP32 精度的 ONNX 模型
├── 學號_FP32.onnx.data  # 若模型過大會產生此檔案（可選）
├── 學號_INT8.onnx       # INT8 量化後的 ONNX 模型
└── 學號_report.pdf      # 報告
```

---

## Topic 1: PyTorch To ONNX（40%）

### 1.1 建立 ResNet18 模型架構（5%）

參考 `resnet18.py` 檔案，在 notebook 中完成 `BasicBlock` 和 `ResNet18` 類別的定義。

`resnet18.py` 中的架構：
- `ResBlock`：殘差區塊，包含兩個 3×3 卷積和 shortcut 連接
- `ResNet`：完整網路，包含 conv1 + 4 個 layer + fc 層

需要將 `ResBlock` 改名為 `BasicBlock`，`ResNet` 改名為 `ResNet18`，確保與 notebook 模板中的命名一致。

### 1.2 載入權重（5%）

使用 `model.load_state_dict()` 載入 `best_model.pth`。

```python
state = torch.load("best_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state)
model.eval()
```

### 1.3 匯出 ONNX（10%）

使用 `torch.onnx.export()` 匯出模型。

```python
torch.onnx.export(
    model,                          # 模型
    dummy_input,                    # 範例輸入 (1, 3, 32, 32)
    "學號_FP32.onnx",               # 輸出檔名
    input_names=["input"],          # 輸入節點名稱
    output_names=["output"],        # 輸出節點名稱
    opset_version=13                # ONNX opset 版本
)
```

### 1.4 靜態量化為 INT8（10%）

使用 `onnxruntime.quantization.quantize_static()` 進行靜態量化。

模板中已提供 `CIFARLikeCalibReader` 作為校準資料讀取器，使用隨機資料進行校準。

```python
from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod

quantize_static(
    model_input="學號_FP32.onnx",
    model_output="學號_INT8.onnx",
    calibration_data_reader=reader,
    quant_format="QOperator",
    per_channel=True,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax
)
```

### 1.5 建立 ONNX Runtime Session（10%）

完成 `build_session()` 函式。

```python
def build_session(model_path, providers):
    return ort.InferenceSession(model_path, providers=providers)
```

---

## Topic 2: Gradio 部署（35%）

### 2.1 圖片前處理函式（10%）

完成 `preprocess()` 函式，將 PIL Image 轉換為模型需要的格式。

處理步驟：
1. 轉換為 RGB 並 resize 到 32×32
2. 轉為 numpy array 並除以 255（normalize 到 0-1）
3. 減去 CIFAR-10 均值，除以標準差
4. 從 HWC 轉換為 CHW 格式
5. 增加 batch 維度

```python
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - CIFAR10_MEAN) / CIFAR10_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr[None, ...]         # 增加 batch 維度 (1, 3, 32, 32)
    return arr
```

CIFAR-10 正規化參數（模板中已提供）：
- CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
- CIFAR10_STD = [0.2470, 0.2435, 0.2616]

### 2.2 FP32 vs INT8 比較函式（10%）

完成 `compare_fp32_int8()` 函式，比較兩個模型的：
- 推論時間
- 輸出結果
- 可自行加入其他比較項目

需要測量推論時間並計算 speedup：

```python
# FP32 推論
t0 = time.time()
out_fp32 = sess_fp32.run([out_fp32], {in_fp32: x})[0]
fp32_ms = (time.time() - t0) * 1000

# INT8 推論
t0 = time.time()
out_int8 = sess_int8.run([out_int8], {in_int8: x})[0]
int8_ms = (time.time() - t0) * 1000
```

### 2.3 建立 Gradio UI 介面（10%）

完成 `gr.Interface()` 的設定。

```python
demo = gr.Interface(
    fn=compare_fp32_int8,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(label="FP32 Top-3"),
        gr.Label(label="INT8 Top-3"),
        gr.Textbox(label="Performance Comparison")
    ],
    title="CIFAR-10 Image Classifier - FP32 vs INT8",
    description="Upload an image to compare FP32 and INT8 model inference"
)
```

### 2.4 產生公開網頁（5%）

使用 `share=True` 啟動 Gradio 並產生公開 URL。

```python
if __name__ == "__main__":
    demo.launch(share=True)
```

注意：公開連結有效期為 72 小時，關閉 Colab 或程式後連結失效。

---

## Topic 3: Report（25%）

### 3.1 跨裝置測試截圖（10%）

用手機、平板等其他裝置連到 Gradio 公開網址，上傳至少 3 張圖片進行測試，將操作結果截圖放入報告。

### 3.2 FP32 vs INT8 分析（10%）

分析並說明 FP32 與 INT8 模型的差異，需附上相關數據或證據，例如：
- 推論時間比較
- 模型檔案大小比較
- 預測結果是否一致
- 準確度差異（如果有）

### 3.3 心得與建議（5%）

撰寫對本次 Lab 的心得與建議，至少 10 字。

---

## 注意事項

1. **輸入大小**：測試圖片要找接近 32×32 或小圖的，太大的圖 resize 後準確率會明顯下降

2. **圖片類別**：僅限 CIFAR-10 的 10 個類別（飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車）

3. **檔案命名**：所有輸出檔案的「學號」部分請替換為你的實際學號

4. **模型檔名一致性**：notebook 模板中使用的是 `image_classifier_model.onnx` 和 `image_classifier_model_int8.onnx`，請統一改為 `學號_FP32.onnx` 和 `學號_INT8.onnx`

5. **Colab 執行**：如果在 Colab 執行，記得先上傳 `best_model.pth` 到 Content 資料夾

---

## resnet18.py 完整參考程式碼

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.inchannel, channels, s))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

---

## 技術重點摘要

### ONNX 概念
- ONNX 是跨框架的通用模型格式
- ONNX Runtime 是高效能推論引擎
- 流程：PyTorch 訓練 → ONNX 匯出 → ONNX Runtime 推論

### 量化（Quantization）
- 目的：FP32 → INT8，減少模型大小約 75%，加速推論 2-4 倍
- 靜態量化需要校準資料來決定 scale 和 zero-point
- 精度損失通常小於 1%

### Gradio 部署
- Gradio 提供網頁介面讓使用者上傳圖片
- 推論仍在伺服器端（你的電腦或 Colab）執行
- `share=True` 產生的公開連結有效期 72 小時
