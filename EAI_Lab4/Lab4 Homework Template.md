# Lab4 - Homework Template

### 1. About Knowledge Distillation (15%)

- What Modes of Distillation is used in this Lab ?

在這個 Lab 中，我們使用了兩種 Knowledge Distillation 的模式：

1. **Response-Based Distillation（響應式蒸餾）**：
   - 利用 Teacher 模型的輸出 logits 作為軟標籤（Soft Labels）
   - 透過溫度參數（Temperature）軟化 softmax 分佈，讓 Student 學習類別之間的關係
   - 使用 KL 散度（KL Divergence）來衡量 Teacher 和 Student 輸出分佈的差異

2. **Feature-Based Distillation（特徵式蒸餾）**：
   - 利用 Teacher 模型的中間層特徵（Intermediate Features）進行知識傳遞
   - 從 ResNet 的 Layer1、Layer2、Layer3、Layer4 提取特徵
   - 透過特徵匹配讓 Student 學習 Teacher 的表示方式
    
- What role do logits play in knowledge distillation? What effect does a higher temperature parameter have on logits conversion ?

**Logits 在知識蒸餾中的角色：**

Logits 是模型最後一層全連接層的原始輸出（未經過 softmax 的數值），在知識蒸餾中扮演非常重要的角色：

1. **包含豐富的類別關係信息**：Logits 不只表達「正確答案是什麼」，還包含了「各個類別之間的相似程度」。例如貓的圖片，logits 可能顯示「貓」的值最高，但「狗」的值也不低（因為都是動物），而「飛機」的值很低。

2. **作為軟標籤的來源**：經過溫度軟化的 softmax(logits/T) 產生軟標籤，這些軟標籤比硬標籤（one-hot）包含更多信息。

**更高的溫度參數（Temperature）的效果：**

溫度 T 的作用是控制 softmax 分佈的「尖銳程度」：

- **T = 1（正常 softmax）**：分佈尖銳，主要集中在最大值
  - 例如：[0.93, 0.05, 0.02] → 幾乎只說「是貓」
  
- **T = 4（較高溫度）**：分佈平滑，保留更多類別間的關係
  - 例如：[0.51, 0.28, 0.21] → 說「主要是貓，但和狗也有點像」

更高的溫度讓 softmax 輸出更加平滑，這樣 Student 可以學到更多 Teacher 對各個類別的「信心程度」和「類別相似性」的隱性知識（Dark Knowledge）。
    
- In Feature-Based Knowledge Distillation, from which parts of the Teacher model do we extract features for knowledge transfer?

在 Feature-Based Knowledge Distillation 中，我們從 Teacher 模型（ResNet50）的以下部分提取特徵：

1. **Layer1 的輸出**：學習邊緣、紋理等底層特徵
2. **Layer2 的輸出**：學習形狀、簡單部件等中層特徵
3. **Layer3 的輸出**：學習更複雜的物體部分
4. **Layer4 的輸出**：學習高層語義特徵

這四層的特徵在 ResNet 的 forward 函數中被保存下來：

```python
feature1 = self.layer1(x)      # 第一層特徵
feature2 = self.layer2(feature1)  # 第二層特徵
feature3 = self.layer3(feature2)  # 第三層特徵
feature4 = self.layer4(feature3)  # 第四層特徵
return x, [feature1, feature2, feature3, feature4]
```

透過這些中間層特徵，Student 不只學習最終的分類結果，還學習 Teacher「如何思考」和「關注哪些位置」。

### 2. Response-Based KD (30%)

Please explain the following:
- How you choose the Temperature and alpha?
- How you design the loss function?

**溫度參數（Temperature = 4.0）的選擇：**

選擇 T = 4.0 的原因如下：

1. **理論依據**：Hinton 等人的原始論文建議溫度範圍在 3-5 之間效果最好
2. **平衡考量**：
   - T 太低（如 1-2）：軟標籤接近硬標籤，蒸餾效果有限
   - T 太高（如 10+）：分佈過於平滑，失去區分性
   - T = 4 是一個良好的平衡點
3. **實驗驗證**：在 CIFAR-10 數據集上，T = 4 能有效傳遞類別間的關係信息

**權重參數（Alpha = 0.7）的選擇：**

選擇 α = 0.7 的原因如下：

1. **蒸餾為主**：70% 的權重給蒸餾損失，讓 Student 主要向 Teacher 學習
2. **保持準確性**：30% 的權重給硬標籤損失，確保 Student 能正確分類
3. **經驗法則**：α 在 0.7-0.9 範圍內通常效果較好

**損失函數的設計：**

Response-Based KD 的損失函數包含兩部分：

```python
def loss_re(student_logits, teacher_logits, target):
    T = 4.0      # 溫度參數
    alpha = 0.7  # 權重參數
    
    # 1. 蒸餾損失（Distillation Loss）
    soft_teacher = F.softmax(teacher_logits / T, dim=1)  # Teacher 軟標籤
    soft_student = F.log_softmax(student_logits / T, dim=1)  # Student 軟標籤
    distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    
    # 2. 硬標籤損失（Hard Label Loss）
    student_loss = F.cross_entropy(student_logits, target)
    
    # 3. 總損失
    loss = alpha * distillation_loss + (1 - alpha) * student_loss
    return loss
```

**設計重點：**

1. **KL 散度衡量分佈差異**：使用 KL Divergence 讓 Student 的輸出分佈接近 Teacher
2. **乘以 T² 補償梯度**：因為溫度會縮小梯度，需要乘回 T² 來平衡
3. **結合硬標籤**：同時使用交叉熵確保 Student 能正確分類

### 3. Feature-based KD (30%)

Please explain the following:
- How you extract features from the choosing intermediate layers?
- How you design the loss function?

**如何從中間層提取特徵：**

在 ResNet 的 forward 函數中，我們在每個 Layer 後面保存其輸出作為特徵：

```python
def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    
    # 保存每一層的輸出作為特徵
    feature1 = self.layer1(x)       # [B, 64/256, 32, 32]
    feature2 = self.layer2(feature1)  # [B, 128/512, 16, 16]
    feature3 = self.layer3(feature2)  # [B, 256/1024, 8, 8]
    feature4 = self.layer4(feature3)  # [B, 512/2048, 4, 4]
    
    x = self.avgpool(feature4)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    
    return x, [feature1, feature2, feature3, feature4]
```

**損失函數的設計：**

由於 ResNet50 和 ResNet18 的特徵通道數不同（例如 Layer4：2048 vs 512），無法直接比較。因此我採用了 **Spatial Attention Transfer** 方法：

```python
def loss_fe(student_features, teacher_features, student_logits, target):
    # 1. 分類損失
    ce_loss = F.cross_entropy(student_logits, target)
    
    # 2. 特徵蒸餾損失
    feature_loss = 0.0
    for student_feat, teacher_feat in zip(student_features, teacher_features):
        # 計算空間注意力圖（沿通道維度求平均）
        student_attention = torch.mean(student_feat, dim=1, keepdim=True)
        student_attention = F.normalize(student_attention.view(student_attention.size(0), -1))
        
        teacher_attention = torch.mean(teacher_feat, dim=1, keepdim=True)
        teacher_attention = F.normalize(teacher_attention.view(teacher_attention.size(0), -1))
        
        # MSE 損失衡量注意力圖差異
        feature_loss += F.mse_loss(student_attention, teacher_attention)
    
    # 3. 總損失
    beta = 1000.0  # 特徵損失權重
    loss = ce_loss + beta * feature_loss
    return loss
```

**設計重點：**

1. **Spatial Attention 解決維度不匹配**：
   - 對通道維度求平均：[B, C, H, W] → [B, 1, H, W]
   - 這樣不管 C 是多少，都能得到相同大小的注意力圖
   - 讓 Student 學習「關注哪些位置」而非「具體數值是什麼」

2. **歸一化處理**：使用 F.normalize 讓比較更穩定

3. **MSE 損失**：簡單有效地衡量注意力圖的差異

4. **β = 1000 的選擇**：因為 MSE 損失值通常很小（0.001 量級），需要放大來平衡分類損失

### 4. Comparison of student models w/ & w/o KD (5%)

Provide results according to the following structure:

|                            | loss     | accuracy |
| -------------------------- | -------- | -------- |
| Teacher from scratch       | 0.43     | 90.90%   |
| Student from scratch       | 0.49     | 90.39%   |
| Response-based student     | 0.84     | 91.35%   |
| Featured-based student     | 1.82     | 91.51%   |

### 5. Implementation Observations and Analysis (20%)

Based on the comparison results above:
- Did any KD method perform unexpectedly? 
- What do you think are the reasons? 
- If not, please share your observations during the implementation process, or what difficulties you encountered and how you solved them?

**觀察到的異常現象：**

是的，有一些出乎意料的結果：

1. **蒸餾後的 Student 超過了 Teacher！**
   - Teacher: 90.90%
   - Response-based Student: 91.35% (+0.45%)
   - Feature-based Student: 91.51% (+0.61%)

2. **Loss 值的反差**
   - 雖然準確率更高，但蒸餾模型的 loss 值反而更大
   - Response-based: 0.84 > Teacher: 0.43
   - Feature-based: 1.82 > Teacher: 0.43

**可能的原因分析：**

1. **Student 超過 Teacher 的原因：**
   - **正則化效果**：蒸餾過程提供了額外的監督信號，類似於正則化，減少了過擬合
   - **軟標籤的優勢**：軟標籤比硬標籤包含更多信息，讓訓練更加平滑
   - **模型容量匹配**：ResNet18 對於 CIFAR-10（32x32 小圖片）可能已經足夠，更大的 ResNet50 反而可能過擬合

2. **Loss 值較大的原因：**
   - 蒸餾損失函數設計不同，包含了額外的 KL 散度或 MSE 項
   - 這些額外的項會增加總 loss 值，但不影響分類準確率
   - Loss 值和準確率之間不是完全正相關的關係

**實現過程中遇到的困難及解決方法：**

1. **特徵維度不匹配問題**
   - 困難：ResNet50 的特徵通道數（256, 512, 1024, 2048）與 ResNet18（64, 128, 256, 512）不同
   - 解決：採用 Spatial Attention Transfer，對通道求平均，只比較「哪裡重要」而非「具體值」

2. **損失函數權重平衡**
   - 困難：Feature-based KD 的特徵損失值很小（0.001 量級），與分類損失（1.0 量級）不平衡
   - 解決：使用較大的權重 β = 1000 來放大特徵損失

3. **多 GPU 訓練設置**
   - 困難：需要正確配置 DataParallel 來利用雙 GPU
   - 解決：在模型加載後立即包裝 DataParallel，並確保優化器使用正確的參數

**總結：**

這次實驗成功驗證了知識蒸餾的有效性。無論是 Response-Based 還是 Feature-Based 方法，都成功讓小模型（ResNet18）超越了基準線，甚至超過了 Teacher（ResNet50）。這說明知識蒸餾不只是「壓縮模型」，還可能帶來正則化等額外好處。
