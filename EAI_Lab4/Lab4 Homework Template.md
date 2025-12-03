# Lab4 - Homework Template

### 1. About Knowledge Distillation (15%)

- What Modes of Distillation is used in this Lab ?

這次 Lab 用了兩種 Knowledge Distillation：

1. **Response-Based Distillation**：
   - 就是讓 Student 去學 Teacher 的輸出 logits
   - 用 Temperature 把 softmax 弄軟一點，這樣可以學到類別之間的關係
   - 用 KL Divergence 來看 Teacher 和 Student 的分佈有多像

2. **Feature-Based Distillation**：
   - 直接拿 Teacher 中間層的 feature 來教
   - 我從 ResNet 的 layer1、layer2、layer3、layer4 抓 feature
   - 讓 Student 學 Teacher 怎麼看圖片、關注哪些地方
    
- What role do logits play in knowledge distillation? What effect does a higher temperature parameter have on logits conversion ?

**Logits 的作用：**

Logits 就是最後 FC layer 吐出來的原始數字，還沒過 softmax。它很重要是因為：

1. 包含類別關係的資訊：比如說一張貓的圖，logits 可能是貓最高，狗次高（因為都是動物），飛機很低。這種「相似度」的資訊就是 Dark Knowledge

2. 可以當 soft label 用：把 logits/T 再做 softmax 就變成軟標籤，比 one-hot 有更多資訊

**Temperature 變高的效果：**

Temperature 就是控制分佈有多「尖」：

- T=1（正常）：分佈很尖，基本上就是 one-hot，例如 [0.93, 0.05, 0.02]
- T=4（較高）：分佈變平滑，保留更多關係，例如 [0.51, 0.28, 0.21]

T 越高，softmax 越平，Student 就能學到更多 Teacher 對各個類別的「信心程度」。
    
- In Feature-Based Knowledge Distillation, from which parts of the Teacher model do we extract features for knowledge transfer?

我從 ResNet50 的這些地方抓 feature：

- **Layer1**：低層特徵（邊緣、紋理）
- **Layer2**：中層特徵（形狀、部件）
- **Layer3**：更複雜的特徵
- **Layer4**：高層語義特徵

實作上就是在 forward 裡面把每一層的輸出都存下來：

```python
feature1 = self.layer1(x)
feature2 = self.layer2(feature1)
feature3 = self.layer3(feature2)
feature4 = self.layer4(feature3)
return x, [feature1, feature2, feature3, feature4]
```

### 2. Response-Based KD (30%)

**Temperature 和 alpha 怎麼選的：**

Temperature 我設 4.0，參考 Hinton 那篇論文說 3-5 效果不錯。太低的話（1-2）軟標籤跟硬標籤差不多，蒸餾意義不大；太高（10+）又會太平滑失去辨識度。實測 4 還蠻好用的。

Alpha 設 0.7，意思是 70% weight 給蒸餾 loss，30% 給 hard label。這樣 Student 主要跟 Teacher 學，但也不會完全忘記正確答案。一般來說 0.7-0.9 都還可以。

**Loss function 設計：**

```python
def loss_re(student_logits, teacher_logits, target):
    T = 4.0
    alpha = 0.7
    
    # 蒸餾 loss：用 KL divergence
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    
    # Hard label loss
    student_loss = F.cross_entropy(student_logits, target)
    
    # 加權
    loss = alpha * distillation_loss + (1 - alpha) * student_loss
    return loss
```

重點是要乘 T²，因為溫度會把梯度縮小，要補償回來。

### 3. Feature-based KD (30%)

**怎麼抓 feature：**

就是在 ResNet 的 forward 裡，每過一個 layer 就把輸出存起來：

```python
def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    
    feature1 = self.layer1(x)       # [B, 64/256, 32, 32]
    feature2 = self.layer2(feature1)  # [B, 128/512, 16, 16]
    feature3 = self.layer3(feature2)  # [B, 256/1024, 8, 8]
    feature4 = self.layer4(feature3)  # [B, 512/2048, 4, 4]
    
    x = self.avgpool(feature4)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    
    return x, [feature1, feature2, feature3, feature4]
```

**Loss function 設計：**

問題是 ResNet50 和 ResNet18 的 channel 數不一樣（比如 layer4 是 2048 vs 512），沒辦法直接比。所以我用 Spatial Attention Transfer：

```python
def loss_fe(student_features, teacher_features, student_logits, target):
    # 分類 loss
    ce_loss = F.cross_entropy(student_logits, target)
    
    # Feature loss
    feature_loss = 0.0
    for student_feat, teacher_feat in zip(student_features, teacher_features):
        # 對 channel 取平均得到 attention map
        student_attention = torch.mean(student_feat, dim=1, keepdim=True)
        student_attention = F.normalize(student_attention.view(student_attention.size(0), -1))
        
        teacher_attention = torch.mean(teacher_feat, dim=1, keepdim=True)
        teacher_attention = F.normalize(teacher_attention.view(teacher_attention.size(0), -1))
        
        # 用 MSE 比較
        feature_loss += F.mse_loss(student_attention, teacher_attention)
    
    beta = 1000.0
    loss = ce_loss + beta * feature_loss
    return loss
```

這樣做的好處是不管 channel 有多少，都可以比較「關注哪裡」而不是「數值多少」。beta 設 1000 是因為 MSE 值太小，要放大才能平衡 CE loss。

### 4. Comparison of student models w/ & w/o KD (5%)

|                            | loss     | accuracy |
| -------------------------- | -------- | -------- |
| Teacher from scratch       | 0.43     | 90.90%   |
| Student from scratch       | 0.49     | 90.39%   |
| Response-based student     | 0.84     | 91.35%   |
| Featured-based student     | 1.82     | 91.51%   |

### 5. Implementation Observations and Analysis (20%)

**有沒有出乎意料的結果？**

有，而且蠻有趣的：

1. Student 蒸餾後反而比 Teacher 還準
   - Teacher: 90.90%
   - Response-based: 91.35% (+0.45%)
   - Feature-based: 91.51% (+0.61%)

2. Loss 值反而變大了
   - 雖然 accuracy 變高，但 loss 變大（0.84、1.82 vs 0.43）

**為什麼會這樣？**

Student 超過 Teacher 可能是因為：
- 蒸餾本身有 regularization 效果，減少 overfitting
- 軟標籤讓訓練更 smooth
- ResNet18 對 CIFAR-10 這種小圖其實就夠了，ResNet50 太大反而可能 overfit

Loss 變大是因為 loss function 本身就不一樣了，加了 KL divergence 或 MSE，當然會比單純的 CE 大。這個不影響 accuracy。

**實作遇到的問題：**

1. **Feature 維度對不上**
   ResNet50 和 ResNet18 的 channel 數差很多，本來想直接用 MSE 比 feature 發現根本不行。後來改用 spatial attention，對 channel 取平均，只比「位置」不比「數值」，才解決。

2. **Loss 權重很難調**
   Feature loss 的值超小（0.001 左右），跟 CE loss（1.0 左右）完全不在一個量級。試了很多 beta 值，最後發現 1000 比較穩。

3. **雙 GPU 設定**
   一開始忘記把 model wrap 成 DataParallel，只用到一張卡。後來才發現要先 load model 再包 DataParallel，optimizer 才能抓對參數。

**心得：**

這次實驗證明 knowledge distillation 真的有用，不只是壓縮模型而已。小模型經過蒸餾不只超過 baseline，還能超過 teacher，這點蠻神奇的。可能是因為有額外的 regularization，或是 soft label 提供更多資訊。總之兩種方法都成功了，Feature-based 稍微好一點點。