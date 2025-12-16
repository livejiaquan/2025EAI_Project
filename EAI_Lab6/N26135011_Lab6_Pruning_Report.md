# Lab 6 - Transformer Pruning Report
  
1. **請說明 get_real_idx 實作部分是怎麼做的** 10%

    因為每一層 pruning 回傳的 idx 是相對於該層輸入的索引，不是原始 patch 的位置。所以第二層以後的 idx 需要透過前一層的 idx 來轉換。

    做法是用 `torch.gather(idxs[i-1], dim=1, index=idxs[i])` 把當前層的相對索引映射回原始位置。另外因為有 fused token 的關係，idx 可能會超出前一層的範圍，所以用 `torch.clamp` 處理這個邊界狀況。

2. **實際在哪些層做了 pruning ?** 10%
    
    看 `keep_rate` 的設定：
    ```
    keep_rate = (1, 1, 1, 0.7) + (1, 1, 0.7) + (1, 1, 0.7) + (1, 1)
    ```
    
    只有 keep_rate < 1 的層才會做 pruning，所以是 Layer 4、Layer 7、Layer 10（索引 3, 6, 9）。每次保留 70% 的 tokens。

3. **如果沒有 get_real_idx 可視化結果會長怎樣，為什麼 ?** 10%
    
    會顯示錯誤的 patch 位置。

    因為第二層 pruning 的 idx 是相對於第一層輸出的 token 順序，不是原始的 196 個 patches。如果直接拿這些索引去 mask 原圖，位置會完全對不上，看起來就是隨機散落的 patches，不會聚焦在物體上。

4. **分析視覺化的圖，這些變化代表著什麼 ?** 10%
    
    從圖可以看到隨著層數增加，保留的 patches 越來越少，而且越來越集中在物體的主體區域。

    Layer 4 還保留蠻多的，Layer 7 開始明顯減少，到 Layer 10 就只剩物體最核心的部分。這代表 CLS token 的 attention 確實能抓到影像中重要的區域，背景那些不重要的就被剪掉了。最後準確率還是 100%，表示這些被剪掉的 token 本來就沒什麼用。
