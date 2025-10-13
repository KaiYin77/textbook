# Lesson 1 | 打開分類器的黑盒
## 分類器是什麼？

分類器就是把輸入資料對應到離散類別的模型。它觀察輸入特徵（Features），使用既定的決策邏輯判斷「這筆資料屬於哪個類別」。

> - iPhone 的 Face ID 會把臉部影像分類成「主人」或「陌生人」，決定裝置是否解鎖。  
> - Tesla FSD 透過鏡頭影像辨識「行人」、「汽車」、「號誌燈」，好讓車輛採取煞車或轉向等動作。

## 分類還是回歸？

在實務上，最常遇到的監督式學習Supervised Learning 任務大多可歸為分類或回歸兩種型態。先確認答案是離散類別還是連續數值，就能決定策略。

| 問題型態 | 典型任務 | 結果型態 | 例子 |
| --- | --- | --- | --- |
| Classification（分類） | 判別資料落在哪個類別 | 離散 | 紅綠燈號誌（紅／黃／綠）、腫瘤是否為惡性 |
| Regression（回歸） | 預測連續數值 | 實數或區間值 | 未來房價、氣溫曲線、股票收盤價 |

想把回歸問題改寫成分類題也很常見，例如把血壓區間切成「高／中／低」三類，再交由分類器處理。

## 模型Model中，甚麼是Training、甚麼是Inference？

架構示意圖如下所示。
![Archeitecture 示意圖](img/arch.png)

> - Training（訓練）：模型讀取大量「特徵＋標籤」的組合，調整參數讓預測更接近真實答案。  
> - Inference（推論）：模型接收尚未看過的資料，只靠訓練好的參數產生預測結果。

了解這兩個階段，有助於你區分「模型在學習」和「模型在回答」時需要的資料與資源。

## 常見術語

> Feature（特徵）
> 特徵是可量化的資料描述。彩色影像可攤平成 1024×1024×3 的張量，每個元素代表紅、綠、藍通道的像素亮度，模型利用這些強度分佈辨識物件或場景。文字則可透過詞向量（word embedding）轉成 300 維向量。
> 簡單說就是模型輸入資料。  
> 例：一張 1024×1024 彩色照片包含 3,145,728 個像素值；詞向量可以讓「cat」與「kitten」的距離只有 0.3，但與「car」的距離高達 1.8。

> Label（標籤）
> 標籤就是人類賦予的答案。  
> ![Label 示意圖](img/label.png)  
> 例：在貓狗分類中，可設定「貓＝0，狗＝1」。本文範例唯一這筆資料的標籤就是 0。

> Logit
> 模型輸出的線性組合結果。
> 輸入 softmax 函數後，可以表達機率分佈。  
> 例：唯一這筆資料的 logits 為 [1.01, 0.56]，差距 0.45，顯示模型更偏向類別 0。

> Loss Function（損失函數）
> 衡量模型預測與真實標籤差異的函數，也決定訓練時要「往哪裡修正」參數。  
> 例：交叉熵損失約 0.49，若未來調整後降到 0.1，就代表預測和真實答案更接近。

> Backpropagation（反向傳播）
> 從損失函數出發沿著模型反向傳遞誤差，計算每個參數的梯度，搭配優化器（如梯度下降）更新權重，模型才能越學越準。  
> 例：梯度運算得到 `grad_b ≈ [-0.389, 0.389]`，代表第一個偏差需要往上調，第二個偏差則往下調。

下面用一個簡單的 NumPy 範例，把這些概念串連起來，並示範線性分類器如何執行一次反向傳播與參數更新。

```python
import numpy as np

# --- Inputs ---------------------------------------------------------------
# 1) 一筆資料，包含三個特徵（feature 向量）
X = np.array([[0.2, 1.1, -0.3]])

# 2) 正確標籤：0 表示「貓」、1 表示「狗」
y = np.array([0])

# 3) 線性分類器的初始權重與偏差（可學參數）
W = np.array([[1.2, -0.7],
              [0.5,  0.9],
              [-0.4, 0.3]])
b = np.array([0.1, -0.2])

# --- Forward pass ---------------------------------------------------------
# Step A: 計算 logits（線性組合 XW + b）
logits = X @ W + b                # ≈ [[1.01, 0.56]]

# Step B: 使用 softmax 把 logits 轉為各類別機率
exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))  # 數值穩定技巧
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)       # ≈ [[0.61, 0.39]]

# Step C: 交叉熵損失（Loss）- 衡量預測與真實標籤的差距
true_probs = probs[np.arange(len(y)), y]
loss = -np.log(true_probs).mean()  # ≈ 0.49

print("Logits:\n", logits)
print("Probabilities:\n", probs)
print("Loss:", loss)

# --- Backward pass --------------------------------------------------------
# Step D: ∂Loss/∂logits（softmax + cross-entropy 的梯度閉式解）
grad_logits = probs.copy()            # ≈ [[0.61, 0.39]]
grad_logits[np.arange(len(y)), y] -= 1  # ≈ [[-0.389, 0.389]]
grad_logits /= len(y)                 # ≈ [[-0.389, 0.389]]（batch=1）

# Step E: 鏈式法則，把梯度傳回權重與偏差
grad_W = X.T @ grad_logits            # ≈ [[-0.0779, 0.0779], [-0.4283, 0.4283], [0.1168, -0.1168]]
grad_b = grad_logits.sum(axis=0)      # ≈ [-0.389, 0.389]

print("Grad W:\n", grad_W)
print("Grad b:", grad_b)

# --- Parameter update -----------------------------------------------------
# Step F: 以梯度下降更新參數（W ← W - η · 梯度）
learning_rate = 0.1
W -= learning_rate * grad_W          # ≈ [[1.208, -0.708], ...]
b -= learning_rate * grad_b          # ≈ [0.139, -0.239]

print("Updated W:\n", W)
print("Updated b:", b)
```

經過一次 forward、loss、backprop 的流程後，我們就得到新的權重與偏差，下一輪再把更新後的參數帶入，分類器就能逐步提升表現。
