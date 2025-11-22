# 🏦 FinTech Alert Account Prediction Pipeline  
**Python 3.13.5 | PU-Learning + RankStack | TimeFix Feature Engineering**

This repository contains a **fully reproducible end‑to‑end pipeline** for the  
**E.SUN Bank 2025 Alert Account Prediction Competition** — including data preprocessing,  
TimeFix feature engineering, PU‑learning classification, RankStack ensemble, model inference,  
and submission file generation.

本專案為玉山銀行 2025 **Alert Account Prediction** 競賽的完整可重現 Pipeline，  
涵蓋資料前處理、TimeFix 特徵工程、PU‑Learning 分類器、RankStack 集成模型、  
預測與 submit 檔案輸出。

---

# 📂 Project Structure 專案目錄結構

```
.
├── data/                                 # 原始資料 + 特徵工程輸出
│   ├── acct_alert.csv
│   ├── acct_predict.csv
│   ├── acct_transaction.csv
│   ├── features_train.csv                # preprocess 產生
│   ├── features_pred.csv                 # preprocess 產生
│   └── features_meta.json                # preprocess 產生
│
├── Preprocess/
│   └── feature_engineering_timefix.py    # 特徵工程（TimeFix + PU-friendly）
│
├── Model/
│   └── model.py                          # PU-Learning + RankStack + LightGBM
│
├── submit/
│   ├── submit_stack_topk.csv             # 最終提交
│   └── acct_predict_out_stack.csv        # 模型輸出（debug/analysis）
│
├── main.py                               # Pipeline 入口（preprocess → model）
├── requirements.txt                      # 套件需求
└── README.md                             # 本文件
```

---

# 🚀 Pipeline Overview 流程概述

```
Raw Data (data/*.csv)
        │
        ▼
[1] Preprocess (feature_engineering_timefix.py)
        ├─ clean & normalize 資料清洗/正規化
        ├─ TimeFix time-window aggregation 時間視窗聚合
        ├─ Hard Negative mining (PU-learning) 未標負例挑選
        ├─ channel / currency wide features 類別展開特徵
        ├─ entropy / activity features 熵值/活躍度特徵
        └→ features_train.csv, features_pred.csv, features_meta.json
        │
        ▼
[2] Model (model.py)
        ├─ LightGBM PU classifier（Meta 模型）
        ├─ Platt scaling 校準
        ├─ Middle-band Ranker（中段機率 Ranker）
        ├─ Score fusion (RankStack)
        └→ submit_stack_topk.csv, acct_predict_out_stack.csv
        │
        ▼
[3] Submit
        ✔ 符合競賽要求之提交 CSV
```

---

# 🧩 Features Included (TimeFix) 特徵工程內容

### ✔ Fundamental Statistics 基礎統計  
- tx_cnt / active_days  
- amt_in_sum / amt_out_sum  
- abs(amount) mean/std/max  
- uniq counterparty  

### ✔ TimeFix 時間修正特徵  
- 5‑min activity entropy（5 分鐘桶熵值）
- peak / night ratio（尖峰/夜間比例）
- min-of-day 分布（每日時間分布）
- recent-window aggregation（近 1–60 天行為）

### ✔ Category Wide Features 類別展開  
- channel_type  
- currency_bucket  

### ✔ PU-Learning Hard Negatives  
- 依據 acct 活躍度、集中度、熵值排序取最可信 U  
- 適用於 Positive‑Unlabeled 競賽情境  

所有特徵與設定將寫入：

```
data/features_meta.json
```

---

# 🤖 Model Architecture  
PU-Learning + RankStack 模型架構

## 1. Meta Model (LightGBM)
- Stratified K‑Fold  
- PU weighted loss  
- Early stopping  
- Output: baseline probability  

## 2. Platt Scaling（機率校準）
- 使用 Logistic Regression  
- Output: `meta_cal`  

## 3. Middle-Band Ranker（中段 Ranker）
只訓練中間機率區間：

```
(0.03, 0.15)
```

- 多 SEED bagging（42 / 73 / 101 / 137）
- 輸出 `rank_score`

## 4. Final Score Fusion 融合

```
final_score = ALPHA * meta_cal + (1 - ALPHA) * rank_score
```

## 5. Top-K Selection
依 Public ACC0 計算陽性比例：

```
RATE = 1 - ACC0_PUBLIC
```

決定 K：

```
predict = 1 if rank in top-K else 0
```

---

# 📦 Installation & Environment 安裝與環境

## 1. 建立 Python 3.13.5 虛擬環境

```bash
python3 -m venv finenv
source finenv/bin/activate
```

## 2. 安裝套件需求

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# ▶️ Run the Entire Pipeline 執行完整流程

只需一行：

```bash
python main.py
```

主程式會：

1. 執行 Preprocess  
2. 產生 features  
3. 執行模型推論  
4. 產生 submit  

輸出位置：

```
submit/submit_stack_topk.csv
submit/acct_predict_out_stack.csv
```

---

# 🗂 Folder Description 資料夾說明

| Folder/File | Description |
|-------------|-------------|
| **data/** | 原始資料 + 特徵工程輸出 |
| **Preprocess/** | TimeFix 特徵工程腳本 |
| **Model/** | RankStack / LightGBM 模型 |
| **submit/** | 最終提交檔 |
| **main.py** | Pipeline 入口 |
| **requirements.txt** | 套件需求 |
| **README.md** | 本文件 |

---

# 🏁 Competition Result 競賽成績

本專案於 **玉山銀行 2025 Alert Account Prediction** 競賽取得：

🎯 **第 36 名 / 790 隊（前 4.5%）**  

- 模型：PU‑Learning + RankStack + TimeFix  
- Public Leaderboard：Top 36  
- Team：TEAM_8294  

---

# 📬 Contact

若你對本專案的架構、特徵工程、模型，  
或如何在其他任務中應用 PU‑Learning / RankStack，  
歡迎提出問題，我會協助你。

