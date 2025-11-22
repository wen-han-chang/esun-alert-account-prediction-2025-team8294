
---

## 📂 `Preprocess/README.md`（請貼到 `Preprocess/README.md`）

```markdown
# 🧮 Preprocess — TimeFix Feature Engineering  
**Transaction aggregation, temporal features, and PU-friendly training keys**

This folder contains the **data preprocessing and feature engineering pipeline**.  
It converts raw transaction-level data into **account-level features** suitable for PU-learning  
and the downstream RankStack model.

此資料夾負責**資料前處理與特徵工程**，  
將原始交易紀錄轉換為適合 PU-Learning 與後續模型使用的 **acct 粒度特徵**。

---

## 📁 Files 檔案說明

### `feature_engineering_timefix.py`

Main script for **reading raw CSVs, normalizing fields, and generating feature matrices**.

此檔案為主要的特徵工程腳本，負責讀取原始 CSV、欄位正規化與產生特徵檔。

It performs the following high-level steps:

1. **Path and environment setup**  
   - Uses `pathlib.Path` to locate the project root and `data/` directory.  
   - Targets:
     - `data/acct_transaction.csv`
     - `data/acct_alert.csv`
     - `data/acct_predict.csv`  
   使用 `Path` 取得專案根目錄與 `data/` 路徑，並讀取三個官方檔案。

2. **Memory-efficient CSV loading**  
   - Uses `read_csv_safely()` to prefer the PyArrow engine when available.  
   - Falls back to pandas C engine with explicit dtypes.  
   透過 `read_csv_safely()` 優先使用 PyArrow，否則退回 pandas 內建引擎並指定欄位型態，以降低記憶體使用。

3. **Date & time normalization (TimeFix)**  
   - Converts `txn_date` / `txn_date_raw` and `event_date` / `event_date_raw`  
     into integer day indices (`txn_day`, `event_day`).  
   - Aligns bases so that transaction and event days are comparable.  
   - Converts `txn_time` into:
     - `min_of_day`
     - `min5_bin` (5-minute bin)
     - `is_night`, `is_peak` flags  
   將日期欄位轉成整數日序，並統一基準；把時間欄位轉成分鐘數、5 分鐘桶與尖峰/夜間標記。

4. **Categorical normalization**  
   - Encodes `is_self_txn` into small integer codes.  
   - Processes `*_acct_type` into in-bank / out-of-bank flags.  
   - Buckets currency into key types (e.g., TWD / USD / OTHER).  
   - Normalizes `channel_type` into categorical with an `UNK` bucket.  
   對自轉帳、帳戶種類、幣別與交易通道做正規化與壓縮編碼。

5. **Winsorization and extreme value handling**  
   - Clips transaction amounts at the 99.5% quantile (symmetric).  
   - Avoids outliers dominating statistics.  
   對金額做 winsorize，避免極端值造成特徵失真。

6. **Long-format transaction table (`tx_long`)**  
   - Builds both **payer** and **payee** views.  
   - For each transaction, creates:
     - `amt_in`, `amt_out`
     - `counterparty`
     - activity-related flags and time bins.  
   建立長表 `tx_long`，同時包含付款方與收款方視角，以及出入金額與對手方資訊。

7. **Feature aggregation**  
   Uses multiple helper functions:

   - `agg_features(df)`  
     - 基礎統計（交易筆數、金額總和、平均、標準差、最大值、活躍天數、對手方數量等）。
   - `wide_count(df, col, prefix)`  
     - 類別變數（如 channel / currency）的 wide encoding + 比例欄位。
   - `counterparty_profile(df)`  
     - 對手方集中度（entropy / Herfindahl 指標）。
   - `timebin_profile(df)`  
     - 時間桶的分布與熵值。
   - `bucket_profile(df, col)`  
     - 任意類別欄位的 entropy / top1 比例。

8. **PU-friendly training key construction**  
   - `build_train_keys(tx, alert, tx_long)`  
   - Builds positive keys from alerts, and hard negative keys from active non-alert accounts.  
   - Ensures a configurable negative-to-positive ratio (`NEG_POS_RATIO`).  
   依據 alert 建立正樣本，並從活躍但未標記帳號中挑選 Hard Negative，  
   形成 PU-Learning 可用的訓練鍵集合。

9. **Windowed transaction selection**  
   - For each account & event_day, selects transactions within `WINDOW_DAYS` before the event.  
   - Applies fallback rules if no data in the default window.  
   針對每個事件日前 `WINDOW_DAYS` 內的交易建立訓練與預測視窗，  
   若視窗內無資料會自動改用較寬視窗或全部歷史。

10. **Output feature files**  
    - `data/features_train.csv`  
      - Columns: `acct`, `label`, `is_unlabeled`, `<features...>`  
    - `data/features_pred.csv`  
      - Columns: `acct`, `<features...>`  
    - `data/features_meta.json`  
      - Contains `feature_cols`, winsorization caps, log-transform info, etc.  

    將訓練與預測特徵寫入 `data/`，同時輸出 meta 資訊以供模型階段使用。

---

## ▶️ How to Run 執行方式

From the project root (專案根目錄下)：

```bash
python -m Preprocess.feature_engineering_timefix
