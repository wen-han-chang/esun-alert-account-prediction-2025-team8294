"""Feature engineering script for account-level fraud detection.

This module:
- reads raw transaction, alert, and prediction CSV files,
- normalizes and converts dates/times into numeric representations,
- builds long-format transaction tables for payer/payee,
- aggregates account-level behavioral features,
- normalizes numeric features (winsorization, log transforms),
- writes `features_train.csv`, `features_pred.csv`, and `features_meta.json`.
"""
import json
import argparse
import numpy as np
import pandas as pd
import sys, gc
from pathlib import Path


# 允許在 Jupyter/VS Code 中執行：移除 Notebook 自動注入的 -f 參數
def _strip_notebook_argv():
    """Remove Jupyter-injected '-f' argv so argparse works in notebooks.

    In Jupyter / VS Code notebooks, a hidden "-f <kernel-file>" argument is
    often injected into sys.argv. This helper removes it to prevent argparse
    from crashing.
    """
    if "-f" in sys.argv:
        i = sys.argv.index("-f")
        del sys.argv[i:i + 2]


_strip_notebook_argv()

# 降低隱性拷貝、節省記憶體
pd.set_option("mode.copy_on_write", True)

# ================== 可調參 ==================
WINDOW_DAYS = 60          # 訓練/預測視窗：事件日前 N 天
NEG_POS_RATIO = 20.0      # Hard negatives：未標示帳號抽樣比
RANDOM_STATE = 42
DRIFT_TOPK = 0

# ================== 路徑設定 ==================
# 這支程式在 preprocess/ 底下，data 在上一層的 data/
BASE_DIR = Path(__file__).resolve().parent.parent   # 專案根目錄（preprocess 的上一層）
DATA_DIR = BASE_DIR / "data"

# 預設輸入檔路徑（可用 CLI 覆寫）
TX_PATH = DATA_DIR / "acct_transaction.csv"
ALERT_PATH = DATA_DIR / "acct_alert.csv"
PRED_PATH = DATA_DIR / "acct_predict.csv"

# 預設輸出檔路徑：也放到 data/ 底下
OUT_TRAIN = DATA_DIR / "features_train.csv"   # acct,label,is_unlabeled,<features...>
OUT_PRED = DATA_DIR / "features_pred.csv"     # acct,<features...>
OUT_META = DATA_DIR / "features_meta.json"    # feature_cols、winsorize caps 等
# ===========================================


# =============== 低記憶體 CSV 讀取器 ===============
def read_csv_safely(path, usecols=None, dtype_hint=None, prefer_arrow=True):
    """Read a CSV file with a memory-efficient backend.

    Parameters
    ----------
    path : str or Path
        File path to the CSV.
    usecols : list of str, optional
        Subset of columns to read.
    dtype_hint : dict, optional
        Mapping of column names to dtypes for the fallback C engine.
    prefer_arrow : bool, default True
        If True, try the pyarrow engine first and fall back to C engine on failure.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe with the requested columns and dtypes.

    Notes
    -----
    Tries pyarrow engine first (typically lower RAM). Falls back to C engine
    with dtype hints on failure.
    """
    if prefer_arrow:
        try:
            import pyarrow  # noqa: F401
            df = pd.read_csv(
                path,
                usecols=usecols,
                engine="pyarrow",
                dtype_backend="pyarrow"
            )
            return df
        except Exception as e:
            print(f"[INFO] Arrow 讀取不可用，改用 C engine（原因：{type(e).__name__}）")

    # 備援：C engine + 明確 dtype + 省記憶體旗標
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=(dtype_hint or {}),
        low_memory=True,
        memory_map=True,
    )
    return df


# =============== 小工具 ===============
def safe_group_apply(gb, func):
    """Apply a groupby function with backward-compatible signature.

    Newer pandas versions support `include_groups=False` for groupby.apply.
    This helper keeps compatibility with older versions.
    """
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)


def _parse_date_col_to_int_day(s: pd.Series):
    """Parse a date-like column into integer day indices.

    Supports integer day indices or date strings (YYYYMMDD / ISO).

    Parameters
    ----------
    s : pandas.Series
        A series holding date-like values.

    Returns
    -------
    tuple
        (day_series, is_datetime, base_date)
        - day_series: Int64 day indices aligned to base_date.
        - is_datetime: whether parsing succeeded as dates.
        - base_date: minimum timestamp used as reference (or None).
    """
    # 支援「整數天序」或「日期字串/yyyymmdd」
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        day = pd.to_numeric(s, errors="coerce").astype("Int64")
        return day, False, None

    s_str = s.astype(str)
    dt = pd.to_datetime(s_str, errors="coerce")

    if dt.isna().all():
        mask8 = s_str.str.len().eq(8) & s_str.str.isnumeric()
        dt2 = pd.to_datetime(s_str.where(mask8, None), format="%Y%m%d", errors="coerce")
        dt = dt.fillna(dt2)

    if dt.notna().sum() == 0:
        return pd.Series([pd.NA] * len(s), dtype="Int64"), False, None

    dt = dt.dt.floor("D")
    base = dt.min()
    day = (dt - base).dt.days.astype("Int64")
    return day, True, base


def signed_log1p(x: pd.Series) -> pd.Series:
    """Signed log1p transform to handle both positive and negative values."""
    return np.sign(x) * np.log1p(np.abs(x))


def compute_caps(df: pd.DataFrame, cols, lo_q=0.005, hi_q=0.995):
    """Compute winsorization caps (quantile bounds) for given numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing numeric features.
    cols : list of str
        Columns to compute caps for.
    lo_q : float, default 0.005
        Lower quantile used as the lower cap.
    hi_q : float, default 0.995
        Upper quantile used as the upper cap.

    Returns
    -------
    dict
        Mapping col -> (lo_cap, hi_cap).
    """
    caps = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) == 0:
            continue
        lo = float(np.quantile(v, lo_q)) if lo_q is not None else None
        hi = float(np.quantile(v, hi_q)) if hi_q is not None else None
        caps[c] = (lo, hi)
    return caps


def apply_caps(df: pd.DataFrame, caps: dict):
    """Apply precomputed winsorization caps to dataframe columns in place."""
    for c, (lo, hi) in caps.items():
        if c not in df.columns:
            continue
        if lo is not None:
            df[c] = np.maximum(df[c], lo)
        if hi is not None:
            df[c] = np.minimum(df[c], hi)


def entropy_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy from a vector of counts."""
    counts = counts.astype(float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# =============== 讀檔 + 正規化（免大規模字串運算；不做去重以避 OOM） ===============
def read_and_normalize(tx_path, alert_path, pred_path, inference_only=False):
    """Load raw CSVs and normalize transaction / alert / prediction tables.

    This function:
    - reads transaction, alert, and prediction CSV files,
    - parses date/time into numeric day indices and minute-of-day bins,
    - converts categorical flags into compact numeric representations,
    - constructs a long-format transaction table for both payer/payee.

    Parameters
    ----------
    tx_path : str or Path
        Path to the transaction CSV.
    alert_path : str or Path
        Path to the alert CSV.
    pred_path : str or Path
        Path to the prediction-account CSV.
    inference_only : bool, default False
        If True, skip reading alerts and do not construct training labels.

    Returns
    -------
    tx : pandas.DataFrame
        Normalized transaction table (wide format).
    alert : pandas.DataFrame
        Alert table with event days (may be empty in inference-only mode).
    pred : pandas.DataFrame
        Prediction-account table.
    tx_long : pandas.DataFrame
        Long-format transaction table with payer/payee views concatenated.
    """
    head_tx = pd.read_csv(tx_path, nrows=0)
    tx_cols_all = set(head_tx.columns.tolist())

    tx_want = [
        "from_acct", "to_acct",
        "txn_date", "txn_date_raw",
        "txn_time", "txn_amt",
        "is_self_txn", "from_acct_type", "to_acct_type",
        "currency_type", "channel_type",
    ]
    tx_usecols = [c for c in tx_want if c in tx_cols_all]

    tx_dtype_hint = {
        "from_acct": "string",
        "to_acct": "string",
        "txn_time": "string",
        "txn_amt": "float32",
        "is_self_txn": "string",
        "from_acct_type": "string",
        "to_acct_type": "string",
        "currency_type": "string",
        "channel_type": "string",
        # txn_date/txn_date_raw 讓後續自行轉 Int64
    }

    tx = read_csv_safely(tx_path, usecols=tx_usecols, dtype_hint=tx_dtype_hint, prefer_arrow=True)

    # ------- alert / pred 同樣容錯 -------
    if not inference_only:
        try:
            head_alert = pd.read_csv(alert_path, nrows=0)
            alert_cols_all = set(head_alert.columns.tolist())
        except Exception:
            alert_cols_all = set()
    else:
        alert_cols_all = set()

    if not inference_only:
        alert_want = ["acct", "event_date", "event_date_raw"]
        alert_usecols = [c for c in alert_want if c in alert_cols_all] or ["acct", "event_date"]
        alert_dtype_hint = {"acct": "string", "event_date": "string", "event_date_raw": "string"}
        alert = read_csv_safely(alert_path, usecols=alert_usecols, dtype_hint=alert_dtype_hint, prefer_arrow=True)
    else:
        alert = pd.DataFrame(columns=["acct", "event_date"])

    pred = read_csv_safely(pred_path, usecols=["acct"], dtype_hint={"acct": "string"}, prefer_arrow=True)

    # 若 txn_time 缺欄位，補空欄（後面時間轉分鐘會處理 NaN）
    if "txn_time" not in tx.columns:
        tx["txn_time"] = pd.Series([None] * len(tx), dtype="string")

    #  不再去重，官方說明重複屬於資料特性；且 drop_duplicates 在 400+ 萬列會 OOM

    # 帳號字串化
    for c in ["from_acct", "to_acct"]:
        if c in tx.columns:
            tx[c] = tx[c].astype("string").str.strip()
    if "acct" in alert.columns:
        alert["acct"] = alert["acct"].astype("string").str.strip()
    pred["acct"] = pred["acct"].astype("string").str.strip()

    # 交易日 / 事件日 → 整數日序
    if "txn_date_raw" in tx.columns and tx["txn_date_raw"].notna().any():
        tx["txn_day"], tx_dt, tx_base = _parse_date_col_to_int_day(tx["txn_date_raw"])
    elif "txn_date" in tx.columns and tx["txn_date"].notna().any():
        tx["txn_day"], tx_dt, tx_base = _parse_date_col_to_int_day(tx["txn_date"])
    else:
        raise ValueError("找不到交易日欄位，需 txn_date_raw 或 txn_date")

    if not inference_only:
        if "event_date_raw" in alert.columns and alert["event_date_raw"].notna().any():
            alert["event_day"], al_dt, al_base = _parse_date_col_to_int_day(alert["event_date_raw"])
        elif "event_date" in alert.columns and alert["event_date"].notna().any():
            alert["event_day"], al_dt, al_base = _parse_date_col_to_int_day(alert["event_date"])
        else:
            raise ValueError("找不到事件日欄位，需 event_date_raw 或 event_date")
    else:
        al_dt = al_base = None

    # 對齊日序基準（若兩者都是日期型）
    if not inference_only and isinstance(tx_base, pd.Timestamp) and isinstance(al_base, pd.Timestamp):
        shift = (al_base - tx_base).days
        alert["event_day"] = (alert["event_day"].astype("Int64") + shift).astype("Int64")

    # 時間轉分鐘 / 5 分鐘桶 / 日間密度
    def _to_min_of_day(s):
        """Convert time to minutes from start of day.

        Supports:
        - 'HH:MM' or 'HH:MM:SS'
        - 'HHMM' digit string
        - integer-like minutes

        Returns NaN on parse failure.
        """
        s = str(s)
        if ":" in s:
            hh, mm, *_ = s.split(":")
            return int(hh) * 60 + int(mm)
        if s.isdigit():
            return int(s[:2]) * 60 + int(s[2:]) if len(s) == 4 else int(s)
        return np.nan

    tx["min_of_day"] = tx["txn_time"].astype("string").apply(_to_min_of_day).astype("Int64")
    tx["min5_bin"] = (tx["min_of_day"] // 5) * 5
    tx["is_night"] = ((tx["min_of_day"] >= 22 * 60) | (tx["min_of_day"] < 6 * 60)).astype("Int64")
    tx["is_peak"] = (((tx["min_of_day"] >= 9 * 60) & (tx["min_of_day"] < 12 * 60)) |
                     ((tx["min_of_day"] >= 14 * 60) & (tx["min_of_day"] < 17 * 60))).astype("Int64")

    # ========= 類別正規化（避免大字串運算造成 OOM） =========
    # is_self_txn → 小整數（int8）
    _map_self = {"Y": 1, "N": 0, "UNK": -1}
    tx["is_self_txn_num"] = (
        tx["is_self_txn"].astype("string").map(_map_self).fillna(-1).astype("int8")
    )

    # *_acct_type：避免 .str 操作，直接轉成數字再判斷 == 1
    def _bank_flag(df, col):
        """Create a bank-ownership flag from acct_type column.

        Parameters
        ----------
        df : pandas.DataFrame
            Transaction dataframe.
        col : str
            Column name that encodes account type.

        Returns
        -------
        pandas.Series
            int8 series where 1 indicates our-bank (type==1), else 0.
        """
        if col not in df.columns:
            return pd.Series(np.zeros(len(df), dtype=np.int8))
        s = pd.to_numeric(df[col], errors="coerce").astype("Int16")
        return (s == 1).astype("int8")

    tx["is_our_bank_from"] = _bank_flag(tx, "from_acct_type")
    tx["is_our_bank_to"] = _bank_flag(tx, "to_acct_type")

    # 用完就釋放原始欄位
    for _c in ("from_acct_type", "to_acct_type"):
        if _c in tx.columns:
            tx.drop(columns=[_c], inplace=True)

    # currency / channel：改用 category 省 RAM（避免 .str.*）
    if "currency_type" in tx.columns:
        cur = tx["currency_type"].astype("category")
        cats = pd.Index(cur.cat.categories.astype(str))
        keep_up = cats.str.upper().isin(["TWD", "USD"])
        keep_set = set(cats[keep_up])
        tx["currency_bucket"] = pd.Categorical(
            np.where(cur.isin(keep_set), cur.astype(str), "OTHER")
        )
        tx.drop(columns=["currency_type"], inplace=True)

    if "channel_type" in tx.columns:
        ch = tx["channel_type"]
        if not isinstance(ch.dtype, pd.CategoricalDtype):
            ch = ch.astype("category")
        if "UNK" in list(ch.cat.categories):
            ch_filled = ch.fillna("UNK")
        else:
            ch_filled = ch.cat.add_categories(["UNK"]).fillna("UNK")
        tx["channel_type"] = ch_filled

    # Winsorize 金額極端值（依全資料估）
    abs_amt = pd.to_numeric(tx["txn_amt"], errors="coerce").abs()
    if abs_amt.notna().any():
        cap = float(np.quantile(abs_amt.dropna().to_numpy(), 0.995))
        tx["txn_amt"] = np.clip(pd.to_numeric(tx["txn_amt"], errors="coerce"), -cap, cap).astype("float32")
        print(f"[WINS] clip txn_amt at ±{cap:.2f}")

    # ---------- 兩視角長表（省記憶體版） ----------
    cols_extra = ["min_of_day", "min5_bin", "is_night", "is_peak"]

    payer = tx[["from_acct", "to_acct", "txn_day", "txn_amt", "is_self_txn_num", "channel_type",
                "currency_bucket", "is_our_bank_from"] + cols_extra].copy()
    payer.rename(columns={"from_acct": "acct", "to_acct": "counterparty", "is_our_bank_from": "is_our_bank"},
                 inplace=True)
    payer["amt_out"] = payer["txn_amt"].astype("float32")
    payer["amt_in"] = np.float32(0.0)

    payee = tx[["to_acct", "from_acct", "txn_day", "txn_amt", "is_self_txn_num", "channel_type",
                "currency_bucket", "is_our_bank_to"] + cols_extra].copy()
    payee.rename(columns={"to_acct": "acct", "from_acct": "counterparty", "is_our_bank_to": "is_our_bank"},
                 inplace=True)
    payee["amt_out"] = np.float32(0.0)
    payee["amt_in"] = payee["txn_amt"].astype("float32")

    for df in (payer, payee):
        for c in ["acct", "counterparty", "channel_type", "currency_bucket"]:
            df[c] = df[c].astype("category")
        for c in ["txn_day", "min_of_day", "min5_bin"]:
            df[c] = df[c].astype("int32")
        for c in ["is_night", "is_peak", "is_self_txn_num", "is_our_bank"]:
            df[c] = df[c].astype("int8")
        df["txn_amt"] = df["txn_amt"].astype("float32")

    tx_long = pd.concat([payer, payee], axis=0, ignore_index=True, copy=False, sort=False)
    tx_long.dropna(subset=["acct", "txn_day"], inplace=True)
    tx_long["acct"] = tx_long["acct"].astype(str).str.strip()
    tx_long["txn_day"] = tx_long["txn_day"].astype(int)
    del payer, payee
    gc.collect()

    return tx, alert, pred, tx_long


# =============== 特徵（穩定款） ===============
def agg_features(df):
    """Aggregate core transaction statistics per account.

    This function groups the long-form transaction table by account and
    computes basic count and amount statistics, plus simple behavioral
    ratios such as self-transfer ratio, in/out ratio, and activity days.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form transaction table with at least:
        'acct', 'txn_day', 'amt_in', 'amt_out', 'txn_amt',
        'counterparty', 'is_self_txn_num', 'is_our_bank',
        'is_night', 'is_peak'.

    Returns
    -------
    pandas.DataFrame
        Wide-form table with one row per account and aggregated features.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["acct"])
    g = df.groupby("acct", as_index=False)
    base = g.agg(
        tx_cnt=("txn_day", "size"),
        in_cnt=("amt_in", lambda s: (s > 0).sum()),
        out_cnt=("amt_out", lambda s: (s > 0).sum()),
        amt_in_sum=("amt_in", "sum"),
        amt_out_sum=("amt_out", "sum"),
        amt_abs_mean=("txn_amt", lambda s: np.mean(np.abs(s))),
        amt_abs_std=("txn_amt", lambda s: np.std(np.abs(s))),
        amt_max=("txn_amt", "max"),
        active_days=("txn_day", lambda s: pd.Series(s).nunique()),
        uniq_ctp=("counterparty", pd.Series.nunique),
        self_ratio=("is_self_txn_num", lambda s: (s == 1).mean() if len(s) > 0 else 0.0),
        ourbank_ratio=("is_our_bank", "mean"),
        night_ratio=("is_night", "mean"),
        peak_ratio=("is_peak", "mean"),
    )
    base["net_flow"] = base["amt_in_sum"] - base["amt_out_sum"]
    base["in_out_ratio"] = np.where(base["out_cnt"] > 0, base["in_cnt"] / base["out_cnt"], 0.0)
    return base


def wide_count(df, col, prefix):
    """Create wide one-hot count and fraction features for a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form transaction table with at least 'acct' and `col`.
    col : str
        Column name of the categorical feature to expand.
    prefix : str
        Prefix to use for generated feature column names.

    Returns
    -------
    pandas.DataFrame
        Wide-form table with one row per account and generated count/pct features.
    """
    if col not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=["acct"])
    tmp = (df.groupby(["acct", col], as_index=False).size().rename(columns={"size": "cnt"}))
    if len(tmp) == 0:
        return pd.DataFrame(columns=["acct"])
    wide = tmp.pivot_table(index="acct", columns=col, values="cnt", aggfunc="sum", fill_value=0)
    wide.columns = [f"{prefix}_{str(c)}" for c in wide.columns]
    total = wide.sum(axis=1).replace(0, np.nan)
    frac = wide.div(total, axis=0).add_prefix(f"{prefix}_pct_")
    return pd.concat([wide, frac], axis=1).reset_index()


def counterparty_profile(df):
    """Compute entropy and concentration metrics of counterparties for one account."""
    vc = df["counterparty"].value_counts(normalize=True)
    if len(vc) == 0:
        return pd.Series({"ctp_entropy": 0.0, "ctp_herfindahl": 0.0})
    p = vc.values.astype(float)
    return pd.Series({
        "ctp_entropy": float(-(p * np.log(p + 1e-12)).sum()),
        "ctp_herfindahl": float((p ** 2).sum())
    })


def timebin_profile(df):
    """Summarize the distribution of transactions over 5-minute time bins."""
    vc = df["min5_bin"].value_counts()
    if len(vc) == 0:
        return pd.Series({"bin5_cnt": 0, "bin5_top1_ratio": 0.0, "bin5_entropy": 0.0})
    p = (vc / vc.sum()).values.astype(float)
    return pd.Series({
        "bin5_cnt": int(vc.shape[0]),
        "bin5_top1_ratio": float(p.max()),
        "bin5_entropy": float(-(p * np.log(p + 1e-12)).sum())
    })


def bucket_profile(df, col):
    """Compute entropy and top-share for a generic categorical bucket column."""
    vc = df[col].value_counts(normalize=True)
    if len(vc) == 0:
        return pd.Series({f"{col}_entropy": 0.0, f"{col}_top1": 0.0})
    p = vc.values.astype(float)
    return pd.Series({
        f"{col}_entropy": float(-(p * np.log(p + 1e-12)).sum()),
        f"{col}_top1": float(p.max())
    })


def build_activity_table(tx_long: pd.DataFrame) -> pd.DataFrame:
    """Build a compact activity table with basic volume and time-entropy features."""
    if len(tx_long) == 0:
        return pd.DataFrame(columns=["acct", "tx_cnt", "uniq_ctp", "bin5_entropy", "amt_abs_std"])
    g = tx_long.groupby("acct", as_index=False).agg(
        tx_cnt=("txn_day", "size"),
        uniq_ctp=("counterparty", pd.Series.nunique),
        amt_abs_std=("txn_amt", lambda s: float(np.std(np.abs(s)))),
    )
    tb = (tx_long.groupby(["acct", "min5_bin"], as_index=False).size()
          .rename(columns={"size": "cnt"}))
    ent = tb.groupby("acct")["cnt"].apply(lambda s: entropy_from_counts(s.values)).reset_index()
    ent.rename(columns={"cnt": "bin5_entropy"}, inplace=True)
    act = g.merge(ent, on="acct", how="left")
    act["bin5_entropy"] = act["bin5_entropy"].fillna(0.0)
    return act


def normalize_numeric_features(X_train: pd.DataFrame, X_pred: pd.DataFrame):
    """Apply winsorization and log transforms to numeric feature columns."""
    win_cols = [c for c in [
        "tx_cnt", "in_cnt", "out_cnt", "active_days", "uniq_ctp",
        "amt_in_sum", "amt_out_sum", "amt_abs_mean", "amt_abs_std", "amt_max", "net_flow",
    ] if c in X_train.columns]
    win_cols += [c for c in X_train.columns
                 if (c.startswith("chan_") or c.startswith("ccy_")) and not ("pct_" in c)]
    win_cols = sorted(set(win_cols))

    caps = compute_caps(X_train, win_cols, lo_q=0.005, hi_q=0.995)
    apply_caps(X_train, caps)
    apply_caps(X_pred, caps)

    log_cols = [c for c in [
        "tx_cnt", "in_cnt", "out_cnt", "active_days", "uniq_ctp",
        "amt_in_sum", "amt_out_sum", "amt_abs_mean", "amt_abs_std", "amt_max",
    ] if c in X_train.columns]
    signed_log_cols = [c for c in ["net_flow"] if c in X_train.columns]

    for c in log_cols:
        X_train[c] = np.log1p(X_train[c].clip(lower=0))
        X_pred[c] = np.log1p(X_pred[c].clip(lower=0))
    for c in signed_log_cols:
        X_train[c] = signed_log1p(X_train[c])
        X_pred[c] = signed_log1p(X_pred[c])

    return X_train, X_pred, caps, log_cols, signed_log_cols


# =============== 建樣本鍵（含 Hard Negatives） ===============
def build_train_keys(tx, alert, tx_long):
    """Construct labeled training keys with hard negative sampling."""
    # 正類
    pos = alert[["acct", "event_day"]].dropna().copy()
    pos["event_day"] = pos["event_day"].astype(int)
    pos = pos.drop_duplicates("acct")
    pos["label"] = 1
    pos["is_unlabeled"] = 0

    # 每帳最後交易日
    acc_last = (pd.concat([
        tx[["from_acct", "txn_day"]].rename(columns={"from_acct": "acct"}),
        tx[["to_acct", "txn_day"]].rename(columns={"to_acct": "acct"})
    ], ignore_index=True)
        .groupby("acct", as_index=False)["txn_day"].max()
        .rename(columns={"txn_day": "last_txn_day"}))

    # U 候選池
    tx_accts = set(pd.concat([tx["from_acct"], tx["to_acct"]]).astype(str))
    pool = sorted(list(tx_accts - set(pos["acct"])))
    neg_full = acc_last[acc_last["acct"].isin(pool)].copy()

    if len(neg_full) == 0:
        any_accts = pd.Series(list(tx_accts)).sample(
            n=min(len(pos), len(tx_accts)), random_state=RANDOM_STATE
        )
        fb = pd.DataFrame({"acct": any_accts}).merge(acc_last, on="acct", how="left")
        fb["event_day"] = fb["last_txn_day"].fillna(tx["txn_day"].max()).astype(int) + 1
        neg_full = fb[["acct", "event_day"]].copy()
    else:
        act = build_activity_table(tx_long)
        neg_full = neg_full.merge(act, on="acct", how="left")
        for col in ["tx_cnt", "uniq_ctp", "bin5_entropy", "amt_abs_std"]:
            neg_full[col] = neg_full[col].fillna(0)
        neg_full["event_day"] = neg_full["last_txn_day"] + 1
        neg_full = neg_full.sort_values(
            ["tx_cnt", "uniq_ctp", "bin5_entropy", "amt_abs_std"],
            ascending=[False, False, False, False]
        )

    need = int(max(1, len(pos) * NEG_POS_RATIO))
    neg = neg_full.head(min(need, len(neg_full)))[["acct", "event_day"]].copy()
    neg["label"] = 0
    neg["is_unlabeled"] = 1

    keys = pd.concat([pos, neg], ignore_index=True).dropna()
    keys["event_day"] = keys["event_day"].astype(int)
    keys = keys.drop_duplicates("acct")
    return keys, acc_last


# =============== 主流程 ===============
def main():
    """Entry point for feature engineering and dataset preparation."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx", default=TX_PATH)
    ap.add_argument("--alert", default=ALERT_PATH)
    ap.add_argument("--pred", default=PRED_PATH)
    ap.add_argument("--inference_only", action="store_true",
                    help="只產生 features_pred.csv（不讀 alert、不輸出 features_train.csv）")
    args, _ = ap.parse_known_args()

    tx, alert, pred, tx_long = read_and_normalize(
        args.tx, args.alert, args.pred, inference_only=args.inference_only
    )

    if not args.inference_only:
        train_keys, acc_last = build_train_keys(tx, alert, tx_long)
    else:
        acc_last = (pd.concat([
            tx[["from_acct", "txn_day"]].rename(columns={"from_acct": "acct"}),
            tx[["to_acct", "txn_day"]].rename(columns={"to_acct": "acct"})
        ], ignore_index=True)
            .groupby("acct", as_index=False)["txn_day"].max()
            .rename(columns={"txn_day": "last_txn_day"}))
        train_keys = None

    # 訓練視窗
    if not args.inference_only:
        tx_train = tx_long.merge(
            train_keys[["acct", "event_day", "label", "is_unlabeled"]],
            on="acct", how="inner"
        )
        tx_train["delta_days"] = tx_train["event_day"] - tx_train["txn_day"]
        tx_win = tx_train[(tx_train["delta_days"] >= 1) &
                          (tx_train["delta_days"] <= WINDOW_DAYS)].copy()
        if len(tx_win) == 0:
            for d in [90]:
                tx_win = tx_train[(tx_train["delta_days"] >= 1) &
                                  (tx_train["delta_days"] <= d)].copy()
                if len(tx_win) > 0:
                    print(f"[WARN] 改用 {d} 天窗：{len(tx_win)} 筆")
                    break
        if len(tx_win) == 0:
            tx_win = tx_train[tx_train["delta_days"] >= 1].copy()
            print(f"[WARN] 改用『事件日前所有歷史』：{len(tx_win)} 筆")

    # 待測視窗（anchor = last+1）
    pred_keys = pred[["acct"]].drop_duplicates().merge(
        (pd.concat([
            tx[["from_acct", "txn_day"]].rename(columns={"from_acct": "acct"}),
            tx[["to_acct", "txn_day"]].rename(columns={"to_acct": "acct"})
        ], ignore_index=True)
            .groupby("acct", as_index=False)["txn_day"].max()
            .rename(columns={"txn_day": "last_txn_day"})),
        on="acct", how="left"
    )
    pred_keys["event_day"] = pred_keys["last_txn_day"].fillna(tx["txn_day"].max()).astype(int) + 1

    tx_pred = tx_long.merge(pred_keys[["acct", "event_day"]], on="acct", how="inner")
    tx_pred["delta_days"] = tx_pred["event_day"] - tx_pred["txn_day"]
    txp_win = tx_pred[(tx_pred["delta_days"] >= 1) &
                      (tx_pred["delta_days"] <= WINDOW_DAYS)].copy()
    if len(txp_win) == 0:
        for d in [90]:
            txp_win = tx_pred[(tx_pred["delta_days"] >= 1) &
                              (tx_pred["delta_days"] <= d)].copy()
            if len(txp_win) > 0:
                print(f"[WARN] 待測改用 {d} 天窗：{len(txp_win)} 筆")
                break
    if len(txp_win) == 0:
        txp_win = tx_pred[tx_pred["delta_days"] >= 1].copy()
        print(f"[WARN] 待測改用『事件日前所有歷史』：{len(txp_win)} 筆")

    # 特徵（訓練）
    if not args.inference_only:
        feat_base = agg_features(tx_win)
        chan_wide = wide_count(tx_win, "channel_type", "chan")
        ccy_wide = wide_count(tx_win, "currency_bucket", "ccy")
        extra_ctp = safe_group_apply(tx_win.groupby("acct"), counterparty_profile).reset_index()
        extra_time = safe_group_apply(tx_win.groupby("acct"), timebin_profile).reset_index()
        extra_chan = safe_group_apply(tx_win.groupby("acct"),
                                      lambda d: bucket_profile(d, "channel_type")).reset_index()
        extra_ccy = safe_group_apply(tx_win.groupby("acct"),
                                     lambda d: bucket_profile(d, "currency_bucket")).reset_index()

        X_train = (train_keys[["acct", "label", "is_unlabeled"]]
                   .merge(feat_base, on="acct", how="left")
                   .merge(chan_wide, on="acct", how="left")
                   .merge(ccy_wide, on="acct", how="left")
                   .merge(extra_ctp, on="acct", how="left")
                   .merge(extra_time, on="acct", how="left")
                   .merge(extra_chan, on="acct", how="left")
                   .merge(extra_ccy, on="acct", how="left")
                   .fillna(0))

    # 特徵（待測）
    feat_pred = agg_features(txp_win)
    chan_pred = wide_count(txp_win, "channel_type", "chan")
    ccy_pred = wide_count(txp_win, "currency_bucket", "ccy")
    extra_ctp_p = safe_group_apply(txp_win.groupby("acct"), counterparty_profile).reset_index()
    extra_time_p = safe_group_apply(txp_win.groupby("acct"), timebin_profile).reset_index()
    extra_chan_p = safe_group_apply(txp_win.groupby("acct"),
                                    lambda d: bucket_profile(d, "channel_type")).reset_index()
    extra_ccy_p = safe_group_apply(txp_win.groupby("acct"),
                                   lambda d: bucket_profile(d, "currency_bucket")).reset_index()

    X_pred = (pred_keys[["acct"]]
              .merge(feat_pred, on="acct", how="left")
              .merge(chan_pred, on="acct", how="left")
              .merge(ccy_pred, on="acct", how="left")
              .merge(extra_ctp_p, on="acct", how="left")
              .merge(extra_time_p, on="acct", how="left")
              .merge(extra_chan_p, on="acct", how="left")
              .merge(extra_ccy_p, on="acct", how="left")
              .fillna(0))

    # 對齊欄位 + 數值正規化 + 輸出
    if not args.inference_only:
        feature_cols = [c for c in X_train.columns if c not in ["acct", "label", "is_unlabeled"]]
        for c in feature_cols:
            if c not in X_pred.columns:
                X_pred[c] = 0
        X_pred = X_pred[["acct"] + feature_cols]

        X_train, X_pred, caps, log_cols, signed_log_cols = normalize_numeric_features(X_train, X_pred)

        X_train.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
        with open(OUT_META, "w", encoding="utf-8") as f:
            json.dump({
                "feature_cols": feature_cols,
                "window_days": WINDOW_DAYS,
                "neg_pos_ratio": NEG_POS_RATIO,
                "caps": caps,
                "log_cols": log_cols,
                "signed_log_cols": signed_log_cols,
                "drift_topk": DRIFT_TOPK,
                "drift_dropped": []
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] 保存: {OUT_TRAIN}, {OUT_META}")
        print("Train shape:", X_train.shape, " Pred shape:", X_pred.shape)
        print("[CHECK] label 分佈：", X_train["label"].value_counts().to_dict())
    else:
        # inference-only：僅輸出待測特徵與空 meta（提醒不可用於訓練）
        feature_cols = [c for c in X_pred.columns if c not in ["acct"]]
        with open(OUT_META, "w", encoding="utf-8") as f:
            json.dump({
                "feature_cols": feature_cols,
                "window_days": WINDOW_DAYS,
                "neg_pos_ratio": None,
                "caps": {},
                "log_cols": [],
                "signed_log_cols": [],
                "drift_topk": 0,
                "drift_dropped": [],
                "note": "inference_only=true；僅供特徵推論，訓練請勿使用此 meta。"
            }, f, ensure_ascii=False, indent=2)
        print("[INFO] inference_only 模式：僅寫出 features_pred.csv / features_meta.json（caps/log 皆為空）")

    X_pred.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
    print(f"[OK] 保存: {OUT_PRED}")


if __name__ == "__main__":
    main()
