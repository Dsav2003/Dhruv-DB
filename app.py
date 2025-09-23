# app.py — IBR Streamlit Dashboard (upload-first) with EDA, Model Lab, Data-Quality experiments,
# leaderboard, permutation importance, partial dependence, simple backtest, and model card export.
# Drop-in ready for GitHub + Streamlit Cloud.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="IBR — AI in Financial Forecasting", layout="wide")

# Optional libs (graceful degradation)
HAS_STATSMODELS = False
try:
    from statsmodels.tsa.stattools import acf as sm_acf
    HAS_STATSMODELS = True
except Exception:
    pass

HAS_SCIPY = False
try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except Exception:
    pass

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =========================
# Helpers
# =========================
@st.cache_data
def load_uploaded_excel(file_bytes, sheet):
    """Read an uploaded Excel file (bytes) and return a cleaned dataframe."""
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    # Drop Excel helper columns like 'Unnamed: xx'
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    # Parse/sort date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def ensure_numeric(df, cols):
    """Coerce selected columns to numeric; non-convertible values become NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_returns(df, price_col="Close"):
    """Add simple and log returns; robust to non-numeric data."""
    if price_col not in df.columns:
        st.error(f"Price column '{price_col}' not found. Please map the correct column in the sidebar.")
        st.stop()
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    if out[price_col].isna().all():
        st.error(f"Column '{price_col}' contains no numeric values. Pick a different column or clean the data.")
        st.stop()
    out["Return"] = out[price_col].pct_change()
    out["LogReturn"] = np.log1p(out["Return"])
    return out

def engineer_features(
    df,
    price_col="Close",
    lags=(1, 5, 10),
    mas=(5, 10, 20),
    vol_windows=(5, 10, 20),
    sentiment_col=None,
    market_col=None,
):
    df = compute_returns(df, price_col=price_col)
    for L in lags:
        df[f"lag_ret_{L}"] = df["Return"].shift(L)
        df[f"lag_close_{L}"] = df[price_col].shift(L)
    for M in mas:
        df[f"sma_{M}"] = df[price_col].rolling(M, min_periods=1).mean()
        df[f"ema_{M}"] = df[price_col].ewm(span=M, adjust=False).mean()
    for W in vol_windows:
        df[f"vol_{W}"] = df["Return"].rolling(W, min_periods=2).std()

    if sentiment_col and sentiment_col in df.columns:
        s = pd.to_numeric(df[sentiment_col], errors="coerce")
        std = s.std(ddof=0)
        df["sentiment_z"] = (s - s.mean()) / (std if (std and std != 0) else 1)

    if market_col and market_col in df.columns:
        df = pd.get_dummies(df, columns=[market_col], drop_first=True)

    return df

def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

def _np_mae(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))

def _np_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def clean_feature_matrix(Xtr, Xte):
    """Drop all-NaN/inf cols; fill NaNs with train medians; align columns."""
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    Xte = Xte[Xtr.columns]  # align
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    return Xtr, Xte

def split_data(df, feature_cols, target_col, test_frac):
    n = len(df)
    cut = int(n * (1 - test_frac))
    Xtr, Xte = df[feature_cols].iloc[:cut], df[feature_cols].iloc[cut:]
    ytr, yte = df[target_col].iloc[:cut], df[target_col].iloc[cut:]
    return Xtr, Xte, ytr, yte, cut

def fit_predict_model(model_name, Xtr, Xte, ytr):
    """
    Fit supported models; return (model, yhat, feature_importance or None).
    Tree models use unscaled features; linear models use StandardScaler.
    """
    linear_like = {"Linear Regression", "Ridge", "Lasso", "ElasticNet"}
    tree_like = {"Random Forest", "GradientBoostingRegressor"}

    fi = None
    if model_name in tree_like:
        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_SEED,
                                          n_jobs=-1, min_samples_leaf=2)
        else:
            model = GradientBoostingRegressor(random_state=RANDOM_SEED)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
        return model, yhat, fi

    elif model_name in linear_like:
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "Lasso":
            model = Lasso(alpha=0.001, max_iter=5000, random_state=RANDOM_SEED)
        else:
            model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_SEED)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        model.fit(Xtr_s, ytr)
        yhat = model.predict(Xte_s)
        # No native feature_importances_ (skip)
        return model, yhat, None

    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_eval(df, target_col, features, model_name="Random Forest", test_frac=0.2):
    n = len(df)
    if n < 10:
        st.error("Not enough rows after feature engineering to train a model.")
        st.stop()
    cut = int(n * (1 - test_frac))
    if cut <= 0 or cut >= n:
        st.error("Invalid test fraction; adjust the slider in the sidebar.")
        st.stop()

    # Split
    Xtr_raw, Xte_raw, ytr, yte, cut = split_data(df, features, target_col, test_frac)
    # Clean matrices
    Xtr, Xte = clean_feature_matrix(Xtr_raw, Xte_raw)

    # Fit/predict
    model, yhat, fi = fit_predict_model(model_name, Xtr, Xte, ytr)

    # Metrics (NumPy only for max compatibility)
    rmse = _np_rmse(yte, yhat)
    mae = _np_mae(yte, yhat)
    try:
        da = directional_accuracy(yte, yhat)
    except Exception:
        da = np.nan

    return {"model": model, "y_test": yte, "y_pred": yhat, "RMSE": rmse, "MAE": mae, "DA": da, "fi": fi, "cut": cut, "Xtr": Xtr, "Xte": Xte}

def data_summary(df):
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notnull().sum(),
        "missing": df.isnull().sum(),
        "missing_%": (df.isnull().mean()*100).round(2),
        "zeros_%": ((df == 0).mean()*100).round(2),
        "nunique": df.nunique(),
    }).sort_index()

# =========================
# UI — Upload-first
# =========================
st.sidebar.title("Upload your IBR data (Excel)")
upl = st.sidebar.file_uploader("Drag & drop your .xlsx", type=["xlsx", "xls"])

if upl is None:
    st.title("IBR Dashboard — AI in Financial Forecasting")
    st.info("👈 Upload your Excel file in the sidebar to begin. The app is designed for your IBR sheet.")
    st.stop()

# Let user pick the sheet (default to 'Cleaned' if present)
with pd.ExcelFile(upl) as xls:
    sheets = xls.sheet_names
default_idx = sheets.index("Cleaned") if "Cleaned" in sheets else 0
sheet_choice = st.sidebar.selectbox("Choose sheet", sheets, index=default_idx)

# Load selected sheet
df_raw = load_uploaded_excel(upl.getvalue(), sheet_choice)

# Coerce common numeric cols
df_raw = ensure_numeric(df_raw, [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_raw.columns])

# Column mapping (in case your column names differ)
st.sidebar.subheader("Column Mapping")
date_col = st.sidebar.selectbox(
    "Date column", options=df_raw.columns, index=list(df_raw.columns).index("Date") if "Date" in df_raw.columns else 0
)
price_col = st.sidebar.selectbox(
    "Price (Close)", options=df_raw.columns, index=list(df_raw.columns).index("Close") if "Close" in df_raw.columns else 0
)
vol_col = st.sidebar.selectbox(
    "Volume", options=["<none>"] + list(df_raw.columns),
    index=(["<none>"] + list(df_raw.columns)).index("Volume") if "Volume" in df_raw.columns else 0,
)
sent_col = st.sidebar.selectbox(
    "Sentiment column", options=["<none>"] + list(df_raw.columns),
    index=(["<none>"] + list(df_raw.columns)).index("Sentiment_Score") if "Sentiment_Score" in df_raw.columns else 0,
)
mkt_col = st.sidebar.selectbox(
    "Market condition column", options=["<none>"] + list(df_raw.columns),
    index=(["<none>"] + list(df_raw.columns)).index("Market_Condition") if "Market_Condition" in df_raw.columns else 0,
)

# Date filter
df = df_raw.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
dmin, dmax = df[date_col].min(), df[date_col].max()
dr = st.sidebar.date_input("Date range", value=[dmin, dmax], min_value=dmin, max_value=dmax)
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    df = df[(df[date_col] >= pd.to_datetime(dr[0])) & (df[date_col] <= pd.to_datetime(dr[1]))].reset_index(drop=True)

# Quality toggles (shown only if flags exist)
st.sidebar.subheader("Data Quality")
miss_col = "Missing_Flag" if "Missing_Flag" in df.columns else None
noise_col = "Noise_Flag" if "Noise_Flag" in df.columns else None
drop_miss = st.sidebar.checkbox("Drop Missing_Flag==1", value=True) if miss_col else False
drop_noise = st.sidebar.checkbox("Drop Noise_Flag==1", value=False) if noise_col else False
if miss_col and drop_miss:
    df = df[df[miss_col] != 1]
if noise_col and drop_noise:
    df = df[df[noise_col] != 1]

# Feature knobs
st.sidebar.subheader("Features")
lags = st.sidebar.multiselect("Lag days", [1, 3, 5, 10], default=[1, 5, 10])
mas = st.sidebar.multiselect("Moving avgs", [5, 10, 20], default=[5, 10, 20])
vols = st.sidebar.multiselect("Vol windows", [5, 10, 20], default=[5, 10, 20])

# =========================
# Build features & target
# =========================
df_feat = df.rename(columns={date_col: "Date", price_col: "Close"}).copy()

# Alias 'Volume' if user mapped a different column name
if vol_col != "<none>" and vol_col in df.columns:
    df_feat["Volume"] = pd.to_numeric(df[vol_col], errors="coerce")

sentiment_in = None if sent_col == "<none>" else sent_col
market_in = None if mkt_col == "<none>" else mkt_col

df_feat = engineer_features(
    df_feat,
    price_col="Close",
    lags=lags,
    mas=mas,
    vol_windows=vols,
    sentiment_col=sentiment_in,
    market_col=market_in,
)
df_feat = df_feat.dropna().reset_index(drop=True)

st.sidebar.subheader("Target & Model")
target = st.sidebar.selectbox("Target", ["Next-day Return", "Next-day Close"])
if target == "Next-day Return":
    df_feat["Target"] = df_feat["Return"].shift(-1)
else:
    df_feat["Target"] = df_feat["Close"].shift(-1)
df_feat = df_feat.dropna().reset_index(drop=True)

# Features for modeling
feature_prefixes = ["lag_ret_", "lag_close_", "sma_", "ema_", "vol_", "sentiment_z", "Volume"]
base_feats = [c for c in df_feat.columns if any(c.startswith(p) for p in feature_prefixes)]
for req in ["Close", "Return"]:
    if (req in df_feat.columns) and (req not in base_feats):
        base_feats.append(req)

# =========================
# Header & KPIs
# =========================
st.title("IBR Dashboard — AI in Financial Forecasting")
st.caption("Aligned to objectives: predictive accuracy • data quality impact • ethical guardrails")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows (raw)", f"{len(df_raw):,}")
with c2:
    st.metric("Rows (modeled)", f"{len(df_feat):,}")
with c3:
    ann_vol = df_feat["Return"].std() * np.sqrt(252) if "Return" in df_feat.columns else np.nan
    st.metric("Annualized Vol", f"{ann_vol:.2%}" if pd.notna(ann_vol) else "—")
with c4:
    cumret = (1 + df_feat.get("Return", pd.Series([0]))).prod() - 1 if "Return" in df_feat.columns else np.nan
    st.metric("Cumulative Return", f"{cumret:.2%}" if pd.notna(cumret) else "—")

# =========================
# Section 1: EDA — Data clarity
# =========================
st.subheader("Data Dictionary & Quality Summary")
st.dataframe(data_summary(df), use_container_width=True)
if "Market_Condition" in df.columns:
    st.caption("Category frequency (raw):")
    st.write(df["Market_Condition"].value_counts(dropna=False))

st.subheader("Exploratory Data Analysis")
left, right = st.columns([2, 1])
with left:
    st.write("Price (Close)")
    fig = plt.figure()
    plt.plot(df_feat["Date"], df_feat["Close"])
    plt.xlabel("Date"); plt.ylabel("Close")
    st.pyplot(fig)
with right:
    st.write("Box-Whisker: Returns")
    fig2 = plt.figure()
    if "Return" in df_feat.columns:
        plt.boxplot(df_feat["Return"].dropna())
    st.pyplot(fig2)

# Rolling stats
st.subheader("Rolling Stats & Regime View")
if "Return" in df_feat.columns:
    roll = df_feat.set_index("Date")[["Return"]].copy()
    roll["vol_20d"] = roll["Return"].rolling(20, min_periods=5).std()
    figr = plt.figure()
    plt.plot(roll.index, roll["vol_20d"])
    plt.ylabel("20-Day Rolling Volatility")
    plt.xlabel("Date")
    st.pyplot(figr)

# ACF (optional)
st.subheader("Return Auto-Correlation (lag 1–10)")
if HAS_STATSMODELS and "Return" in df_feat.columns:
    series = df_feat["Return"].dropna().values
    ac = sm_acf(series, nlags=10, fft=False)
    fig_acf = plt.figure()
    plt.bar(range(len(ac)), ac)
    plt.xticks(range(len(ac)))
    plt.xlabel("Lag"); plt.ylabel("ACF")
    st.pyplot(fig_acf)
else:
    st.info("Install statsmodels to view ACF (optional).")

# Correlation heatmap
num_cols = [c for c in base_feats if df_feat[c].dtype != "O"]
if len(num_cols) > 1:
    st.write("Correlation Heatmap (engineered features)")
    corr = df_feat[num_cols].corr()
    fig3 = plt.figure()
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    st.pyplot(fig3)

# =========================
# Section 2: Model Lab
# =========================
model_name = st.sidebar.selectbox(
    "Model", ["Random Forest", "Linear Regression", "Ridge", "Lasso", "ElasticNet", "GradientBoostingRegressor"]
)
test_frac = st.sidebar.slider("Test fraction", 0.1, 0.5, 0.2, 0.05)

st.subheader("Model Lab — Train & Evaluate")
if len(df_feat) < 40 or len(base_feats) < 3:
    st.warning("Not enough rows/features after filtering. Try fewer lags/MAs or expand date range.")
else:
    res = train_eval(df_feat, "Target", base_feats, model_name=model_name, test_frac=float(test_frac))
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{res['RMSE']:.4f}")
    m2.metric("MAE", f"{res['MAE']:.4f}")
    da_str = "N/A" if (res['DA'] is None or np.isnan(res['DA'])) else f"{res['DA']*100:.1f}%"
    m3.metric("Directional Accuracy", da_str)

    st.write("Prediction vs Actual (Test Window)")
    test_idx = df_feat.index[res["cut"]:]
    plot_df = pd.DataFrame({
        "Date": df_feat.loc[test_idx, "Date"].values,
        "Actual": np.asarray(res["y_test"]).ravel(),
        "Pred": np.asarray(res["y_pred"]).ravel(),
    })
    fig4 = plt.figure()
    plt.plot(plot_df["Date"], plot_df["Actual"], label="Actual")
    plt.plot(plot_df["Date"], plot_df["Pred"], label="Pred")
    plt.xlabel("Date"); plt.ylabel("Target")
    plt.legend()
    st.pyplot(fig4)

    # Feature importance (tree models)
    if res["fi"] is not None:
        st.write("Top Feature Importance")
        fi = res["fi"].head(15)
        fig5 = plt.figure()
        plt.barh(fi.index[::-1], fi.values[::-1])
        plt.xlabel("Importance")
        st.pyplot(fig5)

    # Leaderboard (same split)
    st.subheader("Model Leaderboard (same split)")
    bench_models = ["Random Forest", "Linear Regression", "Ridge", "Lasso", "ElasticNet", "GradientBoostingRegressor"]
    rows = []
    # Prepare split once to keep identical split across models
    Xtr_raw, Xte_raw, ytr, yte, _ = split_data(df_feat, base_feats, "Target", float(test_frac))
    Xtr_b, Xte_b = clean_feature_matrix(Xtr_raw, Xte_raw)
    for name in bench_models:
        mdl, yhat_b, _fi = fit_predict_model(name, Xtr_b, Xte_b, ytr)
        rows.append({
            "Model": name,
            "RMSE": _np_rmse(yte, yhat_b),
            "MAE": _np_mae(yte, yhat_b),
            "DirAcc": directional_accuracy(yte, yhat_b) if "Return" in df_feat.columns else np.nan
        })
    leader = pd.DataFrame(rows).sort_values("RMSE")
    st.dataframe(leader, use_container_width=True)

    # Permutation importance (model-agnostic on test set)
    st.subheader("Permutation Importance (RMSE increase on shuffle)")
    try:
        Xte_pi = res["Xte"].copy()
        yte_pi = res["y_test"]
        base_rmse = _np_rmse(yte_pi, res["model"].predict(Xte_pi))
        pimps = {}
        cols_for_pi = list(Xte_pi.columns)
        rng = np.random.RandomState(RANDOM_SEED)
        for col in cols_for_pi:
            vals = []
            for _ in range(10):
                xcopy = Xte_pi.copy()
                shuf = xcopy[col].values.copy()
                rng.shuffle(shuf)
                xcopy[col] = shuf
                yhat_s = res["model"].predict(xcopy)
                vals.append(_np_rmse(yte_pi, yhat_s))
            pimps[col] = float(np.mean(vals) - base_rmse)
        pimps = pd.Series(pimps).sort_values(ascending=False)
        fig_pi = plt.figure(figsize=(4,3))
        top = pimps.head(12)[::-1]
        plt.barh(top.index, top.values)
        plt.xlabel("Δ RMSE (higher = more important)")
        st.pyplot(fig_pi)
    except Exception as e:
        st.info(f"Permutation importance skipped: {e}")

    # Partial Dependence for top feature (tree models)
    st.subheader("Partial Dependence (Top Feature)")
    if res.get("fi") is not None and len(res["fi"]) > 0:
        top_feat = res["fi"].index[0]
        n = len(df_feat); cut = int(n*(1-float(test_frac)))
        Xtr_pd = df_feat[base_feats].iloc[:cut].copy()
        rng_vals = np.linspace(Xtr_pd[top_feat].quantile(0.05), Xtr_pd[top_feat].quantile(0.95), 30)
        Xgrid = Xtr_pd.sample(min(200, len(Xtr_pd)), random_state=RANDOM_SEED).reset_index(drop=True)
        preds = []
        for v in rng_vals:
            Xtmp = Xgrid.copy()
            Xtmp[top_feat] = v
            preds.append(res["model"].predict(Xtmp).mean())
        fig_pd = plt.figure()
        plt.plot(rng_vals, preds)
        plt.xlabel(top_feat); plt.ylabel("Predicted Target")
        st.pyplot(fig_pd)
    else:
        st.caption("Partial dependence shown when a tree model (with importances) is selected.")

    # Directional confusion (only meaningful for Return target)
    st.subheader("Directional Confusion Matrix")
    def _predicted_return_series(y_pred, df_feat, cut, target_kind):
        if target_kind == "Next-day Return":
            return np.asarray(y_pred).ravel()
        # convert predicted next-day Close → predicted next-day Return (relative to today's Close)
        prev_close = df_feat["Close"].iloc[cut:].values
        pred_close = np.asarray(y_pred).ravel()
        return (pred_close / prev_close) - 1.0

    y_true_dir = None
    y_pred_dir = None
    try:
        n_all = len(df_feat); cut_all = int(n_all*(1-float(test_frac)))
        if target == "Next-day Return":
            y_true_dir = (df_feat["Target"].iloc[cut_all:] > 0).astype(int).values
        else:
            # True next-day return derived from true next-day close
            next_close_true = df_feat["Target"].iloc[cut_all:].values
            today_close = df_feat["Close"].iloc[cut_all:].values
            true_ret = (next_close_true / today_close) - 1.0
            y_true_dir = (true_ret > 0).astype(int)

        y_pred_ret = _predicted_return_series(res["y_pred"], df_feat, cut_all, target)
        y_pred_dir = (y_pred_ret > 0).astype(int)
        TP = int(((y_true_dir==1)&(y_pred_dir==1)).sum())
        TN = int(((y_true_dir==0)&(y_pred_dir==0)).sum())
        FP = int(((y_true_dir==0)&(y_pred_dir==1)).sum())
        FN = int(((y_true_dir==1)&(y_pred_dir==0)).sum())
        cm = pd.DataFrame([[TN, FP],[FN, TP]],
                          columns=["Pred:Down","Pred:Up"],
                          index=["Actual:Down","Actual:Up"])
        st.dataframe(cm)
        precision_up = TP / max(TP+FP,1); recall_up = TP / max(TP+FN,1)
        st.caption(f"Precision(up)={precision_up:.2f}  Recall(up)={recall_up:.2f}")
    except Exception as e:
        st.info(f"Directional view skipped: {e}")

# =========================
# Section 3: Data-Quality Lab (Objective 2)
# =========================
st.subheader("Data-Quality Experiment")
choice = st.radio("Imputation strategy", ["Drop rows", "Forward fill", "Time interpolate"], horizontal=True)
exp_df = df.copy()
num_cols = [c for c in exp_df.columns if exp_df[c].dtype != "O"]

if choice == "Forward fill":
    exp_df[num_cols] = exp_df[num_cols].ffill()
elif choice == "Time interpolate":
    if "Date" in exp_df.columns:
        exp_df = exp_df.set_index(date_col).sort_index()
        exp_df[num_cols] = exp_df[num_cols].interpolate(method="time")
        exp_df = exp_df.reset_index().rename(columns={date_col: date_col})
    else:
        exp_df[num_cols] = exp_df[num_cols].interpolate()

# rebuild features/target with exp_df
exp_feat = exp_df.rename(columns={date_col:"Date", price_col:"Close"}).copy()
if vol_col != "<none>" and vol_col in exp_df.columns:
    exp_feat["Volume"] = pd.to_numeric(exp_df[vol_col], errors="coerce")
exp_feat = engineer_features(exp_feat, price_col="Close", lags=lags, mas=mas, vol_windows=vols,
                             sentiment_col=(None if sent_col=="<none>" else sent_col),
                             market_col=(None if mkt_col=="<none>" else mkt_col)).dropna().reset_index(drop=True)
exp_feat["Target"] = (exp_feat["Return"].shift(-1) if target=="Next-day Return" else exp_feat["Close"].shift(-1))
exp_feat = exp_feat.dropna().reset_index(drop=True)

exp_feats = [c for c in exp_feat.columns if any(c.startswith(p) for p in ["lag_ret_","lag_close_","sma_","ema_","vol_","sentiment_z","Volume"])]
for req in ["Close","Return"]:
    if req in exp_feat.columns and req not in exp_feats: exp_feats.append(req)

if len(exp_feat) >= 40 and len(exp_feats) >= 3:
    res_q = train_eval(exp_feat, "Target", exp_feats, model_name=model_name, test_frac=float(test_frac))
    st.write(pd.DataFrame([{"Strategy": choice, "RMSE": res_q["RMSE"], "MAE": res_q["MAE"], "DirAcc": res_q["DA"]}]))
else:
    st.info("Not enough rows/features post-imputation for this experiment.")

# Noise sensitivity
st.subheader("Noise Sensitivity")
noise_bp = st.slider("Add Gaussian noise (bps on Close)", 0, 50, 0, 5,
                     help="1 bp = 0.01%. Injected as N(0, bps*0.0001).")
if noise_bp > 0:
    tmp = df.copy()
    scale = noise_bp * 0.0001
    tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce") * (1 + np.random.normal(0, scale, size=len(tmp)))
    tmp_feat = tmp.rename(columns={date_col:"Date", price_col:"Close"})
    if vol_col != "<none>" and vol_col in tmp.columns:
        tmp_feat["Volume"] = pd.to_numeric(tmp[vol_col], errors="coerce")
    tmp_feat = engineer_features(tmp_feat, price_col="Close", lags=lags, mas=mas, vol_windows=vols,
                                 sentiment_col=(None if sent_col=="<none>" else sent_col),
                                 market_col=(None if mkt_col=="<none>" else mkt_col)).dropna().reset_index(drop=True)
    tmp_feat["Target"] = (tmp_feat["Return"].shift(-1) if target=="Next-day Return" else tmp_feat["Close"].shift(-1))
    tmp_feat = tmp_feat.dropna().reset_index(drop=True)
    tmp_feats = [c for c in tmp_feat.columns if any(c.startswith(p) for p in ["lag_ret_","lag_close_","sma_","ema_","vol_","sentiment_z","Volume"])]
    for req in ["Close","Return"]:
        if req in tmp_feat.columns and req not in tmp_feats: tmp_feats.append(req)
    if len(tmp_feat) >= 40 and len(tmp_feats) >= 3:
        res_noise = train_eval(tmp_feat, "Target", tmp_feats, model_name=model_name, test_frac=float(test_frac))
        st.write(pd.DataFrame([{"Noise (bps)": noise_bp, "RMSE": res_noise["RMSE"], "MAE": res_noise["MAE"], "DirAcc": res_noise["DA"]}]))

# Train vs Test shift (KS)
st.subheader("Train vs Test Shift Check (KS)")
if HAS_SCIPY:
    def ks_shift(col):
        tr = df_feat[col].iloc[:int(len(df_feat)*(1-float(test_frac)))]
        te = df_feat[col].iloc[int(len(df_feat)*(1-float(test_frac))):]
        tr, te = tr.dropna(), te.dropna()
        if len(tr)>20 and len(te)>20:
            return float(ks_2samp(tr, te).statistic)
        return np.nan
    shift_cols = [c for c in ["Return","sma_5","sma_10","vol_10"] if c in df_feat.columns]
    if shift_cols:
        st.dataframe(pd.DataFrame({"feature": shift_cols, "KS_stat":[ks_shift(c) for c in shift_cols]}), use_container_width=True)
else:
    st.info("Install scipy to compute KS shift (optional).")

# =========================
# Section 4: Simple Backtest (finance linkage)
# =========================
st.subheader("Simple Next-Day Strategy Backtest")
thr = st.slider("Signal threshold (predicted next-day return, %)", 0.0, 1.0, 0.10, 0.05) / 100.0
fee_bp = st.slider("Round-trip cost (bps)", 0, 50, 5, 5) / 10000.0

# Prepare predicted returns from the last trained 'res' if available
if 'res' in locals():
    n_all = len(df_feat); cut_all = int(n_all*(1-float(test_frac)))
    if target == "Next-day Return":
        pred_ret = np.asarray(res["y_pred"]).ravel()
    else:
        # Convert predicted next-day Close → predicted next-day Return (vs today's Close)
        pred_close = np.asarray(res["y_pred"]).ravel()
        today_close = df_feat["Close"].iloc[cut_all:].values
        pred_ret = (pred_close / today_close) - 1.0

    # Align realized next-day return
    realized_ret = df_feat["Return"].shift(-1).iloc[cut_all:].fillna(0).values
    # Signal (long only) when predicted return > threshold
    sig = (pred_ret > thr).astype(int)
    trade = sig * (realized_ret - fee_bp)
    equity = (1 + pd.Series(trade)).cumprod()
    if len(equity) > 0:
        cagr = equity.iloc[-1]**(252/len(equity)) - 1
        sharpe = (np.sqrt(252)*np.mean(trade)/np.std(trade)) if np.std(trade)>0 else np.nan
        dd = (equity / equity.cummax() - 1).min()
        st.write(f"CAGR: **{cagr:.2%}** | Sharpe: **{(sharpe if not np.isnan(sharpe) else 0):.2f}** | Max Drawdown: **{dd:.2%}** | Hit Rate: **{(trade>0).mean():.1%}**")
        fig_bt = plt.figure()
        plt.plot(df_feat["Date"].iloc[cut_all:].values, equity.values)
        plt.ylabel("Equity (1.0 = start)"); plt.xlabel("Date")
        st.pyplot(fig_bt)
    else:
        st.info("Not enough test samples to run the backtest.")
else:
    st.info("Train a model in the Model Lab section to enable the backtest.")

# =========================
# Section 5: Ethics / Governance & Model Card
# =========================
with st.expander("Ethical & Regulatory Checklist"):
    st.markdown("""
- **Transparency**: Feature choices (lags/MAs/vol windows), target definition (next-day), fixed split.
- **Accountability**: Export the model card below; log parameters, data ranges, and metrics.
- **Fairness**: Compare performance under different imputation/noise settings; monitor train-test shifts (KS).
- **Overfitting Guardrails**: Prefer walk-forward/expanding-window CV before any live deployment.
- **Regulatory**: Align with SEBI/ESMA model risk guidance; this dashboard is an educational prototype — **not investment advice**.
""")

st.subheader("Model Card (Download)")
if 'res' in locals():
    card = {
        "dataset": {
            "uploaded_sheet": sheet_choice,
            "rows_raw": int(len(df_raw)),
            "rows_modeled": int(len(df_feat)),
            "date_range": [str(df_feat["Date"].min()), str(df_feat["Date"].max())]
        },
        "columns": {
            "date": date_col,
            "price": price_col,
            "volume": None if vol_col=="<none>" else vol_col,
            "sentiment": None if sent_col=="<none>" else sent_col,
            "market_condition": None if mkt_col=="<none>" else mkt_col
        },
        "features": base_feats,
        "target": target,
        "model": model_name,
        "test_fraction": float(test_frac),
        "metrics": {
            "RMSE": float(res["RMSE"]),
            "MAE": float(res["MAE"]),
            "DirectionalAccuracy": None if (res["DA"] is None or np.isnan(res["DA"])) else float(res["DA"])
        },
        "random_seed": RANDOM_SEED
    }
    import json
    st.download_button("Download model_card.json",
                       data=json.dumps(card, indent=2),
                       file_name="model_card.json",
                       mime="application/json")
    # Optional: predictions CSV
    out_pred = pd.DataFrame({
        "Date": df_feat["Date"].iloc[res["cut"]:].values,
        "y_test": np.asarray(res["y_test"]).ravel(),
        "y_pred": np.asarray(res["y_pred"]).ravel()
    })
    csv_bytes = out_pred.to_csv(index=False).encode()
    st.download_button("Download predictions.csv", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Train a model to enable the Model Card and predictions download.")
