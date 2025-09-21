import streamlit as st
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="IBR — AI in Financial Forecasting", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_uploaded_excel(file_bytes: bytes, sheet: str | int):
    """Read an uploaded Excel file (bytes) and return a cleaned dataframe."""
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    # drop Excel helper columns like 'Unnamed: xx'
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    # parse/sort date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_returns(df, price_col="Close"):
    df = df.copy()
    df["Return"] = df[price_col].pct_change()
    df["LogReturn"] = np.log1p(df["Return"])
    return df

def engineer_features(
    df,
    price_col="Close",
    lags=(1, 5, 10),
    mas=(5, 10, 20),
    vol_windows=(5, 10, 20),
    sentiment_col=None,
    market_col=None,
):
    df = df.copy()
    df = compute_returns(df, price_col=price_col)
    for L in lags:
        df[f"lag_ret_{L}"] = df["Return"].shift(L)
        df[f"lag_close_{L}"] = df[price_col].shift(L)
    for M in mas:
        df[f"sma_{M}"] = df[price_col].rolling(M).mean()
        df[f"ema_{M}"] = df[price_col].ewm(span=M, adjust=False).mean()
    for W in vol_windows:
        df[f"vol_{W}"] = df["Return"].rolling(W).std()

    if sentiment_col and sentiment_col in df.columns:
        s = df[sentiment_col]
        std = s.std(ddof=0)
        df["sentiment_z"] = (s - s.mean()) / (std if std != 0 else 1)

    if market_col and market_col in df.columns:
        df = pd.get_dummies(df, columns=[market_col], drop_first=True)

    return df

def directional_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

def train_eval(df, target_col, features, model_name="Random Forest", test_frac=0.2):
    n = len(df)
    cut = int(n * (1 - test_frac))
    Xtr, Xte = df[features].iloc[:cut], df[features].iloc[cut:]
    ytr, yte = df[target_col].iloc[:cut], df[target_col].iloc[cut:]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=2
        )
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    else:  # Linear Regression
        model = LinearRegression()
        model.fit(Xtr_s, ytr)
        yhat = model.predict(Xte_s)
        fi = None

    rmse = mean_squared_error(yte, yhat, squared=False)
    mae = mean_absolute_error(yte, yhat)
    try:
        da = directional_accuracy(yte, yhat)
    except Exception:
        da = np.nan

    return {"model": model, "y_test": yte, "y_pred": yhat, "RMSE": rmse, "MAE": mae, "DA": da, "fi": fi, "cut": cut}

# -------------------------
# UI — Sidebar (Upload-first)
# -------------------------
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
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)
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

# -------------------------
# Build features & target
# -------------------------
df_feat = df.rename(columns={date_col: "Date", price_col: "Close"}).copy()
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

# -------------------------
# Header & KPIs
# -------------------------
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

# -------------------------
# EDA
# -------------------------
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

# -------------------------
# Model Lab
# -------------------------
model_name = st.sidebar.selectbox("Model", ["Random Forest", "Linear Regression"])
test_frac = st.sidebar.slider("Test fraction", 0.1, 0.5, 0.2, 0.05)

st.subheader("Model Lab — Train & Evaluate")
if len(df_feat) < 40 or len(base_feats) < 3:
    st.warning("Not enough rows/features after filtering. Try fewer lags/MAs or expand date range.")
else:
    res = train_eval(df_feat, "Target", base_feats, model_name=model_name, test_frac=float(test_frac))
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{res['RMSE']:.4f}")
    m2.metric("MAE", f"{res['MAE']:.4f}")
    m3.metric("Directional Accuracy", "N/A" if np.isnan(res['DA']) else f"{res['DA']*100:.1f}%")

    st.write("Prediction vs Actual (Test Window)")
    test_idx = df_feat.index[res["cut"]:]
    plot_df = pd.DataFrame({
        "Date": df_feat.loc[test_idx, "Date"].values,
        "Actual": res["y_test"].values,
        "Pred": res["y_pred"],
    })
    fig4 = plt.figure()
    plt.plot(plot_df["Date"], plot_df["Actual"], label="Actual")
    plt.plot(plot_df["Date"], plot_df["Pred"], label="Pred")
    plt.xlabel("Date"); plt.ylabel("Target")
    plt.legend()
    st.pyplot(fig4)

    if res["fi"] is not None:
        st.write("Top Feature Importance (Random Forest)")
        fi = res["fi"].head(15)
        fig5 = plt.figure()
        plt.barh(fi.index[::-1], fi.values[::-1])
        plt.xlabel("Importance")
        st.pyplot(fig5)

# -------------------------
# Ethics / Governance
# -------------------------
with st.expander("Ethical & Regulatory Checklist"):
    st.markdown("""
- **Transparency**: Document feature choices (lags, MAs, vols), target definition, and test split.
- **Accountability**: Record model type and parameters when exporting results.
- **Fairness**: Check for distribution shifts when toggling data-quality filters.
- **Overfitting Guardrails**: Prefer walk-forward or expanding-window CV before any live decisions.
- **Regulatory**: Align with SEBI/ESMA model risk management guidance.
""")
