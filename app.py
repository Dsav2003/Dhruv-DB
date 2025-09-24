# app.py — IBR Streamlit Dashboard (upload-first) with improved UI/UX, scaling, and robust imputation
# - Plotly charts (responsive, rangeslider, log y, dual panels, tooltips)
# - Feature engineering, multi-model lab, leaderboard, explainability
# - Data-quality lab (imputation, noise, KS shift) — fixed KeyError in time interpolate
# - Simple backtest + governance/model card
# - NOTE: Return Auto-Correlation section removed (no statsmodels dependency)

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional libs (graceful degradation)
HAS_SCIPY = False
try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except Exception:
    pass

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(page_title="IBR — AI in Financial Forecasting", layout="wide", page_icon="📈")

# -------------------- small CSS polish --------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
      .metric-card {border:1px solid rgba(49,51,63,0.2); border-radius:14px; padding:10px 12px; background: rgba(250,250,250,0.7);}
      .metric-card .stMetric {text-align:center;}
      .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
      .stTabs [data-baseweb="tab"] {padding: 10px 14px;}
      .caption-small {font-size: 0.85rem; color: #666;}
      .table-small .dataframe td, .table-small .dataframe th {font-size: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
@st.cache_data
def load_uploaded_excel(file_bytes, sheet):
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_returns(df, price_col="Close"):
    if price_col not in df.columns:
        st.error(f"Price column '{price_col}' not found. Map the correct column in the sidebar.")
        st.stop()
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    if out[price_col].isna().all():
        st.error(f"Column '{price_col}' has no numeric values. Pick another column or clean the data.")
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
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    Xte = Xte[Xtr.columns]
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
        return model, yhat, None

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

    Xtr_raw, Xte_raw, ytr, yte, cut = split_data(df, features, target_col, test_frac)
    Xtr, Xte = clean_feature_matrix(Xtr_raw, Xte_raw)
    model, yhat, fi = fit_predict_model(model_name, Xtr, Xte, ytr)
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

# -------------------- Sidebar (UI Options) --------------------
st.sidebar.header("Upload & Options")
upl = st.sidebar.file_uploader("Upload your Excel (.xlsx/.xls)", type=["xlsx", "xls"])
if upl is None:
    st.title("IBR Dashboard — AI in Financial Forecasting")
    st.info("👈 Upload your Excel file to begin.")
    st.stop()

with pd.ExcelFile(upl) as xls:
    sheets = xls.sheet_names
idx_default = sheets.index("Cleaned") if "Cleaned" in sheets else 0
sheet_choice = st.sidebar.selectbox("Sheet", sheets, index=idx_default)

# Chart options (affect all major charts)
st.sidebar.subheader("Chart Options")
chart_height = st.sidebar.slider("Chart height (px)", 350, 900, 520, 10)
use_log_y = st.sidebar.checkbox("Log scale for price", value=False)
show_rangeslider = st.sidebar.checkbox("Show range slider (time)", value=True)
show_markers = st.sidebar.checkbox("Show markers on lines", value=False)
template_choice = st.sidebar.selectbox("Theme", ["plotly_white", "plotly", "ggplot2", "seaborn", "simple_white", "plotly_dark"], index=0)

# Feature params
st.sidebar.subheader("Feature Engineering")
lags = st.sidebar.multiselect("Lag days", [1, 3, 5, 10], default=[1, 5, 10])
mas = st.sidebar.multiselect("Moving avgs", [5, 10, 20], default=[5, 10, 20])
vols = st.sidebar.multiselect("Vol windows", [5, 10, 20], default=[5, 10, 20])

# Load & map
df_raw = load_uploaded_excel(upl.getvalue(), sheet_choice)
df_raw = ensure_numeric(df_raw, [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_raw.columns])

st.sidebar.subheader("Column Mapping")
date_col = st.sidebar.selectbox("Date", options=df_raw.columns, index=list(df_raw.columns).index("Date") if "Date" in df_raw.columns else 0)
price_col = st.sidebar.selectbox("Price (Close)", options=df_raw.columns, index=list(df_raw.columns).index("Close") if "Close" in df_raw.columns else 0)
vol_col = st.sidebar.selectbox(
    "Volume",
    options=["<none>"] + list(df_raw.columns),
    index=(["<none>"] + list(df_raw.columns)).index("Volume") if "Volume" in df_raw.columns else 0,
)

# ✅ FIXED: Sentiment & Market selectboxes (balanced brackets/parentheses)
sent_col = st.sidebar.selectbox(
    "Sentiment",
    options=["<none>"] + list(df_raw.columns),
    index=(["<none>"] + list(df_raw.columns)).index("Sentiment_Score")
          if "Sentiment_Score" in df_raw.columns else 0,
)
mkt_col = st.sidebar.selectbox(
    "Market condition",
    options=["<none>"] + list(df_raw.columns),
    index=(["<none>"] + list(df_raw.columns)).index("Market_Condition")
          if "Market_Condition" in df_raw.columns else 0,
)

# Date filter
df = df_raw.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
dmin, dmax = df[date_col].min(), df[date_col].max()
dr = st.sidebar.date_input("Date range", value=[dmin, dmax], min_value=dmin, max_value=dmax)
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    df = df[(df[date_col] >= pd.to_datetime(dr[0])) & (df[date_col] <= pd.to_datetime(dr[1]))].reset_index(drop=True)

# Quality toggles
st.sidebar.subheader("Data Quality")
miss_col = "Missing_Flag" if "Missing_Flag" in df.columns else None
noise_col = "Noise_Flag" if "Noise_Flag" in df.columns else None
drop_miss = st.sidebar.checkbox("Drop Missing_Flag==1", value=True) if miss_col else False
drop_noise = st.sidebar.checkbox("Drop Noise_Flag==1", value=False) if noise_col else False
if miss_col and drop_miss:
    df = df[df[miss_col] != 1]
if noise_col and drop_noise:
    df = df[df[noise_col] != 1]

# ========================= Build features & target =========================
df_feat = df.rename(columns={date_col: "Date", price_col: "Close"}).copy()
if vol_col != "<none>" and vol_col in df.columns:
    df_feat["Volume"] = pd.to_numeric(df[vol_col], errors="coerce")
sentiment_in = None if sent_col == "<none>" else sent_col
market_in   = None if mkt_col == "<none>" else mkt_col

df_feat = engineer_features(
    df_feat, price_col="Close", lags=lags, mas=mas, vol_windows=vols,
    sentiment_col=sentiment_in, market_col=market_in
).dropna().reset_index(drop=True)

st.sidebar.subheader("Target & Model")
target = st.sidebar.selectbox("Target", ["Next-day Return", "Next-day Close"])
if target == "Next-day Return":
    df_feat["Target"] = df_feat["Return"].shift(-1)
else:
    df_feat["Target"] = df_feat["Close"].shift(-1)
df_feat = df_feat.dropna().reset_index(drop=True)

feature_prefixes = ["lag_ret_", "lag_close_", "sma_", "ema_", "vol_", "sentiment_z", "Volume"]
base_feats = [c for c in df_feat.columns if any(c.startswith(p) for p in feature_prefixes)]
for req in ["Close", "Return"]:
    if (req in df_feat.columns) and (req not in base_feats):
        base_feats.append(req)

# ========================= Header & KPIs =========================
st.title("IBR Dashboard — AI in Financial Forecasting")
k1, k2, k3, k4 = st.columns(4)
with k1:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rows (raw)", f"{len(df_raw):,}")
        st.markdown("</div>", unsafe_allow_html=True)
with k2:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        # ✅ cleaned line
        st.metric("Rows (modeled)", f"{len(df_feat):,}")
        st.markdown("</div>", unsafe_allow_html=True)
with k3:
    ann_vol = df_feat["Return"].std() * np.sqrt(252) if "Return" in df_feat.columns else np.nan
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Annualized Vol", "—" if pd.isna(ann_vol) else f"{ann_vol:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)
with k4:
    cumret = (1 + df_feat.get("Return", pd.Series([0]))).prod() - 1 if "Return" in df_feat.columns else np.nan
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cumulative Return", "—" if pd.isna(cumret) else f"{cumret:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

# ========================= Tabs =========================
tab_overview, tab_eda, tab_model, tab_quality, tab_backtest, tab_govern = st.tabs(
    ["Overview", "EDA", "Model Lab", "Data Quality", "Backtest", "Governance"]
)

# ---------- Overview (Price + Volume) ----------
with tab_overview:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.72, 0.28]
    )
    has_ohlc = all(col in df_feat.columns for col in ["Open", "High", "Low", "Close"])
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df_feat["Date"], open=df_feat["Open"], high=df_feat["High"],
                low=df_feat["Low"], close=df_feat["Close"], name="OHLC",
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_feat["Date"], y=df_feat["Close"],
                mode="lines+markers" if show_markers else "lines",
                name="Close"
            ),
            row=1, col=1
        )

    if "Volume" in df_feat.columns:
        fig.add_trace(
            go.Bar(x=df_feat["Date"], y=df_feat["Volume"], name="Volume", opacity=0.6),
            row=2, col=1
        )

    fig.update_yaxes(type="log" if use_log_y else "linear", row=1, col=1, title="Price")
    fig.update_yaxes(title="Volume", row=2, col=1)
    fig.update_layout(
        template=template_choice,
        height=chart_height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=show_rangeslider,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: use the range slider / zoom to focus on key intervals.")

# ---------- EDA ----------
with tab_eda:
    c1, c2 = st.columns([2, 1])
    with c1:
        if "Return" in df_feat.columns:
            roll = df_feat.set_index("Date")[["Return"]].copy()
            roll["vol_20d"] = roll["Return"].rolling(20, min_periods=5).std()
            figv = px.line(roll.reset_index(), x="Date", y="vol_20d", template=template_choice)
            figv.update_layout(height=chart_height, margin=dict(l=10, r=10, t=10, b=10))
            figv.update_yaxes(title="20D Rolling Volatility")
            st.plotly_chart(figv, use_container_width=True)
    with c2:
        if "Return" in df_feat.columns:
            figb = px.box(df_feat, y="Return", template=template_choice, points=False)
            figb.update_layout(height=chart_height, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figb, use_container_width=True)

    num_cols = [c for c in base_feats if df_feat[c].dtype != "O"]
    if len(num_cols) > 1:
        corr = df_feat[num_cols].corr()
        figh = px.imshow(
            corr, text_auto=False, aspect="auto", template=template_choice, color_continuous_scale="RdBu_r",
        )
        figh.update_layout(height=min(chart_height, 600), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figh, use_container_width=True)

    st.markdown("### Data Dictionary & Quality Summary")
    st.dataframe(data_summary(df), use_container_width=True, height=320)

# ---------- Model Lab ----------
with tab_model:
    model_name = st.selectbox(
        "Model", ["Random Forest", "Linear Regression", "Ridge", "Lasso", "ElasticNet", "GradientBoostingRegressor"]
    )
    test_frac = st.slider("Test fraction", 0.1, 0.5, 0.2, 0.05, help="Held-out proportion at the end of the series.")

    if len(df_feat) < 40 or len(base_feats) < 3:
        st.warning("Not enough rows/features after filtering. Try fewer lags/MAs or expand date range.")
    else:
        res = train_eval(df_feat, "Target", base_feats, model_name=model_name, test_frac=float(test_frac))
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{res['RMSE']:.4f}")
        m2.metric("MAE", f"{res['MAE']:.4f}")
        da_str = "N/A" if (res['DA'] is None or np.isnan(res['DA'])) else f"{res['DA']*100:.1f}%"
        m3.metric("Directional Accuracy", da_str)

        test_idx = df_feat.index[res["cut"]:]
        pv = pd.DataFrame({
            "Date": df_feat.loc[test_idx, "Date"].values,
            "Actual": np.asarray(res["y_test"]).ravel(),
            "Pred": np.asarray(res["y_pred"]).ravel(),
        })
        figpv = go.Figure()
        figpv.add_trace(go.Scatter(x=pv["Date"], y=pv["Actual"], name="Actual", mode="lines"))
        figpv.add_trace(go.Scatter(x=pv["Date"], y=pv["Pred"], name="Pred", mode="lines"))
        figpv.update_layout(
            template=template_choice, height=chart_height, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(figpv, use_container_width=True)

        if res["fi"] is not None and len(res["fi"]) > 0:
            top_fi = res["fi"].head(15)[::-1]
            figfi = px.bar(x=top_fi.values, y=top_fi.index, orientation="h", template=template_choice,
                           labels={"x": "Importance", "y": "Feature"})
            figfi.update_layout(height=min(chart_height, 600), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figfi, use_container_width=True)

        st.markdown("#### Model Leaderboard (same split)")
        bench_models = ["Random Forest", "Linear Regression", "Ridge", "Lasso", "ElasticNet", "GradientBoostingRegressor"]
        rows = []
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

        st.markdown("#### Permutation Importance (Δ RMSE on shuffle)")
        if res["fi"] is not None and len(res["fi"]) > 0:  # tree models only
            try:
                Xte_pi = res["Xte"].copy()
                yte_pi = res["y_test"]
                base_rmse = _np_rmse(yte_pi, res["model"].predict(Xte_pi))
                pimps = {}
                rng = np.random.RandomState(RANDOM_SEED)
                for col in Xte_pi.columns:
                    vals = []
                    for _ in range(10):
                        xcopy = Xte_pi.copy()
                        shuf = xcopy[col].values.copy()
                        rng.shuffle(shuf)
                        xcopy[col] = shuf
                        vals.append(_np_rmse(yte_pi, res["model"].predict(xcopy)))
                    pimps[col] = float(np.mean(vals) - base_rmse)
                ser = pd.Series(pimps).sort_values(ascending=False).head(12)[::-1]
                figpi = px.bar(x=ser.values, y=ser.index, orientation="h",
                               labels={"x": "Δ RMSE", "y": "Feature"}, template=template_choice)
                figpi.update_layout(height=min(chart_height, 550), margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(figpi, use_container_width=True)
            except Exception as e:
                st.caption(f"Permutation importance skipped: {e}")
        else:
            st.caption("Permutation importance is available for tree models (with feature importances).")

        st.markdown("#### Partial Dependence (Top Feature)")
        if res.get("fi") is not None and len(res["fi"]) > 0:
            top_feat = res["fi"].index[0]
            n = len(df_feat); cutpd = int(n*(1-float(test_frac)))
            Xtr_pd = df_feat[base_feats].iloc[:cutpd].copy()
            rng_vals = np.linspace(Xtr_pd[top_feat].quantile(0.05), Xtr_pd[top_feat].quantile(0.95), 30)
            Xgrid = Xtr_pd.sample(min(200, len(Xtr_pd)), random_state=RANDOM_SEED).reset_index(drop=True)
            preds = []
            for v in rng_vals:
                Xtmp = Xgrid.copy(); Xtmp[top_feat] = v
                preds.append(res["model"].predict(Xtmp).mean())
            figpd = px.line(x=rng_vals, y=preds, labels={"x": top_feat, "y": "Predicted Target"}, template=template_choice)
            figpd.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figpd, use_container_width=True)
        else:
            st.caption("Partial dependence shown when a tree model (with importances) is selected.")

# ---------- Data Quality ----------
with tab_quality:
    st.markdown("### Imputation & Noise Experiments")
    colq1, colq2 = st.columns(2)
    with colq1:
        choice = st.radio("Imputation strategy", ["Drop rows", "Forward fill", "Time interpolate"], horizontal=True)
    with colq2:
        noise_bp = st.slider("Add Gaussian noise (bps on Close)", 0, 50, 0, 5)

    exp_df = df.copy()

    # Robust numeric detection that excludes datetime/object
    num_cols = exp_df.select_dtypes(include=[np.number]).columns.tolist()

    # Apply imputation choices safely
    if choice == "Forward fill":
        if num_cols:
            exp_df[num_cols] = exp_df[num_cols].ffill()

    elif choice == "Time interpolate":
        if date_col in exp_df.columns:
            # Use DatetimeIndex for method='time'
            exp_df = exp_df.set_index(date_col).sort_index()
            # Recompute numeric columns AFTER changing index (date is no longer a column)
            num_cols_idx = exp_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols_idx:
                exp_df[num_cols_idx] = exp_df[num_cols_idx].interpolate(method="time")
            # Restore date column
            exp_df = exp_df.reset_index()
        else:
            # Fallback: interpolate numerics without time
            num_cols_idx = exp_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols_idx:
                exp_df[num_cols_idx] = exp_df[num_cols_idx].interpolate()

    # Optional: noise injection on Close (in basis points)
    if noise_bp > 0:
        scale = noise_bp * 0.0001
        exp_df[price_col] = pd.to_numeric(exp_df[price_col], errors="coerce") * (1 + np.random.normal(0, scale, size=len(exp_df)))

    # Rebuild features/target for experiment df
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
        res_q = train_eval(exp_feat, "Target", exp_feats, model_name="Random Forest", test_frac=0.2)
        st.dataframe(pd.DataFrame([{"Strategy": choice, "Noise(bps)": noise_bp, "RMSE": res_q["RMSE"], "MAE": res_q["MAE"], "DirAcc": res_q["DA"]}]), use_container_width=True)
    else:
        st.info("Not enough rows/features post-imputation for this experiment.")

    st.markdown("### Train vs Test Shift (KS)")
    if HAS_SCIPY:
        ks_test_frac = float(test_frac) if 'test_frac' in locals() else 0.2
        def ks_shift(col):
            cut_idx = int(len(df_feat)*(1-ks_test_frac))
            tr = df_feat[col].iloc[:cut_idx]
            te = df_feat[col].iloc[cut_idx:]
            tr, te = tr.dropna(), te.dropna()
            if len(tr)>20 and len(te)>20:
                return float(ks_2samp(tr, te).statistic)
            return np.nan
        shift_cols = [c for c in ["Return","sma_5","sma_10","vol_10"] if c in df_feat.columns]
        if shift_cols:
            st.dataframe(pd.DataFrame({"feature": shift_cols, "KS_stat":[ks_shift(c) for c in shift_cols]}), use_container_width=True)
    else:
        st.caption("Install `scipy` to compute KS shift (optional).")

# ---------- Backtest ----------
with tab_backtest:
    st.markdown("### Simple Next-Day Strategy Backtest")
    if 'res' not in locals():
        st.info("Train a model in the Model Lab tab to enable the backtest.")
    else:
        thr = st.slider("Signal threshold (predicted next-day return, %)", 0.0, 1.0, 0.10, 0.05) / 100.0
        fee_bp = st.slider("Round-trip cost (bps)", 0, 50, 5, 5) / 10000.0

        cut_idx = res["cut"]
        if target == "Next-day Return":
            pred_ret = np.asarray(res["y_pred"]).ravel()
        else:
            pred_close = np.asarray(res["y_pred"]).ravel()
            today_close = df_feat["Close"].iloc[cut_idx:].values
            pred_ret = (pred_close / today_close) - 1.0

        realized_ret = df_feat["Return"].shift(-1).iloc[cut_idx:].fillna(0).values
        sig = (pred_ret > thr).astype(int)
        trade = sig * (realized_ret - fee_bp)
        equity = (1 + pd.Series(trade)).cumprod()

        if len(equity) > 0:
            cagr = equity.iloc[-1]**(252/len(equity)) - 1
            sharpe = (np.sqrt(252)*np.mean(trade)/np.std(trade)) if np.std(trade)>0 else np.nan
            dd = (equity / equity.cummax() - 1).min()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CAGR", f"{cagr:.2%}")
            m2.metric("Sharpe", "—" if pd.isna(sharpe) else f"{sharpe:.2f}")
            m3.metric("Max Drawdown", f"{dd:.2%}")
            m4.metric("Hit Rate", f"{(trade>0).mean():.1%}")

            fig_bt = px.line(x=df_feat["Date"].iloc[cut_idx:], y=equity.values, template=template_choice,
                             labels={"x":"Date", "y":"Equity"})
            fig_bt.update_layout(height=chart_height, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.info("Not enough test samples to run the backtest.")

# ---------- Governance ----------
with tab_govern:
    st.markdown("### Ethical & Regulatory Checklist")
    st.markdown("""
- **Transparency**: Feature choices (lags/MAs/vol windows), target definition (next-day), fixed split.
- **Accountability**: Export the model card; log parameters, data ranges, and metrics.
- **Fairness**: Compare performance under different imputation/noise settings; monitor train–test shifts (KS).
- **Overfitting Guardrails**: Prefer walk-forward/expanding-window CV before any live deployment.
- **Regulatory**: Align with SEBI/ESMA model risk guidance; this dashboard is an educational prototype — **not investment advice**.
    """)
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
            "model": model_name if 'model_name' in locals() else None,
            "test_fraction": float(test_frac) if 'test_frac' in locals() else None,
            "metrics": {
                "RMSE": float(res["RMSE"]),
                "MAE": float(res["MAE"]),
                "DirectionalAccuracy": None if (res['DA'] is None or np.isnan(res['DA'])) else float(res["DA"])
            },
            "random_seed": RANDOM_SEED
        }
        import json
        st.download_button("Download model_card.json",
                           data=json.dumps(card, indent=2),
                           file_name="model_card.json",
                           mime="application/json")
    else:
        st.caption("Train a model to enable the Model Card export.")
