import streamlit as st
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

import sys
import subprocess
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure required packages (allowed to install)
def ensure_package(pkg):
    try:
        __import__(pkg)
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            __import__(pkg)
        except Exception as e:
            raise RuntimeError(f"Failed to install or import {pkg}: {e}")

for pkg in ("vnstock", "scipy", "plotly"):
    try:
        ensure_package(pkg)
    except Exception as e:
        st.error(str(e))
        st.stop()

from vnstock import Quote
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Defaults / Config
# -----------------------------
# Master ticker list (>=100). Includes VN30 + popular midcaps. Expand as needed.
ALL_TICKERS = [
    "VIC","VHM","VNM","VCB","BID","CTG","HPG","MWG","FPT","SAB","MSN","VJC","PNJ","SSI","STB",
    "TPB","MBB","BVH","GAS","ROS","DXG","NVL","KDH","PLX","BSR","PNC","VRE","REE","ACB","HDB",
    "TPC","KBC","HPX","VHC","VICF","FLC","AGG","MSH","VIX","HSG","PDR","SBT","LTG","TCH",
    "GMD","VCG","VCS","VGC","VND","VDS","DBC","HNG","KDC","NLG","IJC","VICB","CTD",
    "CII","PVD","PVS","VCM","TCB","VIB","SHB","SFG","FRT","THD","VOS","HT1","GEX","BCM","VFG",
    "VGG","DGC","DIG","IDC","IDJ","ITC","JVC","KSB","LHG","NKG","OGC","PHR","QNS","SMC","SZC",
    "DPM","VCI","BWE","VPK","IDI","SJS","KBC2","MWG1","SBT2","REE2","VIC3","SAB2","SBTG"
]
# Remove duplicates while preserving order
seen = set()
ALL_TICKERS = [x for x in ALL_TICKERS if not (x in seen or seen.add(x))]

DEFAULT_SELECTION = ['FPT', 'HPG', 'VCB', 'SSI', 'MWG', 'VIC', 'GAS', 'VNM']

BENCH = "VNINDEX"
RFR = 0.03
INTERVAL = "1D"

HORIZON_MAP = {
    "6 Months": 0.5,
    "1 Year": 1,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
    "15 Years": 15
}
TODAY = pd.Timestamp.today().normalize()

# -----------------------------
# Helpers: Data fetch
# -----------------------------
@st.cache_data
def fetch_close(series_symbol: str, start: str, end: str, interval: str = "1D"):
    try:
        q = Quote(source="kbs", symbol=series_symbol)
        df = q.history(start=start, end=end, interval=interval)
        if df is None or df.empty:
            raise RuntimeError(f"No data for {series_symbol}")
        # normalize index
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
        close_col = None
        for c in ("close", "Close", "adjClose", "close_price"):
            if c in df.columns:
                close_col = c
                break
        if close_col is None:
            close_col = df.columns[-1]
        series = df[close_col].rename(series_symbol).sort_index()
        return series
    except Exception as e:
        raise RuntimeError(f"Error fetching {series_symbol}: {e}")

@st.cache_data
def fetch_all(symbols_tuple, bench, start, end, interval="1D"):
    symbols = list(symbols_tuple)
    series_list = []
    for s in symbols + [bench]:
        ser = fetch_close(s, start=start, end=end, interval=interval)
        series_list.append(ser)
    df = pd.concat(series_list, axis=1)
    return df

# -----------------------------
# UI: Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

tickers = st.sidebar.multiselect(
    "Select tickers (type to search)",
    options=ALL_TICKERS,
    default=[t for t in DEFAULT_SELECTION if t in ALL_TICKERS],
    help="Type to search, or click to select. Master list contains many Vietnamese tickers."
)
if len(tickers) < 1:
    st.sidebar.error("Select at least one ticker")
    st.stop()

# Mode: Manual vs Auto
mode = st.sidebar.radio("Mode", options=["Auto Optimization", "Manual Allocation"])

horizon_label = st.sidebar.selectbox("Lookback", options=list(HORIZON_MAP.keys()), index=2)
years = HORIZON_MAP[horizon_label]
# Compute start date robustly including fractional years (for 6 months)
if years < 1:
    start_date = (TODAY - pd.DateOffset(months=int(round(years * 12)))).strftime("%Y-%m-%d")
else:
    start_date = (TODAY - pd.DateOffset(years=int(years))).strftime("%Y-%m-%d")
end_date = TODAY.strftime("%Y-%m-%d")

# Auto strategy controls
strategy = None
run_optim = False
if mode == "Auto Optimization":
    strategy = st.sidebar.selectbox("Strategy", options=["Maximize Sharpe", "Minimum Volatility", "Equal Weight"], index=0)
    run_optim = st.sidebar.button("Run Optimization")

# -----------------------------
# Fetch data
# -----------------------------
with st.spinner("Fetching price data..."):
    try:
        raw_df = fetch_all(tuple(tickers), BENCH, start_date, end_date, INTERVAL)
    except Exception as e:
        st.error("Data fetch failed: " + str(e))
        st.error(traceback.format_exc())
        st.stop()

# -----------------------------
# Data alignment & cleaning
# -----------------------------
first_valid = raw_df.apply(lambda col: col.first_valid_index())
relevant_idx = first_valid.loc[list(tickers) + [BENCH]]

if relevant_idx.isnull().any():
    missing = relevant_idx[relevant_idx.isnull()].index.tolist()
    st.warning(f"Symbols with no data in selected horizon removed: {missing}")
    tickers = [t for t in tickers if t not in missing]
    if BENCH in missing:
        st.error(f"Benchmark {BENCH} has no data in selected horizon. Abort.")
        st.stop()
    raw_df = raw_df.loc[:, tickers + [BENCH]]
    first_valid = raw_df.apply(lambda col: col.first_valid_index())
    relevant_idx = first_valid.loc[list(tickers) + [BENCH]]

latest_start = max(relevant_idx.values)
aligned = raw_df.loc[latest_start:].ffill().dropna(how="any")
if aligned.empty:
    st.error("No overlapping data after alignment. Try different tickers or shorter horizon.")
    st.stop()
aligned = aligned.loc[:, tickers + [BENCH]]

# -----------------------------
# Quant processing (vectorized)
# -----------------------------
returns = aligned.pct_change().dropna(how="any")

annual_mean = returns.mean() * 252
annual_cov = returns.cov() * 252

bench_var = returns[BENCH].var()
cov_with_bench = returns.cov().loc[tickers, BENCH]
betas = cov_with_bench / bench_var

# -----------------------------
# Optimization helpers
# -----------------------------
num_assets = len(tickers)

def build_mean_cov():
    mean_array = annual_mean.loc[tickers].values
    cov_matrix = annual_cov.loc[tickers, tickers].values
    return mean_array, cov_matrix

def optimize_max_sharpe():
    mean_array, cov_matrix = build_mean_cov()
    def port_return(w):
        return float(np.dot(w, mean_array))
    def port_vol(w):
        return float(np.sqrt(w.T.dot(cov_matrix).dot(w)))
    def neg_sharpe(w):
        vol = port_vol(w)
        if vol == 0:
            return 1e6
        return - (port_return(w) - RFR) / vol
    bounds = tuple((0.0,1.0) for _ in range(num_assets))
    constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})
    x0 = np.array([1.0/num_assets]*num_assets)
    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={'maxiter':1000})
    if not res.success:
        st.warning("Optimization warning: " + res.message)
    w = pd.Series(res.x, index=tickers)
    return w.clip(lower=0) / res.x.sum()

def optimize_min_vol():
    mean_array, cov_matrix = build_mean_cov()
    def vol_obj(w):
        return float(np.sqrt(w.T.dot(cov_matrix).dot(w)))
    bounds = tuple((0.0,1.0) for _ in range(num_assets))
    constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})
    x0 = np.array([1.0/num_assets]*num_assets)
    res = minimize(vol_obj, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={'maxiter':1000})
    if not res.success:
        st.warning("Optimization warning: " + res.message)
    w = pd.Series(res.x, index=tickers)
    return w.clip(lower=0) / res.x.sum()

def equal_weight():
    w = pd.Series([1.0/num_assets]*num_assets, index=tickers)
    return w

# Run default optimization initially to populate values
weights_auto = equal_weight()
try:
    if strategy == "Maximize Sharpe":
        weights_auto = optimize_max_sharpe()
    elif strategy == "Minimum Volatility":
        weights_auto = optimize_min_vol()
    elif strategy == "Equal Weight":
        weights_auto = equal_weight()
except Exception as e:
    st.warning("Auto optimization initial run failed: " + str(e))

# If user clicked Run Optimization, recompute weights_auto
if mode == "Auto Optimization" and run_optim:
    try:
        if strategy == "Maximize Sharpe":
            weights_auto = optimize_max_sharpe()
        elif strategy == "Minimum Volatility":
            weights_auto = optimize_min_vol()
        elif strategy == "Equal Weight":
            weights_auto = equal_weight()
    except Exception as e:
        st.error("Optimization failed: " + str(e))
        st.error(traceback.format_exc())
        st.stop()

# -----------------------------
# Manual allocation editor
# -----------------------------
weights_manual = None
if mode == "Manual Allocation":
    # build editable dataframe
    df_manual = pd.DataFrame({"Ticker": tickers})
    df_manual["weight"] = 1.0 / len(tickers)
    # show editable table
    edited = st.data_editor(df_manual, num_rows="fixed", use_container_width=True)
    # sanitize: clip negatives to zero, then normalize
    w_vals = edited["weight"].clip(lower=0).astype(float)
    if w_vals.sum() == 0:
        # if user zeroed all weights, fallback to equal weight
        w_norm = np.array([1.0/len(tickers)]*len(tickers))
    else:
        w_norm = (w_vals / w_vals.sum()).to_numpy()
    weights_manual = pd.Series(w_norm, index=edited["Ticker"].tolist())

# -----------------------------
# Choose final weights (manual vs auto)
# -----------------------------
if mode == "Manual Allocation":
    weights_used = weights_manual
else:
    weights_used = weights_auto

# Ensure weights_used index order matches tickers
weights_used = weights_used.reindex(tickers).fillna(0)
weights_used = weights_used.clip(lower=0)
if weights_used.sum() == 0:
    weights_used = pd.Series([1.0/len(tickers)]*len(tickers), index=tickers)
weights_used = weights_used / weights_used.sum()

# -----------------------------
# Metrics and series using weights_used
# -----------------------------
mean_array = annual_mean.loc[tickers].values
cov_matrix = annual_cov.loc[tickers, tickers].values

def port_return_from_weights(w_series):
    return float(np.dot(w_series.values, mean_array))

def port_vol_from_weights(w_series):
    return float(np.sqrt(w_series.values.T.dot(cov_matrix).dot(w_series.values)))

exp_annual_return = port_return_from_weights(weights_used)
annual_volatility = port_vol_from_weights(weights_used)
sharpe = (exp_annual_return - RFR) / annual_volatility if annual_volatility != 0 else np.nan

bench_returns = returns[BENCH]
bench_annual_return = bench_returns.mean() * 252
bench_annual_vol = bench_returns.std() * np.sqrt(252)
bench_sharpe = (bench_annual_return - RFR) / bench_annual_vol if bench_annual_vol != 0 else np.nan

port_daily = returns[tickers].dot(weights_used)
port_cum = (1 + port_daily).cumprod()
bench_cum = (1 + bench_returns).cumprod()

port_norm = port_cum / port_cum.iloc[0]
bench_norm = bench_cum / bench_cum.iloc[0]

running_max = port_norm.cummax()
drawdown = port_norm / running_max - 1
mdd = float(drawdown.min())

bench_running_max = bench_norm.cummax()
bench_dd = bench_norm / bench_running_max - 1
bench_mdd = float(bench_dd.min())

# -----------------------------
# Correlation matrix
# -----------------------------
corr = returns[tickers].corr()

# -----------------------------
# UI: Main panel
# -----------------------------
st.title("Portfolio Optimization Sandbox")

# Equity curve
st.subheader("Equity Curve")
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=port_norm.index, y=port_norm.values, mode="lines",
                            name="Optimized Portfolio", line=dict(color="#00CCFF", width=2)))
fig_eq.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm.values, mode="lines",
                            name="Benchmark (VNINDEX)", line=dict(color="#888888", width=2, dash="dash")))
fig_eq.update_layout(
    yaxis_title="Cumulative (normalized)",
    xaxis_title="Date",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=40, b=40)
)
st.plotly_chart(fig_eq, width="stretch")

# Key metrics
st.subheader("Key Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Expected Annual Return", f"{exp_annual_return:.2%}", delta=f"{(exp_annual_return - bench_annual_return):.2%}")
m2.metric("Annual Volatility", f"{annual_volatility:.2%}", delta=f"{(annual_volatility - bench_annual_vol):.2%}")
m3.metric("Sharpe Ratio", f"{sharpe:.3f}", delta=f"{(sharpe - bench_sharpe):.3f}")
m4.metric("Max Drawdown (MDD)", f"{mdd:.2%}", delta=f"{(mdd - bench_mdd):.2%}")

# Correlation matrix
st.subheader("Correlation Matrix")
fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu', zmin=-1, zmax=1)
fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
st.plotly_chart(fig_corr, width="stretch")

# Allocation & details
st.subheader("Allocation & Details")
left, right = st.columns([2, 2])

with left:
    # Weights dataframe: ensure ticker column visible
    weights_df = weights_used.rename("weight").reset_index().rename(columns={'index': 'Ticker', 0: 'weight'})
    weights_df["pct"] = (weights_df["weight"] * 100).round(2)
    weights_col_config = {
        "weight": st.column_config.ProgressColumn("Weight", min_value=0.0, max_value=1.0, format="%.2f"),
        "pct": st.column_config.NumberColumn("Percent", format="%.2f")
    }
    st.dataframe(weights_df.sort_values("pct", ascending=False), hide_index=True, column_config=weights_col_config, width="stretch")

    beta_df = betas.rename(BENCH).to_frame("beta").reset_index().rename(columns={'index':'Ticker'})
    beta_col_config = {
        "beta": st.column_config.NumberColumn("Beta", format="%.2f")
    }
    st.dataframe(beta_df.sort_values("beta", ascending=False), hide_index=True, column_config=beta_col_config, width="stretch")

with right:
    display_weights = weights_used[weights_used >= 0.01].copy()
    if display_weights.sum() < 0.9999:
        others_val = 1.0 - display_weights.sum()
        others_series = pd.Series({"Others": others_val})
        pie_series = pd.concat([display_weights, others_series])
    else:
        pie_series = display_weights
    pie_df = pie_series.reset_index().rename(columns={'index':'asset', 0:'weight'})
    pie_df.columns = ["asset", "weight"]
    fig_pie = px.pie(pie_df, names="asset", values="weight", hole=0.45, title="Portfolio Allocation")
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12, showlegend=False)
    fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, width="stretch")

# Underwater plot
st.subheader("Underwater Plot (Drawdown)")
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode="lines",
                            line=dict(color="red"), fill='tozeroy', name="Drawdown"))
fig_dd.update_layout(yaxis_tickformat=".0%", xaxis_title="Date", yaxis_title="Drawdown",
                     template="plotly_white", showlegend=False, margin=dict(l=40, r=20, t=10, b=40))
st.plotly_chart(fig_dd, width="stretch")

# Data snapshot
with st.expander("Data snapshot"):
    st.dataframe(aligned.tail(), width="stretch")
    compare_df = pd.DataFrame({"Portfolio": port_norm, "Benchmark": bench_norm})
    st.dataframe(compare_df.tail(), width="stretch")

# No footer notes printed