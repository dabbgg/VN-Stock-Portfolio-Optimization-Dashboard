import streamlit as st
import pandas as pd
import numpy as np
from vnstock import Quote
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# Custom CSS to remove padding, hide menu, and make Plotly transparent
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stPlotlyChart {
            background-color: rgba(0,0,0,0) !important;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants & Config
# -----------------------------
ALL_TICKERS = [
    "VIC","VHM","VNM","VCB","BID","CTG","HPG","MWG","FPT","SAB","MSN","VJC","PNJ","SSI","STB",
    "TPB","MBB","BVH","GAS","ROS","DXG","NVL","KDH","PLX","BSR","PNC","VRE","REE","ACB","HDB",
    "TPC","KBC","HPX","VHC","VICF","FLC","AGG","MSH","VIX","HSG","PDR","SBT","LTG","TCH",
    "GMD","VCG","VCS","VGC","VND","VDS","DBC","HNG","KDC","NLG","IJC","VICB","CTD",
    "CII","PVD","PVS","VCM","TCB","VIB","SHB","SFG","FRT","THD","VOS","HT1","GEX","BCM","VFG",
    "VGG","DGC","DIG","IDC","IDJ","ITC","JVC","KSB","LHG","NKG","OGC","PHR","QNS","SMC","SZC",
    "DPM","VCI","BWE","VPK","IDI","SJS","KBC2","MWG1","SBT2","REE2","VIC3","SAB2","SBTG"
]
ALL_TICKERS = sorted(list(set(ALL_TICKERS)))
DEFAULT_SELECTION = ['FPT', 'HPG', 'VCB', 'SSI', 'MWG']
BENCH = "VNINDEX"
HORIZON_MAP = {
    "6 Months": 0.5,
    "1 Year": 1,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
}

# -----------------------------
# Session State Initialization
# -----------------------------
if "active_weights" not in st.session_state:
    st.session_state.active_weights = pd.Series(
        [1.0 / len(DEFAULT_SELECTION)] * len(DEFAULT_SELECTION),
        index=DEFAULT_SELECTION
    )
if "calc_mode" not in st.session_state:
    st.session_state.calc_mode = "Manual"
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0

# -----------------------------
# Backend Logic
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close(symbol, start, end):
    try:
        q = Quote(source="kbs", symbol=symbol)
        df = q.history(start=start, end=end, interval="1D")
        if df is None or df.empty:
            return None
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
        close_col = next((c for c in ("close", "Close", "adjClose", "close_price") if c in df.columns), df.columns[-1])
        return df[close_col].rename(symbol).sort_index()
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(symbols, bench, start, end):
    series_list = []
    for s in symbols + [bench]:
        ser = fetch_close(s, start, end)
        if ser is not None:
            series_list.append(ser)
    if not series_list:
        return None
    df = pd.concat(series_list, axis=1)
    return df

def optimize_max_sharpe(mean_array, cov_matrix, rfr):
    num_assets = len(mean_array)
    def neg_sharpe(w):
        ret = np.dot(w, mean_array)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(ret - rfr) / vol if vol != 0 else 1e6
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    res = minimize(neg_sharpe, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else np.array([1./num_assets]*num_assets)

def optimize_min_vol(cov_matrix):
    num_assets = cov_matrix.shape[0]
    def vol_obj(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    res = minimize(vol_obj, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else np.array([1./num_assets]*num_assets)

def run_monte_carlo(mean_ret, cov_mat, rfr, num_portfolios=5000):
    num_assets = len(mean_ret)
    weights = np.random.random((num_portfolios, num_assets))
    weights /= weights.sum(axis=1)[:, np.newaxis]
    port_returns = weights @ mean_ret
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_mat, weights))
    sharpes = (port_returns - rfr) / port_vols
    return np.array([port_returns, port_vols, sharpes])

# -----------------------------
# Main UI
# -----------------------------
st.title("Portfolio Optimization Dashboard")
st.markdown("Investment-grade asset allocation and risk analysis")

left_col, right_col = st.columns([1, 2.5], gap="large")

with left_col:
    with st.container(border=True):
        st.subheader("Asset Selection")
        selected_tickers = st.multiselect(
            "Tickers",
            options=ALL_TICKERS,
            default=[t for t in DEFAULT_SELECTION if t in ALL_TICKERS]
        )

    with st.container(border=True):
        st.subheader("Time Horizon")
        horizon_label = st.radio("Select Period", options=list(HORIZON_MAP.keys()), horizontal=True, index=1)
        years = HORIZON_MAP[horizon_label]

    with st.container(border=True):
        st.subheader("Mode & Weights")
        rfr_pct = st.number_input("Risk-Free Rate (%)", value=3.0, step=0.1)
        rfr = rfr_pct / 100.0

        tab1, tab2 = st.tabs(["Manual Allocation", "Auto Optimize"])

        with tab1:
            current_weights = st.session_state.active_weights
            editor_data = []
            for t in selected_tickers:
                w = current_weights.get(t, 0.0)
                editor_data.append({"Ticker": t, "Weight (%)": w * 100.0})
            editor_df = pd.DataFrame(editor_data)

            edited_df = st.data_editor(
                editor_df,
                column_config={
                    "Weight (%)": st.column_config.NumberColumn(
                        "Weight (%)", min_value=0.0, max_value=100.0, format="%.2f%%"
                    )
                },
                hide_index=True,
                use_container_width=True,
                key=f"manual_editor_{st.session_state.editor_version}"
            )

            total_weight = edited_df["Weight (%)"].sum()
            st.write(f"Total: **{total_weight:.2f}%**")

            def reset_equal_weight():
                if selected_tickers:
                    equal_w = 1.0 / len(selected_tickers)
                    st.session_state.active_weights = pd.Series(
                        [equal_w] * len(selected_tickers),
                        index=selected_tickers
                    )
                    st.session_state.calc_mode = "Manual"
                    st.session_state.editor_version += 1
                    st.rerun()

            st.button("⚖️ Equal Weight", use_container_width=True, on_click=reset_equal_weight)

            if abs(total_weight - 100.0) <= 0.01:
                new_weights = edited_df.set_index("Ticker")["Weight (%)"] / 100.0
                st.session_state.active_weights = new_weights
                st.session_state.calc_mode = "Manual"
            else:
                st.error("Total must be exactly 100%")
                st.stop()

        with tab2:
            with st.form(key="optimize_form"):
                strategy = st.selectbox("Strategy", ["Maximize Sharpe", "Minimum Volatility"])
                submit = st.form_submit_button("Run Optimization", use_container_width=True)
                if submit:
                    if not selected_tickers:
                        st.warning("Please select at least one ticker.")
                    else:
                        today = pd.Timestamp.today().normalize()
                        start_date = (today - pd.DateOffset(years=years)).strftime("%Y-%m-%d") if years >= 1 else (today - pd.DateOffset(months=int(years*12))).strftime("%Y-%m-%d")
                        end_date = today.strftime("%Y-%m-%d")
                        raw_df = fetch_all(selected_tickers, BENCH, start_date, end_date)
                        if raw_df is None or raw_df.empty:
                            st.error("No data found for selected tickers.")
                        else:
                            first_valid = raw_df.apply(lambda col: col.first_valid_index())
                            latest_start = first_valid.max()
                            aligned = raw_df.loc[latest_start:].ffill().dropna(how="any")
                            if aligned.empty or BENCH not in aligned.columns:
                                st.error("Insufficient data for optimization.")
                            else:
                                returns = aligned.pct_change().dropna()
                                asset_returns = returns[selected_tickers]
                                mean_ret = asset_returns.mean() * 252
                                cov_mat = asset_returns.cov() * 252
                                if strategy == "Maximize Sharpe":
                                    opt_weights = optimize_max_sharpe(mean_ret.values, cov_mat.values, rfr)
                                else:
                                    opt_weights = optimize_min_vol(cov_mat.values)
                                st.session_state.active_weights = pd.Series(opt_weights, index=selected_tickers)
                                st.session_state.calc_mode = "Auto"
                                st.session_state.editor_version += 1
                                st.success(f"Optimization complete: {strategy}")

    with st.container(border=True):
        st.subheader("Quick Stats")
        stats_placeholder = st.empty()

# -----------------------------
# Data Processing
# -----------------------------
if not selected_tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

today = pd.Timestamp.today().normalize()
start_date = (today - pd.DateOffset(years=years)).strftime("%Y-%m-%d") if years >= 1 else (today - pd.DateOffset(months=int(years*12))).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

with st.spinner("Fetching data..."):
    raw_df = fetch_all(selected_tickers, BENCH, start_date, end_date)

if raw_df is None or raw_df.empty:
    st.error("No data found for selected tickers.")
    st.stop()

first_valid = raw_df.apply(lambda col: col.first_valid_index())
latest_start = first_valid.max()
aligned = raw_df.loc[latest_start:].ffill().dropna(how="any")

if aligned.empty:
    st.error("No overlapping data found for the selected assets.")
    st.stop()

if BENCH not in aligned.columns:
    st.error(f"Benchmark {BENCH} data missing.")
    st.stop()

returns = aligned.pct_change().dropna()
asset_returns = returns[selected_tickers]
bench_returns = returns[BENCH]

mean_ret = asset_returns.mean() * 252
cov_mat = asset_returns.cov() * 252

# Monte Carlo Simulation
with st.spinner("Running Monte Carlo simulation..."):
    mc_results = run_monte_carlo(mean_ret.values, cov_mat.values, rfr)

# Centralized weights
weights = st.session_state.active_weights.reindex(selected_tickers).fillna(0.0).values
weights_series = pd.Series(weights, index=selected_tickers)

# -----------------------------
# Portfolio Metrics (Buy-and-Hold)
# -----------------------------
cum_asset_returns = (1 + asset_returns).cumprod()
port_values = (cum_asset_returns * weights).sum(axis=1)
port_norm = port_values / port_values.iloc[0]
port_daily_ret = port_norm.pct_change().fillna(0)

bench_cum = (1 + bench_returns).cumprod()
bench_norm = bench_cum / bench_cum.iloc[0]

ann_ret = port_daily_ret.mean() * 252
ann_vol = port_daily_ret.std() * np.sqrt(252)
sharpe = (ann_ret - rfr) / ann_vol if ann_vol != 0 else 0
mdd = (port_norm / port_norm.cummax() - 1).min()

# -----------------------------
# Right Column Visuals
# -----------------------------
with right_col:
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=port_norm.index, y=port_norm.values, name="Portfolio", line=dict(color="#00CCFF", width=3)))
    fig_eq.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm.values, name="VNINDEX", line=dict(color="#888888", width=2, dash="dash")))
    fig_eq.update_layout(
        title="Normalized Equity Curve (Buy & Hold)",
        xaxis_title="Date", yaxis_title="Value (Base 1.0)",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    m_col, c_col = st.columns([1, 1])
    with m_col:
        st.subheader("Key Metrics")
        st.metric("Annual Return", f"{ann_ret:.2%}")
        st.metric("Annual Volatility", f"{ann_vol:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Max Drawdown", f"{mdd:.2%}")

    with c_col:
        df_pie = pd.DataFrame({"Asset": selected_tickers, "Weight": weights})
        df_pie = df_pie[df_pie["Weight"] > 0.001]
        fig_pie = px.pie(df_pie, names="Asset", values="Weight", hole=0.45)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# Monte Carlo & Efficient Frontier
# -----------------------------
st.divider()
st.subheader("Monte Carlo Simulation & Efficient Frontier")

mc_df = pd.DataFrame(mc_results.T, columns=['Return', 'Volatility', 'Sharpe'])
max_sharpe_idx = mc_df['Sharpe'].idxmax()
min_vol_idx = mc_df['Volatility'].idxmin()

fig_mc = go.Figure()

fig_mc.add_trace(go.Scatter(
    x=mc_df['Volatility'], y=mc_df['Return'],
    mode='markers',
    marker=dict(
        color=mc_df['Sharpe'],
        colorscale='Turbo',
        size=5,
        opacity=0.6,
        line=dict(width=0),
        showscale=True,
        colorbar=dict(title='Sharpe Ratio')
    ),
    name='Simulated Portfolios',
    hovertemplate="<b>Return:</b> %{y:.2%}<br><b>Volatility:</b> %{x:.2%}<br><b>Sharpe:</b> %{marker.color:.2f}<extra></extra>"
))

fig_mc.add_trace(go.Scatter(
    x=[mc_df.loc[max_sharpe_idx, 'Volatility']], y=[mc_df.loc[max_sharpe_idx, 'Return']],
    mode='markers',
    marker=dict(color='red', size=12, symbol='star'),
    name='Max Sharpe',
    hovertemplate="<b>Max Sharpe Portfolio</b><br><b>Return:</b> %{y:.2%}<br><b>Volatility:</b> %{x:.2%}<extra></extra>"
))

fig_mc.add_trace(go.Scatter(
    x=[mc_df.loc[min_vol_idx, 'Volatility']], y=[mc_df.loc[min_vol_idx, 'Return']],
    mode='markers',
    marker=dict(color='blue', size=12, symbol='diamond'),
    name='Min Volatility',
    hovertemplate="<b>Min Volatility Portfolio</b><br><b>Return:</b> %{y:.2%}<br><b>Volatility:</b> %{x:.2%}<extra></extra>"
))

fig_mc.update_layout(
    xaxis_title="Volatility (Std Dev)",
    yaxis_title="Return",
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    xaxis=dict(tickformat=".1%"),
    yaxis=dict(tickformat=".1%")
)
st.plotly_chart(fig_mc, use_container_width=True)

# Populate Quick Stats in Left Col
with left_col:
    indiv_returns = (aligned[selected_tickers].iloc[-1] / aligned[selected_tickers].iloc[0]) - 1
    best_stock = indiv_returns.idxmax()
    best_val = indiv_returns.max()
    worst_stock = indiv_returns.idxmin()
    worst_val = indiv_returns.min()

    with stats_placeholder.container():
        s1, s2 = st.columns(2)
        s1.metric("Best Stock", best_stock, f"{best_val:.2%}")
        s2.metric("Worst Stock", worst_stock, f"{worst_val:.2%}")

# -----------------------------
# Bottom Section: Correlation Matrix
# -----------------------------
st.divider()
st.subheader("Asset Correlation Matrix")
corr_matrix = asset_returns.corr()
fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1,
    aspect="auto"
)
fig_corr.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=20, b=20),
    height=600
)
st.plotly_chart(fig_corr, use_container_width=True)
