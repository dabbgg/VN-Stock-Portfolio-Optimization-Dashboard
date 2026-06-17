# VN Stock Portfolio Optimization Dashboard

A professional, institutional-grade portfolio analysis and optimization tool for the Vietnamese stock market. This application allows users to construct portfolios, optimize asset allocation based on quantitative strategies, and analyze risk-adjusted performance against the VNINDEX benchmark.

## 🎯 Purpose
The dashboard is designed for quantitative analysts and investors to:
- **Analyze** the historical performance of a custom selection of Vietnamese equities.
- **Optimize** asset weights to either maximize the Sharpe Ratio or minimize overall portfolio volatility.
- **Visualize** risk metrics, including Max Drawdown and asset correlations.
- **Compare** portfolio equity curves against the market benchmark (VNINDEX) on a normalized basis.

## 🛠 Methodology

### 1. Data Processing & Alignment
To ensure mathematical integrity, the app implements a strict data alignment pipeline:
- **Common Window**: The analysis window is sliced from the **latest common start date** across all selected assets and the benchmark. This prevents "look-ahead bias" and ensures no `NaN` values exist in the return series.
- **Forward Filling**: Missing values are handled via forward-filling (`ffill`) to account for non-trading days or liquidity gaps.

### 2. Performance Metrics
All metrics are annualized assuming **252 trading days** per year.

- **Annualized Return**: 
  $$\text{Annual Return} = \text{Mean Daily Return} \times 252$$
- **Annualized Volatility**: 
  $$\text{Annual Volatility} = \text{Std Dev of Daily Returns} \times \sqrt{252}$$
- **Sharpe Ratio**: 
  $$\text{Sharpe Ratio} = \frac{\text{Annual Return} - \text{Risk Free Rate}}{\text{Annual Volatility}}$$
  *(Default Risk-Free Rate: 3%)*
- **Max Drawdown (MDD)**: 
  The maximum peak-to-trough decline of the normalized equity curve, representing the worst possible loss from a historical peak.

### 3. Optimization Strategies
The app utilizes the `scipy.optimize` library (SLSQP algorithm) to solve constrained optimization problems:

- **Maximize Sharpe**: Finds the weight vector $\mathbf{w}$ that maximizes the ratio of excess return to volatility, subject to $\sum w_i = 1$ and $0 \le w_i \le 1$.
- **Minimum Volatility**: Finds the weight vector $\mathbf{w}$ that minimizes the portfolio variance $\mathbf{w}^T \Sigma \mathbf{w}$, where $\Sigma$ is the annualized covariance matrix.
- **Equal Weight**: A naive $1/N$ allocation strategy.

## 📊 Data Source & Availability

- **Source**: Data is fetched via the `vnstock` Python library, utilizing the **KBS** data provider.
- **Coverage**: Includes major Vietnamese tickers (VN30 and popular mid-caps).
- **Availability**: Data is subject to the availability of the KBS API. Real-time updates are typically available during Vietnam trading hours (09:00 - 15:00 GMT+7).

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Streamlit
- vnstock
- scipy
- plotly
- pandas
- numpy

### Installation & Execution
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy plotly scipy vnstock
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## ⚖️ Disclaimer
This tool is for educational and analytical purposes only. It does not constitute financial advice. Past performance is not indicative of future results.