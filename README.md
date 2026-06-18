# VN Stock Portfolio Optimization Dashboard

## 📊 Tổng quan

Dashboard tối ưu hóa danh mục đầu tư cho thị trường chứng khoán Việt Nam. Ứng dụng cho phép phân bổ tài sản dựa trên dữ liệu lịch sử, chạy mô phỏng Monte Carlo và trực quan hóa rủi ro.

---

## 🔧 Các hàm (Functions) & Logic 


### 1. Optimization Functions

#### `optimize_max_sharpe(mean_array, cov_matrix, rfr)`
- **Mục đích:** Tìm danh mục có Sharpe Ratio cao nhất
- **Công thức:**
  ```
  Sharpe Ratio = (E[Rp] - Rf) / σp
  ```
  Trong đó:
  - `E[Rp]` = Expected return của danh mục = w^T × μ
  - `Rf` = Risk-free rate
  - `σp` = Độ lệch chuẩn của danh mục = √(w^T × Σ × w)
  - `w` = Vector trọng số
  - `μ` = Vector expected returns
  - `Σ` = Ma trận hiệp phương sai
- **Phương pháp:** Sử dụng SciPy `minimize` với phương pháp SLSQP
- **Constraints:** Tổng trọng số = 1 (∑w_i = 1), trọng số >= 0 (w_i ≥ 0)

#### `optimize_min_vol(cov_matrix)`
- **Mục đích:** Tìm danh mục có độ biến động (volatility) thấp nhất
- **Công thức:**
  ```
  Minimize: σp = √(w^T × Σ × w)
  ```
- **Phương pháp:** Tương tự nhưng tối thiểu hóa volatility thay vì maximize Sharpe

---

### 2. Monte Carlo Simulation

#### `run_monte_carlo(mean_ret, cov_mat, rfr, num_portfolios=5000)`
- **Mục đích:** Mô phỏng hàng nghìn danh mục để vẽ Efficient Frontier
- **Logic:**
  1. Sinh ngẫu nhiên `num_portfolios` bộ trọng số
  2. Chuẩn hóa mỗi bộ trọng số để tổng = 1
  3. Tính return và volatility cho mỗi danh mục
- **Vectorization:** Sử dụng NumPy matrix multiplication và `np.einsum` để tính toàn bộ 5000 danh mục trong vài milliseconds:
  ```python
  port_returns = weights @ mean_ret  # Matrix multiplication
  port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_mat, weights))
  ```
- **Output:** Array [returns, volatilities, sharpes] cho 5000 danh mục

---

### 3. Portfolio Metrics Calculations

#### Buy-and-Hold Logic
Khác với naive rebalancing (tái cân bằng hàng ngày), ứng dụng này mô phỏng **Buy-and-Hold** thực tế:

```python
cum_asset_returns = (1 + asset_returns).cumprod()  # Tích lũy lợi nhuận
port_values = (cum_asset_returns * weights).sum(axis=1)  # Giá trị danh mục theo thời gian
```

- **Điểm khác biệt:** Trọng số tự động drift theo thời gian dựa trên cumulative returns của từng asset
- **Kết quả:** Phản ánh đúng поведение của danh mục đầu tư thực tế (không có tái cân bằng giả định)

#### Các chỉ số tính toán:

| Chỉ số | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Annual Return** | `ann_ret = port_daily_ret.mean() × 252` | Lợi nhuận hàng năm |
| **Annual Volatility** | `ann_vol = port_daily_ret.std() × √252` | Độ biến động hàng năm |
| **Sharpe Ratio** | `sharpe = (ann_ret - RFR) / ann_vol` | Lợi nhiệp điều chỉnh rủi ro |
| **Max Drawdown** | `mdd = (port_norm / port_norm.cummax() - 1).min()` | Mức sụt giảm lớn nhất |

---


## 🎯 Cách sử dụng

### Chế độ Manual Allocation:
1. Nhập trọng số (%) cho từng mã chứng khoán
2. Tổng phải = 100%
3. Click "Equal Weight" để reset về trọng số bằng nhau

### Chế độ Auto Optimize:
1. Chọn strategy: "Maximize Sharpe" hoặc "Minimum Volatility"
2. Click "Run Optimization"
3. Kết quả sẽ hiển thị trên chart và metrics

### Xem kết quả:
- **Equity Curve:** So sánh danh mục với VNINDEX
- **Metrics:** Annual Return, Volatility, Sharpe, Max Drawdown
- **Pie Chart:** Phân bổ tài sản hiện tại
- **Monte Carlo:** Efficient Frontier với 5000 simulated portfolios
- **Correlation Matrix:** Tương quan giữa các assets

---

## 🛠 Cài đặt

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Requirements:
```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
plotly>=5.18.0
vnstock>=0.2.0