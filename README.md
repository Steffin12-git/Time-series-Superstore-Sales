
# Time Series Forecasting — Superstore Sales

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-1.x-150458?logo=pandas\&logoColor=white) ![statsmodels](https://img.shields.io/badge/statsmodels-0.13-FF9900) ![pmdarima](https://img.shields.io/badge/pmdarima-auto--arima-1f77b4)

---

## Project Summary

This project performs **time series forecasting** on a retail "Superstore" sales dataset. The objective is to model historical monthly sales and produce a 12-month sales forecast that stakeholders can use for inventory planning, staffing and promotions.

The notebook explores data aggregation, decomposition, stationarity checks, and compares two families of models:

* **ARIMA** (auto\_arima → seasonal ARIMA)
* **Holt–Winters Exponential Smoothing** (multiple specifications)

After evaluating model performance using RMSE and MAE on the historical fitted values, the **Holt–Winters model with Multiplicative Trend** was selected as the final production model due to substantially lower RMSE.

---

## Dataset & Preprocessing

* Source file: `Superstore Sales Dataset.csv` (order-level sales data)
* Aggregation: sales were grouped by **month end** using `pd.Grouper(key='Order Date', freq='ME')` to create the monthly time series.

**Time range & basic stats** (monthly aggregation):

* Periods: **48 months**
* Start: **2015-01-31**
* End: **2018-12-31**
* Mean monthly sales: **≈ 47,115.35**
* Median monthly sales: **≈ 39,202.13**
* Minimum: **4,519.89** (2015-02-28)
* Maximum: **117,938.16** (2018-11-30)

📈 **Monthly Sales Over Time**
![Monthly Sales](./images/monthly%20sales%20over%20time.png)

---

## Exploratory Analysis & Stationarity

* The series was plotted and visually shows a **clear seasonal pattern** (yearly seasonality) with strong peaks around **November–December** (holiday season), and an overall trend component.
* **Seasonal decomposition** (additive decomposition) was performed to separate trend, seasonality and residuals.

📊 **Trend, Seasonality & Residual Decomposition**
![Trend Decomposition](./images/sales%20trend,%20seasnal%20and%20resid%20trend%20analysis.png)

**Stationarity check (ADF Test):**

* ADF Statistic: **-4.4161**
* p-value: **0.000278**
* Conclusion: **The series is stationary (p < 0.01)** — no differencing was required for ARIMA in this implementation.

---

## Models Implemented

### 1) ARIMA (pmdarima.auto\_arima → statsmodels ARIMA)

* `pmdarima.auto_arima` was used to suggest orders (seasonal order with m=12).
* The chosen ARIMA model was fit and used to compute a 12-month forecast and confidence intervals.

📈 **ARIMA Forecast with Confidence Intervals**
![ARIMA Forecast](./images/sales%20forcast%20with%20ARIMA%20and%20multple%20confidence%20intervel.png)

**ARIMA performance on historical fitted values:**

* MSE: **372,198,461.44**
* RMSE: **19,292.45**
* MAE: **14,432.38**

**ARIMA 12-month forecast (2019):**

| Month | Forecast (USD) |
| ----- | -------------: |
| Jan   |      71,457.29 |
| Feb   |      55,170.68 |
| Mar   |      75,354.27 |
| Apr   |      63,712.97 |
| May   |      68,065.12 |
| Jun   |      70,562.39 |
| Jul   |      68,781.96 |
| Aug   |      79,738.55 |
| Sep   |      94,101.27 |
| Oct   |      88,767.50 |
| Nov   |     113,674.50 |
| Dec   |  **92,191.07** |

---

### 2) Holt–Winters Exponential Smoothing

Multiple specifications were fit and compared (Additive/Multiplicative trend/seasonal variations, and a fixed-parameter configuration).

📊 **Model Comparison & Residuals (ARIMA vs Holt–Winters)**
![Model Comparison](./images/model%20and%20best%20model%20comparison%20and%20also%20residual.png)

**Model comparison (selected results):**

* Fixed Parameters: RMSE ≈ 9,783.60
* Additive Seasonal: RMSE ≈ 11,758.67
* **Multiplicative Trend: RMSE ≈ 9,762.29** ← *Best by RMSE*

**Chosen Model:** Holt–Winters with **Multiplicative Trend**

* RMSE: **9,762.29**
* MAE: **7,841.90**

**Holt–Winters 12-month forecast (2019):**

| Month | Forecast (USD) |
| ----- | -------------: |
| Jan   |      49,435.71 |
| Feb   |      41,459.51 |
| Mar   |      73,220.56 |
| Apr   |      60,904.08 |
| May   |      67,886.54 |
| Jun   |      63,574.16 |
| Jul   |      66,020.93 |
| Aug   |      66,663.18 |
| Sep   |     105,201.49 |
| Oct   |      78,140.77 |
| Nov   |     115,255.95 |
| Dec   | **116,513.16** |

---

## Outputs saved in the notebook

* `monthly_sales` — aggregated monthly series
* `auto_model` / `result` — ARIMA model fit
* `forcast` — ARIMA forecast with CI
* `specifications`, `results`, `best_model`, `hw_model` — Holt–Winters variants and best selection
* `hw_forcast` — Holt–Winters 12-month forecast
* `final_result` — Actual, Fitted, Residuals comparison

---

## Key Takeaways

* Implemented a reproducible time-series pipeline.
* Compared ARIMA vs Holt–Winters, with Holt–Winters **Multiplicative Trend** emerging best.
* Forecast captures strong **seasonality (Nov–Dec peaks)**, valuable for retail inventory and staffing decisions.

---

## Next Steps

1. Time-series cross-validation.
2. Automated hyperparameter tuning for Holt–Winters.
3. Incorporating exogenous variables (holidays, promotions).
4. Benchmarking against ML models (XGBoost, LSTM, Transformers).
5. Improved prediction intervals (bootstrapped or quantile models).