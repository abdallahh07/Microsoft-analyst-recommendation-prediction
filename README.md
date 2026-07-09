<div align="center">

# Abdallah Hashad

### Quantitative Finance × Machine Learning

**Data Scientist**

</div>

---

# Microsoft Analyst Recommendation Prediction

A production-ready machine learning system that predicts whether Wall Street analysts will be **Bullish or Not Bullish** on Microsoft (MSFT) based on fundamental financial data, macroeconomic indicators, and DCF valuation analysis.

---

## Problem Statement

Analyst recommendations drive significant market movements. This project answers:

> "Given Microsoft's financial health, valuation, and macroeconomic environment — will analysts recommend buying the stock?"

Using CFA-level financial analysis combined with machine learning to understand what drives analyst sentiment beyond traditional valuation metrics.

---

## Key Insight

Microsoft's stock price has consistently traded **above its DCF intrinsic value** yet analysts remain overwhelmingly Bullish. This project demonstrates that traditional DCF valuation alone doesn't explain analyst recommendations — analysts are pricing in future AI growth and cloud expansion that backward-looking models can't capture.

---

## Dataset

16 CSV files from Yahoo Finance covering Microsoft (NASDAQ: MSFT) from 1986 to present:

| File                         | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| MSFT_upgrades_downgrades.csv | Analyst recommendation history (target variable) |
| MSFT_income.csv              | Annual income statements                         |
| MSFT_balance.csv             | Annual balance sheet statements                  |
| MSFT_cashflow.csv            | Annual cash flow statements                      |
| MSFT_price.csv               | Daily stock price history                        |
| MSFT_dividends.csv           | Dividend payment history                         |
| + 10 more files              | Institutional holders, earnings dates, etc.      |

---

## Target Variable

Built from `MSFT_upgrades_downgrades.csv` — analyst `ToGrade` mapped to binary:

```
Bullish     → 1  (Buy, Outperform, Overweight, Strong Buy)
Not Bullish → 0  (Hold, Neutral, Sell, Underperform)
```

**Class Distribution:** 93.6% Bullish, 6.4% Not Bullish — heavily imbalanced, handled with `scale_pos_weight=14`

---

## Features (20 total)

### Profitability Ratios

- Gross Margin, EBIT Margin, Net Profit Margin
- Revenue Growth Rate, R&D as % of Revenue

### Liquidity Ratios

- Current Ratio, Quick Ratio

### Leverage Ratios

- Debt to Equity, Debt to Assets

### Efficiency Ratios

- ROE, ROA, Cash Flow Margin, FCF Margin

### DCF Valuation (CFA-Level Feature Engineering)

- FCF Per Share
- Intrinsic Value (Gordon Growth Model: FCF / (WACC - g))
- Stock Price
- Margin of Safety

### Macroeconomic Features

- US GDP, US GDP Growth Rate
- Federal Funds Rate (FEDFUNDS)

---

## Financial Analysis Highlights

| Metric         | 2022   | 2023   | 2024   |
| -------------- | ------ | ------ | ------ |
| Revenue Growth | -      | 6.9%   | 15.7%  |
| Gross Margin   | 68.4%  | 68.9%  | 69.8%  |
| Net Margin     | 36.7%  | 34.1%  | 36.0%  |
| Current Ratio  | 1.78   | 1.77   | 1.27   |
| Debt/Equity    | 0.37   | 0.29   | 0.25   |
| ROE            | 43.7%  | 35.1%  | 32.8%  |
| Free Cash Flow | $65.1B | $59.5B | $74.1B |

### DCF Valuation vs Market Price

| Year | Intrinsic Value | Stock Price | Margin of Safety |
| ---- | --------------- | ----------- | ---------------- |
| 2022 | $174.57         | $239.82     | -37.4%           |
| 2023 | $160.05         | $376.04     | -134.9%          |
| 2024 | $199.27         | $421.50     | -111.5%          |

**Key Finding:** Microsoft trades significantly above intrinsic value yet analysts remain Bullish — traditional DCF doesn't capture AI and cloud growth optionality.

---

## Models Evaluated

| Model               | AUC-ROC           |
| ------------------- | ----------------- |
| LightGBM            | **0.7218** ← Best |
| RandomForest        | 0.7218            |
| Logistic Regression | 0.7218            |

**Evaluation Metric:** AUC-ROC (accuracy is misleading with 93.6% class imbalance)

**Top Features by Importance:** Gross Margin, EBIT Margin, Stock Price

---

## Tech Stack

| Tool                | Purpose                          |
| ------------------- | -------------------------------- |
| Python 3.13         | Core language                    |
| Pandas, NumPy       | Data manipulation                |
| Scikit-learn        | ML pipeline, preprocessing       |
| LightGBM            | Gradient boosting classifier     |
| pandas-datareader   | FRED macro data (GDP, Fed rates) |
| FastAPI             | REST API framework               |
| Uvicorn             | ASGI web server                  |
| Pydantic            | Data validation                  |
| Joblib              | Model serialization              |
| Matplotlib, Seaborn | Visualizations                   |
| PyYAML              | Configuration management         |

---

## Project Structure

```
Microsoft-analyst-recommendation-prediction/
├── microsoft_model/
│   ├── config/
│   │   ├── config.py          # Reads and validates config
│   │   └── config.yml         # All settings and parameters
│   ├── processing/
│   │   ├── data_manager.py    # Loads all 5 CSV files
│   │   └── features.py        # Feature engineering pipeline
│   ├── pipeline.py            # Sklearn pipeline definition
│   └── trained_models/
│       └── model.pkl          # Saved trained model
├── app/
│   ├── main.py                # FastAPI app entry point
│   ├── api.py                 # API endpoints
│   └── schemas.py             # Input/output schemas
├── data/                      # Raw CSV files
├── charts/                    # EDA visualizations
├── research.ipynb             # Full EDA and modeling notebook
├── train_pipeline.py          # Training script
└── predict.py                 # Prediction function
```

---

## API

Built with **FastAPI** and served with **Uvicorn**.

### Run Locally

```bash
python train_pipeline.py  # Train and save model
uvicorn app.main:app --reload --port 8002  # Start API
```

Visit `http://localhost:8002/docs` for interactive documentation.

### Endpoints

**GET /api/v1/health**

```json
{ "status": "ok" }
```

**POST /api/v1/predict**

Request:

```json
{
  "Gross_Margin": 68.9,
  "EBIT_Margin": 43.1,
  "Net_Margin": 34.1,
  "Revenue_Growth": 6.9,
  "RD_Pct": 12.8,
  "Current_Ratio": 1.77,
  "Quick_Ratio": 1.75,
  "Debt_To_Equity": 0.29,
  "Debt_To_Assets": 0.15,
  "ROE": 35.1,
  "ROA": 17.6,
  "Cash_Flow_Margin": 41.3,
  "FCF_Margin": 28.1,
  "GDP": 27811.0,
  "GDP_Growth": 1.5,
  "FEDFUNDS": 5.02,
  "FCF_Per_Share": 8.0,
  "Intrinsic_Value": 160.0,
  "Stock_Price": 376.0,
  "Margin_Of_Safety": -134.9
}
```

Response:

```json
{
  "prediction": "Bullish",
  "probability": 0.8923
}
```

---

## Limitations & Future Work

- **Small dataset:** Only 3 years of annual financial data (329 rows after merge) — quarterly data would significantly improve performance
- **Single stock:** Model trained only on MSFT — future work would extend to S&P 500
- **Static WACC:** DCF uses fixed 9% WACC — dynamic WACC calculation would improve intrinsic value accuracy
- **Future:** Add LSTM for time series price prediction, sentiment analysis on earnings call transcripts

---

## Author

**Abdallah Hashad** — Self-taught ML Engineer | FMVA | Law Degree  
Combining CFA-level financial analysis with machine learning for fintech applications.  
GitHub: [abdallahh07](https://github.com/abdallahh07)
