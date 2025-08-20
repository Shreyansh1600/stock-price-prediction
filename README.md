# Stock Price Prediction Dashboard

A minimal data science project that predicts stock prices using machine learning and historical CSV data.

## Features

- **CSV data loading** from sample dataset or user uploads
- **Simple ML model** (Linear Regression) for price prediction
- **Interactive dashboard** built with Streamlit
- **Visual charts** showing historical prices and predictions
- **Model performance metrics** (R² score, MSE)
- **No API dependencies** - works offline with CSV files

## Setup

1. **Clone/Download** this project
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open browser** and go to `http://localhost:8501`

## Usage

1. Choose data source: "Sample Dataset" or "Upload CSV"
2. If uploading CSV, ensure it has Date, Close, and Volume columns
3. Select prediction days (1-30)
4. Click "Analyze Stock Data"
5. View results: charts, predictions, and model performance

## Sample Data

The project includes `sample_data.csv` with realistic stock data from 2023. You can also upload your own CSV files from sources like:

- **Kaggle datasets** (S&P 500, stock market data)
- **Yahoo Finance** (download historical data)
- **Alpha Vantage** (export to CSV)
- **Any financial data provider**

## CSV Format Required

```csv
Date,Open,High,Low,Close,Volume
2023-01-03,130.28,130.90,124.17,125.07,112117500
2023-01-04,126.89,128.66,125.08,126.36,89113600
...
```

**Required columns**: Date, Close, Volume  
**Optional columns**: Open, High, Low

## Project Structure

```
├── app.py              # Main Streamlit application
├── sample_data.csv     # Sample historical stock data
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technical Details

- **Data Source**: CSV files (local or uploaded)
- **ML Algorithm**: Linear Regression
- **Features**: Moving averages, price changes, volume changes
- **Frontend**: Streamlit with Plotly charts
- **Dependencies**: Only 5 essential packages (no API dependencies)

## Benefits

- **Works offline** - no internet required
- **No API limits** - use any CSV dataset
- **Fast loading** - instant data access
- **Flexible** - supports various CSV formats
- **Educational** - perfect for learning ML concepts

## Note

This is an educational project. Do not use predictions for actual trading decisions.
