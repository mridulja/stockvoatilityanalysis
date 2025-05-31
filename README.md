# Stock Volatility Analyzer

An interactive web application for analyzing stock volatility across different timeframes using Streamlit and Yahoo Finance data.

## Features

- **Interactive Date Selection**: Choose any date range with minimum 90 days
- **Multiple Timeframes**: Hourly, daily, and weekly analysis
- **Trading Hours Filter**: Select specific hours for hourly analysis
- **Multi-Ticker Comparison**: Compare up to 5 stocks simultaneously
- **Real-time Data**: Uses Yahoo Finance for live market data
- **Interactive Charts**: Candlestick charts with volatility analysis
- **Statistical Analysis**: ATR, volatility, percentiles, and correlation
- **Beautiful UI**: Modern, responsive design with tabbed interface

## Installation

1. **Clone or download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Version
Run the original command-line analysis:
```bash
python stock_analyst.py
```

### Streamlit Web App
Launch the interactive web application:
```bash
streamlit run streamlit_stock_app.py
```

This will open your browser to `http://localhost:8501` with the interactive interface.

## Application Interface

### Sidebar Controls
- **Stock Tickers**: Select up to 5 stocks (default: SPY, QQQ)
- **Date Range**: Choose start and end dates (minimum 90 days)
- **Timeframes**: Enable/disable hourly, daily, weekly analysis
- **Trading Hours**: Filter hourly data to specific time ranges

### Main Interface Tabs

1. **📊 Summary**: Overview table with key metrics and expandable insights
2. **📈 Price Charts**: Interactive candlestick charts with range analysis
3. **🔍 Detailed Stats**: Complete statistical breakdown by timeframe
4. **⚖️ Comparison**: Cross-ticker ATR/volatility comparison and correlation

## Key Metrics Calculated

- **Range**: High - Low for each time period
- **ATR (Average True Range)**: 14-period moving average of ranges
- **Volatility**: Standard deviation of ranges
- **Coefficient of Variation**: Normalized volatility measure
- **Percentiles**: 25th, 50th (median), 75th percentiles
- **Correlation**: Cross-asset range correlation analysis

## Default Tickers Available

Popular ETFs and stocks included:
- **ETFs**: SPY, QQQ
- **Tech Stocks**: AAPL, MSFT, TSLA, NVDA, GOOGL, AMZN, META, NFLX

## Data Source

- **Yahoo Finance**: Free, real-time market data
- **No API Keys Required**: Uses yfinance library
- **Data Caching**: 5-minute cache for improved performance

## Requirements

- Python 3.7+
- Internet connection for real-time data
- See `requirements.txt` for package dependencies

## Notes

- Minimum 90 days of data required for meaningful statistical analysis
- Hourly data available for recent periods (typically last 2 years)
- Market hours filtering available for hourly analysis
- Data automatically handles weekends and holidays
- Performance optimized with caching and progress indicators

## Troubleshooting

1. **No data error**: Check ticker symbol and date range
2. **Slow loading**: Reduce number of tickers or date range
3. **Missing hourly data**: Some older periods may not have hourly data
4. **Connection issues**: Verify internet connection for Yahoo Finance

---

**Happy Trading! 📈** 