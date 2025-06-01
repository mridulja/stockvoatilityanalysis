# 📊 Enhanced Stock Volatility Analyzer with Advanced Options Strategy

A comprehensive Streamlit-based application for stock market analysis, volatility measurement, and data-driven options trading strategies. This tool combines technical analysis, statistical modeling, and market condition assessment to provide professional-grade trading insights.

## 🚀 Key Features

### 📈 Multi-Timeframe Volatility Analysis
- **Hourly Analysis**: Intraday volatility patterns with customizable trading hours
- **Daily Analysis**: Traditional daily range analysis with enhanced ATR calculations
- **Weekly Analysis**: Longer-term volatility trends and patterns

### 🤖 AI-Powered Analysis & Recommendations (NEW!)
- **GPT-4o-mini Integration**: Professional trading analysis using OpenAI's latest model
- **Automatic Analysis**: AI insights generated automatically after options strategy completion
- **Natural Language Summaries**: Human-readable interpretations of complex data
- **Actionable Recommendations**: Specific trading actions and risk management
- **Market Condition Assessment**: AI-powered market overview and timing guidance
- **Risk Management Insights**: Position sizing and stop-loss recommendations

### 🎯 Advanced Options Strategy Engine
- **95% Probability Range**: Statistical confidence intervals using Z-score methodology
- **PUT Spread Recommendations**: Data-driven strike price suggestions
- **Custom Strike Analysis**: User-defined strike price risk assessment
- **VIX-Based Trade Approval**: Market condition filtering for trade safety

### 📊 Enhanced Technical Analysis
- **True Range Calculations**: Accurate ATR using max(H-L, |H-C_prev|, |L-C_prev|)
- **Interactive Charts**: Plotly-based candlestick charts with volatility overlays
- **Cross-Ticker Comparison**: Multi-stock analysis and correlation matrices
- **VIX Integration**: Real-time market condition assessment

### 🌡️ Market Condition Assessment
- **VIX Zones**: Color-coded market conditions with trading recommendations
- **Risk Management**: Automated trade approval/rejection based on volatility levels
- **Market Timeline**: Historical VIX condition tracking

## 🛠️ Technical Implementation

### Core Technologies
- **Streamlit**: Web application framework
- **yfinance**: Real-time stock data retrieval
- **Plotly**: Interactive charting and visualization
- **scipy.stats**: Statistical probability calculations
- **pandas/numpy**: Data manipulation and analysis

### Statistical Methods
- **95% Confidence Intervals**: Using Z-score of 1.96 for probability ranges
- **Normal Distribution Modeling**: Price movement probability calculations
- **True Range Formula**: Enhanced volatility measurement
- **Correlation Analysis**: Multi-asset relationship assessment

### Architecture
```
streamlit_stock_app_complete.py (Main Application)
├── Data Fetching (yfinance)
├── Volatility Calculations (True Range/ATR)
├── Probability Modeling (scipy.stats)
├── VIX Analysis (Market Conditions)
├── Options Strategy Engine
└── Interactive Visualization (Plotly)
```

## 📋 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Internet connection for real-time data

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Stock_Analysis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run streamlit_stock_app_complete.py
   ```

5. **Access the app**: Open your browser to `http://localhost:8501`

### 🤖 **AI Analysis Setup (Optional)**
To enable AI-powered analysis and recommendations:

1. **Get OpenAI API Key**:
   - Visit https://platform.openai.com/api-keys
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Set Environment Variable**:
   ```bash
   # Option 1: Set environment variable
   export OPENAI_API_KEY=your_api_key_here
   
   # Option 2: Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. **Restart the application** to enable AI features

4. **AI Features Available**:
   - 🧠 **Options Strategy Analysis**: Comprehensive trading recommendations
   - 🌟 **Market Summary**: Overall market condition assessment
   - 📋 **Risk Management**: Position sizing and stop-loss guidance
   - 🎯 **Entry/Exit Criteria**: Specific trading actions

## 📖 Usage Guide

### 🔧 Initial Setup
1. **Select Tickers**: Choose up to 5 stock symbols for analysis
2. **Date Range**: Minimum 90 days required for statistical significance
3. **Timeframes**: Enable hourly, daily, and/or weekly analysis
4. **Trading Hours**: Customize hours for hourly analysis (default: 9 AM - 4 PM)

### 📊 Running Analysis
1. Click **"🚀 Run Enhanced Analysis"** in the sidebar
2. Wait for data fetching and calculations to complete
3. Explore results across 6 comprehensive tabs

### 📋 Analysis Tabs

#### 1. 📊 Summary Tab
- **ATR Overview**: Enhanced Average True Range for all timeframes
- **Data Quality**: Validation indicators and sample sizes
- **Quick Metrics**: Key volatility statistics at a glance

#### 2. 📈 Price Charts Tab
- **Interactive Candlesticks**: OHLC price action with zoom capabilities
- **Volatility Overlay**: ATR lines and range analysis
- **VIX Integration**: Market condition visualization (daily charts)

#### 3. 🔍 Detailed Stats Tab
- **Statistical Breakdown**: Count, mean, std, percentiles for each timeframe
- **ATR Quality**: Calculation windows and validity indicators
- **Multi-Ticker Display**: Side-by-side comparison format

#### 4. ⚖️ Comparison Tab
- **ATR Comparison**: Cross-ticker volatility charts
- **Correlation Heatmap**: Asset relationship visualization
- **Relative Analysis**: Volatility ranking and relationships

#### 5. 📉 VIX Analysis Tab
- **Market Conditions**: Real-time VIX assessment with color coding
- **Trading Recommendations**: Risk-based position sizing guidance
- **Historical Timeline**: VIX condition tracking over time

#### 6. 🎯 Options Strategy Tab (Enhanced)
- **95% Probability Range**: Statistical confidence intervals
- **Strike Recommendations**: Data-driven PUT spread suggestions
- **Custom Analysis**: User-defined strike price assessment
- **Risk Assessment**: Multi-level probability zones
- **🤖 AI Analysis**: Professional trading recommendations (if OpenAI API key configured)

### 🤖 **AI-Powered Features**

#### Options Strategy AI Analysis
1. Complete the **Enhanced Options Strategy** analysis
2. **AI analysis runs automatically** - no additional steps needed!
3. Review comprehensive AI insights displayed at the top including:
   - **Executive Summary**: Overall strategy assessment
   - **Risk Management**: Position sizing and stop-loss guidance
   - **Market Conditions**: VIX impact and timing considerations
   - **Actionable Recommendations**: Specific trading actions

#### Market Summary AI Analysis
1. Navigate to the **📊 Summary** tab
2. Check **"🧠 Generate AI Market Overview"**
3. Click **"🚀 Generate Market Summary"**
4. Review AI-powered market assessment covering:
   - **Volatility Environment**: Overall market conditions
   - **Trading Opportunities**: Risk/reward scenarios
   - **Key Recommendations**: Strategic guidance for options traders

## 🔬 Advanced Features

### Statistical Modeling
- **Probability Distributions**: Normal distribution modeling for price movements
- **Safety Scoring**: Risk assessment based on probability thresholds
- **Time Scaling**: Adjustment factors for daily vs weekly options

### Risk Management
- **Automated Filtering**: VIX-based trade rejection
- **Multiple Confidence Levels**: 90%, 95%, 99% probability ranges
- **Custom Strike Testing**: User-defined risk assessment

### Data Quality
- **Real-time Validation**: Data completeness checks
- **Error Handling**: Comprehensive exception management
- **Cache Management**: 5-minute TTL for optimal performance

## 📁 Project Structure

```
Stock_Analysis/
├── streamlit_stock_app_complete.py    # Main application (FINAL VERSION)
├── stock_analyst.py                   # Command-line version
├── requirements.txt                   # Python dependencies
├── README.md                         # This documentation
├── .gitignore                       # Git ignore rules
└── venv/                           # Virtual environment
```

## 🔧 Configuration

### Timeframe Settings
- **Hourly**: 1-hour intervals with customizable trading hours
- **Daily**: 1-day intervals with True Range calculations
- **Weekly**: 1-week intervals for longer-term analysis

### Default Parameters
- **ATR Window**: 14 periods (industry standard)
- **Probability Threshold**: 10% for options recommendations
- **Cache TTL**: 5 minutes for data freshness
- **Minimum Date Range**: 90 days for statistical significance

## 🐛 Troubleshooting

### Common Issues
1. **"No data available"**: Check internet connection and ticker symbols
2. **"Insufficient data"**: Increase date range to minimum 90 days
3. **Charts not loading**: Refresh browser or check JavaScript settings
4. **VIX unavailable**: Network issue - analysis continues without VIX

### Performance Tips
- Limit to 5 tickers maximum for optimal performance
- Use shorter date ranges for faster analysis
- Clear browser cache if experiencing issues

## 🚀 Future Enhancements

### Planned Features
- **Real-time Alerts**: Price and volatility notifications
- **Portfolio Analysis**: Multi-position risk assessment
- **Backtesting**: Historical strategy performance
- **Export Functionality**: PDF reports and data export

### Technical Improvements
- **Database Integration**: Historical data storage
- **API Expansion**: Multiple data providers
- **Mobile Optimization**: Responsive design improvements

## 📞 Support

For technical support or feature requests:
1. Check this documentation first
2. Review the troubleshooting section
3. Open an issue with detailed error descriptions

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using real-time market data.

---

**Built with ❤️ using Streamlit, Plotly, and Python** 