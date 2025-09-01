# 📊 Enhanced Stock Volatility Analyzer with Advanced Options Strategy & AI Chart Analysis

A comprehensive Streamlit-based application for stock market analysis, volatility measurement, data-driven options trading strategies, and AI-powered chart analysis. This tool combines technical analysis, statistical modeling, market condition assessment, and cutting-edge AI vision models to provide professional-grade trading insights.

## 🚀 Key Features

### 📈 Multi-Timeframe Volatility Analysis
- **Hourly Analysis**: Intraday volatility patterns with customizable trading hours
- **Daily Analysis**: Traditional daily range analysis with enhanced ATR calculations
- **Weekly Analysis**: Longer-term volatility trends and patterns

### 🖼️ **AI-Powered Chart Analysis (NEW!)**
- **GPT-5-mini Vision Model**: Advanced image analysis using OpenAI's latest multimodal model
- **Chart Pattern Recognition**: Automatic identification of technical patterns (wedges, triangles, channels)
- **Support/Resistance Mapping**: AI-detected key price levels and breakout points
- **Options Strategy Recommendations**: Tailored strategies based on chart analysis
- **Risk Management Guidelines**: AI-generated position sizing and stop-loss recommendations
- **Structured Output**: Pydantic-validated JSON responses for consistent analysis
- **Text Cleaning**: Advanced post-processing for professional formatting

### 🤖 **Enhanced AI-Powered Analysis & Recommendations**
- **GPT-5-nano Integration**: Professional trading analysis using OpenAI's latest model for tabs 1-8
- **GPT-5-mini Vision**: Advanced chart analysis with image processing capabilities
- **Automatic Analysis**: AI insights generated automatically after options strategy completion
- **Natural Language Summaries**: Human-readable interpretations of complex data
- **Actionable Recommendations**: Specific trading actions and risk management
- **Market Condition Assessment**: AI-powered market overview and timing guidance
- **Risk Management Insights**: Position sizing and stop-loss recommendations
- **Comprehensive Debug Logging**: Real-time console output for development and troubleshooting

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
- **OpenAI GPT Models**: Advanced AI analysis and chart interpretation
- **Pydantic**: Structured data validation and output formatting

### Statistical Methods
- **95% Confidence Intervals**: Using Z-score of 1.96 for probability ranges
- **Normal Distribution Modeling**: Price movement probability calculations
- **True Range Formula**: Enhanced volatility measurement
- **Correlation Analysis**: Multi-asset relationship assessment

### **New Modular Architecture**
```
Stock_Analysis/
├── streamlit_stock_app_complete.py    # Main application entry point
├── core/                              # Core analysis modules
│   ├── chart_analyzer.py             # GPT-5-mini chart analysis engine
│   ├── calculations.py               # Statistical calculations
│   ├── charts.py                     # Chart generation utilities
│   ├── data_fetchers.py             # Data retrieval functions
│   └── styling.py                    # UI styling and themes
├── tabs/                             # Modular tab components
│   ├── tab1_summary.py              # Master analysis center
│   ├── tab2_price_charts.py         # Interactive price charts
│   ├── tab3_detailed_stats.py       # Statistical analysis
│   ├── tab4_comparison.py           # Multi-ticker comparison
│   ├── tab5_vix_analysis.py         # VIX market conditions
│   ├── tab6_options_strategy.py     # Options strategy engine
│   ├── tab7_put_spread_analysis.py  # PUT spread analysis
│   ├── tab8_iron_condor_playbook.py # Iron condor strategies
│   └── tab9_chart_analysis.py       # AI chart analysis (GPT-5-mini)
├── shared/                           # Shared utilities
│   └── ai_formatter.py              # AI response formatting
├── config/                           # Configuration management
│   └── settings.py                  # Application settings
├── llm_analysis.py                  # GPT-5-nano text analysis
├── llm_input_formatters.py          # LLM input formatting
└── requirements.txt                  # Python dependencies
```

## 📋 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Internet connection for real-time data
- OpenAI API key for AI features

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

4. **Set up OpenAI API key** (required for AI features):
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

5. **Run the application**:
   ```bash
   streamlit run streamlit_stock_app_complete.py
   ```

6. **Access the app**: Open your browser to `http://localhost:8501`

### 🤖 **AI Analysis Setup (Required for Full Features)**
To enable AI-powered analysis and chart interpretation:

1. **Get OpenAI API Key**:
   - Visit https://platform.openai.com/api-keys
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Set Environment Variable**:
   ```bash
   # Option 1: Set environment variable
   export OPENAI_API_KEY=your_api_key_here
   
   # Option 2: Create .env file (Recommended)
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. **Restart the application** to enable AI features

4. **AI Models Available**:
   - 🖼️ **GPT-5-mini**: Chart analysis with vision capabilities (Tab 9)
   - 🧠 **GPT-5-nano**: Text-based analysis for all other tabs (Tabs 1-8)
   - 🔄 **Automatic Fallback**: Seamless model switching for reliability

## 📖 Usage Guide

### 🔧 Initial Setup
1. **Select Tickers**: Choose up to 5 stock symbols for analysis
2. **Date Range**: Minimum 90 days required for statistical significance
3. **Timeframes**: Enable hourly, daily, and/or weekly analysis
4. **Trading Hours**: Customize hours for hourly analysis (default: 9 AM - 4 PM)

### 📊 Running Analysis
1. Click **"🚀 Run Enhanced Analysis"** in the sidebar
2. Wait for data fetching and calculations to complete
3. Explore results across 9 comprehensive tabs

### 📋 Analysis Tabs

#### 1. 📊 Summary Tab (Master Analysis Center)
- **Master Analysis**: One-click comprehensive analysis across all modules
- **ATR Overview**: Enhanced Average True Range for all timeframes
- **Data Quality**: Validation indicators and sample sizes
- **Quick Metrics**: Key volatility statistics at a glance
- **🤖 AI Analysis**: GPT-5-nano powered market insights and strategy recommendations

#### 2. 📈 Price Charts Tab
- **Interactive Candlesticks**: OHLC price action with zoom capabilities
- **Volatility Overlay**: ATR lines and range analysis
- **VIX Integration**: Market condition visualization (daily charts)
- **🤖 AI Analysis**: Technical analysis and trading recommendations

#### 3. 🔍 Detailed Stats Tab
- **Statistical Breakdown**: Count, mean, std, percentiles for each timeframe
- **ATR Quality**: Calculation windows and validity indicators
- **Multi-Ticker Display**: Side-by-side comparison format
- **🤖 AI Analysis**: Statistical interpretation and risk assessment

#### 4. ⚖️ Comparison Tab
- **ATR Comparison**: Cross-ticker volatility charts
- **Correlation Heatmap**: Asset relationship visualization
- **Relative Analysis**: Volatility ranking and relationships
- **🤖 AI Analysis**: Comparative market insights

#### 5. 📉 VIX Analysis Tab
- **Market Conditions**: Real-time VIX assessment with color coding
- **Trading Recommendations**: Risk-based position sizing guidance
- **Historical Timeline**: VIX condition tracking over time
- **🤖 AI Analysis**: Market condition interpretation

#### 6. 🎯 Options Strategy Tab (Enhanced)
- **95% Probability Range**: Statistical confidence intervals
- **Strike Recommendations**: Data-driven PUT spread suggestions
- **Custom Analysis**: User-defined strike price assessment
- **Risk Assessment**: Multi-level probability zones
- **🤖 AI Analysis**: Professional trading recommendations using GPT-5-nano

#### 7. 📐 PUT Spread Analysis Tab
- **Advanced PUT Spreads**: Comprehensive strategy analysis
- **Risk Assessment**: Probability of profit calculations
- **🤖 AI Analysis**: PUT spread strategy insights

#### 8. 🦅 Iron Condor Playbook Tab
- **Iron Condor Strategies**: Multi-leg options strategies
- **Probability Analysis**: Risk/reward calculations
- **🤖 AI Analysis**: Iron condor strategy recommendations

#### 9. 🖼️ **Chart Analysis Tab (NEW!)**
- **Image Upload**: Support for PNG, JPG chart images
- **GPT-5-mini Vision**: Advanced AI chart interpretation
- **Pattern Recognition**: Automatic technical pattern identification
- **Support/Resistance**: AI-detected key price levels
- **Options Strategies**: Tailored recommendations based on chart analysis
- **Risk Management**: AI-generated position sizing and stop-loss guidance
- **Structured Output**: Pydantic-validated JSON responses
- **Debug Console**: Real-time analysis progress and model information

### 🤖 **AI-Powered Features**

#### Chart Analysis (Tab 9)
1. **Upload Chart Image**: PNG or JPG format (max 10MB)
2. **Select Analysis Mode**: Quick or Deep Technical Analysis
3. **Custom System Prompt**: Optional guidance for AI analysis
4. **AI Processing**: GPT-5-mini analyzes chart with vision capabilities
5. **Results**: Structured analysis with patterns, levels, and strategies

#### Text Analysis (Tabs 1-8)
1. **Automatic AI Integration**: AI analysis runs automatically after strategy completion
2. **GPT-5-nano Processing**: Advanced text analysis for trading insights
3. **Comprehensive Coverage**: Market assessment, risk management, and recommendations

#### AI Model Configuration
- **Primary Model**: GPT-5-nano for text analysis (tabs 1-8)
- **Vision Model**: GPT-5-mini for chart analysis (tab 9)
- **Fallback System**: Automatic model switching for reliability
- **Debug Logging**: Real-time console output for development

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

### **AI Model Management**
- **Dynamic Parameter Selection**: Automatic GPT-4 vs GPT-5 parameter handling
- **Response Validation**: Pydantic schema validation for structured output
- **Text Cleaning**: Advanced post-processing for professional formatting
- **Token Tracking**: Comprehensive usage monitoring and logging

## 📁 Project Structure

```
Stock_Analysis/
├── streamlit_stock_app_complete.py    # Main application entry point
├── core/                              # Core analysis modules
│   ├── chart_analyzer.py             # GPT-5-mini chart analysis engine
│   ├── calculations.py               # Statistical calculations
│   ├── charts.py                     # Chart generation utilities
│   ├── data_fetchers.py             # Data retrieval functions
│   └── styling.py                    # UI styling and themes
├── tabs/                             # Modular tab components
│   ├── tab1_summary.py              # Master analysis center
│   ├── tab2_price_charts.py         # Interactive price charts
│   ├── tab3_detailed_stats.py       # Statistical analysis
│   ├── tab4_comparison.py           # Multi-ticker comparison
│   ├── tab5_vix_analysis.py         # VIX market conditions
│   ├── tab6_options_strategy.py     # Options strategy engine
│   ├── tab7_put_spread_analysis.py  # PUT spread analysis
│   ├── tab8_iron_condor_playbook.py # Iron condor strategies
│   └── tab9_chart_analysis.py       # AI chart analysis (GPT-5-mini)
├── shared/                           # Shared utilities
│   └── ai_formatter.py              # AI response formatting
├── config/                           # Configuration management
│   └── settings.py                  # Application settings
├── llm_analysis.py                  # GPT-5-nano text analysis
├── llm_input_formatters.py          # LLM input formatting
├── requirements.txt                  # Python dependencies
├── .env                             # Environment variables (create this)
└── README.md                        # This documentation
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

### **AI Model Settings**
- **GPT-5-nano**: Primary model for text analysis (tabs 1-8)
- **GPT-5-mini**: Vision model for chart analysis (tab 9)
- **Fallback Models**: GPT-4o-mini for reliability
- **Debug Logging**: Comprehensive console output for development

## 🐛 Troubleshooting

### Common Issues
1. **"No data available"**: Check internet connection and ticker symbols
2. **"Insufficient data"**: Increase date range to minimum 90 days
3. **Charts not loading**: Refresh browser or check JavaScript settings
4. **VIX unavailable**: Network issue - analysis continues without VIX
5. **AI analysis fails**: Check OpenAI API key in .env file
6. **Empty AI responses**: Verify model availability and API limits

### **AI-Specific Issues**
1. **"API key not found"**: Create .env file with OPENAI_API_KEY
2. **"Model not available"**: Check OpenAI model access and billing
3. **Empty chart analysis**: Verify image format (PNG/JPG) and file size
4. **Debug output missing**: Check console for real-time logging

### Performance Tips
- Limit to 5 tickers maximum for optimal performance
- Use shorter date ranges for faster analysis
- Clear browser cache if experiencing issues
- Monitor OpenAI API usage and rate limits

## 🚀 Future Enhancements

### Planned Features
- **Real-time Alerts**: Price and volatility notifications
- **Portfolio Analysis**: Multi-position risk assessment
- **Backtesting**: Historical strategy performance
- **Export Functionality**: PDF reports and data export
- **Additional AI Models**: Claude, Gemini integration options

### Technical Improvements
- **Database Integration**: Historical data storage
- **API Expansion**: Multiple data providers
- **Mobile Optimization**: Responsive design improvements
- **Advanced AI Features**: Multi-modal analysis and predictions

## 📞 Support

For technical support or feature requests:
1. Check this documentation first
2. Review the troubleshooting section
3. Check console debug output for detailed error information
4. Open an issue with detailed error descriptions and debug logs

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using real-time market data and OpenAI API usage.

---

**Built with ❤️ using Streamlit, Plotly, Python, and OpenAI GPT Models** 