"""
Enhanced Stock Volatility Analyzer with Advanced Options Strategy

This Streamlit application provides comprehensive stock market analysis with a focus on volatility 
measurement and options trading strategies. It combines technical analysis, statistical modeling, 
and market condition assessment to provide data-driven trading recommendations.

Key Features:
- Multi-timeframe volatility analysis (hourly, daily, weekly)
- Enhanced ATR (Average True Range) calculations with True Range methodology
- VIX market condition assessment and trading recommendations  
- Advanced options strategy with 95% probability range calculations
- Interactive charts with Plotly for price action and volatility visualization
- Cross-ticker comparison and correlation analysis
- Probability-based PUT spread recommendations using statistical modeling

Technical Implementation:
- Uses scipy.stats for probability distribution calculations
- Implements True Range formula: max(H-L, |H-C_prev|, |L-C_prev|)
- Calculates 95% confidence intervals using Z-score of 1.96
- VIX-based market condition filtering for trade approval
- Session state persistence for analysis results

Author: Mridul Jain
Date: 2025
Version: 2.0 - Enhanced with 95% probability analysis
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from scipy import stats
import warnings
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will rely on system environment variables

# Import LLM analysis functionality
try:
    from llm_analysis import get_llm_analyzer, format_vix_data_for_llm, format_confidence_levels_for_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import Put Spread Analysis functionality
try:
    from put_spread_analysis import (
        PutSpreadAnalyzer, 
        format_percentage, 
        format_currency, 
        get_next_friday, 
        get_same_day_expiry
    )
    PUT_SPREAD_AVAILABLE = True
except ImportError:
    PUT_SPREAD_AVAILABLE = False

# Import Iron Condor Analysis functionality
try:
    from iron_condor_analysis import (
        IronCondorAnalyzer, 
        format_currency as ic_format_currency, 
        format_percentage as ic_format_percentage, 
        get_next_friday as ic_get_next_friday,
        get_next_monthly_expiry
    )
    from iron_condor_charts import (
        create_iron_condor_pnl_chart,
        create_strategy_comparison_chart as ic_create_strategy_comparison_chart,
        create_pop_distribution_chart as ic_create_pop_distribution_chart,
        create_trade_management_dashboard,
        create_volatility_impact_chart,
        create_earnings_impact_analysis
    )
    IRON_CONDOR_AVAILABLE = True
except ImportError:
    IRON_CONDOR_AVAILABLE = False

# Import modular tab functionality
try:
    from tabs.tab1_summary import render_summary_tab
    TAB1_MODULAR = True
except ImportError:
    TAB1_MODULAR = False

try:
    from tabs.tab2_price_charts import render_price_charts_tab
    TAB2_MODULAR = True
except ImportError:
    TAB2_MODULAR = False

try:
    from tabs.tab3_detailed_stats import render_detailed_stats_tab
    TAB3_MODULAR = True
except ImportError:
    TAB3_MODULAR = False

try:
    from tabs.tab4_comparison import render_comparison_tab
    TAB4_MODULAR = True
except ImportError:
    TAB4_MODULAR = False

# Fallback functions for Put Spread Analysis
def get_same_day_expiry():
    """Fallback function for same day expiry"""
    return date.today().strftime('%Y-%m-%d')

def get_next_friday():
    """Fallback function for next Friday expiry"""
    today = date.today()
    days_ahead = 4 - today.weekday()  # Friday is weekday 4
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

def format_percentage(value):
    """Fallback function for formatting percentages"""
    return f"{value*100:.1f}%"

def format_currency(value):
    """Fallback function for formatting currency"""
    return f"${value:.2f}"

# Try to import Put Spread Analysis functions at module level - override fallbacks if available
try:
    from put_spread_analysis import (
        PutSpreadAnalyzer, 
        format_percentage as psa_format_percentage, 
        format_currency as psa_format_currency, 
        get_next_friday as psa_get_next_friday, 
        get_same_day_expiry as psa_get_same_day_expiry
    )
    # Override fallback functions with real ones if available
    format_percentage = psa_format_percentage
    format_currency = psa_format_currency
    get_next_friday = psa_get_next_friday
    get_same_day_expiry = psa_get_same_day_expiry
    PUT_SPREAD_AVAILABLE = True
except ImportError:
    # Use fallback functions defined above
    pass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Volatility Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary-color: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --secondary-color: #06b6d4;
        --accent-color: #f59e0b;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-primary: #ffffff;
        --background-secondary: #f8fafc;
        --background-tertiary: #f1f5f9;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-light: #94a3b8;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        --gradient-secondary: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    }
    
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 3rem;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    
    /* VIX Condition Styling with Modern Colors */
    .vix-calm { 
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1rem; 
        border-radius: 12px; 
        border-left: 4px solid var(--success-color);
        box-shadow: var(--shadow-sm);
    }
    .vix-normal { 
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1rem; 
        border-radius: 12px; 
        border-left: 4px solid var(--secondary-color);
        box-shadow: var(--shadow-sm);
    }
    .vix-choppy { 
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1rem; 
        border-radius: 12px; 
        border-left: 4px solid var(--warning-color);
        box-shadow: var(--shadow-sm);
    }
    .vix-volatile { 
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        padding: 1rem; 
        border-radius: 12px; 
        border-left: 4px solid var(--error-color);
        box-shadow: var(--shadow-sm);
    }
    .vix-extreme { 
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        color: white;
        padding: 1rem; 
        border-radius: 12px; 
        border-left: 4px solid #dc2626;
        box-shadow: var(--shadow-lg);
    }
    
    .strike-recommend { 
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem; 
        border-radius: 16px; 
        border: 2px solid var(--success-color);
        box-shadow: var(--shadow-lg);
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .strike-recommend::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }
    
    /* Button Styling */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease-in-out;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def get_vix_condition(vix_value):
    """Determine market condition based on VIX level"""
    if pd.isna(vix_value):
        return "Unknown", "vix-normal", "🤷"
    
    if vix_value < 15:
        return "Calm Markets - Clean Trend", "vix-calm", "🟢"
    elif 15 <= vix_value < 19:
        return "Normal Markets - Trendy", "vix-normal", "🔵"
    elif 19 <= vix_value < 26:
        return "Choppy Market - Proceed with Caution", "vix-choppy", "🟡"
    elif 26 <= vix_value < 36:
        return "High Volatility - Big Swings, Don't Trade", "vix-volatile", "🔴"
    else:
        return "Extreme Volatility - Very Bad Day, DO NOT TRADE", "vix-extreme", "🚨"

def should_trade(vix_value):
    """Determine if trading is recommended based on VIX"""
    if pd.isna(vix_value):
        return False, "Unknown VIX - Cannot assess"
    
    if vix_value < 26:
        return True, "Trading conditions acceptable"
    else:
        return False, "VIX too high - Avoid trading"

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, start_date, end_date, interval='1h'):
    """Fetch stock data with caching"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            return None
            
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure consistent column names
        data.columns = data.columns.str.title()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_vix_data(start_date, end_date):
    """Fetch VIX data with caching"""
    try:
        vix_data = yf.download("^VIX", start=start_date, end=end_date, interval='1d', progress=False)
        
        if vix_data.empty:
            return None
            
        # Handle multi-level columns
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.droplevel(1)
        
        # Ensure consistent column names
        vix_data.columns = vix_data.columns.str.title()
        
        # Add market condition analysis
        vix_data['VIX_Close'] = vix_data['Close']
        vix_data['Market_Condition'] = vix_data['VIX_Close'].apply(lambda x: get_vix_condition(x)[0])
        vix_data['Condition_Color'] = vix_data['VIX_Close'].apply(lambda x: get_vix_condition(x)[1])
        vix_data['Condition_Icon'] = vix_data['VIX_Close'].apply(lambda x: get_vix_condition(x)[2])
        
        return vix_data[['VIX_Close', 'Market_Condition', 'Condition_Color', 'Condition_Icon']]
    except Exception as e:
        st.warning(f"Could not fetch VIX data: {str(e)}")
        return None

@st.cache_data(ttl=300)
def get_current_price(ticker):
    """Get current/latest price for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d', interval='1m')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        else:
            # Fallback to daily data
            hist = stock.history(period='5d')
            return hist['Close'].iloc[-1] if not hist.empty else None
    except:
        return None

@st.cache_data(ttl=300)
def validate_ticker(ticker):
    """Validate if a ticker exists and can be traded"""
    try:
        stock = yf.Ticker(ticker)
        # Try to get basic info and recent price
        info = stock.info
        hist = stock.history(period='5d')
        
        if hist.empty or not info:
            return False, "No data available"
        
        # Check if it's a valid stock with basic info
        if 'longName' in info or 'shortName' in info:
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            return True, f"${current_price:.2f}" if current_price else "Valid"
        else:
            return False, "Invalid ticker"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def calculate_true_range(data):
    """Calculate True Range for ATR calculation"""
    if data is None or len(data) < 2:
        return None
    
    data = data.copy()
    
    # True Range calculation: max of (H-L, |H-C_prev|, |L-C_prev|)
    data['prev_close'] = data['Close'].shift(1)
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = abs(data['High'] - data['prev_close'])
    data['tr3'] = abs(data['Low'] - data['prev_close'])
    
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # For the first row, use simple range since no previous close
    data.loc[data.index[0], 'true_range'] = data.loc[data.index[0], 'tr1']
    
    return data['true_range']

def calculate_volatility_metrics(data, timeframe='hourly', vix_data=None):
    """Calculate volatility metrics for given timeframe"""
    if data is None or data.empty:
        return None
    
    # Calculate ranges and true range
    data = data.copy()
    data['range'] = data['High'] - data['Low']
    
    # Calculate true range for better ATR
    true_range = calculate_true_range(data)
    if true_range is not None:
        data['true_range'] = true_range
    else:
        data['true_range'] = data['range']  # Fallback to simple range
    
    # Resample based on timeframe
    if timeframe == 'daily':
        resampled = data.resample('D').agg({
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last', 
            'Volume': 'sum',
            'range': 'max',
            'true_range': 'max'
        })
        resampled['range'] = resampled['High'] - resampled['Low']
        daily_tr = calculate_true_range(resampled)
        if daily_tr is not None:
            resampled['true_range'] = daily_tr
    elif timeframe == 'weekly':
        resampled = data.resample('W').agg({
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last', 
            'Volume': 'sum',
            'range': 'max',
            'true_range': 'max'
        })
        resampled['range'] = resampled['High'] - resampled['Low']
        weekly_tr = calculate_true_range(resampled)
        if weekly_tr is not None:
            resampled['true_range'] = weekly_tr
    else:  # hourly
        resampled = data
    
    # Remove any rows with NaN values
    resampled = resampled.dropna()
    
    if len(resampled) == 0:
        return None
    
    # Calculate statistics
    range_stats = resampled['range'].describe(percentiles=[.25, .5, .75])
    
    # Calculate ATR (Average True Range)
    atr_window = min(14, len(resampled))
    if atr_window > 0 and not resampled['true_range'].isna().all():
        atr = resampled['true_range'].rolling(window=atr_window).mean().iloc[-1]
        if pd.isna(atr) and len(resampled) >= 1:
            atr = resampled['true_range'].mean()
    else:
        atr = 0
    
    # Calculate additional metrics
    volatility = resampled['range'].std()
    cv = volatility / range_stats['mean'] if range_stats['mean'] > 0 else 0
    
    # Add VIX data if available for daily analysis
    if timeframe == 'daily' and vix_data is not None:
        resampled = resampled.join(vix_data, how='left')
    
    return {
        'stats': range_stats,
        'atr': atr if not pd.isna(atr) else 0,
        'volatility': volatility if not pd.isna(volatility) else 0,
        'coefficient_variation': cv,
        'data': resampled,
        'atr_window': atr_window
    }

def create_price_chart(data, ticker, timeframe, show_vix=False):
    """Create interactive price chart with ranges and VIX overlay"""
    if data is None or data.empty:
        return None
    
    # Determine number of subplots
    rows = 3 if (show_vix and 'VIX_Close' in data.columns) else 2
    subplot_titles = [f'{ticker} Price Action ({timeframe.title()})', 'Range Analysis']
    if show_vix and 'VIX_Close' in data.columns:
        subplot_titles.append('VIX Analysis')
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=[0.5, 0.25, 0.25] if rows == 3 else [0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'] if 'Open' in data.columns else data['Close'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f'{ticker} Price'
        ),
        row=1, col=1
    )
    
    # Range bar chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['range'],
            name='Range',
            marker_color='rgba(99, 102, 241, 0.7)',
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Add ATR line
    if len(data) > 1 and 'true_range' in data.columns:
        atr_window = min(14, len(data))
        atr_line = data['true_range'].rolling(window=atr_window).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=atr_line,
                mode='lines',
                name=f'{atr_window}-period ATR',
                line=dict(color='#ef4444', width=3)
            ),
            row=2, col=1
        )
    
    # Add VIX chart if available
    if show_vix and 'VIX_Close' in data.columns and rows == 3:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['VIX_Close'],
                mode='lines+markers',
                name='VIX',
                line=dict(color='#8b5cf6', width=3),
                marker=dict(size=5, color='#8b5cf6')
            ),
            row=3, col=1
        )
        
        # Add VIX level zones
        fig.add_hline(y=15, line_dash="dash", line_color="#10b981", opacity=0.6, row=3)
        fig.add_hline(y=19, line_dash="dash", line_color="#06b6d4", opacity=0.6, row=3)
        fig.add_hline(y=25, line_dash="dash", line_color="#f59e0b", opacity=0.6, row=3)
        fig.add_hline(y=35, line_dash="dash", line_color="#ef4444", opacity=0.6, row=3)
    
    fig.update_layout(
        title=f'{ticker} - {timeframe.title()} Analysis with Enhanced ATR',
        xaxis_rangeslider_visible=False,
        height=800 if rows == 3 else 600,
        showlegend=True
    )
    
    return fig

def create_comparison_chart(results_dict, metric='atr'):
    """Create comparison chart between multiple tickers"""
    tickers = list(results_dict.keys())
    timeframes = ['hourly', 'daily', 'weekly']
    
    fig = go.Figure()
    
    for ticker in tickers:
        values = []
        for tf in timeframes:
            if tf in results_dict[ticker] and results_dict[ticker][tf] and metric in results_dict[ticker][tf]:
                val = results_dict[ticker][tf][metric]
                values.append(val if not pd.isna(val) and val > 0 else 0)
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=ticker,
            x=timeframes,
            y=values,
            text=[f'${v:.2f}' if v > 0 else 'N/A' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f'{metric.upper()} Comparison Across Timeframes (Enhanced Calculation)',
        xaxis_title='Timeframe',
        yaxis_title=f'{metric.upper()} Value ($)',
        barmode='group',
        height=400
    )
    
    return fig

# Import the working function from core module
from core.charts import create_enhanced_price_chart

def create_enhanced_price_chart_backup(data, ticker, timeframe, chart_type='Candlestick', show_volume=True, indicators=None, vix_data=None):
    """Create enhanced interactive price chart with technical indicators"""
    if data is None or data.empty:
        return None
    
    # Determine subplot configuration
    rows = 3 if show_volume else 2
    subplot_titles = [f'{ticker} - {chart_type} Chart ({timeframe.title()})', 'Technical Indicators']
    if show_volume:
        subplot_titles.append('Volume')
    
    # Configure row heights
    if show_volume:
        row_heights = [0.6, 0.25, 0.15]
    else:
        row_heights = [0.7, 0.3]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}]] * rows
    )
    
    # Main price chart based on type
    if chart_type == 'Candlestick':
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'] if 'Open' in data.columns else data['Close'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f'{ticker} Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
    elif chart_type == 'Line':
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{ticker} Close',
                line=dict(color='#2196f3', width=2)
            ),
            row=1, col=1
        )
    elif chart_type == 'Area':
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                fill='tonexty',
                mode='lines',
                name=f'{ticker} Close',
                line=dict(color='#2196f3', width=1),
                fillcolor='rgba(33, 150, 243, 0.1)'
            ),
            row=1, col=1
        )
    elif chart_type == 'OHLC':
        fig.add_trace(
            go.Ohlc(
                x=data.index,
                open=data['Open'] if 'Open' in data.columns else data['Close'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f'{ticker} OHLC'
            ),
            row=1, col=1
        )
    
    # Add technical indicators if specified
    if indicators:
        # Moving Averages
        if indicators.get('sma_20') and 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#ff9800', width=2, dash='solid')
                ),
                row=1, col=1
            )
        
        if indicators.get('sma_50') and 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#e91e63', width=2, dash='solid')
                ),
                row=1, col=1
            )
        
        if indicators.get('ema_12') and 'EMA_12' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_12'],
                    mode='lines',
                    name='EMA 12',
                    line=dict(color='#4caf50', width=2, dash='dot')
                ),
                row=1, col=1
            )
        
        if indicators.get('ema_26') and 'EMA_26' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_26'],
                    mode='lines',
                    name='EMA 26',
                    line=dict(color='#9c27b0', width=2, dash='dot')
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if indicators.get('bollinger_bands') and all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='Bollinger Bands',
                    line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='#607d8b', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # ATR Bands
        if indicators.get('atr_bands') and all(col in data.columns for col in ['ATR_Upper', 'ATR_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ATR_Upper'],
                    mode='lines',
                    name='ATR Upper',
                    line=dict(color='rgba(255, 152, 0, 0.6)', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ATR_Lower'],
                    mode='lines',
                    name='ATR Bands',
                    line=dict(color='rgba(255, 152, 0, 0.6)', width=1, dash='dash')
                ),
                row=1, col=1
            )
    
    # Technical indicators subplot (ATR)
    if 'true_range' in data.columns:
        atr_window = min(14, len(data))
        atr_line = data['true_range'].rolling(window=atr_window).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=atr_line,
                mode='lines',
                name=f'ATR ({atr_window})',
                line=dict(color='#f44336', width=2)
            ),
            row=2, col=1
        )
        
        # Add ATR average line
        atr_avg = atr_line.mean()
        fig.add_hline(
            y=atr_avg,
            line_dash="dash",
            line_color="#f44336",
            opacity=0.5,
            row=2,
            annotation_text=f"ATR Avg: ${atr_avg:.2f}"
        )
    
    # Volume subplot
    if show_volume and 'Volume' in data.columns:
        colors = ['#26a69a' if close >= open else '#ef5350' 
                 for close, open in zip(data['Close'], data['Open'] if 'Open' in data.columns else data['Close'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=3 if show_volume else 2, col=1
        )
        
        # Add volume moving average
        if len(data) >= 20:
            vol_ma = data['Volume'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=vol_ma,
                    mode='lines',
                    name='Volume MA(20)',
                    line=dict(color='#ff9800', width=2)
                ),
                row=3 if show_volume else 2, col=1
            )
    
    # VIX overlay if requested
    if indicators and indicators.get('vix_overlay') and vix_data is not None:
        # Add VIX to a secondary y-axis on the main chart
        fig.add_trace(
            go.Scatter(
                x=vix_data.index,
                y=vix_data['VIX_Close'],
                mode='lines',
                name='VIX',
                line=dict(color='#9c27b0', width=2, dash='dot'),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Update layout for secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title='VIX',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - Enhanced {chart_type} Analysis ({timeframe.title()})',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=rows, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="ATR ($)", row=2, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

def main():
    """Main Streamlit application with 8 comprehensive tabs"""

    # Header with modern gradient design
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div class="main-header">🎯 Stock Volatility Analyzer</div>
        <p style="font-size: 1.25rem; color: var(--text-secondary); font-weight: 400; margin-top: -1rem;">
            Advanced Options Strategy with 95% Probability Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'vix_data' not in st.session_state:
        st.session_state.vix_data = None
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ['SPY', 'QQQ']
    
    # Sidebar for controls
    st.sidebar.header("Analysis Parameters")
    
    # Ticker selection
    default_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'GLD']
    selected_tickers = st.sidebar.multiselect(
        "Select Stock Tickers",
        options=default_tickers + ['NVDA', 'GOOGL', 'AMZN', 'META', 'NFLX', 'IWM', 'DIA'],
        default=['SPY', 'QQQ'],
        help="Choose up to 5 tickers for comparison"
    )
    
    # Date selection
    st.sidebar.subheader("Date Range Selection")
    
    min_date = date.today() - timedelta(days=365*2)
    max_date = date.today()
    default_start = date.today() - timedelta(days=90)
    
    start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=start_date, max_value=max_date)
    
    # Validate minimum 90 days
    date_range = (end_date - start_date).days
    if date_range < 90:
        st.sidebar.error("Please select at least 90 days of data for meaningful analysis")
        return
    
    st.sidebar.success(f"Selected range: {date_range} days")
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    include_hourly = st.sidebar.checkbox("Include Hourly Analysis", value=True)
    include_daily = st.sidebar.checkbox("Include Daily Analysis", value=True)
    include_weekly = st.sidebar.checkbox("Include Weekly Analysis", value=True)
    include_vix = st.sidebar.checkbox("Include VIX Analysis", value=True)
    
    # Analysis button
    if st.sidebar.button("🚀 Run Enhanced Analysis", type="primary"):
        if not selected_tickers:
            st.error("Please select at least one ticker")
            return
        
        # Run analysis
        results = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(selected_tickers):
            progress_bar.progress((i + 1) / len(selected_tickers))
            results[ticker] = {}
            
            # Fetch data for different timeframes
            timeframes = []
            if include_hourly:
                timeframes.append(('hourly', '1h'))
            if include_daily:
                timeframes.append(('daily', '1d'))
            if include_weekly:
                timeframes.append(('weekly', '1wk'))
            
            for tf_name, tf_interval in timeframes:
                data = fetch_stock_data(ticker, start_date, end_date, tf_interval)
                if data is not None:
                    results[ticker][tf_name] = calculate_volatility_metrics(data, tf_name)
                else:
                    results[ticker][tf_name] = None
        
        progress_bar.empty()
        
        # Fetch VIX data if requested
        vix_data = None
        if include_vix:
            vix_data = fetch_vix_data(start_date, end_date)
        
        # Store results in session state
        st.session_state.results = results
        st.session_state.vix_data = vix_data
        st.session_state.selected_tickers = selected_tickers
    
    # Get results from session state
    results = getattr(st.session_state, 'results', {})
    vix_data = getattr(st.session_state, 'vix_data', None)
    session_tickers = getattr(st.session_state, 'selected_tickers', selected_tickers)
    
    # CREATE TABS - Show all 8 tabs
    if results and len(results) > 0:
        st.success("📈 Analysis Complete! All tabs are now available.")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Summary", 
            "📈 Price Charts", 
            "🔍 Detailed Stats", 
            "⚖️ Comparison", 
            "📉 VIX Analysis",
            "🎯 Options Strategy",
            "📐 Put Spread Analysis",
            "🦅 Iron Condor Playbook"
        ])
        
        with tab1:
            # Use modular Tab 1 if available, otherwise fallback to original
            if TAB1_MODULAR:
                render_summary_tab(results, vix_data, session_tickers)
            else:
                st.warning("⚠️ Modular Tab 1 not available. Using fallback implementation.")
                st.subheader("📊 Summary Tab (Fallback)")
                st.info("Please ensure tabs/tab1_summary.py is available for full functionality.")
        
        with tab2:
            # Use modular Tab 2 if available, otherwise fallback to original
            if TAB2_MODULAR:
                render_price_charts_tab(results, vix_data, session_tickers)
            else:
                st.warning("⚠️ Modular Tab 2 not available. Using fallback implementation.")
                st.subheader("📈 Price Charts Tab (Fallback)")
                st.info("Please ensure tabs/tab2_price_charts.py is available for full functionality.")
        
        with tab3:
            # Use modular Tab 3 if available, otherwise fallback to original
            if TAB3_MODULAR:
                render_detailed_stats_tab(results, vix_data, session_tickers)
            else:
                st.warning("⚠️ Modular Tab 3 not available. Using fallback implementation.")
                st.subheader("🔍 Detailed Statistical Analysis")
                
                stats_ticker = st.selectbox("Select ticker for stats:", session_tickers, key="stats_ticker")
                
                if stats_ticker in results:
                    for timeframe in ['hourly', 'daily', 'weekly']:
                        if timeframe in results[stats_ticker] and results[stats_ticker][timeframe]:
                            st.markdown(f"#### {timeframe.title()} Analysis for {stats_ticker}")
                            
                            data = results[stats_ticker][timeframe]['data']
                            if data is not None and not data.empty:
                                stats = results[stats_ticker][timeframe]['stats']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean Range", f"${stats['mean']:.2f}")
                                with col2:
                                    st.metric("Standard Deviation", f"${stats['std']:.2f}")
                                with col3:
                                    st.metric("75th Percentile", f"${stats['75%']:.2f}")
                
                st.info("Please ensure tabs/tab3_detailed_stats.py is available for full functionality.")
        
        with tab4:
            # Use modular Tab 4 if available, otherwise fallback to original
            if TAB4_MODULAR:
                render_comparison_tab(results, vix_data, session_tickers)
            else:
                st.warning("⚠️ Modular Tab 4 not available. Using fallback implementation.")
                st.subheader("⚖️ Multi-Ticker Comparison (Fallback)")
                
                if len(session_tickers) > 1:
                    comparison_metric = st.selectbox(
                        "Select metric to compare:",
                        ['atr', 'volatility'],
                        format_func=lambda x: {'atr': 'Average True Range (ATR)', 'volatility': 'Standard Deviation'}[x],
                        key="comparison_metric"
                    )
                    
                    comparison_chart = create_comparison_chart(results, comparison_metric)
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)
                else:
                    st.info("Please select multiple tickers for comparison analysis")
                
                st.info("Please ensure tabs/tab4_comparison.py is available for full functionality.")
        
        with tab5:
            st.subheader("📉 VIX Market Condition Analysis")
            
            if vix_data is not None:
                current_vix = vix_data['VIX_Close'].iloc[-1]
                condition, condition_class, icon = get_vix_condition(current_vix)
                trade_ok, trade_msg = should_trade(current_vix)
                
                st.markdown(f"""
                <div class="{condition_class}">
                    <h3>{icon} Current VIX: {current_vix:.2f}</h3>
                    <p><strong>Market Condition:</strong> {condition}</p>
                    <p><strong>Trading Recommendation:</strong> {trade_msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # VIX statistics
                vix_stats = vix_data['VIX_Close'].describe()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean VIX", f"{vix_stats['mean']:.2f}")
                with col2:
                    st.metric("Std Dev", f"{vix_stats['std']:.2f}")
                with col3:
                    st.metric("Min VIX", f"{vix_stats['min']:.2f}")
                with col4:
                    st.metric("Max VIX", f"{vix_stats['max']:.2f}")
            else:
                st.info("VIX analysis not available. Enable VIX analysis in sidebar and run analysis.")
        
        with tab6:
            st.subheader("🎯 Advanced Options Strategy")
            
            strategy_ticker = st.selectbox("Select ticker for options:", session_tickers, key="strategy_ticker")
            
            if strategy_ticker in results:
                current_price = get_current_price(strategy_ticker)
                
                if current_price:
                    st.markdown(f"### 📊 Options Analysis for {strategy_ticker}")
                    st.markdown(f"**Current Price**: ${current_price:.2f}")
                    
                    st.success("✅ Options strategy framework ready")
                    st.info("📊 Advanced probability calculations can be implemented here")
                else:
                    st.error(f"Could not fetch current price for {strategy_ticker}")
        
        with tab7:
            st.subheader("📐 Advanced Put Spread Analysis")
            
            if PUT_SPREAD_AVAILABLE:
                st.success("✅ Put Spread Analysis module is available")
                st.info("📊 Put spread functionality ready for implementation")
            else:
                st.warning("Put spread analysis module not available. Install put_spread_analysis.py")
        
        with tab8:
            st.subheader("🦅 Iron Condor Trading Playbook")
            
            if IRON_CONDOR_AVAILABLE:
                st.success("✅ Iron Condor Analysis module is available") 
                st.info("📊 Iron Condor functionality ready for implementation")
            else:
                st.warning("Iron Condor analysis module not available. Install iron_condor_analysis.py")
    
    else:
        # No analysis run yet
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
                    padding: 2rem; border-radius: 16px; margin: 2rem 0; 
                    border-left: 6px solid var(--warning-color); box-shadow: var(--shadow-md);">
            <h2 style="color: var(--warning-color); margin: 0; font-weight: 700;">
                📊 Enhanced Stock Volatility Analyzer
            </h2>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                ℹ️ Click "Run Enhanced Analysis" in the sidebar to unlock all 8 tabs with comprehensive features.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show available features
        st.markdown("### 🚀 Available Features:")
        
        features = [
            "📊 **Summary Tab**: ATR analysis and market summaries",
            "📈 **Price Charts**: Interactive candlestick charts with enhanced ATR",
            "🔍 **Detailed Stats**: Comprehensive statistical analysis",
            "⚖️ **Comparison**: Multi-ticker comparative analysis", 
            "📉 **VIX Analysis**: Market condition assessment",
            "🎯 **Options Strategy**: 95% probability analysis framework",
            "📐 **Put Spread Analysis**: Advanced spread strategy analysis",
            "🦅 **Iron Condor Playbook**: Complete IC trading system"
        ]
        
        for feature in features:
            st.markdown(f"- {feature}")
        
        st.info("💡 **Instructions**: Select tickers and date range in the sidebar, then click 'Run Enhanced Analysis' to access all features.")

if __name__ == "__main__":
    main() 