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
    
    .metric-container {
        background: var(--background-secondary);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease-in-out;
    }
    
    .metric-container:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
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
    
    .trade-recommend { 
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem; 
        border-radius: 16px; 
        border-left: 6px solid var(--primary-color);
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
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
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--background-tertiary);
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
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--background-secondary);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: var(--text-secondary);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    /* Metric Styling */
    [data-testid="metric-container"] {
        background: var(--background-secondary);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
    }
    
    /* Code blocks */
    .stCodeBlock {
        font-family: 'JetBrains Mono', monospace;
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        border-radius: 8px;
        border-color: var(--border-color);
    }
    
    .stTextInput > div > div {
        border-radius: 8px;
        border-color: var(--border-color);
    }
    
    /* Data tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background: var(--gradient-primary);
        border-radius: 4px;
    }
    
    /* Custom success/warning/error styling */
    .success-highlight {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid var(--success-color);
        color: #065f46;
        font-weight: 500;
    }
    
    .warning-highlight {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid var(--warning-color);
        color: #92400e;
        font-weight: 500;
    }
    
    .error-highlight {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid var(--error-color);
        color: #991b1b;
        font-weight: 500;
    }
    
    /* Sidebar header styling */
    .css-1lcbmhc {
        background: var(--gradient-secondary);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Chart container styling */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
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

def calculate_probability_distribution(historical_data, current_price, timeframe='daily', lookback_days=14):
    """Calculate probability distribution for price movements"""
    if historical_data is None or len(historical_data) < lookback_days:
        return None
    
    # Get recent data for analysis
    recent_data = historical_data.tail(lookback_days).copy()
    
    # Calculate daily/weekly returns
    if timeframe == 'weekly':
        # For weekly, resample to weekly data first
        weekly_data = recent_data.resample('W').agg({'Close': 'last'})
        returns = weekly_data['Close'].pct_change().dropna()
    else:
        # Daily returns
        returns = recent_data['Close'].pct_change().dropna()
    
    if len(returns) < 3:
        return None
    
    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Calculate ATR-based volatility
    true_range = calculate_true_range(recent_data)
    if true_range is not None:
        atr = true_range.mean()
        atr_std = true_range.std()
    else:
        atr = recent_data['High'].subtract(recent_data['Low']).mean()
        atr_std = recent_data['High'].subtract(recent_data['Low']).std()
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'atr': atr,
        'atr_std': atr_std,
        'current_price': current_price,
        'sample_size': len(returns)
    }

def calculate_strike_probabilities(prob_dist, target_strikes, timeframe='daily'):
    """Calculate probability of hitting target strike prices"""
    if prob_dist is None:
        return None
    
    current_price = prob_dist['current_price']
    mean_return = prob_dist['mean_return']
    std_return = prob_dist['std_return']
    
    # Adjust for timeframe
    if timeframe == 'weekly':
        # Scale for 5 trading days
        time_factor = 5
    else:
        # Single day
        time_factor = 1
    
    # Adjust statistics for timeframe
    adj_mean = mean_return * time_factor
    adj_std = std_return * np.sqrt(time_factor)
    
    probabilities = {}
    
    for strike in target_strikes:
        # Calculate required return to hit strike
        required_return = (strike - current_price) / current_price
        
        # Calculate probability using normal distribution
        # Probability of price going BELOW strike (for PUT analysis)
        z_score = (required_return - adj_mean) / adj_std
        prob_below = stats.norm.cdf(z_score)
        
        probabilities[strike] = {
            'prob_below': prob_below,
            'prob_above': 1 - prob_below,
            'required_return': required_return,
            'z_score': z_score
        }
    
    return probabilities

def generate_strike_recommendations(current_price, prob_dist, target_prob=0.10, timeframe='daily', num_strikes=5):
    """Generate strike price recommendations with target probability"""
    if prob_dist is None:
        return None
    
    mean_return = prob_dist['mean_return']
    std_return = prob_dist['std_return']
    
    # Adjust for timeframe
    if timeframe == 'weekly':
        time_factor = 5
    else:
        time_factor = 1
    
    adj_mean = mean_return * time_factor
    adj_std = std_return * np.sqrt(time_factor)
    
    # Calculate z-score for target probability
    z_score = stats.norm.ppf(target_prob)
    
    # Calculate required return for target probability
    required_return = adj_mean + (z_score * adj_std)
    
    # Calculate target strike price
    target_strike = current_price * (1 + required_return)
    
    # Generate range of strikes around target
    strike_range = current_price * 0.02  # 2% range
    strikes = []
    
    for i in range(num_strikes):
        offset = (i - num_strikes//2) * (strike_range / num_strikes)
        strike = target_strike + offset
        strikes.append(round(strike, 2))
    
    # Calculate probabilities for all strikes
    strike_probs = calculate_strike_probabilities(prob_dist, strikes, timeframe)
    
    recommendations = []
    for strike in strikes:
        if strike in strike_probs:
            prob_info = strike_probs[strike]
            recommendations.append({
                'strike': strike,
                'distance_from_current': strike - current_price,
                'distance_pct': ((strike - current_price) / current_price) * 100,
                'prob_below': prob_info['prob_below'],
                'prob_above': prob_info['prob_above'],
                'safety_score': 1 - prob_info['prob_below']  # Higher is safer for PUT selling
            })
    
    # Sort by safety score (descending)
    recommendations.sort(key=lambda x: x['safety_score'], reverse=True)
    
    return recommendations

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

def create_vix_analysis_chart(vix_data):
    """Create VIX analysis chart with market condition zones"""
    if vix_data is None or vix_data.empty:
        return None
    
    fig = go.Figure()
    
    # VIX line
    fig.add_trace(go.Scatter(
        x=vix_data.index,
        y=vix_data['VIX_Close'],
        mode='lines+markers',
        name='VIX',
        line=dict(color='#8b5cf6', width=3),
        marker=dict(size=5, color='#8b5cf6')
    ))
    
    # Add condition zones
    fig.add_hrect(y0=0, y1=15, fillcolor="green", opacity=0.1, annotation_text="Calm Markets")
    fig.add_hrect(y0=15, y1=19, fillcolor="blue", opacity=0.1, annotation_text="Normal Markets")
    fig.add_hrect(y0=19, y1=25, fillcolor="yellow", opacity=0.1, annotation_text="Choppy Markets")
    fig.add_hrect(y0=25, y1=35, fillcolor="orange", opacity=0.1, annotation_text="High Volatility")
    fig.add_hrect(y0=35, y1=100, fillcolor="red", opacity=0.1, annotation_text="Extreme Volatility")
    
    fig.update_layout(
        title="VIX Analysis - Market Condition Zones",
        xaxis_title="Date",
        yaxis_title="VIX Level",
        height=400,
        showlegend=True
    )
    
    return fig

def create_probability_chart(recommendations, current_price, ticker):
    """Create probability visualization chart"""
    if not recommendations:
        return None
    
    strikes = [r['strike'] for r in recommendations]
    probs_below = [r['prob_below'] * 100 for r in recommendations]
    safety_scores = [r['safety_score'] * 100 for r in recommendations]
    
    fig = go.Figure()
    
    # Probability bars
    fig.add_trace(go.Bar(
        x=strikes,
        y=probs_below,
        name='Probability Below (%)',
        marker_color='rgba(239, 68, 68, 0.7)',
        opacity=0.8
    ))
    
    # Safety score line
    fig.add_trace(go.Scatter(
        x=strikes,
        y=safety_scores,
        mode='lines+markers',
        name='Safety Score (%)',
        line=dict(color='#10b981', width=4),
        marker=dict(size=10, color='#10b981'),
        yaxis='y2'
    ))
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", line_color="#6366f1", 
                  annotation_text=f"Current: ${current_price:.2f}")
    
    # Add 10% probability line
    fig.add_hline(y=10, line_dash="dash", line_color="#ef4444", 
                  annotation_text="10% Target")
    
    fig.update_layout(
        title=f'{ticker} - Strike Price Probability Analysis',
        xaxis_title='Strike Price ($)',
        yaxis_title='Probability Below (%)',
        yaxis2=dict(
            title='Safety Score (%)',
            overlaying='y',
            side='right'
        ),
        height=500,
        showlegend=True
    )
    
    return fig

def main():
    # Header with modern gradient design
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div class="main-header">🎯 Stock Volatility Analyzer</div>
        <p style="font-size: 1.25rem; color: var(--text-secondary); font-weight: 400; margin-top: -1rem;">
            Advanced Options Strategy with 95% Probability Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables to prevent scoping errors
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'vix_data' not in st.session_state:
        st.session_state.vix_data = None
    if 'include_vix' not in st.session_state:
        st.session_state.include_vix = False
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ['SPY', 'QQQ']
    
    # Sidebar for controls
    st.sidebar.header("Analysis Parameters")
    
    # Ticker selection
    default_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'GLD', 'NDX']
    selected_tickers = st.sidebar.multiselect(
        "Select Stock Tickers",
        options=default_tickers + ['NVDA', 'GOOGL', 'AMZN', 'META', 'NFLX', 'IWM', 'DIA'],
        default=['SPY', 'QQQ'],
        help="Choose up to 5 tickers for comparison"
    )
    
    if len(selected_tickers) > 5:
        st.sidebar.warning("Please select maximum 5 tickers for better performance")
        selected_tickers = selected_tickers[:5]
    
    # Date selection
    st.sidebar.subheader("Date Range Selection")
    
    min_date = date.today() - timedelta(days=365*2)
    max_date = date.today()
    default_start = date.today() - timedelta(days=90)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=start_date,
        max_value=max_date
    )
    
    # Validate minimum 90 days
    date_range = (end_date - start_date).days
    if date_range < 90:
        st.sidebar.markdown('<div class="error-highlight">Please select at least 90 days of data for meaningful analysis</div>', unsafe_allow_html=True)
        return
    
    st.sidebar.markdown(f'<div class="success-highlight">Selected range: {date_range} days</div>', unsafe_allow_html=True)
    
    # Time window selection
    st.sidebar.subheader("Analysis Timeframes")
    include_hourly = st.sidebar.checkbox("Include Hourly Analysis", value=True)
    include_daily = st.sidebar.checkbox("Include Daily Analysis", value=True)
    include_weekly = st.sidebar.checkbox("Include Weekly Analysis", value=True)
    include_vix = st.sidebar.checkbox("Include VIX Analysis", value=True)
    
    # Hour selection for hourly data
    if include_hourly:
        hour_range = st.sidebar.slider(
            "Trading Hours Range (for hourly analysis)",
            min_value=0,
            max_value=23,
            value=(9, 16),
            help="Select the range of hours to include in analysis (24-hour format)"
        )
    
    # Analysis button
    if st.sidebar.button("🚀 Run Enhanced Analysis", type="primary"):
        if not selected_tickers:
            st.error("Please select at least one ticker")
            return
        
        # Fetch VIX data first if requested
        vix_data = None
        if include_vix:
            with st.spinner("Fetching VIX data..."):
                vix_data = fetch_vix_data(start_date, end_date)
                if vix_data is not None:
                    st.sidebar.markdown('<div class="success-highlight">✅ VIX data loaded successfully</div>', unsafe_allow_html=True)
                else:
                    st.sidebar.markdown('<div class="warning-highlight">⚠️ VIX data unavailable</div>', unsafe_allow_html=True)
        
        # Main content area
        results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_operations = len(selected_tickers) * sum([include_hourly, include_daily, include_weekly])
        current_operation = 0
        
        for ticker in selected_tickers:
            status_text.text(f"Analyzing {ticker}...")
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
                current_operation += 1
                progress_bar.progress(current_operation / total_operations)
                
                # Fetch data
                data = fetch_stock_data(ticker, start_date, end_date, tf_interval)
                
                if data is not None:
                    # Filter hours for hourly data
                    if tf_name == 'hourly' and include_hourly:
                        data = data.between_time(f"{hour_range[0]:02d}:00", f"{hour_range[1]:02d}:00")
                    
                    # Calculate metrics
                    results[ticker][tf_name] = calculate_volatility_metrics(data, tf_name, vix_data)
                else:
                    results[ticker][tf_name] = None
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.results = results
        st.session_state.vix_data = vix_data
        st.session_state.include_vix = include_vix
        st.session_state.selected_tickers = selected_tickers
    
    # Check if we have results in session state
    results = getattr(st.session_state, 'results', {})
    vix_data = getattr(st.session_state, 'vix_data', None)
    include_vix = getattr(st.session_state, 'include_vix', False)
    session_tickers = getattr(st.session_state, 'selected_tickers', selected_tickers)
    
    # CREATE TABS - ALWAYS AVAILABLE
    if results and len(results) > 0:
        # Full analysis with all tabs
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    padding: 2rem; border-radius: 16px; margin: 2rem 0; 
                    border-left: 6px solid var(--primary-color); box-shadow: var(--shadow-md);">
            <h2 style="color: var(--primary-color); margin: 0; font-weight: 700;">
                📈 Enhanced Analysis Results
            </h2>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Analysis results from your last run. All tabs are now available!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Summary", 
            "📈 Price Charts", 
            "🔍 Detailed Stats", 
            "⚖️ Comparison", 
            "📉 VIX Analysis",
            "🎯 Options Strategy"
        ])
        
        with tab1:
            st.subheader("Enhanced Volatility Summary with ATR Explanation")
            
            # ATR Explanation
            with st.expander("📚 What is ATR (Average True Range)?"):
                st.markdown("""
                **ATR (Average True Range)** measures market volatility by calculating the average of true ranges over a specified period.
                
                **True Range** is the maximum of:
                - Current High - Current Low
                - |Current High - Previous Close|
                - |Current Low - Previous Close|
                
                **Why ATR is important:**
                - 📊 Provides normalized volatility measure
                - 🎯 Helps set stop-loss levels
                - 📈 Indicates market activity levels
                - 🔄 Adjusts for gaps and limit moves
                
                **ATR Interpretation:**
                - **Higher ATR** = More volatile, larger price swings
                - **Lower ATR** = Less volatile, smaller price movements
                - **Trending ATR** = Changing market character
                """)
            
            # Create summary table
            summary_data = []
            for ticker in session_tickers:
                row = {'Ticker': ticker}
                for tf in ['hourly', 'daily', 'weekly']:
                    if tf in results[ticker] and results[ticker][tf]:
                        atr_val = results[ticker][tf]['atr']
                        vol_val = results[ticker][tf]['volatility']
                        atr_window = results[ticker][tf]['atr_window']
                        
                        row[f'{tf.title()} ATR'] = f"${atr_val:.2f}" if atr_val > 0 else "Insufficient Data"
                        row[f'{tf.title()} Volatility'] = f"${vol_val:.2f}" if vol_val > 0 else "N/A"
                        row[f'{tf.title()} ATR Window'] = f"{atr_window} periods"
                    else:
                        row[f'{tf.title()} ATR'] = "No Data"
                        row[f'{tf.title()} Volatility'] = "No Data"
                        row[f'{tf.title()} ATR Window'] = "N/A"
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Interactive Price Charts with Enhanced ATR")
            
            # Chart selection
            chart_ticker = st.selectbox("Select ticker for detailed chart:", session_tickers)
            chart_timeframe = st.selectbox(
                "Select timeframe:",
                [tf for tf in ['hourly', 'daily', 'weekly'] if tf in results[chart_ticker] and results[chart_ticker][tf]]
            )
            
            if chart_timeframe in results[chart_ticker] and results[chart_ticker][chart_timeframe]:
                chart_data = results[chart_ticker][chart_timeframe]['data']
                show_vix_chart = include_vix and chart_timeframe == 'daily' and vix_data is not None
                
                fig = create_price_chart(chart_data, chart_ticker, chart_timeframe, show_vix_chart)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Chart explanation
                    st.info("""
                    **Chart Elements:**
                    - 🕯️ **Candlesticks**: Price action (Open, High, Low, Close)
                    - 📊 **Blue Bars**: Range (High - Low) for each period
                    - 🔴 **Red Line**: Enhanced ATR (True Range average)
                    - 🟣 **Purple Line**: VIX levels (daily charts only)
                    - **Horizontal Lines**: VIX condition thresholds
                    """)
        
        with tab3:
            st.subheader("🔍 Detailed Statistical Analysis")
            
            for ticker in session_tickers:
                st.write(f"### {ticker}")
                
                valid_timeframes = [tf for tf in ['hourly', 'daily', 'weekly'] if tf in results[ticker] and results[ticker][tf]]
                cols = st.columns(len(valid_timeframes))
                
                for col_idx, tf in enumerate(valid_timeframes):
                    with cols[col_idx]:
                        st.write(f"**{tf.title()} Range Statistics**")
                        stats = results[ticker][tf]['stats']
                        atr = results[ticker][tf]['atr']
                        atr_window = results[ticker][tf]['atr_window']
                        
                        st.write(f"- Count: {stats['count']:.0f}")
                        st.write(f"- Mean: ${stats['mean']:.2f}")
                        st.write(f"- Std: ${stats['std']:.2f}")
                        st.write(f"- Min: ${stats['min']:.2f}")
                        st.write(f"- 25%: ${stats['25%']:.2f}")
                        st.write(f"- 50%: ${stats['50%']:.2f}")
                        st.write(f"- 75%: ${stats['75%']:.2f}")
                        st.write(f"- Max: ${stats['max']:.2f}")
                        st.write(f"- **ATR ({atr_window}p): ${atr:.2f}**")
                        
                        # ATR quality indicator
                        if atr > 0:
                            st.success("✅ Valid ATR")
                        else:
                            st.error("❌ ATR calculation failed")
        
        with tab4:
            st.subheader("⚖️ Cross-Ticker Comparison")
            
            if len(session_tickers) > 1:
                # ATR comparison
                atr_fig = create_comparison_chart(results, 'atr')
                st.plotly_chart(atr_fig, use_container_width=True)
                
                # Volatility comparison
                vol_fig = create_comparison_chart(results, 'volatility')
                st.plotly_chart(vol_fig, use_container_width=True)
                
                # Correlation analysis
                st.subheader("📊 Range Correlation Analysis")
                
                # Get daily data for correlation
                correlation_data = {}
                for ticker in session_tickers:
                    if 'daily' in results[ticker] and results[ticker]['daily']:
                        correlation_data[ticker] = results[ticker]['daily']['data']['range']
                
                if len(correlation_data) > 1:
                    corr_df = pd.DataFrame(correlation_data).corr()
                    
                    # Create correlation heatmap
                    fig_corr = px.imshow(
                        corr_df,
                        text_auto=True,
                        aspect="auto",
                        title="Daily Range Correlation Matrix",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Select multiple tickers to see comparison charts")
        
        with tab5:
            st.subheader("📉 VIX Analysis & Market Conditions")
            
            if include_vix and vix_data is not None:
                # VIX chart
                vix_fig = create_vix_analysis_chart(vix_data)
                if vix_fig:
                    st.plotly_chart(vix_fig, use_container_width=True)
                
                # VIX statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 VIX Statistics")
                    vix_stats = vix_data['VIX_Close'].describe()
                    for stat, value in vix_stats.items():
                        st.write(f"**{stat.title()}**: {value:.2f}")
                
                with col2:
                    st.subheader("🎯 Trading Recommendations")
                    
                    # Current condition
                    current_vix = vix_data['VIX_Close'].iloc[-1]
                    condition, _, icon = get_vix_condition(current_vix)
                    
                    st.markdown(f"**Current Condition**: {icon} {condition}")
                    
                    # Recommendations based on VIX
                    if current_vix < 15:
                        st.success("🟢 **Recommended**: Normal position sizing, trend following strategies work well")
                    elif 15 <= current_vix < 19:
                        st.info("🔵 **Recommended**: Standard trading approach, good for most strategies")
                    elif 19 <= current_vix < 26:
                        st.warning("🟡 **Caution**: Reduce position sizes, avoid breakout trades")
                    elif 26 <= current_vix < 36:
                        st.error("🔴 **High Risk**: Consider staying out, if trading use very small positions")
                    else:
                        st.error("🚨 **Extreme Risk**: DO NOT TRADE - Wait for volatility to calm down")
                
                # VIX condition timeline
                st.subheader("📅 VIX Condition Timeline")
                condition_colors = {
                    'Calm Markets - Clean Trend': 'green',
                    'Normal Markets - Trendy': 'blue', 
                    'Choppy Market - Proceed with Caution': 'orange',
                    'High Volatility - Big Swings, Don\'t Trade': 'red',
                    'Extreme Volatility - Very Bad Day, DO NOT TRADE': 'darkred'
                }
                
                fig_timeline = go.Figure()
                for condition in condition_colors.keys():
                    condition_data = vix_data[vix_data['Market_Condition'] == condition]
                    if not condition_data.empty:
                        fig_timeline.add_trace(go.Scatter(
                            x=condition_data.index,
                            y=condition_data['VIX_Close'],
                            mode='markers',
                            name=condition,
                            marker=dict(color=condition_colors[condition], size=8)
                        ))
                
                fig_timeline.update_layout(
                    title="VIX Levels by Market Condition",
                    xaxis_title="Date",
                    yaxis_title="VIX Level",
                    height=400
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
            else:
                st.info("VIX analysis not available. Enable 'Include VIX Analysis' in the sidebar to see market condition analysis.")
        
        with tab6:
            st.subheader("🎯 Enhanced Options Trading Strategy")
            
            # Strategy configuration
            st.markdown("### 📅 Trade Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                strategy_ticker = st.selectbox(
                    "Select ticker for options strategy:",
                    session_tickers,
                    help="Choose the ticker you want to trade options on"
                )
                
                trade_date = st.date_input(
                    "Select trade date:",
                    value=date.today(),
                    min_value=date.today() - timedelta(days=7),
                    max_value=date.today() + timedelta(days=30),
                    help="Date when you plan to enter the trade"
                )
            
            with col2:
                strategy_timeframe = st.selectbox(
                    "Options expiry timeframe:",
                    ['daily', 'weekly'],
                    help="Daily = same day expiry, Weekly = end of week expiry"
                )
                
                target_probability = st.slider(
                    "Target probability threshold (%):",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Maximum acceptable probability of strike being hit"
                ) / 100
            
            # Enhanced 95% Probability Range Display
            if strategy_ticker in results and 'daily' in results[strategy_ticker] and results[strategy_ticker]['daily'] is not None:
                try:
                    current_price = get_current_price(strategy_ticker)
                    daily_data = results[strategy_ticker]['daily']['data']
                    atr = results[strategy_ticker]['daily']['atr']
                    
                    if current_price and atr > 0:
                        # Calculate 95% probability range using Z = 1.96
                        z_score_95 = 1.96
                        upper_bound = current_price + (z_score_95 * atr)
                        lower_bound = current_price - (z_score_95 * atr)
                        range_width = upper_bound - lower_bound
                        
                        st.markdown("### 📊 95% Probability Price Range")
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Lower Bound (95%)", f"${lower_bound:.2f}", f"-${current_price - lower_bound:.2f}")
                        with col3:
                            st.metric("Upper Bound (95%)", f"${upper_bound:.2f}", f"+${upper_bound - current_price:.2f}")
                        with col4:
                            st.metric("Range Width", f"${range_width:.2f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show which data source is being used
                        st.info(f"📊 Using **daily** data for {strategy_timeframe} options strategy analysis")
                        
                        # Visual representation
                        st.info(f"""
                        **95% Confidence Range**: There is a 95% probability that {strategy_ticker} will trade between 
                        **${lower_bound:.2f}** and **${upper_bound:.2f}** based on daily ATR of ${atr:.2f}.
                        
                        - **2.5% chance** price goes below ${lower_bound:.2f}
                        - **2.5% chance** price goes above ${upper_bound:.2f}
                        - **Range represents ±{z_score_95} standard deviations** from current price
                        - **Data source**: daily analysis
                        """)
                        
                except Exception as e:
                    st.warning(f"Could not calculate 95% range: {str(e)}")
            
            # Custom strike testing
            st.markdown("### 🎯 Custom Strike Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                custom_strikes_input = st.text_input(
                    "Enter custom strikes (comma-separated):",
                    placeholder="e.g., 580, 575, 570",
                    help="Enter specific strike prices you want to analyze"
                )
            
            with col2:
                num_recommendations = st.slider(
                    "Number of strike recommendations:",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="How many strike prices to recommend"
                )
            
            if st.button("🚀 Generate Enhanced Options Strategy", type="primary"):
                try:
                    st.info("🔄 Generating enhanced options strategy with 95% probability analysis...")
                    
                    # Get current price
                    with st.spinner("Fetching current price..."):
                        current_price = get_current_price(strategy_ticker)
                    
                    if current_price is None:
                        st.error(f"❌ Could not fetch current price for {strategy_ticker}")
                        return
                    
                    st.success(f"✅ Current price fetched: ${current_price:.2f}")
                    
                    # Enhanced analysis with 95% probability ranges
                    if strategy_ticker in results and 'daily' in results[strategy_ticker] and results[strategy_ticker]['daily'] is not None:
                        # Select appropriate data based on timeframe
                        if strategy_timeframe == 'weekly' and 'weekly' in results[strategy_ticker] and results[strategy_ticker]['weekly'] is not None:
                            analysis_data = results[strategy_ticker]['weekly']['data']
                            atr = results[strategy_ticker]['weekly']['atr']
                            data_source = "weekly"
                        else:
                            analysis_data = results[strategy_ticker]['daily']['data']
                            atr = results[strategy_ticker]['daily']['atr']
                            data_source = "daily"
                        
                        # Show which data source is being used
                        st.info(f"📊 Using **{data_source}** data for {strategy_timeframe} options strategy analysis")
                        
                        # Calculate 95% probability range
                        z_score_95 = 1.96
                        upper_bound_95 = current_price + (z_score_95 * atr)
                        lower_bound_95 = current_price - (z_score_95 * atr)
                        
                        # Calculate 90% probability range for comparison
                        z_score_90 = 1.645
                        upper_bound_90 = current_price + (z_score_90 * atr)
                        lower_bound_90 = current_price - (z_score_90 * atr)
                        
                        # Calculate 99% probability range for ultra-conservative
                        z_score_99 = 2.576
                        upper_bound_99 = current_price + (z_score_99 * atr)
                        lower_bound_99 = current_price - (z_score_99 * atr)
                        
                        st.markdown("### 📈 Enhanced Probability Analysis")
                        
                        # Display multiple confidence levels
                        conf_data = []
                        for conf_level, z_score, upper, lower in [
                            ("90%", z_score_90, upper_bound_90, lower_bound_90),
                            ("95%", z_score_95, upper_bound_95, lower_bound_95),
                            ("99%", z_score_99, upper_bound_99, lower_bound_99)
                        ]:
                            conf_data.append({
                                'Confidence Level': conf_level,
                                'Lower Bound': f"${lower:.2f}",
                                'Upper Bound': f"${upper:.2f}",
                                'Range Width': f"${upper - lower:.2f}",
                                'PUT Strike Zone': f"Below ${lower:.2f}",
                                'Risk Level': 'Conservative' if conf_level == '99%' else 'Moderate' if conf_level == '95%' else 'Aggressive'
                            })
                        
                        conf_df = pd.DataFrame(conf_data)
                        st.dataframe(conf_df, use_container_width=True)
                        
                        # VIX Assessment
                        trade_approved = True
                        if include_vix and vix_data is not None and not vix_data.empty:
                            try:
                                latest_vix = vix_data['VIX_Close'].iloc[-1]
                                can_trade, trade_reason = should_trade(latest_vix)
                                condition, color_class, icon = get_vix_condition(latest_vix)
                                
                                st.markdown("### 🌡️ Market Condition Assessment")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Current VIX", f"{latest_vix:.2f}")
                                    st.markdown(f'<div class="{color_class}">{icon} {condition}</div>', unsafe_allow_html=True)
                                with col2:
                                    if can_trade:
                                        st.success(f"✅ **TRADE APPROVED**: {trade_reason}")
                                    else:
                                        st.error(f"❌ **TRADE NOT RECOMMENDED**: {trade_reason}")
                                        trade_approved = False
                            except Exception as e:
                                st.warning(f"⚠️ VIX assessment error: {str(e)}")
                        
                        # Enhanced Strike Recommendations based on 95% probability
                        st.markdown("### 🎯 Enhanced PUT Strike Recommendations")
                        
                        # Calculate probability distribution for more accurate analysis
                        lookback_days = 10 if strategy_timeframe == 'weekly' else 14
                        prob_dist = calculate_probability_distribution(
                            analysis_data, current_price, strategy_timeframe, lookback_days
                        )
                        
                        if prob_dist:
                            # Generate recommendations using both ATR-based and probability-based methods
                            recommendations = generate_strike_recommendations(
                                current_price, prob_dist, target_probability, strategy_timeframe, num_recommendations
                            )
                            
                            if recommendations:
                                # Enhanced recommendations table with 95% probability zones
                                enhanced_rec_data = []
                                for rec in recommendations:
                                    # Determine which probability zone the strike falls into
                                    strike_price = rec['strike']
                                    if strike_price >= lower_bound_99:
                                        prob_zone = "Ultra Safe (>99%)"
                                        zone_color = "🟢"
                                    elif strike_price >= lower_bound_95:
                                        prob_zone = "Very Safe (95-99%)"
                                        zone_color = "🟢"
                                    elif strike_price >= lower_bound_90:
                                        prob_zone = "Safe (90-95%)"
                                        zone_color = "🟡"
                                    else:
                                        prob_zone = "Risky (<90%)"
                                        zone_color = "🔴"
                                    
                                    safety_color = "🟢" if rec['safety_score'] > 0.95 else "🟡" if rec['safety_score'] > 0.90 else "🔴"
                                    
                                    enhanced_rec_data.append({
                                        'Strike': f"${rec['strike']:.2f}",
                                        'Distance': f"${rec['distance_from_current']:.2f}",
                                        'Distance %': f"{rec['distance_pct']:.1f}%",
                                        'Prob Below': f"{rec['prob_below']:.1%}",
                                        'Safety Score': f"{safety_color} {rec['safety_score']:.1%}",
                                        'Probability Zone': f"{zone_color} {prob_zone}",
                                        'Recommendation': 'RECOMMENDED' if rec['prob_below'] <= target_probability else 'RISKY'
                                    })
                                
                                enhanced_rec_df = pd.DataFrame(enhanced_rec_data)
                                st.dataframe(enhanced_rec_df, use_container_width=True)
                                
                                # Best recommendation with enhanced analysis
                                best_rec = recommendations[0]
                                st.markdown('<div class="strike-recommend">', unsafe_allow_html=True)
                                st.markdown(f"""
                                ### 🏆 ENHANCED RECOMMENDATION: ${best_rec['strike']:.2f} PUT
                                
                                **Statistical Analysis:**
                                - **Current Price**: ${current_price:.2f}
                                - **Strike Distance**: ${best_rec['distance_from_current']:.2f} ({best_rec['distance_pct']:.1f}%)
                                - **Probability of Hit**: {best_rec['prob_below']:.1%}
                                - **Safety Score**: {best_rec['safety_score']:.1%}
                                
                                **95% Probability Analysis:**
                                - **95% Lower Bound**: ${lower_bound_95:.2f}
                                - **Strike vs 95% Bound**: {"✅ SAFE" if best_rec['strike'] >= lower_bound_95 else "⚠️ RISKY"}
                                - **ATR-Based Range**: ±${z_score_95 * atr:.2f}
                                
                                **Risk Assessment:**
                                - **Timeframe**: {strategy_timeframe.title()}
                                - **Market Condition**: {condition if 'condition' in locals() else 'VIX not available'}
                                - **Overall Risk**: {"LOW" if best_rec['strike'] >= lower_bound_95 and trade_approved else "MEDIUM" if best_rec['strike'] >= lower_bound_90 else "HIGH"}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Probability visualization
                                try:
                                    prob_fig = create_probability_chart(recommendations, current_price, strategy_ticker)
                                    if prob_fig:
                                        st.plotly_chart(prob_fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Chart generation failed: {str(e)}")
                        
                        # Custom strikes analysis with 95% probability assessment
                        if custom_strikes_input:
                            try:
                                custom_strikes = [float(x.strip()) for x in custom_strikes_input.split(',')]
                                st.markdown("### 🔍 Enhanced Custom Strike Analysis")
                                
                                custom_enhanced_data = []
                                for strike in custom_strikes:
                                    distance = current_price - strike
                                    distance_pct = (distance / current_price) * 100
                                    
                                    # Assess against 95% probability zones
                                    if strike >= lower_bound_99:
                                        prob_assessment = "🟢 Ultra Safe (>99%)"
                                        risk_level = "VERY LOW"
                                    elif strike >= lower_bound_95:
                                        prob_assessment = "🟢 Very Safe (95-99%)"
                                        risk_level = "LOW"
                                    elif strike >= lower_bound_90:
                                        prob_assessment = "🟡 Safe (90-95%)"
                                        risk_level = "MEDIUM"
                                    else:
                                        prob_assessment = "🔴 Risky (<90%)"
                                        risk_level = "HIGH"
                                    
                                    custom_enhanced_data.append({
                                        'Strike': f"${strike:.2f}",
                                        'Distance': f"${distance:.2f}",
                                        'Distance %': f"{distance_pct:.1f}%",
                                        'vs 95% Bound': f"${strike - lower_bound_95:.2f}",
                                        'Probability Zone': prob_assessment,
                                        'Risk Level': risk_level
                                    })
                                
                                custom_enhanced_df = pd.DataFrame(custom_enhanced_data)
                                st.dataframe(custom_enhanced_df, use_container_width=True)
                                
                            except ValueError:
                                st.error("❌ Invalid strike format. Please use comma-separated numbers")
                        
                        st.success("✅ Enhanced options strategy analysis complete!")
                        
                    else:
                        if strategy_timeframe == 'weekly':
                            st.error(f"❌ No weekly data available for {strategy_ticker}")
                            st.error("Please ensure weekly analysis was included in the main analysis")
                        else:
                            st.error(f"❌ No daily data available for {strategy_ticker}")
                            st.error("Please ensure daily analysis was included in the main analysis")
                
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        # No analysis run yet - show standalone Options Strategy
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
                    padding: 2rem; border-radius: 16px; margin: 2rem 0; 
                    border-left: 6px solid var(--warning-color); box-shadow: var(--shadow-md);">
            <h2 style="color: var(--warning-color); margin: 0; font-weight: 700;">
                📊 Enhanced Stock Volatility Analyzer
            </h2>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                ℹ️ Run the Enhanced Analysis first to see all features, or use Basic Options Strategy below.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create standalone Options Strategy container
        st.subheader("🎯 Basic Options Trading Strategy")
        
        # Strategy configuration
        st.markdown("### 📅 Trade Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_ticker = st.selectbox(
                "Select ticker for options strategy:",
                selected_tickers if selected_tickers else ['SPY', 'QQQ', 'GLD', 'NDX'],
                help="Choose the ticker you want to trade options on",
                key="standalone_ticker"
            )
            
            trade_date = st.date_input(
                "Select trade date:",
                value=date.today(),
                min_value=date.today() - timedelta(days=7),
                max_value=date.today() + timedelta(days=30),
                help="Date when you plan to enter the trade",
                key="standalone_date"
            )
        
        with col2:
            strategy_timeframe = st.selectbox(
                "Options expiry timeframe:",
                ['daily', 'weekly'],
                help="Daily = same day expiry, Weekly = end of week expiry",
                key="standalone_timeframe"
            )
            
            target_probability = st.slider(
                "Target probability threshold (%):",
                min_value=1,
                max_value=20,
                value=10,
                help="Maximum acceptable probability of strike being hit",
                key="standalone_probability"
            ) / 100
        
        # Custom strike testing
        st.markdown("### 🎯 Custom Strike Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            custom_strikes_input = st.text_input(
                "Enter custom strikes (comma-separated):",
                placeholder="e.g., 580, 575, 570",
                help="Enter specific strike prices you want to analyze",
                key="standalone_strikes"
            )
        
        with col2:
            num_recommendations = st.slider(
                "Number of strike recommendations:",
                min_value=3,
                max_value=10,
                value=5,
                help="How many strike prices to recommend",
                key="standalone_num_recs"
            )
        
        if st.button("🚀 Generate Basic Options Strategy", type="primary", key="standalone_button"):
            try:
                st.info("🔄 Generating basic options strategy...")
                st.warning("⚠️ This is basic analysis. Run Enhanced Analysis for 95% probability calculations!")
                
                # Get current price
                with st.spinner("Fetching current price..."):
                    current_price = get_current_price(strategy_ticker)
                
                if current_price is None:
                    st.error(f"❌ Could not fetch current price for {strategy_ticker}")
                    st.error("Please try again or select a different ticker")
                    return
                
                st.success(f"✅ Current price fetched: ${current_price:.2f}")
                
                st.markdown(f"### 📊 Basic Analysis for {strategy_ticker}")
                st.info(f"**Current Price**: ${current_price:.2f} | **Trade Date**: {trade_date} | **Timeframe**: {strategy_timeframe}")
                
                # Generate basic PUT strike recommendations
                st.markdown("### 🎯 Basic PUT Strategy")
                
                # Calculate distance-based strikes (2%, 4%, 6%, 8%, 10% below current price)
                strikes_below = []
                for i in range(num_recommendations):
                    pct_below = (i + 1) * 2
                    strike = current_price * (1 - pct_below/100)
                    strikes_below.append(round(strike, 2))
                
                # Create recommendations table
                st.markdown("### 📋 Suggested PUT Strike Prices")
                strike_data = []
                for i, strike in enumerate(strikes_below):
                    distance = current_price - strike
                    distance_pct = (distance / current_price) * 100
                    
                    strike_data.append({
                        'Rank': i + 1,
                        'Strike Price': f"${strike:.2f}",
                        'Distance': f"${distance:.2f}",
                        'Distance %': f"{distance_pct:.1f}%",
                        'Risk Assessment': 'Conservative' if distance_pct > 5 else 'Aggressive'
                    })
                
                strike_df = pd.DataFrame(strike_data)
                st.dataframe(strike_df, use_container_width=True)
                
                # Display best recommendation
                best_strike = strikes_below[0]
                st.markdown('<div class="strike-recommend">', unsafe_allow_html=True)
                st.markdown(f"""
                ### 🏆 BASIC RECOMMENDATION: ${best_strike:.2f} PUT
                - **Current Price**: ${current_price:.2f}
                - **Strike Distance**: ${current_price - best_strike:.2f} ({((current_price - best_strike)/current_price)*100:.1f}%)
                - **Strategy**: Sell PUT at ${best_strike:.2f}
                - **Timeframe**: {strategy_timeframe.title()}
                
                ⚠️ **Note**: This is a basic recommendation. For 95% probability analysis and VIX conditions, run Enhanced Analysis first.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Process custom strikes if provided
                if custom_strikes_input:
                    try:
                        custom_strikes = [float(x.strip()) for x in custom_strikes_input.split(',')]
                        st.info(f"Analyzing custom strikes: {custom_strikes}")
                        
                        st.markdown("### 🔍 Custom Strike Analysis")
                        
                        custom_data = []
                        for strike in custom_strikes:
                            distance = current_price - strike
                            distance_pct = (distance / current_price) * 100
                            
                            # Basic risk assessment based on distance
                            if distance_pct > 5:
                                risk_level = 'LOW'
                            elif distance_pct > 2:
                                risk_level = 'MEDIUM'
                            else:
                                risk_level = 'HIGH'
                            
                            custom_data.append({
                                'Strike': f"${strike:.2f}",
                                'Distance': f"${distance:.2f}",
                                'Distance %': f"{distance_pct:.1f}%",
                                'Risk Level': risk_level
                            })
                        
                        custom_df = pd.DataFrame(custom_data)
                        st.dataframe(custom_df, use_container_width=True)
                    
                    except ValueError:
                        st.error("❌ Invalid strike format. Please use comma-separated numbers")
                    except Exception as e:
                        st.error(f"❌ Custom strikes analysis failed: {str(e)}")
                
                st.success("✅ Basic options strategy analysis complete!")
                st.info("💡 Run the Enhanced Analysis first for 95% probability calculations, VIX analysis, and all advanced features!")
                
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 