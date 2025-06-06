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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Volatility Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .vix-calm { background-color: #d4edda; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-normal { background-color: #cce5ff; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-choppy { background-color: #fff3cd; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-volatile { background-color: #f8d7da; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-extreme { background-color: #f5c6cb; padding: 0.5rem; border-radius: 0.25rem; }
    .trade-recommend { background-color: #e7f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; }
    .strike-recommend { background-color: #f0fff4; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #28a745; }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
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

@st.cache_data(ttl=300)  # Cache for 5 minutes
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
            'range': 'max',  # Use max range within the day
            'true_range': 'max'  # Use max true range within the day
        })
        resampled['range'] = resampled['High'] - resampled['Low']
        # Recalculate true range for daily data
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
        # Recalculate true range for weekly data
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
    
    # Calculate ATR (Average True Range) - more accurate
    atr_window = min(14, len(resampled))
    if atr_window > 0 and not resampled['true_range'].isna().all():
        atr = resampled['true_range'].rolling(window=atr_window).mean().iloc[-1]
        if pd.isna(atr) and len(resampled) >= 1:
            atr = resampled['true_range'].mean()  # Fallback to simple mean
    else:
        atr = 0
    
    # Calculate additional metrics
    volatility = resampled['range'].std()
    cv = volatility / range_stats['mean'] if range_stats['mean'] > 0 else 0
    
    # Add VIX data if available for daily analysis
    if timeframe == 'daily' and vix_data is not None:
        # Align VIX data with stock data
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
            marker_color='lightblue',
            opacity=0.7
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
                line=dict(color='red', width=2)
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
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
        
        # Add VIX level zones
        fig.add_hline(y=15, line_dash="dash", line_color="green", opacity=0.5, row=3)
        fig.add_hline(y=19, line_dash="dash", line_color="blue", opacity=0.5, row=3)
        fig.add_hline(y=25, line_dash="dash", line_color="orange", opacity=0.5, row=3)
        fig.add_hline(y=35, line_dash="dash", line_color="red", opacity=0.5, row=3)
    
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
        line=dict(color='purple', width=3),
        marker=dict(size=5)
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
        marker_color='lightcoral',
        opacity=0.7
    ))
    
    # Safety score line
    fig.add_trace(go.Scatter(
        x=strikes,
        y=safety_scores,
        mode='lines+markers',
        name='Safety Score (%)',
        line=dict(color='green', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", line_color="blue", 
                  annotation_text=f"Current: ${current_price:.2f}")
    
    # Add 10% probability line
    fig.add_hline(y=10, line_dash="dash", line_color="red", 
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
    # Header
    st.markdown('<div class="main-header">📊 Enhanced Stock Volatility Analyzer with Options Strategy</div>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("Analysis Parameters")
    
    # Ticker selection
    default_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
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
    
    min_date = date.today() - timedelta(days=365*2)  # 2 years ago
    max_date = date.today()
    
    # Default to 90 days
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
        st.sidebar.error("Please select at least 90 days of data for meaningful analysis")
        return
    
    st.sidebar.success(f"Selected range: {date_range} days")
    
    # Time window selection
    st.sidebar.subheader("Analysis Timeframes")
    include_hourly = st.sidebar.checkbox("Include Hourly Analysis", value=True)
    include_daily = st.sidebar.checkbox("Include Daily Analysis", value=True)
    include_weekly = st.sidebar.checkbox("Include Weekly Analysis", value=True)
    include_vix = st.sidebar.checkbox("Include VIX Analysis", value=True, help="Add VIX market condition analysis")
    
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
                    st.sidebar.success("✅ VIX data loaded successfully")
                else:
                    st.sidebar.warning("⚠️ VIX data unavailable")
        
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
        
        # Store results in session state so they persist
        st.session_state.results = results
        st.session_state.vix_data = vix_data
        st.session_state.include_vix = include_vix
        st.session_state.selected_tickers = selected_tickers
    
    # Check if we have results in session state
    results = getattr(st.session_state, 'results', {})
    vix_data = getattr(st.session_state, 'vix_data', None)
    include_vix = getattr(st.session_state, 'include_vix', False)
    selected_tickers = getattr(st.session_state, 'selected_tickers', selected_tickers)
    
    # Main content area - always show this
    if results:
        # Display results
        st.header("📈 Enhanced Analysis Results")
        
        # Create tabs for different views
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
            for ticker in selected_tickers:
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
            chart_ticker = st.selectbox("Select ticker for detailed chart:", selected_tickers)
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
            
            for ticker in selected_tickers:
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
            
            if len(selected_tickers) > 1:
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
                for ticker in selected_tickers:
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
    else:
        # Show just the Options Strategy when no analysis has been run
        st.header("📊 Enhanced Stock Volatility Analyzer with Options Strategy")
        st.info("ℹ️ Run the Enhanced Analysis first to see all features, or use Options Strategy below.")
        
        # Create just the Options Strategy container
        tab6 = st.container()
    
    # Always show Options Strategy (either in tab6 or container)
    with tab6:
        st.subheader("🎯 Options Trading Strategy Recommendations")
        
        # Strategy configuration
        st.markdown("### 📅 Trade Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_ticker = st.selectbox(
                "Select ticker for options strategy:",
                selected_tickers,
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
        
        if st.button("🚀 Generate Options Strategy", type="primary"):
            # IMMEDIATE DEBUG - Show we entered the button handler
            st.write("🔧 DEBUG: Button clicked! Starting options strategy generation...")
            st.write(f"🔧 DEBUG: Button pressed at {datetime.now()}")
            
            try:
                st.write("🔧 DEBUG: Entered try block successfully")
                st.info("🔄 Generating options strategy...")
                
                # Debug: Check basic variables
                st.write(f"🔧 DEBUG: strategy_ticker = {strategy_ticker}")
                st.write(f"🔧 DEBUG: strategy_timeframe = {strategy_timeframe}")
                st.write(f"🔧 DEBUG: target_probability = {target_probability}")
                st.write(f"🔧 DEBUG: trade_date = {trade_date}")
                
                # Debug: Check if results exist
                st.write(f"🔧 DEBUG: 'results' variable exists: {'results' in locals()}")
                if 'results' in locals():
                    st.write(f"🔧 DEBUG: results.keys() = {list(results.keys())}")
                    st.write(f"🔧 DEBUG: len(results) = {len(results)}")
                else:
                    st.error("🔧 DEBUG: ERROR - 'results' variable not found!")
                    st.error("🔧 This means you need to run the main analysis first!")
                    st.stop()
                
                # Debug: Check if we have the required data
                st.write(f"🔧 DEBUG: Selected ticker: {strategy_ticker}")
                st.write(f"🔧 DEBUG: Available results keys: {list(results.keys())}")
                
                st.write("🔧 DEBUG: About to fetch current price...")
                
                # Get current price
                with st.spinner("Fetching current price..."):
                    st.write("🔧 DEBUG: Inside spinner for current price...")
                    current_price = get_current_price(strategy_ticker)
                    st.write(f"🔧 DEBUG: get_current_price returned: {current_price}")
                
                if current_price is None:
                    st.error(f"❌ Could not fetch current price for {strategy_ticker}")
                    st.error("Please try again or select a different ticker")
                    st.warning("⚠️ Try selecting a different ticker or check your internet connection")
                    st.write("🔧 DEBUG: Exiting due to current_price = None")
                    return
                
                st.success(f"✅ Current price fetched: ${current_price:.2f}")
                st.write(f"🔧 DEBUG: Successfully got current price: ${current_price:.2f}")
                
                st.markdown(f"### 📊 Analysis for {strategy_ticker}")
                st.info(f"**Current Price**: ${current_price:.2f} | **Trade Date**: {trade_date} | **Timeframe**: {strategy_timeframe}")
                
                st.write("🔧 DEBUG: About to start VIX assessment...")
                
                # VIX Assessment
                trade_approved = True
                st.write(f"🔧 DEBUG: include_vix = {'include_vix' in locals()}")
                if 'include_vix' in locals():
                    st.write(f"🔧 DEBUG: include_vix value = {include_vix}")
                st.write(f"🔧 DEBUG: vix_data exists = {'vix_data' in locals()}")
                if 'vix_data' in locals():
                    st.write(f"🔧 DEBUG: vix_data is not None = {vix_data is not None}")
                    if vix_data is not None:
                        st.write(f"�� DEBUG: vix_data is not empty = {not vix_data.empty}")
                
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
                                st.warning("You can still continue with the analysis, but exercise extra caution.")
                                trade_approved = False
                    except Exception as e:
                        st.warning(f"⚠️ VIX assessment error: {str(e)}")
                        trade_approved = True
                else:
                    st.warning("⚠️ VIX data not available - proceeding with analysis")
                    trade_approved = True
                
                # Continue with analysis regardless of VIX (but warn user)
                if not trade_approved:
                    st.error("⚠️ **HIGH RISK CONDITIONS DETECTED** - Use extreme caution if proceeding")
                
                # Get historical data for probability analysis
                st.write(f"Debug: Checking if {strategy_ticker} in results: {strategy_ticker in results}")
                if strategy_ticker in results:
                    st.write(f"Debug: Available timeframes for {strategy_ticker}: {list(results[strategy_ticker].keys())}")
                
                if strategy_ticker in results and 'daily' in results[strategy_ticker] and results[strategy_ticker]['daily'] is not None:
                    try:
                        daily_data = results[strategy_ticker]['daily']['data']
                        st.success(f"✅ Found {len(daily_data)} days of historical data")
                        
                        # Calculate probability distribution
                        lookback_days = 10 if strategy_timeframe == 'weekly' else 14
                        
                        with st.spinner("Calculating probability distribution..."):
                            prob_dist = calculate_probability_distribution(
                                daily_data, current_price, strategy_timeframe, lookback_days
                            )
                        
                        if prob_dist is None:
                            st.error("❌ Insufficient data for probability analysis")
                            st.error(f"Need at least {lookback_days} days of data for {strategy_timeframe} analysis")
                            
                            # Show available data info instead of stopping
                            if strategy_ticker in results:
                                available_timeframes = [tf for tf in results[strategy_ticker].keys() if results[strategy_ticker][tf] is not None]
                                st.info(f"Available timeframes for {strategy_ticker}: {available_timeframes}")
                                
                                # Try to suggest alternatives
                                if 'daily' in available_timeframes:
                                    daily_data_len = len(results[strategy_ticker]['daily']['data'])
                                    st.info(f"You have {daily_data_len} days of daily data. Try reducing the lookback period or selecting a different timeframe.")
                                
                            st.warning("⚠️ Cannot generate options strategy with current data. Please try:")
                            st.markdown("""
                            - Select a longer date range (at least 30 days)
                            - Ensure 'Include Daily Analysis' is checked
                            - Try a different ticker with more data
                            """)
                            return  # Skip to the end instead of stopping the app
                        
                        st.success(f"✅ Probability distribution calculated using {prob_dist['sample_size']} periods")
                        
                        # Display ATR and volatility info
                        st.markdown("### 📈 Volatility Analysis")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("ATR", f"${prob_dist['atr']:.2f}")
                        with col2:
                            st.metric("ATR Std Dev", f"${prob_dist['atr_std']:.2f}")
                        with col3:
                            st.metric("Return Volatility", f"{prob_dist['std_return']:.2%}")
                        with col4:
                            st.metric("Sample Size", f"{prob_dist['sample_size']} periods")
                        
                        # Generate strike recommendations
                        with st.spinner("Generating strike recommendations..."):
                            recommendations = generate_strike_recommendations(
                                current_price, prob_dist, target_probability, strategy_timeframe, num_recommendations
                            )
                        
                        if recommendations and len(recommendations) > 0:
                            st.success(f"✅ Generated {len(recommendations)} strike recommendations")
                            
                            st.markdown("### 🎯 Strike Price Recommendations")
                            
                            # Create recommendations table
                            rec_data = []
                            for rec in recommendations:
                                safety_color = "🟢" if rec['safety_score'] > 0.95 else "🟡" if rec['safety_score'] > 0.90 else "🔴"
                                rec_data.append({
                                    'Strike': f"${rec['strike']:.2f}",
                                    'Distance': f"${rec['distance_from_current']:.2f}",
                                    'Distance %': f"{rec['distance_pct']:.1f}%",
                                    'Prob Below': f"{rec['prob_below']:.1%}",
                                    'Safety Score': f"{safety_color} {rec['safety_score']:.1%}",
                                    'Recommendation': 'RECOMMENDED' if rec['prob_below'] <= target_probability else 'RISKY'
                                })
                            
                            rec_df = pd.DataFrame(rec_data)
                            st.dataframe(rec_df, use_container_width=True)
                            
                            # Best recommendation
                            best_rec = recommendations[0]
                            st.markdown('<div class="strike-recommend">', unsafe_allow_html=True)
                            st.markdown(f"""
                            ### 🏆 BEST RECOMMENDATION: ${best_rec['strike']:.2f}
                            - **Distance from current**: ${best_rec['distance_from_current']:.2f} ({best_rec['distance_pct']:.1f}%)
                            - **Probability of being hit**: {best_rec['prob_below']:.1%}
                            - **Safety score**: {best_rec['safety_score']:.1%}
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Probability visualization
                            try:
                                prob_fig = create_probability_chart(recommendations, current_price, strategy_ticker)
                                if prob_fig:
                                    st.plotly_chart(prob_fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Chart generation failed: {str(e)}")
                            
                            # Custom strikes analysis
                            if custom_strikes_input:
                                try:
                                    custom_strikes = [float(x.strip()) for x in custom_strikes_input.split(',')]
                                    st.info(f"Analyzing custom strikes: {custom_strikes}")
                                    
                                    custom_probs = calculate_strike_probabilities(prob_dist, custom_strikes, strategy_timeframe)
                                    
                                    if custom_probs:
                                        st.markdown("### 🔍 Custom Strike Analysis")
                                        
                                        custom_data = []
                                        for strike, prob_info in custom_probs.items():
                                            distance = strike - current_price
                                            distance_pct = (distance / current_price) * 100
                                            
                                            custom_data.append({
                                                'Strike': f"${strike:.2f}",
                                                'Distance': f"${distance:.2f}",
                                                'Distance %': f"{distance_pct:.1f}%",
                                                'Prob Below': f"{prob_info['prob_below']:.1%}",
                                                'Prob Above': f"{prob_info['prob_above']:.1%}",
                                                'Risk Assessment': 'LOW' if prob_info['prob_below'] <= target_probability else 'HIGH'
                                            })
                                        
                                        custom_df = pd.DataFrame(custom_data)
                                        st.dataframe(custom_df, use_container_width=True)
                                
                                except ValueError:
                                    st.error("❌ Invalid strike format. Please use comma-separated numbers (e.g., 580, 575, 570)")
                                except Exception as e:
                                    st.error(f"❌ Custom strikes analysis failed: {str(e)}")
                            
                            # Strategy summary
                            st.markdown("### 📋 PUT Spread Strategy Summary")
                            
                            try:
                                protection_strike = best_rec['strike'] - (prob_dist['atr'] * 0.5)
                                
                                st.markdown('<div class="trade-recommend">', unsafe_allow_html=True)
                                st.markdown(f"""
                                **Recommended PUT Spread Strategy for {strategy_ticker}:**
                                
                                🎯 **Short PUT**: ${best_rec['strike']:.2f} (Collect premium)
                                🛡️ **Long PUT**: ${protection_strike:.2f} (Protection)
                                
                                **Risk Assessment:**
                                - Probability of short strike being hit: {best_rec['prob_below']:.1%}
                                - Expected max profit: Premium collected
                                - Expected max loss: Strike spread - Premium
                                - Break-even: Short strike - Premium collected
                                
                                **Market Conditions:**
                                - Current Price: ${current_price:.2f}
                                - ATR: ${prob_dist['atr']:.2f}
                                - Timeframe: {strategy_timeframe.title()}
                                {"- VIX Level: " + f"{latest_vix:.2f} ({condition})" if 'latest_vix' in locals() else "- VIX: Not available"}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"❌ Strategy summary generation failed: {str(e)}")
                            
                        else:
                            st.error("❌ No valid strike recommendations could be generated")
                            st.error("This might be due to insufficient data or extreme market conditions")
                        
                    except Exception as e:
                        st.write("🔧 DEBUG: CAUGHT EXCEPTION!")
                        st.write(f"🔧 DEBUG: Exception type: {type(e).__name__}")
                        st.write(f"🔧 DEBUG: Exception message: {str(e)}")
                        st.error(f"❌ Unexpected error: {str(e)}")
                        st.error("Please try again or contact support")
                        # Show the full traceback for debugging
                        import traceback
                        st.write("🔧 DEBUG: Full traceback:")
                        st.code(traceback.format_exc())
                        
                else:
                    st.write("🔧 DEBUG: No daily data available branch")
                    st.error(f"❌ No daily data available for {strategy_ticker}")
                    st.error("Please ensure you've run the main analysis first and selected 'Include Daily Analysis'")
                    
                    # Show what data is available
                    if strategy_ticker in results:
                        available_timeframes = [tf for tf in results[strategy_ticker].keys() if results[strategy_ticker][tf] is not None]
                        st.info(f"Available timeframes: {available_timeframes}")
                    else:
                        st.info("No data found for this ticker. Please run the main analysis first.")
            
            except Exception as e:
                st.write("🔧 DEBUG: CAUGHT EXCEPTION!")
                st.write(f"🔧 DEBUG: Exception type: {type(e).__name__}")
                st.write(f"🔧 DEBUG: Exception message: {str(e)}")
                st.error(f"❌ Unexpected error: {str(e)}")
                st.error("Please try again or contact support")
                # Show the full traceback for debugging
                import traceback
                st.write("🔧 DEBUG: Full traceback:")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 