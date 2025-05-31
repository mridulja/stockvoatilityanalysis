import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
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

def main():
    # Header
    st.markdown('<div class="main-header">📊 Enhanced Stock Volatility Analyzer with VIX</div>', unsafe_allow_html=True)
    
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
        
        # Display results
        st.header("📈 Enhanced Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "📈 Price Charts", "🔍 Detailed Stats", "⚖️ Comparison", "📉 VIX Analysis"])
        
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
            
            # VIX Market Conditions Summary
            if include_vix and vix_data is not None:
                st.subheader("🌡️ Current Market Conditions (VIX-based)")
                
                latest_vix = vix_data['VIX_Close'].iloc[-1] if not vix_data.empty else None
                if latest_vix is not None:
                    condition, color_class, icon = get_vix_condition(latest_vix)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.metric("Current VIX", f"{latest_vix:.2f}")
                    with col2:
                        st.markdown(f'<div class="{color_class}">{icon} {condition}</div>', unsafe_allow_html=True)
                    with col3:
                        # Calculate VIX trend
                        if len(vix_data) >= 5:
                            vix_trend = vix_data['VIX_Close'].iloc[-1] - vix_data['VIX_Close'].iloc[-5]
                            trend_icon = "📈" if vix_trend > 0 else "📉" if vix_trend < 0 else "➡️"
                            st.metric("5-Day VIX Trend", f"{vix_trend:+.2f}", delta=f"{trend_icon}")
                
                # VIX condition distribution
                if not vix_data.empty:
                    st.subheader("📊 VIX Condition Distribution (Selected Period)")
                    condition_counts = vix_data['Market_Condition'].value_counts()
                    
                    fig_pie = px.pie(
                        values=condition_counts.values,
                        names=condition_counts.index,
                        title="Market Condition Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Key insights
            st.subheader("🔍 Key Insights")
            for ticker in selected_tickers:
                with st.expander(f"{ticker} Enhanced Analysis"):
                    for tf in ['hourly', 'daily', 'weekly']:
                        if tf in results[ticker] and results[ticker][tf]:
                            metrics = results[ticker][tf]
                            st.write(f"**{tf.title()} Metrics:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                atr_val = metrics['atr']
                                st.metric("ATR", f"${atr_val:.2f}" if atr_val > 0 else "N/A")
                            with col2:
                                vol_val = metrics['volatility']
                                st.metric("Volatility", f"${vol_val:.2f}" if vol_val > 0 else "N/A")
                            with col3:
                                st.metric("CV", f"{metrics['coefficient_variation']:.2%}")
                            with col4:
                                st.metric("ATR Window", f"{metrics['atr_window']} periods")
                            
                            # Add interpretation
                            if atr_val > 0:
                                if tf == 'daily':
                                    if atr_val > 5:
                                        interpretation = "🔴 High volatility - Consider wider stops"
                                    elif atr_val > 2:
                                        interpretation = "🟡 Moderate volatility - Normal trading"
                                    else:
                                        interpretation = "🟢 Low volatility - Tight stops possible"
                                    st.info(f"Interpretation: {interpretation}")
        
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

if __name__ == "__main__":
    main() 