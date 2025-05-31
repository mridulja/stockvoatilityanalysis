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
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

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

def calculate_volatility_metrics(data, timeframe='hourly'):
    """Calculate volatility metrics for given timeframe"""
    if data is None or data.empty:
        return None
    
    # Calculate ranges
    data = data.copy()
    data['range'] = data['High'] - data['Low']
    
    # Resample based on timeframe
    if timeframe == 'daily':
        resampled = data.resample('D').agg({'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
        resampled['range'] = resampled['High'] - resampled['Low']
    elif timeframe == 'weekly':
        resampled = data.resample('W').agg({'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
        resampled['range'] = resampled['High'] - resampled['Low']
    else:  # hourly
        resampled = data
    
    # Calculate statistics
    range_stats = resampled['range'].describe(percentiles=[.25, .5, .75])
    
    # Calculate ATR (Average True Range)
    atr = resampled['range'].rolling(window=min(14, len(resampled))).mean().iloc[-1] if len(resampled) >= 1 else 0
    
    # Calculate additional metrics
    volatility = resampled['range'].std()
    cv = volatility / range_stats['mean'] if range_stats['mean'] > 0 else 0  # Coefficient of variation
    
    return {
        'stats': range_stats,
        'atr': atr,
        'volatility': volatility,
        'coefficient_variation': cv,
        'data': resampled
    }

def create_price_chart(data, ticker, timeframe):
    """Create interactive price chart with ranges"""
    if data is None or data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price Action ({timeframe.title()})', 'Range Analysis'),
        row_heights=[0.7, 0.3]
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
    if len(data) > 14:
        atr_line = data['range'].rolling(window=14).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=atr_line,
                mode='lines',
                name='14-period ATR',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=f'{ticker} - {timeframe.title()} Analysis',
        xaxis_rangeslider_visible=False,
        height=600,
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
            if results_dict[ticker][tf] and metric in results_dict[ticker][tf]:
                values.append(results_dict[ticker][tf][metric])
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=ticker,
            x=timeframes,
            y=values,
            text=[f'${v:.2f}' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f'{metric.upper()} Comparison Across Timeframes',
        xaxis_title='Timeframe',
        yaxis_title=f'{metric.upper()} Value ($)',
        barmode='group',
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">📊 Stock Volatility Analyzer</div>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("Analysis Parameters")
    
    # Ticker selection
    default_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
    selected_tickers = st.sidebar.multiselect(
        "Select Stock Tickers",
        options=default_tickers + ['NVDA', 'GOOGL', 'AMZN', 'META', 'NFLX'],
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
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if not selected_tickers:
            st.error("Please select at least one ticker")
            return
        
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
                    results[ticker][tf_name] = calculate_volatility_metrics(data, tf_name)
                else:
                    results[ticker][tf_name] = None
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.header("Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Price Charts", "🔍 Detailed Stats", "⚖️ Comparison"])
        
        with tab1:
            st.subheader("Volatility Summary")
            
            # Create summary table
            summary_data = []
            for ticker in selected_tickers:
                row = {'Ticker': ticker}
                for tf in ['hourly', 'daily', 'weekly']:
                    if tf in results[ticker] and results[ticker][tf]:
                        row[f'{tf.title()} ATR'] = f"${results[ticker][tf]['atr']:.2f}"
                        row[f'{tf.title()} Volatility'] = f"${results[ticker][tf]['volatility']:.2f}"
                    else:
                        row[f'{tf.title()} ATR'] = "N/A"
                        row[f'{tf.title()} Volatility'] = "N/A"
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Key insights
            st.subheader("Key Insights")
            for ticker in selected_tickers:
                with st.expander(f"{ticker} Analysis"):
                    for tf in ['hourly', 'daily', 'weekly']:
                        if tf in results[ticker] and results[ticker][tf]:
                            metrics = results[ticker][tf]
                            st.write(f"**{tf.title()} Metrics:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ATR", f"${metrics['atr']:.2f}")
                            with col2:
                                st.metric("Volatility", f"${metrics['volatility']:.2f}")
                            with col3:
                                st.metric("CV", f"{metrics['coefficient_variation']:.2%}")
        
        with tab2:
            st.subheader("Interactive Price Charts")
            
            # Chart selection
            chart_ticker = st.selectbox("Select ticker for detailed chart:", selected_tickers)
            chart_timeframe = st.selectbox(
                "Select timeframe:",
                [tf for tf in ['hourly', 'daily', 'weekly'] if tf in results[chart_ticker] and results[chart_ticker][tf]]
            )
            
            if chart_timeframe in results[chart_ticker] and results[chart_ticker][chart_timeframe]:
                chart_data = results[chart_ticker][chart_timeframe]['data']
                fig = create_price_chart(chart_data, chart_ticker, chart_timeframe)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Detailed Statistical Analysis")
            
            for ticker in selected_tickers:
                st.write(f"### {ticker}")
                
                cols = st.columns(len([tf for tf in ['hourly', 'daily', 'weekly'] if tf in results[ticker] and results[ticker][tf]]))
                col_idx = 0
                
                for tf in ['hourly', 'daily', 'weekly']:
                    if tf in results[ticker] and results[ticker][tf]:
                        with cols[col_idx]:
                            st.write(f"**{tf.title()} Range Statistics**")
                            stats = results[ticker][tf]['stats']
                            st.write(f"- Count: {stats['count']:.0f}")
                            st.write(f"- Mean: ${stats['mean']:.2f}")
                            st.write(f"- Std: ${stats['std']:.2f}")
                            st.write(f"- Min: ${stats['min']:.2f}")
                            st.write(f"- 25%: ${stats['25%']:.2f}")
                            st.write(f"- 50%: ${stats['50%']:.2f}")
                            st.write(f"- 75%: ${stats['75%']:.2f}")
                            st.write(f"- Max: ${stats['max']:.2f}")
                        col_idx += 1
        
        with tab4:
            st.subheader("Cross-Ticker Comparison")
            
            if len(selected_tickers) > 1:
                # ATR comparison
                atr_fig = create_comparison_chart(results, 'atr')
                st.plotly_chart(atr_fig, use_container_width=True)
                
                # Volatility comparison
                vol_fig = create_comparison_chart(results, 'volatility')
                st.plotly_chart(vol_fig, use_container_width=True)
                
                # Correlation analysis
                st.subheader("Range Correlation Analysis")
                
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
                        title="Daily Range Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Select multiple tickers to see comparison charts")

if __name__ == "__main__":
    main() 