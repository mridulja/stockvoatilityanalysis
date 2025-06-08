"""
Tab 3: Detailed Statistical Analysis - Stock Volatility Analyzer

This module provides comprehensive statistical analysis including:
- Advanced volatility metrics and distributions
- Correlation analysis and risk assessment
- Probability-based trading ranges
- Statistical significance testing
- Risk-adjusted performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Import unified AI formatter
try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, display_ai_setup_instructions, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

# Import LLM analysis functionality
try:
    from llm_analysis import get_llm_analyzer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import utility functions
try:
    from core import get_vix_condition
except ImportError:
    # Fallback function if core module not available
    def get_vix_condition(vix_value):
        """Fallback VIX condition function"""
        if pd.isna(vix_value):
            return "Unknown", "vix-normal", "🤷"
        if vix_value < 15:
            return "Calm Markets", "vix-calm", "🟢"
        elif 15 <= vix_value < 19:
            return "Normal Markets", "vix-normal", "🔵"
        elif 19 <= vix_value < 26:
            return "Choppy Market", "vix-choppy", "🟡"
        elif 26 <= vix_value < 36:
            return "High Volatility", "vix-volatile", "🔴"
        else:
            return "Extreme Volatility", "vix-extreme", "🚨"

def calculate_advanced_statistics(data, timeframe='daily'):
    """
    Calculate advanced statistical metrics for the given data
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV
        timeframe (str): Timeframe for analysis
        
    Returns:
        dict: Advanced statistical metrics
    """
    if data is None or data.empty:
        return None
    
    # Calculate returns
    data = data.copy()
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate range metrics
    data['Range'] = data['High'] - data['Low']
    data['True_Range'] = np.maximum.reduce([
        data['High'] - data['Low'],
        np.abs(data['High'] - data['Close'].shift(1)),
        np.abs(data['Low'] - data['Close'].shift(1))
    ])
    
    # Remove NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 10:  # Need sufficient data
        return None
    
    # Basic statistics
    returns = clean_data['Returns']
    log_returns = clean_data['Log_Returns']
    ranges = clean_data['Range']
    true_ranges = clean_data['True_Range']
    prices = clean_data['Close']
    
    # Calculate comprehensive metrics
    stats_dict = {
        # Price Statistics
        'current_price': prices.iloc[-1],
        'price_mean': prices.mean(),
        'price_std': prices.std(),
        'price_min': prices.min(),
        'price_max': prices.max(),
        
        # Return Statistics
        'return_mean': returns.mean(),
        'return_std': returns.std(),
        'return_skewness': returns.skew(),
        'return_kurtosis': returns.kurtosis(),
        'return_min': returns.min(),
        'return_max': returns.max(),
        
        # Log Return Statistics
        'log_return_mean': log_returns.mean(),
        'log_return_std': log_returns.std(),
        
        # Range Statistics  
        'range_mean': ranges.mean(),
        'range_std': ranges.std(),
        'range_median': ranges.median(),
        'range_q25': ranges.quantile(0.25),
        'range_q75': ranges.quantile(0.75),
        'range_max': ranges.max(),
        
        # True Range Statistics
        'atr': true_ranges.rolling(min(14, len(true_ranges))).mean().iloc[-1],
        'atr_std': true_ranges.std(),
        'atr_median': true_ranges.median(),
        'true_range_max': true_ranges.max(),
        
        # Volatility Metrics
        'realized_volatility': returns.std() * np.sqrt(252),  # Annualized
        'parkinson_volatility': np.sqrt(np.log(data['High'] / data['Low']).pow(2).mean() * 252),
        'garman_klass_volatility': np.sqrt(252 * (
            0.5 * np.log(data['High'] / data['Low']).pow(2) - 
            (2 * np.log(2) - 1) * np.log(data['Close'] / data['Open']).pow(2)
        ).mean()) if 'Open' in data.columns else None,
        
        # Risk Metrics
        'value_at_risk_5': returns.quantile(0.05),
        'value_at_risk_1': returns.quantile(0.01),
        'expected_shortfall_5': returns[returns <= returns.quantile(0.05)].mean(),
        'expected_shortfall_1': returns[returns <= returns.quantile(0.01)].mean(),
        'max_drawdown': calculate_max_drawdown(prices),
        
        # Distribution Tests
        'normality_test': stats.normaltest(returns.dropna()),
        'autocorr_lag1': calculate_autocorr_lag1(returns.dropna()) if len(returns.dropna()) > 10 else 0,
        
        # Trend Metrics
        'trend_strength': calculate_trend_strength(prices),
        'momentum_1w': (prices.iloc[-1] / prices.iloc[-min(5, len(prices))]) - 1 if len(prices) >= 5 else 0,
        'momentum_1m': (prices.iloc[-1] / prices.iloc[-min(21, len(prices))]) - 1 if len(prices) >= 21 else 0,
        
        # Probability Ranges (95% confidence)
        'prob_range_daily': calculate_probability_range(returns, confidence=0.95),
        'prob_range_weekly': calculate_probability_range(returns, confidence=0.95, periods=5),
        'prob_range_monthly': calculate_probability_range(returns, confidence=0.95, periods=21),
        
        # Data quality
        'data_points': len(clean_data),
        'missing_data_pct': (len(data) - len(clean_data)) / len(data) * 100,
        'timeframe': timeframe
    }
    
    return stats_dict

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    cumulative = prices / prices.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_trend_strength(prices, window=20):
    """Calculate trend strength using linear regression slope"""
    if len(prices) < window:
        window = len(prices)
    
    recent_prices = prices.iloc[-window:].values
    x = np.arange(len(recent_prices))
    
    # Linear regression
    slope, _, r_value, _, _ = stats.linregress(x, recent_prices)
    
    # Normalize slope by average price
    avg_price = recent_prices.mean()
    normalized_slope = (slope / avg_price) * 100  # Percentage per period
    
    return normalized_slope * (r_value ** 2)  # Weight by R-squared

def calculate_probability_range(returns, confidence=0.95, periods=1):
    """Calculate probability-based price range"""
    if len(returns.dropna()) < 10:
        return {'lower': 0, 'upper': 0}
    
    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Adjust for multiple periods (square root rule)
    adjusted_std = std_return * np.sqrt(periods)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha/2)
    
    lower_return = mean_return - z_score * adjusted_std
    upper_return = mean_return + z_score * adjusted_std
    
    return {
        'lower': lower_return,
        'upper': upper_return,
        'lower_pct': lower_return * 100,
        'upper_pct': upper_return * 100
    }

def calculate_autocorr_lag1(returns):
    """Calculate lag-1 autocorrelation manually"""
    if len(returns) < 2:
        return 0
    
    try:
        # Simple lag-1 autocorrelation calculation
        returns_lag1 = returns.shift(1).dropna()
        returns_aligned = returns[1:]
        
        if len(returns_aligned) < 2:
            return 0
        
        correlation = np.corrcoef(returns_aligned, returns_lag1)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    except:
        return 0

def create_statistical_distribution_chart(data, ticker, timeframe):
    """Create distribution analysis charts"""
    if data is None or data.empty:
        return None
    
    # Calculate returns
    returns = data['Close'].pct_change().dropna() * 100  # Convert to percentage
    
    if len(returns) < 10:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{ticker} Return Distribution',
            f'Q-Q Plot vs Normal Distribution',
            f'Returns Over Time',
            f'Cumulative Returns'
        ],
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # 1. Histogram with normal overlay
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=30,
            name='Actual Returns',
            opacity=0.7,
            marker_color='rgba(99, 102, 241, 0.7)',
            histnorm='probability density'
        ),
        row=1, col=1
    )
    
    # Overlay normal distribution
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    y_norm = stats.norm.pdf(x_norm, returns.mean(), returns.std())
    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # 2. Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
    sample_quantiles = np.sort(returns)
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='rgba(99, 102, 241, 0.6)', size=4)
        ),
        row=1, col=2
    )
    
    # Add diagonal reference line
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Normal',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )
    
    # 3. Returns over time
    fig.add_trace(
        go.Scatter(
            x=data.index[1:],  # Skip first NaN
            y=returns,
            mode='lines',
            name='Daily Returns',
            line=dict(color='rgba(99, 102, 241, 0.8)')
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 4. Cumulative returns
    cumulative_returns = (1 + data['Close'].pct_change()).cumprod() - 1
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=cumulative_returns * 100,  # Convert to percentage
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='rgba(16, 185, 129, 0.8)', width=2),
            fill='tonexty'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Statistical Distribution Analysis ({timeframe.title()})',
        height=600,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Return %", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Return %", row=2, col=2)
    
    return fig

def create_correlation_heatmap(results, session_tickers, timeframe='daily'):
    """Create correlation heatmap between tickers"""
    if len(session_tickers) < 2:
        return None
    
    # Collect returns data
    returns_data = {}
    
    for ticker in session_tickers:
        if ticker in results and timeframe in results[ticker] and results[ticker][timeframe]:
            data = results[ticker][timeframe]['data']
            if data is not None and not data.empty:
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 0:
                    returns_data[ticker] = returns
    
    if len(returns_data) < 2:
        return None
    
    # Create DataFrame with aligned dates
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 10:
        return None
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
        showscale=True,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=f'Correlation Matrix - {timeframe.title()} Returns',
        width=600,
        height=500,
        xaxis_title="Tickers",
        yaxis_title="Tickers"
    )
    
    return fig

def create_risk_metrics_chart(stats_data, session_tickers):
    """Create risk metrics comparison chart"""
    if not stats_data:
        return None
    
    # Prepare data for visualization
    risk_metrics = []
    
    for ticker in session_tickers:
        if ticker in stats_data and stats_data[ticker]:
            metrics = stats_data[ticker]
            risk_metrics.append({
                'Ticker': ticker,
                'Volatility (Ann.)': metrics.get('realized_volatility', 0) * 100,
                'VaR 5%': abs(metrics.get('value_at_risk_5', 0)) * 100,
                'VaR 1%': abs(metrics.get('value_at_risk_1', 0)) * 100,
                'Max Drawdown': abs(metrics.get('max_drawdown', 0)) * 100,
                'Sharpe Ratio': (metrics.get('return_mean', 0) / metrics.get('return_std', 1)) * np.sqrt(252) if metrics.get('return_std', 0) > 0 else 0
            })
    
    if not risk_metrics:
        return None
    
    df = pd.DataFrame(risk_metrics)
    
    # Create subplot with multiple metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Annualized Volatility (%)', 'Value at Risk (%)', 'Max Drawdown (%)', 'Sharpe Ratio'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Volatility
    fig.add_trace(
        go.Bar(
            x=df['Ticker'],
            y=df['Volatility (Ann.)'],
            name='Volatility',
            marker_color='rgba(99, 102, 241, 0.8)'
        ),
        row=1, col=1
    )
    
    # VaR comparison
    fig.add_trace(
        go.Bar(
            x=df['Ticker'],
            y=df['VaR 5%'],
            name='VaR 5%',
            marker_color='rgba(251, 191, 36, 0.8)'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=df['Ticker'],
            y=df['VaR 1%'],
            name='VaR 1%',
            marker_color='rgba(239, 68, 68, 0.8)'
        ),
        row=1, col=2
    )
    
    # Max Drawdown
    fig.add_trace(
        go.Bar(
            x=df['Ticker'],
            y=df['Max Drawdown'],
            name='Max DD',
            marker_color='rgba(239, 68, 68, 0.6)'
        ),
        row=2, col=1
    )
    
    # Sharpe Ratio
    colors = ['green' if x > 0 else 'red' for x in df['Sharpe Ratio']]
    fig.add_trace(
        go.Bar(
            x=df['Ticker'],
            y=df['Sharpe Ratio'],
            name='Sharpe',
            marker_color=colors
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Risk Metrics Comparison',
        height=600,
        showlegend=False
    )
    
    return fig

def render_detailed_stats_tab(results, vix_data, session_tickers):
    """
    Render the Detailed Statistical Analysis tab
    
    Args:
        results (dict): Analysis results from the main app
        vix_data (pd.DataFrame): VIX data  
        session_tickers (list): List of selected tickers
    """
    
    st.subheader("🔍 Advanced Statistical Analysis")
    
    # Ticker selection for detailed analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_ticker = st.selectbox(
            "Select ticker for detailed analysis:",
            session_tickers,
            key="detailed_stats_ticker"
        )
    
    with col2:
        timeframe_detailed = st.selectbox(
            "Select timeframe:",
            ['daily', 'weekly', 'hourly'],
            key="detailed_stats_timeframe"
        )
    
    if selected_ticker not in results:
        st.error(f"No data available for {selected_ticker}")
        return
    
    # Calculate advanced statistics for all tickers
    st.markdown("### 📊 Computing Advanced Statistics...")
    
    advanced_stats = {}
    for ticker in session_tickers:
        if ticker in results and timeframe_detailed in results[ticker] and results[ticker][timeframe_detailed]:
            data = results[ticker][timeframe_detailed]['data']
            advanced_stats[ticker] = calculate_advanced_statistics(data, timeframe_detailed)
    
    # === MAIN ANALYSIS FOR SELECTED TICKER ===
    if selected_ticker in advanced_stats and advanced_stats[selected_ticker]:
        stats = advanced_stats[selected_ticker]
        
        st.markdown(f"### 📈 Detailed Analysis: {selected_ticker} ({timeframe_detailed.title()})")
        
        # === SECTION 1: KEY METRICS OVERVIEW ===
        st.markdown("#### 💎 Key Statistical Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Current Price",
                f"${stats['current_price']:.2f}",
                delta=f"{stats['momentum_1w']*100:+.1f}% (1W)" if abs(stats['momentum_1w']) > 0.001 else None
            )
        
        with col2:
            st.metric(
                "Realized Vol (Ann.)",
                f"{stats['realized_volatility']*100:.1f}%",
                help="Annualized volatility based on daily returns"
            )
        
        with col3:
            st.metric(
                "ATR",
                f"${stats['atr']:.2f}",
                delta=f"{(stats['atr']/stats['current_price'])*100:.1f}% of price"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{stats['max_drawdown']*100:.1f}%",
                help="Maximum peak-to-trough decline"
            )
        
        with col5:
            sharpe = (stats['return_mean'] / stats['return_std']) * np.sqrt(252) if stats['return_std'] > 0 else 0
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                help="Risk-adjusted return measure"
            )
        
        # === SECTION 2: RETURN DISTRIBUTION ANALYSIS ===
        st.markdown("#### 📊 Return Distribution Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Distribution chart
            if selected_ticker in results and timeframe_detailed in results[selected_ticker]:
                data = results[selected_ticker][timeframe_detailed]['data']
                dist_chart = create_statistical_distribution_chart(data, selected_ticker, timeframe_detailed)
                if dist_chart:
                    st.plotly_chart(dist_chart, use_container_width=True)
        
        with col2:
            st.markdown("##### 📈 Distribution Statistics")
            
            # Normality test
            normality_stat, normality_p = stats['normality_test']
            is_normal = normality_p > 0.05
            
            st.write(f"**Skewness**: {stats['return_skewness']:.3f}")
            st.write(f"**Kurtosis**: {stats['return_kurtosis']:.3f}")
            st.write(f"**Normality Test**: {'✅ Normal' if is_normal else '❌ Non-Normal'}")
            st.write(f"**P-value**: {normality_p:.4f}")
            
            # Risk metrics
            st.markdown("##### ⚠️ Risk Metrics")
            st.write(f"**VaR (5%)**: {stats['value_at_risk_5']*100:.2f}%")
            st.write(f"**VaR (1%)**: {stats['value_at_risk_1']*100:.2f}%")
            st.write(f"**Expected Shortfall (5%)**: {stats['expected_shortfall_5']*100:.2f}%")
            
            # Trend analysis
            st.markdown("##### 📈 Trend Analysis")
            trend_strength = stats['trend_strength']
            if trend_strength > 0.1:
                trend_desc = "🟢 Strong Uptrend"
            elif trend_strength < -0.1:
                trend_desc = "🔴 Strong Downtrend"
            elif abs(trend_strength) > 0.05:
                trend_desc = "🟡 Moderate Trend"
            else:
                trend_desc = "⚪ Sideways/No Trend"
            
            st.write(f"**Trend Strength**: {trend_desc}")
            st.write(f"**1W Momentum**: {stats['momentum_1w']*100:+.1f}%")
            st.write(f"**1M Momentum**: {stats['momentum_1m']*100:+.1f}%")
        
        # === SECTION 3: PROBABILITY RANGES ===
        st.markdown("#### 🎯 Probability-Based Trading Ranges (95% Confidence)")
        
        current_price = stats['current_price']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📅 Daily Range")
            daily_prob = stats['prob_range_daily']
            lower_price = current_price * (1 + daily_prob['lower'])
            upper_price = current_price * (1 + daily_prob['upper'])
            
            st.write(f"**Lower Bound**: ${lower_price:.2f}")
            st.write(f"**Upper Bound**: ${upper_price:.2f}")
            st.write(f"**Range**: ${upper_price - lower_price:.2f}")
            st.write(f"**Range %**: {(upper_price/lower_price - 1)*100:.1f}%")
        
        with col2:
            st.markdown("##### 📅 Weekly Range")
            weekly_prob = stats['prob_range_weekly']
            lower_price = current_price * (1 + weekly_prob['lower'])
            upper_price = current_price * (1 + weekly_prob['upper'])
            
            st.write(f"**Lower Bound**: ${lower_price:.2f}")
            st.write(f"**Upper Bound**: ${upper_price:.2f}")
            st.write(f"**Range**: ${upper_price - lower_price:.2f}")
            st.write(f"**Range %**: {(upper_price/lower_price - 1)*100:.1f}%")
        
        with col3:
            st.markdown("##### 📅 Monthly Range")
            monthly_prob = stats['prob_range_monthly']
            lower_price = current_price * (1 + monthly_prob['lower'])
            upper_price = current_price * (1 + monthly_prob['upper'])
            
            st.write(f"**Lower Bound**: ${lower_price:.2f}")
            st.write(f"**Upper Bound**: ${upper_price:.2f}")
            st.write(f"**Range**: ${upper_price - lower_price:.2f}")
            st.write(f"**Range %**: {(upper_price/lower_price - 1)*100:.1f}%")
        
        # === SECTION 4: AI ANALYSIS ===
        _render_ai_analysis_section(selected_ticker, timeframe_detailed, stats, advanced_stats, session_tickers, results)
        
        # === SECTION 5: MULTI-TICKER COMPARISON ===
        if len(session_tickers) > 1:
            st.markdown("### ⚖️ Multi-Ticker Statistical Comparison")
            
            # Risk metrics comparison chart
            risk_chart = create_risk_metrics_chart(advanced_stats, session_tickers)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
            
            # Correlation analysis
            st.markdown("#### 🔗 Correlation Analysis")
            
            correlation_chart = create_correlation_heatmap(results, session_tickers, timeframe_detailed)
            if correlation_chart:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(correlation_chart, use_container_width=True)
                
                with col2:
                    st.markdown("##### 📊 Correlation Insights")
                    st.markdown("""
                    **Understanding Correlations:**
                    - **+1.0**: Perfect positive correlation
                    - **0.0**: No linear relationship  
                    - **-1.0**: Perfect negative correlation
                    
                    **Trading Applications:**
                    - **High correlation (>0.7)**: Similar risk factors
                    - **Low correlation (<0.3)**: Good for diversification
                    - **Negative correlation (<-0.3)**: Natural hedge
                    """)
            
            # Statistical comparison table
            st.markdown("#### 📋 Statistical Comparison Table")
            
            comparison_data = []
            for ticker in session_tickers:
                if ticker in advanced_stats and advanced_stats[ticker]:
                    s = advanced_stats[ticker]
                    comparison_data.append({
                        'Ticker': ticker,
                        'Current Price': f"${s['current_price']:.2f}",
                        'Ann. Volatility': f"{s['realized_volatility']*100:.1f}%",
                        'ATR': f"${s['atr']:.2f}",
                        'ATR %': f"{(s['atr']/s['current_price'])*100:.1f}%",
                        'Sharpe Ratio': f"{(s['return_mean']/s['return_std'])*np.sqrt(252):.2f}" if s['return_std'] > 0 else "N/A",
                        'Max Drawdown': f"{s['max_drawdown']*100:.1f}%",
                        'VaR (5%)': f"{abs(s['value_at_risk_5'])*100:.1f}%",
                        'Skewness': f"{s['return_skewness']:.2f}",
                        'Kurtosis': f"{s['return_kurtosis']:.2f}",
                        'Data Points': s['data_points']
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, height=300)
        
        # === SECTION 6: STATISTICAL EXPLANATIONS ===
        with st.expander("📚 Understanding Advanced Statistics"):
            st.markdown("""
            ### 📊 Statistical Metrics Explained
            
            #### Return Distribution Metrics
            - **Skewness**: Measures asymmetry of return distribution
              - Positive: More extreme positive returns
              - Negative: More extreme negative returns  
              - Zero: Symmetric distribution
            
            - **Kurtosis**: Measures tail heaviness
              - High: More extreme events ("fat tails")
              - Low: Fewer extreme events
              - Normal distribution has kurtosis = 3
            
            #### Risk Metrics
            - **Value at Risk (VaR)**: Maximum expected loss at given confidence level
              - VaR 5%: Loss not expected to exceed this 95% of the time
              - VaR 1%: Loss not expected to exceed this 99% of the time
            
            - **Expected Shortfall**: Average loss when VaR is exceeded
              - Also called Conditional VaR (CVaR)
              - Measures tail risk beyond VaR
            
            - **Maximum Drawdown**: Largest peak-to-trough decline
              - Key risk measure for position sizing
              - Historical worst-case scenario
            
            #### Volatility Measures
            - **Realized Volatility**: Standard deviation of returns (annualized)
            - **Parkinson Volatility**: Based on high-low range (more efficient)
            - **Garman-Klass Volatility**: Uses OHLC data (most efficient)
            
            #### Trading Applications
            - **Position Sizing**: Use volatility and VaR for risk management
            - **Options Strategy**: Higher volatility = higher premiums
            - **Stop Losses**: Set stops based on ATR or volatility
            - **Profit Targets**: Use probability ranges for realistic targets
            """)
    
    else:
        st.error(f"Unable to calculate advanced statistics for {selected_ticker}. Check data availability.")


def _render_ai_analysis_section(selected_ticker, timeframe_detailed, stats_dict, advanced_stats, session_tickers, results):
    """
    Render the AI Analysis section for the Detailed Statistics tab
    
    Args:
        selected_ticker (str): Selected ticker symbol
        timeframe_detailed (str): Selected timeframe
        stats_dict (dict): Advanced statistics for the selected ticker
        advanced_stats (dict): All ticker advanced statistics
        session_tickers (list): List of all tickers
        results (dict): Analysis results from main app
    """
    
    st.markdown("### 🤖 AI-Powered Statistical Analysis")
    
    # Check if LLM is available
    try:
        from llm_analysis import get_llm_analyzer
        LLM_AVAILABLE = True
    except ImportError:
        LLM_AVAILABLE = False
    
    if LLM_AVAILABLE:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Use form to prevent page refresh
            with st.form("ai_stats_analysis_form"):
                ai_button_clicked = st.form_submit_button(
                    "🤖 Generate AI Statistical Analysis", 
                    type="primary", 
                    help="Get AI-powered statistical insights and trading recommendations"
                )
            
            if ai_button_clicked:
                # Prepare comprehensive data for AI analysis
                normality_stat, normality_p = stats_dict['normality_test']
                is_normal = normality_p > 0.05
                sharpe_ratio = (stats_dict['return_mean'] / stats_dict['return_std']) * np.sqrt(252) if stats_dict['return_std'] > 0 else 0
                
                # Determine trend description
                trend_strength = stats_dict['trend_strength']
                if trend_strength > 0.1:
                    trend_desc = "Strong Uptrend"
                elif trend_strength < -0.1:
                    trend_desc = "Strong Downtrend"
                elif abs(trend_strength) > 0.05:
                    trend_desc = "Moderate Trend"
                else:
                    trend_desc = "Sideways/No Trend"
                
                # Risk assessment
                var_5_pct = abs(stats_dict['value_at_risk_5']) * 100
                max_dd_pct = abs(stats_dict['max_drawdown']) * 100
                
                if var_5_pct > 3 or max_dd_pct > 20:
                    risk_level = "HIGH RISK"
                elif var_5_pct > 1.5 or max_dd_pct > 10:
                    risk_level = "MODERATE RISK"
                else:
                    risk_level = "LOW RISK"
                
                # Volatility regime
                realized_vol_pct = stats_dict['realized_volatility'] * 100
                if realized_vol_pct > 40:
                    vol_regime = "EXTREME VOLATILITY"
                elif realized_vol_pct > 25:
                    vol_regime = "HIGH VOLATILITY"
                elif realized_vol_pct > 15:
                    vol_regime = "MODERATE VOLATILITY"
                else:
                    vol_regime = "LOW VOLATILITY"
                
                # Probability ranges
                daily_prob = stats_dict['prob_range_daily']
                weekly_prob = stats_dict['prob_range_weekly']
                monthly_prob = stats_dict['prob_range_monthly']
                current_price = stats_dict['current_price']
                
                ai_analysis_data = {
                    'ticker': selected_ticker,
                    'timeframe': timeframe_detailed,
                    'statistical_summary': {
                        'current_price': current_price,
                        'realized_volatility_pct': realized_vol_pct,
                        'atr': stats_dict['atr'],
                        'atr_percentage': (stats_dict['atr'] / current_price) * 100,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown_pct': max_dd_pct,
                        'skewness': stats_dict['return_skewness'],
                        'kurtosis': stats_dict['return_kurtosis'],
                        'is_normal_distribution': is_normal,
                        'normality_p_value': normality_p
                    },
                    'risk_metrics': {
                        'var_5_pct': var_5_pct,
                        'var_1_pct': abs(stats_dict['value_at_risk_1']) * 100,
                        'expected_shortfall_5_pct': abs(stats_dict['expected_shortfall_5']) * 100,
                        'risk_level': risk_level,
                        'volatility_regime': vol_regime
                    },
                    'trend_analysis': {
                        'trend_strength': trend_strength,
                        'trend_description': trend_desc,
                        'momentum_1w_pct': stats_dict['momentum_1w'] * 100,
                        'momentum_1m_pct': stats_dict['momentum_1m'] * 100
                    },
                    'probability_ranges': {
                        'daily_range_pct': (daily_prob['upper'] - daily_prob['lower']) * 100,
                        'weekly_range_pct': (weekly_prob['upper'] - weekly_prob['lower']) * 100,
                        'monthly_range_pct': (monthly_prob['upper'] - monthly_prob['lower']) * 100,
                        'daily_lower': current_price * (1 + daily_prob['lower']),
                        'daily_upper': current_price * (1 + daily_prob['upper']),
                        'weekly_lower': current_price * (1 + weekly_prob['lower']),
                        'weekly_upper': current_price * (1 + weekly_prob['upper']),
                        'monthly_lower': current_price * (1 + monthly_prob['lower']),
                        'monthly_upper': current_price * (1 + monthly_prob['upper'])
                    },
                    'data_quality': {
                        'data_points': stats_dict['data_points'],
                        'missing_data_pct': stats_dict.get('missing_data_pct', 0)
                    }
                }
                
                # Generate AI analysis
                with st.spinner("🤖 AI is analyzing statistical patterns..."):
                    try:
                        llm_analyzer = get_llm_analyzer()
                        
                        # Create comprehensive prompt for statistical analysis
                        analysis_prompt = f"""
                        Analyze the following advanced statistical data for {selected_ticker} and provide professional trading insights:

                        **STATISTICAL OVERVIEW ({timeframe_detailed} timeframe)**
                        • Current Price: ${current_price:.2f}
                        • Realized Volatility: {realized_vol_pct:.1f}% (annualized) - {vol_regime}
                        • ATR: ${stats_dict['atr']:.2f} ({(stats_dict['atr']/current_price)*100:.1f}% of price)
                        • Sharpe Ratio: {sharpe_ratio:.2f} (risk-adjusted returns)
                        • Maximum Drawdown: {max_dd_pct:.1f}%

                        **RETURN DISTRIBUTION ANALYSIS**
                        • Distribution: {'Normal' if is_normal else 'Non-Normal'} (p-value: {normality_p:.4f})
                        • Skewness: {stats_dict['return_skewness']:.3f} ({'Positive tail bias' if stats_dict['return_skewness'] > 0.1 else 'Negative tail bias' if stats_dict['return_skewness'] < -0.1 else 'Symmetric'})
                        • Kurtosis: {stats_dict['return_kurtosis']:.3f} ({'Fat tails' if stats_dict['return_kurtosis'] > 3.5 else 'Thin tails' if stats_dict['return_kurtosis'] < 2.5 else 'Normal tails'})

                        **RISK ASSESSMENT - {risk_level}**
                        • VaR (5%): {var_5_pct:.2f}% (95% confidence daily loss limit)
                        • VaR (1%): {abs(stats_dict['value_at_risk_1'])*100:.2f}% (99% confidence daily loss limit)
                        • Expected Shortfall (5%): {abs(stats_dict['expected_shortfall_5'])*100:.2f}% (average loss when VaR exceeded)

                        **TREND & MOMENTUM**
                        • Trend Strength: {trend_desc} ({trend_strength:+.2f}%)
                        • 1-Week Momentum: {stats_dict['momentum_1w']*100:+.1f}%
                        • 1-Month Momentum: {stats_dict['momentum_1m']*100:+.1f}%

                        **PROBABILITY-BASED TRADING RANGES (95% Confidence)**
                        • Daily Range: ${current_price*(1+daily_prob['lower']):.2f} - ${current_price*(1+daily_prob['upper']):.2f} ({(daily_prob['upper']-daily_prob['lower'])*100:.1f}% range)
                        • Weekly Range: ${current_price*(1+weekly_prob['lower']):.2f} - ${current_price*(1+weekly_prob['upper']):.2f} ({(weekly_prob['upper']-weekly_prob['lower'])*100:.1f}% range)
                        • Monthly Range: ${current_price*(1+monthly_prob['lower']):.2f} - ${current_price*(1+monthly_prob['upper']):.2f} ({(monthly_prob['upper']-monthly_prob['lower'])*100:.1f}% range)

                        **DATA QUALITY**
                        • Sample Size: {stats_dict['data_points']} observations
                        • Missing Data: {stats_dict.get('missing_data_pct', 0):.1f}%

                        Please provide a comprehensive analysis covering:

                        1. **Statistical Interpretation** (What do these metrics tell us about {selected_ticker}?)
                        2. **Risk Assessment** (Position sizing recommendations based on volatility and VaR)
                        3. **Trading Strategy Implications** (Best strategies for this statistical profile)
                        4. **Probability Range Usage** (How to use the 95% confidence ranges for entries/exits)
                        5. **Portfolio Considerations** (Diversification and correlation insights)

                        Focus on practical trading applications and risk management. Be specific about how traders should interpret and act on these statistical insights.
                        """
                        
                        # Try different method names for AI analysis
                        ai_response = None
                        if hasattr(llm_analyzer, 'analyze'):
                            ai_response = llm_analyzer.analyze(analysis_prompt)
                        elif hasattr(llm_analyzer, 'get_analysis'):
                            ai_response = llm_analyzer.get_analysis(analysis_prompt)
                        elif hasattr(llm_analyzer, 'generate_analysis'):
                            ai_response = llm_analyzer.generate_analysis(analysis_prompt)
                        elif hasattr(llm_analyzer, 'chat'):
                            ai_response = llm_analyzer.chat(analysis_prompt)
                        elif hasattr(llm_analyzer, 'query'):
                            ai_response = llm_analyzer.query(analysis_prompt)
                        elif callable(llm_analyzer):
                            ai_response = llm_analyzer(analysis_prompt)
                        else:
                            ai_response = "AI Statistical Analysis functionality needs to be configured. Please check llm_analysis.py for the correct method name."
                        
                        if ai_response:
                            # Store in session state for persistence
                            if 'ai_stats_analysis' not in st.session_state:
                                st.session_state.ai_stats_analysis = {}
                            st.session_state.ai_stats_analysis[f"{selected_ticker}_{timeframe_detailed}"] = ai_response
                            st.success("✅ AI Statistical Analysis completed!")
                        else:
                            st.warning("⚠️ AI Statistical Analysis could not be generated. Check LLM configuration.")
                        
                    except Exception as e:
                        st.error(f"AI Statistical Analysis Error: {str(e)}")
                        st.info("💡 AI analysis requires proper LLM configuration. Check llm_analysis.py setup.")
        
        with col2:
            # Display AI analysis if available
            analysis_key = f"{selected_ticker}_{timeframe_detailed}"
            if 'ai_stats_analysis' in st.session_state and analysis_key in st.session_state.ai_stats_analysis:
                ai_content = st.session_state.ai_stats_analysis[analysis_key]
                
                if AI_FORMATTER_AVAILABLE:
                    # Use unified AI formatter with statistical analysis styling
                    display_ai_analysis(
                        ai_content=ai_content,
                        analysis_type="Statistical Analysis",
                        tab_color=get_tab_color("statistical"),
                        analysis_key=analysis_key,
                        session_key="ai_stats_analysis",
                        regenerate_key="regenerate_stats_ai",
                        clear_key="clear_stats_ai",
                        show_debug=True,
                        show_metadata=True
                    )
                else:
                    # Fallback to simple display
                    st.markdown("#### 🧠 AI Statistical Analysis Results")
                    content_text = str(ai_content.get('content', ai_content)) if isinstance(ai_content, dict) else str(ai_content)
                    st.markdown(content_text)
                    
                    # Simple action buttons
                    col_regen, col_clear = st.columns(2)
                    with col_regen:
                        if st.button("🔄 Regenerate", key="regenerate_stats_fallback"):
                            if analysis_key in st.session_state.ai_stats_analysis:
                                del st.session_state.ai_stats_analysis[analysis_key]
                                st.rerun()
                    with col_clear:
                        if st.button("🗑️ Clear", key="clear_stats_fallback"):
                            if analysis_key in st.session_state.ai_stats_analysis:
                                del st.session_state.ai_stats_analysis[analysis_key]
                                st.rerun()
            
            else:
                if AI_FORMATTER_AVAILABLE:
                    # Use unified placeholder
                    display_ai_placeholder(
                        analysis_type="Statistical Analysis",
                        features_list=[
                            "Distribution analysis and normality assessment",
                            "VaR-based position sizing recommendations",
                            "Skewness and kurtosis trading implications",
                            "Volatility regime identification and strategy selection",
                            "Probability-based trading range calculations",
                            "Risk-adjusted performance metrics and correlation insights"
                        ]
                    )
                else:
                    # Fallback placeholder
                    st.info("👆 Click 'Generate AI Statistical Analysis' to get intelligent insights on statistical patterns and risk metrics")
    
    else:
        if AI_FORMATTER_AVAILABLE:
            # Use unified setup instructions
            display_ai_setup_instructions("Statistical Analysis")
        else:
            # Fallback setup instructions
            st.info("🤖 AI Statistical Analysis not available. Install and configure `llm_analysis.py` for intelligent statistical insights.")