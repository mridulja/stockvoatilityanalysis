"""
Put Spread Analysis Tab for Streamlit Stock Analysis Application

This module provides comprehensive put spread analysis including:
- Bull Put Spreads and Bear Put Spreads
- Real-time options data analysis
- Probability of success calculations
- P&L visualization and risk management
- Advanced Greeks analysis
- Market condition assessment

Author: Enhanced Stock Analysis System
Date: 2025
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # 5% risk-free rate

# Check for AI analysis availability
try:
    from llm_analysis import get_llm_analyzer, format_vix_data_for_llm
    from llm_input_formatters import format_put_spread_data_for_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

def get_next_friday():
    """Get next Friday date"""
    today = date.today()
    days_ahead = 4 - today.weekday()  # Friday is weekday 4
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return today + timedelta(days=days_ahead)

def get_monthly_expiry():
    """Get next monthly options expiry (3rd Friday)"""
    today = date.today()
    # Find 3rd Friday of current month
    first_day = today.replace(day=1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    
    # If 3rd Friday has passed, get next month's 3rd Friday
    if third_friday <= today:
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_friday = next_month + timedelta(days=(4 - next_month.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
    
    return third_friday

@st.cache_data(ttl=300)
def fetch_options_data(ticker, expiry_date):
    """Fetch options data for a specific ticker and expiry"""
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        
        if not options_dates:
            return None, None
        
        # Find closest expiry date
        target_date = pd.to_datetime(expiry_date).date()
        available_dates = [pd.to_datetime(d).date() for d in options_dates]
        closest_date = min(available_dates, key=lambda x: abs((x - target_date).days))
        
        # Get options chain
        chain = stock.option_chain(closest_date.strftime('%Y-%m-%d'))
        return chain.calls, chain.puts
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        return None, None

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put price"""
    try:
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        put_price = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
        return max(put_price, 0)
    except:
        return 0

def calculate_put_spread_greeks(S, K_short, K_long, T, r, sigma, spread_type='bull'):
    """Calculate Greeks for put spreads"""
    def put_delta(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        return -stats.norm.cdf(-d1)
    
    def put_gamma(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        return stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def put_theta(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        theta = (-S*stats.norm.pdf(d1)*sigma/(2*np.sqrt(T)) 
                + r*K*np.exp(-r*T)*stats.norm.cdf(-d2))
        return theta / 365  # Per day
    
    def put_vega(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        return S * stats.norm.pdf(d1) * np.sqrt(T) / 100
    
    try:
        # Calculate Greeks for both legs
        short_delta = put_delta(S, K_short, T, r, sigma)
        long_delta = put_delta(S, K_long, T, r, sigma)
        
        short_gamma = put_gamma(S, K_short, T, r, sigma)
        long_gamma = put_gamma(S, K_long, T, r, sigma)
        
        short_theta = put_theta(S, K_short, T, r, sigma)
        long_theta = put_theta(S, K_long, T, r, sigma)
        
        short_vega = put_vega(S, K_short, T, r, sigma)
        long_vega = put_vega(S, K_long, T, r, sigma)
        
        # Net Greeks for the spread
        if spread_type == 'bull':
            # Bull put spread: Short higher strike, Long lower strike
            net_delta = short_delta - long_delta
            net_gamma = short_gamma - long_gamma
            net_theta = short_theta - long_theta
            net_vega = short_vega - long_vega
        else:
            # Bear put spread: Long higher strike, Short lower strike
            net_delta = long_delta - short_delta
            net_gamma = long_gamma - short_gamma
            net_theta = long_theta - short_theta
            net_vega = long_vega - short_vega
        
        return {
            'delta': net_delta,
            'gamma': net_gamma,
            'theta': net_theta,
            'vega': net_vega
        }
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

def analyze_put_spread(current_price, short_strike, long_strike, short_premium, long_premium, 
                      days_to_expiry, volatility, spread_type='bull'):
    """Comprehensive put spread analysis"""
    
    T = days_to_expiry / 365.0
    
    # Calculate net premium
    if spread_type == 'bull':
        # Bull put spread: collect premium (short premium > long premium)
        net_credit = short_premium - long_premium
        max_profit = net_credit
        max_loss = (short_strike - long_strike) - net_credit
        breakeven = short_strike - net_credit
        
        # Probability calculations
        prob_max_profit = stats.norm.cdf((np.log(short_strike/current_price)) / (volatility*np.sqrt(T)))
        prob_breakeven = stats.norm.cdf((np.log(breakeven/current_price)) / (volatility*np.sqrt(T)))
        
    else:
        # Bear put spread: pay premium (long premium > short premium)
        net_debit = long_premium - short_premium
        max_profit = (short_strike - long_strike) - net_debit
        max_loss = net_debit
        breakeven = short_strike - net_debit
        
        # Probability calculations
        prob_max_profit = 1 - stats.norm.cdf((np.log(long_strike/current_price)) / (volatility*np.sqrt(T)))
        prob_breakeven = 1 - stats.norm.cdf((np.log(breakeven/current_price)) / (volatility*np.sqrt(T)))
    
    # Calculate Greeks
    greeks = calculate_put_spread_greeks(current_price, short_strike, long_strike, T, RISK_FREE_RATE, volatility, spread_type)
    
    # Risk-reward metrics
    risk_reward_ratio = max_profit / abs(max_loss) if max_loss != 0 else 0
    
    return {
        'spread_type': spread_type,
        'net_premium': net_credit if spread_type == 'bull' else -net_debit,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'breakeven': breakeven,
        'prob_max_profit': prob_max_profit,
        'prob_breakeven': prob_breakeven,
        'risk_reward_ratio': risk_reward_ratio,
        'greeks': greeks,
        'days_to_expiry': days_to_expiry
    }

def create_put_spread_pnl_chart(current_price, spread_analysis, short_strike, long_strike):
    """Create P&L diagram for put spread"""
    
    # Price range for P&L calculation
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    pnl_values = []
    
    spread_type = spread_analysis['spread_type']
    net_premium = spread_analysis['net_premium']
    
    for price in price_range:
        if spread_type == 'bull':
            # Bull put spread P&L
            short_pnl = min(0, price - short_strike)  # Short put P&L
            long_pnl = max(0, long_strike - price)   # Long put P&L
            total_pnl = net_premium + short_pnl - long_pnl
        else:
            # Bear put spread P&L
            short_pnl = min(0, price - short_strike)  # Short put P&L
            long_pnl = max(0, long_strike - price)   # Long put P&L
            total_pnl = net_premium - short_pnl + long_pnl
        
        pnl_values.append(total_pnl)
    
    # Create chart
    fig = go.Figure()
    
    # P&L line
    fig.add_trace(go.Scatter(
        x=price_range,
        y=pnl_values,
        mode='lines',
        name='P&L at Expiration',
        line=dict(color='#2196f3', width=3)
    ))
    
    # Profit/Loss zones
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Breakeven line
    fig.add_vline(x=spread_analysis['breakeven'], line_dash="dot", line_color="orange", 
                  annotation_text=f"Breakeven: ${spread_analysis['breakeven']:.2f}")
    
    # Current price line
    fig.add_vline(x=current_price, line_dash="solid", line_color="black", 
                  annotation_text=f"Current: ${current_price:.2f}")
    
    # Strike lines
    fig.add_vline(x=short_strike, line_dash="dash", line_color="red", opacity=0.7,
                  annotation_text=f"Short: ${short_strike:.2f}")
    fig.add_vline(x=long_strike, line_dash="dash", line_color="green", opacity=0.7,
                  annotation_text=f"Long: ${long_strike:.2f}")
    
    # Fill profit/loss areas
    profit_mask = np.array(pnl_values) > 0
    loss_mask = np.array(pnl_values) < 0
    
    if np.any(profit_mask):
        fig.add_scatter(x=price_range[profit_mask], y=np.array(pnl_values)[profit_mask],
                       fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', 
                       line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Profit Zone')
    
    if np.any(loss_mask):
        fig.add_scatter(x=price_range[loss_mask], y=np.array(pnl_values)[loss_mask],
                       fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', 
                       line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Loss Zone')
    
    fig.update_layout(
        title=f'{spread_analysis["spread_type"].title()} Put Spread P&L Diagram',
        xaxis_title='Stock Price at Expiration ($)',
        yaxis_title='Profit/Loss ($)',
        height=500,
        showlegend=True
    )
    
    return fig

def create_probability_analysis_chart(current_price, spread_analysis, volatility, days_to_expiry):
    """Create probability analysis visualization"""
    
    T = days_to_expiry / 365.0
    
    # Generate price distribution
    price_range = np.linspace(current_price * 0.5, current_price * 1.5, 200)
    
    # Log-normal distribution parameters
    mu = np.log(current_price) + (RISK_FREE_RATE - 0.5 * volatility**2) * T
    sigma = volatility * np.sqrt(T)
    
    # Calculate probability density
    pdf_values = stats.lognorm.pdf(price_range, s=sigma, scale=np.exp(mu))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Price Probability Distribution', 'Cumulative Probability'],
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # PDF plot
    fig.add_trace(go.Scatter(
        x=price_range, y=pdf_values,
        fill='tonexty', fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#6366f1', width=2),
        name='Price Distribution'
    ), row=1, col=1)
    
    # Add key price levels
    fig.add_vline(x=current_price, line_dash="solid", line_color="black", row=1,
                  annotation_text=f"Current: ${current_price:.2f}")
    fig.add_vline(x=spread_analysis['breakeven'], line_dash="dot", line_color="orange", row=1,
                  annotation_text=f"Breakeven: ${spread_analysis['breakeven']:.2f}")
    
    # CDF plot
    cdf_values = stats.lognorm.cdf(price_range, s=sigma, scale=np.exp(mu))
    fig.add_trace(go.Scatter(
        x=price_range, y=cdf_values * 100,
        line=dict(color='#ef4444', width=2),
        name='Cumulative Probability (%)'
    ), row=2, col=1)
    
    # Add probability lines
    breakeven_prob = stats.lognorm.cdf(spread_analysis['breakeven'], s=sigma, scale=np.exp(mu)) * 100
    fig.add_hline(y=breakeven_prob, line_dash="dot", line_color="orange", row=2,
                  annotation_text=f"P(Below Breakeven): {breakeven_prob:.1f}%")
    
    fig.update_layout(
        title='Put Spread Probability Analysis',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Stock Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability (%)", row=2, col=1)
    
    return fig

def find_optimal_put_spreads(current_price, puts_data, spread_type='bull', min_dte=7):
    """Find optimal put spread combinations"""
    
    if puts_data is None or puts_data.empty:
        return []
    
    # Filter liquid options
    liquid_puts = puts_data[
        (puts_data['volume'] > 10) & 
        (puts_data['openInterest'] > 50) &
        (puts_data['bid'] > 0.05)
    ].copy()
    
    if len(liquid_puts) < 2:
        return []
    
    optimal_spreads = []
    
    # Find spreads based on type
    if spread_type == 'bull':
        # Bull put spread: Short higher strike, Long lower strike
        for i, short_put in liquid_puts.iterrows():
            short_strike = short_put['strike']
            short_premium = (short_put['bid'] + short_put['ask']) / 2
            
            # Find suitable long strikes (lower than short strike)
            long_candidates = liquid_puts[
                (liquid_puts['strike'] < short_strike) &
                (liquid_puts['strike'] >= short_strike * 0.85)  # Max 15% width
            ]
            
            for j, long_put in long_candidates.iterrows():
                long_strike = long_put['strike']
                long_premium = (long_put['bid'] + long_put['ask']) / 2
                
                # Calculate spread metrics
                net_credit = short_premium - long_premium
                max_profit = net_credit
                max_loss = (short_strike - long_strike) - net_credit
                
                if net_credit > 0 and max_loss > 0:  # Valid bull put spread
                    risk_reward = max_profit / max_loss
                    
                    optimal_spreads.append({
                        'type': 'Bull Put',
                        'short_strike': short_strike,
                        'long_strike': long_strike,
                        'short_premium': short_premium,
                        'long_premium': long_premium,
                        'net_credit': net_credit,
                        'max_profit': max_profit,
                        'max_loss': max_loss,
                        'risk_reward': risk_reward,
                        'breakeven': short_strike - net_credit,
                        'width': short_strike - long_strike,
                        'short_volume': short_put['volume'],
                        'long_volume': long_put['volume']
                    })
    
    else:  # bear spread
        # Bear put spread: Long higher strike, Short lower strike
        for i, long_put in liquid_puts.iterrows():
            long_strike = long_put['strike']
            long_premium = (long_put['bid'] + long_put['ask']) / 2
            
            # Find suitable short strikes (lower than long strike)
            short_candidates = liquid_puts[
                (liquid_puts['strike'] < long_strike) &
                (liquid_puts['strike'] >= long_strike * 0.85)  # Max 15% width
            ]
            
            for j, short_put in short_candidates.iterrows():
                short_strike = short_put['strike']
                short_premium = (short_put['bid'] + short_put['ask']) / 2
                
                # Calculate spread metrics
                net_debit = long_premium - short_premium
                max_profit = (long_strike - short_strike) - net_debit
                max_loss = net_debit
                
                if net_debit > 0 and max_profit > 0:  # Valid bear put spread
                    risk_reward = max_profit / max_loss
                    
                    optimal_spreads.append({
                        'type': 'Bear Put',
                        'short_strike': short_strike,
                        'long_strike': long_strike,
                        'short_premium': short_premium,
                        'long_premium': long_premium,
                        'net_debit': net_debit,
                        'max_profit': max_profit,
                        'max_loss': max_loss,
                        'risk_reward': risk_reward,
                        'breakeven': long_strike - net_debit,
                        'width': long_strike - short_strike,
                        'short_volume': short_put['volume'],
                        'long_volume': long_put['volume']
                    })
    
    # Sort by risk-reward ratio
    optimal_spreads.sort(key=lambda x: x['risk_reward'], reverse=True)
    
    return optimal_spreads[:10]  # Return top 10

def render_put_spread_analysis_tab(results, vix_data, session_tickers):
    """Render the Put Spread Analysis tab"""
    
    st.markdown("## 📐 Advanced Put Spread Analysis")
    st.markdown("*Comprehensive put spread strategy analysis with real-time options data*")
    
    # === TICKER SELECTION ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Universal ticker input
        ticker_input = st.text_input(
            "🎯 Enter Stock/ETF Symbol:",
            value="SPY",
            help="Enter any valid stock or ETF symbol (e.g., AAPL, SPY, QQQ, TSLA)",
            key="put_spread_ticker"
        ).upper()
        
        # Validate ticker
        if ticker_input:
            is_valid, validation_msg = validate_ticker(ticker_input)
            if is_valid:
                st.success(f"✅ {ticker_input} - {validation_msg}")
                current_price = get_current_price(ticker_input)
            else:
                st.error(f"❌ {validation_msg}")
                return
        else:
            st.warning("Please enter a ticker symbol")
            return
    
    with col2:
        st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
    
    # === EXPIRY SELECTION ===
    st.markdown("#### 📅 Expiry Selection")
    
    expiry_col1, expiry_col2, expiry_col3 = st.columns(3)
    
    with expiry_col1:
        # Initialize expiry type in session state if not exists
        if "put_spread_expiry_type_value" not in st.session_state:
            st.session_state.put_spread_expiry_type_value = "Next Friday"
        
        expiry_type = st.selectbox(
            "Expiry Type:",
            ["Next Friday", "Monthly Expiry", "Custom Date"],
            index=["Next Friday", "Monthly Expiry", "Custom Date"].index(st.session_state.put_spread_expiry_type_value),
            key="put_spread_expiry_type",
            on_change=lambda: setattr(st.session_state, 'put_spread_expiry_type_value', st.session_state.put_spread_expiry_type)
        )
    
    with expiry_col2:
        if expiry_type == "Next Friday":
            selected_expiry = get_next_friday()
        elif expiry_type == "Monthly Expiry":
            selected_expiry = get_monthly_expiry()
        else:  # Custom Date
            min_date = date.today() + timedelta(days=1)
            max_date = date.today() + timedelta(days=365)
            selected_expiry = st.date_input(
                "Select Custom Date:",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="put_spread_custom_date"
            )
        
        st.write(f"**Selected:** {selected_expiry}")
    
    with expiry_col3:
        days_to_expiry = (selected_expiry - date.today()).days
        st.metric("Days to Expiry", f"{days_to_expiry} days")
    
    # === FETCH OPTIONS DATA ===
    st.markdown("#### 🔄 Options Data")
    
    if st.button("🚀 Fetch Options Data", type="primary", key="fetch_put_options"):
        with st.spinner("Fetching options data..."):
            calls_data, puts_data = fetch_options_data(ticker_input, selected_expiry)
            
            if puts_data is not None and not puts_data.empty:
                st.session_state[f'puts_data_{ticker_input}'] = puts_data
                st.session_state[f'calls_data_{ticker_input}'] = calls_data
                st.success(f"✅ Loaded {len(puts_data)} put options")
            else:
                st.error("❌ Could not fetch options data")
                return
    
    # Check if we have options data
    puts_data_key = f'puts_data_{ticker_input}'
    if puts_data_key not in st.session_state:
        st.info("👆 Click 'Fetch Options Data' to begin analysis")
        return
    
    puts_data = st.session_state[puts_data_key]
    
    # === STRATEGY SELECTION ===
    st.markdown("#### 🎯 Strategy Selection")
    
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        spread_type = st.selectbox(
            "Put Spread Type:",
            ["Bull Put Spread", "Bear Put Spread"],
            help="Bull Put = Bullish bias, Bear Put = Bearish bias",
            key="put_spread_type"
        )
    
    with strategy_col2:
        analysis_mode = st.selectbox(
            "Analysis Mode:",
            ["Manual Selection", "Quick Analysis"],
            key="put_analysis_mode"
        )
    
    # === VOLATILITY ESTIMATION ===
    st.markdown("#### 📊 Volatility Analysis")
    
    vol_col1, vol_col2, vol_col3 = st.columns(3)
    
    # Calculate historical volatility
    try:
        hist_data = yf.download(ticker_input, period='30d', progress=False)
        if not hist_data.empty:
            returns = hist_data['Close'].pct_change().dropna()
            historical_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            # Ensure we have a numeric value
            if hasattr(historical_volatility, 'iloc'):
                historical_volatility = historical_volatility.iloc[0] if len(historical_volatility) > 0 else 0.3
        else:
            historical_volatility = 0.3
    except:
        historical_volatility = 0.3
    
    # Ensure historical_volatility is a number
    if pd.isna(historical_volatility) or not isinstance(historical_volatility, (int, float)):
        historical_volatility = 0.3
    
    with vol_col1:
        st.metric("Historical Vol (30D)", f"{historical_volatility*100:.1f}%")
    
    with vol_col2:
        # Get VIX for market context
        current_vix = None
        if vix_data is not None and not vix_data.empty:
            current_vix = vix_data['VIX_Close'].iloc[-1]
            st.metric("Current VIX", f"{current_vix:.2f}")
        else:
            st.metric("VIX", "N/A")
    
    with vol_col3:
        # Allow volatility override
        volatility_override = st.number_input(
            "Volatility Override:",
            min_value=0.05,
            max_value=2.0,
            value=float(historical_volatility),
            step=0.01,
            format="%.3f",
            help="Override volatility for calculations",
            key="put_vol_override"
        )
    
    # === STRIKE SELECTION ===
    st.markdown("#### ⚙️ Strike Selection")
    
    # Filter liquid puts for manual selection
    liquid_puts = puts_data[
        (puts_data['volume'] > 5) & 
        (puts_data['openInterest'] > 20) &
        (puts_data['bid'] > 0.01)
    ].copy() if not puts_data.empty else pd.DataFrame()
    
    if liquid_puts.empty:
        st.error("No liquid put options found")
        return
    
    manual_col1, manual_col2 = st.columns(2)
    
    available_strikes = sorted(liquid_puts['strike'].unique(), reverse=True)
    
    with manual_col1:
        short_strike = st.selectbox(
            "Short Put Strike:",
            available_strikes,
            index=0,
            key="manual_short_strike"
        )
        
        short_put_data = liquid_puts[liquid_puts['strike'] == short_strike].iloc[0]
        short_premium = (short_put_data['bid'] + short_put_data['ask']) / 2
        st.write(f"Premium: ${short_premium:.2f}")
    
    with manual_col2:
        # Filter long strikes based on spread type
        if spread_type == "Bull Put Spread":
            long_strikes = [s for s in available_strikes if s < short_strike]
        else:
            long_strikes = [s for s in available_strikes if s > short_strike]
        
        if not long_strikes:
            st.error("No suitable long strikes available")
            return
        
        long_strike = st.selectbox(
            "Long Put Strike:",
            long_strikes,
            key="manual_long_strike"
        )
        
        long_put_data = liquid_puts[liquid_puts['strike'] == long_strike].iloc[0]
        long_premium = (long_put_data['bid'] + long_put_data['ask']) / 2
        st.write(f"Premium: ${long_premium:.2f}")
    
    # === COMPREHENSIVE ANALYSIS ===
    st.markdown("#### 📈 Comprehensive Put Spread Analysis")
    
    # Perform detailed analysis
    spread_analysis = analyze_put_spread(
        current_price, short_strike, long_strike, short_premium, long_premium,
        days_to_expiry, volatility_override, 
        'bull' if spread_type == "Bull Put Spread" else 'bear'
    )
    
    # Display key metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Max Profit", f"${spread_analysis['max_profit']:.2f}")
        st.metric("Max Loss", f"${abs(spread_analysis['max_loss']):.2f}")
    
    with metrics_col2:
        st.metric("Breakeven", f"${spread_analysis['breakeven']:.2f}")
        st.metric("Risk/Reward", f"{spread_analysis['risk_reward_ratio']:.2f}")
    
    with metrics_col3:
        st.metric("Prob. Max Profit", f"{spread_analysis['prob_max_profit']*100:.1f}%")
        st.metric("Prob. Breakeven", f"{spread_analysis['prob_breakeven']*100:.1f}%")
    
    with metrics_col4:
        net_premium = spread_analysis['net_premium']
        premium_type = "Credit" if net_premium > 0 else "Debit"
        st.metric(f"Net {premium_type}", f"${abs(net_premium):.2f}")
        st.metric("Spread Width", f"${abs(short_strike - long_strike):.2f}")
    
    # === GREEKS ANALYSIS ===
    st.markdown("#### 🔢 Greeks Analysis")
    
    greeks = spread_analysis['greeks']
    greeks_col1, greeks_col2, greeks_col3, greeks_col4 = st.columns(4)
    
    with greeks_col1:
        st.metric("Delta", f"{greeks['delta']:.3f}")
    with greeks_col2:
        st.metric("Gamma", f"{greeks['gamma']:.4f}")
    with greeks_col3:
        st.metric("Theta", f"{greeks['theta']:.3f}")
    with greeks_col4:
        st.metric("Vega", f"{greeks['vega']:.3f}")
    
    # === TRADE RECOMMENDATION ===
    st.markdown("#### 💡 Trade Recommendation")
    
    # Market condition assessment
    trade_quality = "Unknown"
    trade_color = "blue"
    
    if current_vix is not None:
        if spread_analysis['prob_max_profit'] > 0.6 and spread_analysis['risk_reward_ratio'] > 0.3:
            if current_vix < 25:
                trade_quality = "Excellent"
                trade_color = "green"
            else:
                trade_quality = "Good (High Vol)"
                trade_color = "orange"
        elif spread_analysis['prob_max_profit'] > 0.5 and spread_analysis['risk_reward_ratio'] > 0.2:
            trade_quality = "Fair"
            trade_color = "yellow"
        else:
            trade_quality = "Poor"
            trade_color = "red"
    
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 8px; border-left: 4px solid {trade_color}; 
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">
        <h4>🎯 {spread_type} Recommendation: <span style="color: {trade_color};">{trade_quality}</span></h4>
        <p><strong>Strategy:</strong> {spread_type}</p>
        <p><strong>Strikes:</strong> Short ${short_strike:.2f} / Long ${long_strike:.2f}</p>
        <p><strong>Expected Outcome:</strong> {spread_analysis['prob_max_profit']*100:.1f}% chance of max profit</p>
        <p><strong>Risk Assessment:</strong> Max risk ${abs(spread_analysis['max_loss']):.2f} for max reward ${spread_analysis['max_profit']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # === RISK MANAGEMENT ===
    st.markdown("#### ⚠️ Risk Management Guidelines")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("""
        **Entry Rules:**
        - ✅ VIX < 30 for optimal conditions
        - ✅ Probability of profit > 50%
        - ✅ Risk/Reward ratio > 0.25
        - ✅ Sufficient liquidity in both strikes
        """)
    
    with risk_col2:
        st.markdown("""
        **Exit Rules:**
        - 🎯 Take profits at 50% of max profit
        - ⛔ Cut losses at 200% of credit received
        - ⏰ Close at 21 DTE if profitable
        - 📈 Adjust if underlying moves against position
        """)
    
    # === SENSITIVITY ANALYSIS ===
    with st.expander("🔬 Advanced Sensitivity Analysis"):
        st.markdown("#### Scenario Analysis")
        
        scenarios = [
            ("Bullish (+10%)", current_price * 1.1),
            ("Neutral (0%)", current_price),
            ("Bearish (-10%)", current_price * 0.9),
            ("Crash (-20%)", current_price * 0.8)
        ]
        
        scenario_results = []
        for scenario_name, scenario_price in scenarios:
            # Calculate P&L for scenario
            if spread_type == "Bull Put Spread":
                short_pnl = min(0, scenario_price - short_strike)
                long_pnl = max(0, long_strike - scenario_price)
                total_pnl = spread_analysis['net_premium'] + short_pnl - long_pnl
            else:
                short_pnl = min(0, scenario_price - short_strike)
                long_pnl = max(0, long_strike - scenario_price)
                total_pnl = spread_analysis['net_premium'] - short_pnl + long_pnl
            
            scenario_results.append({
                'Scenario': scenario_name,
                'Stock Price': f"${scenario_price:.2f}",
                'P&L': f"${total_pnl:.2f}",
                'Return %': f"{(total_pnl / abs(spread_analysis['max_loss'])) * 100:.1f}%"
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        st.dataframe(scenario_df, use_container_width=True)
    
    # === AI ANALYSIS SECTION ===
    st.markdown("### 🤖 AI-Powered Put Spread Analysis")
    
    if LLM_AVAILABLE:
        ai_col1, ai_col2 = st.columns([1, 3])
        
        with ai_col1:
            if st.button("🧠 Generate Put Spread AI Analysis", type="primary", key="generate_put_spread_ai"):
                _generate_put_spread_ai_analysis(
                    ticker_input, current_price, selected_expiry, days_to_expiry,
                    spread_type, spread_analysis, volatility_override, current_vix
                )
        
        with ai_col2:
            _display_put_spread_ai_results()
    elif AI_FORMATTER_AVAILABLE:
        # Show placeholder when LLM not available but formatter is
        display_ai_placeholder("Put Spread Analysis")
    else:
        st.warning("⚠️ AI analysis not available - install LLM modules for AI insights")

def _generate_put_spread_ai_analysis(ticker, current_price, expiry_date, days_to_expiry, 
                                   spread_type, spread_analysis, volatility, current_vix):
    """Generate AI analysis for Put Spread strategy"""
    
    if not LLM_AVAILABLE:
        st.warning("⚠️ AI analysis not available - missing LLM modules")
        return
    
    try:
        # Format data for LLM
        put_spread_data = format_put_spread_data_for_llm(
            ticker=ticker,
            current_price=current_price,
            expiry_date=expiry_date,
            days_to_expiry=days_to_expiry,
            spread_type=spread_type,
            spread_analysis=spread_analysis,
            volatility=volatility,
            current_vix=current_vix
        )
        
        # Get LLM analyzer
        analyzer = get_llm_analyzer()
        
        # Generate analysis
        with st.spinner("🧠 AI analyzing Put Spread strategy..."):
            
            analysis_prompt = f"""
            Analyze this Put Spread trading opportunity and provide detailed insights:
            
            {put_spread_data}
            
            Please provide:
            1. Strategy Assessment (Excellent/Good/Fair/Poor) with reasoning
            2. Market Condition Analysis and how it affects this trade
            3. Risk/Reward evaluation and probability assessment
            4. Entry and exit recommendations
            5. Key risks and mitigation strategies
            6. Alternative strategies if this one isn't optimal
            
            Format your response with clear sections and actionable insights.
            """
            
            ai_response = analyzer.generate_analysis(analysis_prompt)
            
            # Store in session state
            st.session_state['put_spread_ai_analysis'] = ai_response
            st.session_state['put_spread_ai_timestamp'] = datetime.now()
            
    except Exception as e:
        st.error(f"❌ AI analysis failed: {str(e)}")

def _display_put_spread_ai_results():
    """Display stored AI analysis results with proper formatting"""
    
    if 'put_spread_ai_analysis' in st.session_state:
        ai_content = st.session_state['put_spread_ai_analysis']
        
        if AI_FORMATTER_AVAILABLE:
            # Use the shared AI formatter for consistent display
            display_ai_analysis(
                ai_content=ai_content,
                analysis_type="Put Spread Analysis",
                tab_color=get_tab_color("Put Spread"),
                analysis_key="put_spread_analysis",
                session_key="put_spread_ai_analysis",
                regenerate_key="regen_put_spread_ai",
                clear_key="clear_put_spread_ai",
                show_debug=True,
                show_metadata=True
            )
        else:
            # Fallback to basic display
            timestamp = st.session_state.get('put_spread_ai_timestamp', datetime.now())
            
            st.markdown("#### 🤖 AI Put Spread Analysis")
            st.markdown(f"*Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
            
            # Display AI analysis in an expandable container
            with st.container():
                st.markdown(ai_content)
            
            # Option to regenerate
            if st.button("🔄 Regenerate AI Analysis", key="regen_put_spread_ai"):
                if 'put_spread_ai_analysis' in st.session_state:
                    del st.session_state['put_spread_ai_analysis']
                st.rerun()
    else:
        if AI_FORMATTER_AVAILABLE:
            display_ai_placeholder(
                analysis_type="Put Spread Analysis",
                features_list=[
                    "Bull/Bear put spread strategy analysis and recommendations",
                    "Probability of profit calculations using Black-Scholes model",
                    "Risk/reward analysis with breakeven calculations",
                    "Greeks analysis and position management guidance",
                    "Market timing recommendations and optimal entry/exit points",
                    "Alternative spread strategies and portfolio optimization"
                ]
            )
        else:
            st.info("🤖 No AI analysis available. Generate analysis using the button above.")

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