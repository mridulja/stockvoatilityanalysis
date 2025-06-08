"""
Advanced Options Strategy Tab for Stock Volatility Analyzer

This module provides comprehensive options analysis including:
- 95% probability analysis framework with statistical modeling
- Black-Scholes options pricing and Greeks calculations
- Advanced strike selection algorithms based on probability targets
- Multi-strategy options analysis (Calls, Puts, Spreads, Strangles, Iron Condors)
- Risk/reward analysis with probability of profit calculations
- VIX-based strategy recommendations
- AI-powered options market analysis and trade recommendations

Author: Enhanced by AI Assistant
Date: 2025
Version: 1.0
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
from scipy.stats import norm
import warnings
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Import shared functions
try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

# Import LLM analysis functionality
try:
    from llm_analysis import get_llm_analyzer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Constants for options calculations
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # 5% risk-free rate (can be made dynamic)

def get_next_friday():
    """Get next Friday's date for weekly options"""
    today = date.today()
    days_ahead = 4 - today.weekday()  # Friday is weekday 4
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return today + timedelta(days=days_ahead)

def get_monthly_expiry():
    """Get next monthly expiry (3rd Friday of next month)"""
    today = date.today()
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month + 1, day=1)
    
    # Find 3rd Friday
    first_day_weekday = next_month.weekday()
    first_friday = 1 + (4 - first_day_weekday) % 7
    third_friday = first_friday + 14
    
    return next_month.replace(day=third_friday)

@st.cache_data(ttl=300)
def fetch_options_data(ticker, expiry_date):
    """Fetch options chain data for a given ticker and expiry"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get options chain
        options_chain = stock.option_chain(expiry_date.strftime('%Y-%m-%d'))
        
        if options_chain.calls.empty and options_chain.puts.empty:
            return None, None
            
        return options_chain.calls, options_chain.puts
    
    except Exception as e:
        st.warning(f"Could not fetch options data for {ticker}: {str(e)}")
        return None, None

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:  # put
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
             r * K * np.exp(-r * T) * norm.cdf(d2 if option_type.lower() == 'call' else -d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def calculate_implied_volatility(market_price, S, K, T, r, option_type='call', max_iterations=100):
    """Calculate implied volatility using Newton-Raphson method"""
    if T <= 0:
        return 0
    
    # Initial guess
    sigma = 0.3
    
    for i in range(max_iterations):
        if option_type.lower() == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)
        
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T) / 100
        
        price_diff = price - market_price
        
        if abs(price_diff) < 0.001:  # Tolerance
            break
            
        if vega == 0:
            break
            
        sigma = sigma - price_diff / (vega * 100)
        
        if sigma <= 0:
            sigma = 0.001
    
    return max(sigma, 0.001)

def calculate_probability_analysis(current_price, target_price, days_to_expiry, volatility):
    """Calculate 95% probability analysis using statistical modeling"""
    if days_to_expiry <= 0:
        return {}
    
    # Convert to annual terms
    time_to_expiry = days_to_expiry / 365.25
    
    # Calculate expected price movement using lognormal distribution
    drift = RISK_FREE_RATE - 0.5 * volatility ** 2
    expected_log_return = drift * time_to_expiry
    std_log_return = volatility * np.sqrt(time_to_expiry)
    
    # Calculate probabilities
    log_ratio = np.log(target_price / current_price)
    z_score = (log_ratio - expected_log_return) / std_log_return
    
    # Probability calculations
    prob_above = 1 - norm.cdf(z_score)
    prob_below = norm.cdf(z_score)
    
    # 95% confidence interval
    z_95 = 1.96
    log_lower_95 = expected_log_return - z_95 * std_log_return
    log_upper_95 = expected_log_return + z_95 * std_log_return
    
    price_lower_95 = current_price * np.exp(log_lower_95)
    price_upper_95 = current_price * np.exp(log_upper_95)
    
    # Expected price
    expected_price = current_price * np.exp(expected_log_return + 0.5 * std_log_return ** 2)
    
    return {
        'expected_price': expected_price,
        'prob_above_target': prob_above,
        'prob_below_target': prob_below,
        'price_lower_95': price_lower_95,
        'price_upper_95': price_upper_95,
        'confidence_interval_95': (price_lower_95, price_upper_95),
        'volatility_used': volatility,
        'days_to_expiry': days_to_expiry
    }

def analyze_option_strategy(current_price, strike_price, option_price, option_type, days_to_expiry, volatility):
    """Analyze an individual option strategy"""
    time_to_expiry = days_to_expiry / 365.25
    
    # Calculate theoretical price and Greeks
    if option_type.lower() == 'call':
        theoretical_price = black_scholes_call(current_price, strike_price, time_to_expiry, RISK_FREE_RATE, volatility)
    else:
        theoretical_price = black_scholes_put(current_price, strike_price, time_to_expiry, RISK_FREE_RATE, volatility)
    
    greeks = calculate_greeks(current_price, strike_price, time_to_expiry, RISK_FREE_RATE, volatility, option_type)
    
    # Calculate implied volatility
    implied_vol = calculate_implied_volatility(option_price, current_price, strike_price, time_to_expiry, RISK_FREE_RATE, option_type)
    
    # Calculate profit/loss scenarios
    if option_type.lower() == 'call':
        breakeven = strike_price + option_price
        max_profit = "Unlimited"
        max_loss = option_price
        prob_profit_target = 1 - norm.cdf(np.log(breakeven / current_price) / (volatility * np.sqrt(time_to_expiry)))
    else:  # put
        breakeven = strike_price - option_price
        max_profit = strike_price - option_price
        max_loss = option_price
        prob_profit_target = norm.cdf(np.log(breakeven / current_price) / (volatility * np.sqrt(time_to_expiry)))
    
    return {
        'theoretical_price': theoretical_price,
        'market_price': option_price,
        'implied_volatility': implied_vol,
        'greeks': greeks,
        'breakeven': breakeven,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'prob_profit': prob_profit_target,
        'intrinsic_value': max(current_price - strike_price, 0) if option_type.lower() == 'call' else max(strike_price - current_price, 0),
        'time_value': option_price - max(current_price - strike_price, 0) if option_type.lower() == 'call' else option_price - max(strike_price - current_price, 0)
    }

def find_optimal_strikes(current_price, calls_data, puts_data, target_delta=None, target_prob=None):
    """Find optimal strike prices based on criteria"""
    if calls_data is None or puts_data is None:
        return {}
    
    recommendations = {}
    
    try:
        # ATM strikes
        atm_call = calls_data.iloc[(calls_data['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts_data.iloc[(puts_data['strike'] - current_price).abs().argsort()[:1]]
        
        if not atm_call.empty and not atm_put.empty:
            recommendations['atm_call'] = {
                'strike': atm_call.iloc[0]['strike'],
                'price': atm_call.iloc[0]['lastPrice'],
                'volume': atm_call.iloc[0].get('volume', 0),
                'openInterest': atm_call.iloc[0].get('openInterest', 0)
            }
            
            recommendations['atm_put'] = {
                'strike': atm_put.iloc[0]['strike'],
                'price': atm_put.iloc[0]['lastPrice'],
                'volume': atm_put.iloc[0].get('volume', 0),
                'openInterest': atm_put.iloc[0].get('openInterest', 0)
            }
        
        # OTM strikes for covered calls and cash-secured puts
        otm_calls = calls_data[calls_data['strike'] > current_price * 1.02].head(5)
        otm_puts = puts_data[puts_data['strike'] < current_price * 0.98].tail(5)
        
        if not otm_calls.empty:
            best_otm_call = otm_calls.iloc[0]
            recommendations['otm_call'] = {
                'strike': best_otm_call['strike'],
                'price': best_otm_call['lastPrice'],
                'volume': best_otm_call.get('volume', 0),
                'openInterest': best_otm_call.get('openInterest', 0)
            }
        
        if not otm_puts.empty:
            best_otm_put = otm_puts.iloc[-1]
            recommendations['otm_put'] = {
                'strike': best_otm_put['strike'],
                'price': best_otm_put['lastPrice'],
                'volume': best_otm_put.get('volume', 0),
                'openInterest': best_otm_put.get('openInterest', 0)
            }
    
    except Exception as e:
        st.warning(f"Error finding optimal strikes: {str(e)}")
    
    return recommendations

def create_options_pnl_chart(current_price, strategy_data, expiry_date):
    """Create P&L chart for options strategy"""
    # Create price range
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    
    fig = go.Figure()
    
    for strategy_name, strategy_info in strategy_data.items():
        pnl_values = []
        
        for price in price_range:
            if strategy_info['type'].lower() == 'call':
                if strategy_info['position'] == 'long':
                    pnl = max(price - strategy_info['strike'], 0) - strategy_info['premium']
                else:  # short
                    pnl = strategy_info['premium'] - max(price - strategy_info['strike'], 0)
            else:  # put
                if strategy_info['position'] == 'long':
                    pnl = max(strategy_info['strike'] - price, 0) - strategy_info['premium']
                else:  # short
                    pnl = strategy_info['premium'] - max(strategy_info['strike'] - price, 0)
            
            pnl_values.append(pnl)
        
        fig.add_trace(go.Scatter(
            x=price_range,
            y=pnl_values,
            mode='lines',
            name=strategy_name,
            line=dict(width=3)
        ))
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", line_color="gray", 
                  annotation_text=f"Current: ${current_price:.2f}")
    
    # Add break-even lines
    fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=f'📊 Options Strategy P&L Analysis - Expiry: {expiry_date}',
        xaxis_title='Stock Price at Expiry ($)',
        yaxis_title='Profit/Loss ($)',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_probability_chart(current_price, prob_analysis):
    """Create probability distribution chart"""
    if not prob_analysis:
        return None
    
    # Create price range around current price
    lower_bound = current_price * 0.7
    upper_bound = current_price * 1.3
    price_range = np.linspace(lower_bound, upper_bound, 200)
    
    # Calculate probability density
    volatility = prob_analysis['volatility_used']
    time_to_expiry = prob_analysis['days_to_expiry'] / 365.25
    
    mean_log_return = (RISK_FREE_RATE - 0.5 * volatility ** 2) * time_to_expiry
    std_log_return = volatility * np.sqrt(time_to_expiry)
    
    log_returns = np.log(price_range / current_price)
    prob_density = norm.pdf(log_returns, mean_log_return, std_log_return) / price_range
    
    fig = go.Figure()
    
    # Add probability density curve
    fig.add_trace(go.Scatter(
        x=price_range,
        y=prob_density,
        mode='lines',
        name='Price Probability Distribution',
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#6366f1', width=3)
    ))
    
    # Add current price
    fig.add_vline(x=current_price, line_dash="solid", line_color="#ef4444", 
                  annotation_text=f"Current: ${current_price:.2f}")
    
    # Add expected price
    fig.add_vline(x=prob_analysis['expected_price'], line_dash="dash", line_color="#10b981",
                  annotation_text=f"Expected: ${prob_analysis['expected_price']:.2f}")
    
    # Add 95% confidence interval
    fig.add_vrect(
        x0=prob_analysis['price_lower_95'], x1=prob_analysis['price_upper_95'],
        fillcolor="rgba(16, 185, 129, 0.2)",
        layer="below", line_width=0,
        annotation_text="95% Confidence"
    )
    
    fig.update_layout(
        title=f'📈 Price Probability Distribution ({prob_analysis["days_to_expiry"]} days)',
        xaxis_title='Stock Price ($)',
        yaxis_title='Probability Density',
        height=400,
        showlegend=True
    )
    
    return fig

def render_options_strategy_tab(results, vix_data, session_tickers):
    """Render the complete Options Strategy tab"""
    
    st.markdown("### 🎯 Advanced Options Strategy Analysis")
    st.markdown("*Comprehensive options analysis with 95% probability modeling and Greeks calculations*")
    
    # === TICKER SELECTION ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy_ticker = st.selectbox(
            "Select ticker for options analysis:",
            session_tickers,
            key="options_strategy_ticker",
            help="Choose a ticker for comprehensive options analysis"
        )
    
    with col2:
        # Get current price
        current_price = None
        if strategy_ticker:
            try:
                stock = yf.Ticker(strategy_ticker)
                hist = stock.history(period='1d')
                current_price = hist['Close'].iloc[-1] if not hist.empty else None
            except:
                current_price = None
        
        if current_price:
            st.metric("Current Price", f"${current_price:.2f}")
        else:
            st.error("Could not fetch current price")
            return
    
    # === EXPIRY SELECTION ===
    st.markdown("#### ⏰ Expiry Selection")
    
    expiry_col1, expiry_col2, expiry_col3 = st.columns(3)
    
    with expiry_col1:
        expiry_type = st.selectbox(
            "Expiry Type:",
            ["Next Friday", "Monthly", "Custom"],
            help="Select expiration timeframe"
        )
    
    with expiry_col2:
        if expiry_type == "Next Friday":
            selected_expiry = get_next_friday()
        elif expiry_type == "Monthly":
            selected_expiry = get_monthly_expiry()
        else:  # Custom
            selected_expiry = st.date_input(
                "Custom Expiry:",
                value=get_next_friday(),
                min_value=date.today() + timedelta(days=1)
            )
        
        st.write(f"**Selected:** {selected_expiry}")
    
    with expiry_col3:
        days_to_expiry = (selected_expiry - date.today()).days
        st.metric("Days to Expiry", days_to_expiry)
    
    if days_to_expiry <= 0:
        st.error("Please select a future expiry date")
        return
    
    # === FETCH OPTIONS DATA ===
    with st.spinner(f"Fetching options data for {strategy_ticker}..."):
        calls_data, puts_data = fetch_options_data(strategy_ticker, selected_expiry)
    
    if calls_data is None or puts_data is None:
        st.error(f"No options data available for {strategy_ticker} on {selected_expiry}")
        return
    
    # === VOLATILITY ANALYSIS ===
    st.markdown("#### 📊 Volatility Analysis")
    
    vol_col1, vol_col2, vol_col3 = st.columns(3)
    
    # Calculate historical volatility
    try:
        hist_data = yf.download(strategy_ticker, period='30d', progress=False)
        if not hist_data.empty:
            returns = hist_data['Close'].pct_change().dropna()
            historical_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            # Ensure we have a numeric value, not a Series
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
        st.metric("Historical Volatility (30D)", f"{historical_volatility*100:.1f}%")
    
    with vol_col2:
        # Get VIX for market volatility context
        current_vix = None
        if vix_data is not None and not vix_data.empty:
            current_vix = vix_data['VIX_Close'].iloc[-1]
            st.metric("Current VIX", f"{current_vix:.2f}")
        else:
            st.metric("VIX", "N/A")
    
    with vol_col3:
        # Allow user to override volatility
        volatility_override = st.number_input(
            "Volatility Override:",
            min_value=0.05,
            max_value=2.0,
            value=historical_volatility,
            step=0.01,
            format="%.3f",
            help="Override volatility for calculations"
        )
    
    # === 95% PROBABILITY ANALYSIS ===
    st.markdown("#### 📈 95% Probability Analysis")
    
    # User can set target price for probability calculations
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        target_price = st.number_input(
            "Target Price for Probability Analysis:",
            min_value=current_price * 0.5,
            max_value=current_price * 2.0,
            value=current_price,
            step=0.01,
            format="%.2f"
        )
    
    with prob_col2:
        prob_analysis = calculate_probability_analysis(current_price, target_price, days_to_expiry, volatility_override)
        
        if prob_analysis:
            st.metric(
                f"Probability Above ${target_price:.2f}",
                f"{prob_analysis['prob_above_target']*100:.1f}%"
            )
    
    # Display 95% confidence interval
    if prob_analysis:
        conf_col1, conf_col2, conf_col3 = st.columns(3)
        
        with conf_col1:
            st.metric("Expected Price", f"${prob_analysis['expected_price']:.2f}")
        with conf_col2:
            st.metric("95% CI Lower", f"${prob_analysis['price_lower_95']:.2f}")
        with conf_col3:
            st.metric("95% CI Upper", f"${prob_analysis['price_upper_95']:.2f}")
        
        # Create probability chart
        prob_chart = create_probability_chart(current_price, prob_analysis)
        if prob_chart:
            st.plotly_chart(prob_chart, use_container_width=True)
    
    # === OPTIMAL STRIKES ANALYSIS ===
    st.markdown("#### 🎯 Optimal Strike Analysis")
    
    optimal_strikes = find_optimal_strikes(current_price, calls_data, puts_data)
    
    if optimal_strikes:
        strike_col1, strike_col2 = st.columns(2)
        
        with strike_col1:
            st.markdown("**📞 Recommended Calls:**")
            
            if 'atm_call' in optimal_strikes:
                st.markdown(f"""
                **ATM Call**: ${optimal_strikes['atm_call']['strike']:.2f}
                - Price: ${optimal_strikes['atm_call']['price']:.2f}
                - Volume: {optimal_strikes['atm_call']['volume']:,}
                - OI: {optimal_strikes['atm_call']['openInterest']:,}
                """)
            
            if 'otm_call' in optimal_strikes:
                st.markdown(f"""
                **OTM Call**: ${optimal_strikes['otm_call']['strike']:.2f}
                - Price: ${optimal_strikes['otm_call']['price']:.2f}
                - Volume: {optimal_strikes['otm_call']['volume']:,}
                - OI: {optimal_strikes['otm_call']['openInterest']:,}
                """)
        
        with strike_col2:
            st.markdown("**📉 Recommended Puts:**")
            
            if 'atm_put' in optimal_strikes:
                st.markdown(f"""
                **ATM Put**: ${optimal_strikes['atm_put']['strike']:.2f}
                - Price: ${optimal_strikes['atm_put']['price']:.2f}
                - Volume: {optimal_strikes['atm_put']['volume']:,}
                - OI: {optimal_strikes['atm_put']['openInterest']:,}
                """)
            
            if 'otm_put' in optimal_strikes:
                st.markdown(f"""
                **OTM Put**: ${optimal_strikes['otm_put']['strike']:.2f}
                - Price: ${optimal_strikes['otm_put']['price']:.2f}
                - Volume: {optimal_strikes['otm_put']['volume']:,}
                - OI: {optimal_strikes['otm_put']['openInterest']:,}
                """)
    
    # === STRATEGY ANALYZER ===
    st.markdown("#### 🔍 Strategy Analyzer")
    
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        selected_strategy = st.selectbox(
            "Select Strategy to Analyze:",
            ["Long Call", "Long Put", "Covered Call", "Cash-Secured Put", "Custom"],
            help="Choose an options strategy for detailed analysis"
        )
    
    with strategy_col2:
        if selected_strategy != "Custom":
            if selected_strategy in ["Long Call", "Covered Call"] and 'atm_call' in optimal_strikes:
                selected_strike = optimal_strikes['atm_call']['strike']
                selected_premium = optimal_strikes['atm_call']['price']
                option_type = 'call'
            elif selected_strategy in ["Long Put", "Cash-Secured Put"] and 'atm_put' in optimal_strikes:
                selected_strike = optimal_strikes['atm_put']['strike']
                selected_premium = optimal_strikes['atm_put']['price']
                option_type = 'put'
            else:
                selected_strike = current_price
                selected_premium = 1.0
                option_type = 'call'
            
            st.metric("Selected Strike", f"${selected_strike:.2f}")
            st.metric("Premium", f"${selected_premium:.2f}")
        else:
            selected_strike = st.number_input("Strike Price:", value=current_price, step=0.50)
            selected_premium = st.number_input("Option Premium:", value=1.0, step=0.05)
            option_type = st.selectbox("Option Type:", ["call", "put"])
    
    # Analyze selected strategy
    if selected_strike and selected_premium:
        strategy_analysis = analyze_option_strategy(
            current_price, selected_strike, selected_premium, 
            option_type, days_to_expiry, volatility_override
        )
        
        if strategy_analysis:
            st.markdown("##### 📊 Strategy Analysis Results")
            
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                st.metric("Theoretical Price", f"${strategy_analysis['theoretical_price']:.2f}")
                st.metric("Breakeven", f"${strategy_analysis['breakeven']:.2f}")
                st.metric("Probability of Profit", f"{strategy_analysis['prob_profit']*100:.1f}%")
            
            with analysis_col2:
                st.metric("Implied Volatility", f"{strategy_analysis['implied_volatility']*100:.1f}%")
                st.metric("Intrinsic Value", f"${strategy_analysis['intrinsic_value']:.2f}")
                st.metric("Time Value", f"${strategy_analysis['time_value']:.2f}")
            
            with analysis_col3:
                st.metric("Max Loss", f"${strategy_analysis['max_loss']:.2f}")
                if isinstance(strategy_analysis['max_profit'], str):
                    st.metric("Max Profit", strategy_analysis['max_profit'])
                else:
                    st.metric("Max Profit", f"${strategy_analysis['max_profit']:.2f}")
            
            # Greeks display
            st.markdown("##### 🔢 Greeks Analysis")
            greeks = strategy_analysis['greeks']
            
            greeks_col1, greeks_col2, greeks_col3, greeks_col4, greeks_col5 = st.columns(5)
            
            with greeks_col1:
                st.metric("Delta", f"{greeks['delta']:.3f}")
            with greeks_col2:
                st.metric("Gamma", f"{greeks['gamma']:.4f}")
            with greeks_col3:
                st.metric("Theta", f"{greeks['theta']:.3f}")
            with greeks_col4:
                st.metric("Vega", f"{greeks['vega']:.3f}")
            with greeks_col5:
                st.metric("Rho", f"{greeks['rho']:.3f}")
    
    # === AI ANALYSIS SECTION ===
    st.markdown("### 🤖 AI-Powered Options Analysis")
    
    if LLM_AVAILABLE and AI_FORMATTER_AVAILABLE:
        ai_col1, ai_col2 = st.columns([1, 3])
        
        with ai_col1:
            if st.button("🧠 Generate Options AI Analysis", type="primary", key="generate_options_ai"):
                _generate_options_ai_analysis(
                    strategy_ticker, current_price, selected_expiry, days_to_expiry,
                    volatility_override, prob_analysis, optimal_strikes, 
                    strategy_analysis if 'strategy_analysis' in locals() else {},
                    current_vix
                )
        
        with ai_col2:
            _display_options_ai_results()
    
    elif AI_FORMATTER_AVAILABLE:
        st.info("🤖 AI analysis requires the LLM analyzer. Please ensure llm_analysis.py is available.")
    else:
        st.info("🤖 AI analysis requires the unified AI formatter. Please ensure shared/ai_formatter.py is available.")
    
    # === OPTIONS EDUCATION SECTION ===
    with st.expander("📚 Options Strategy Guide"):
        st.markdown("""
        ### 🎯 Options Strategy Quick Reference
        
        **Basic Strategies:**
        
        **📞 Long Call**
        - Bullish strategy with unlimited upside potential
        - Max Loss: Premium paid
        - Breakeven: Strike + Premium
        - Best when: Expecting significant upward movement
        
        **📉 Long Put**
        - Bearish strategy with high profit potential
        - Max Loss: Premium paid
        - Breakeven: Strike - Premium
        - Best when: Expecting significant downward movement
        
        **🛡️ Covered Call**
        - Income generation on existing stock holdings
        - Max Profit: Strike - Stock Cost + Premium
        - Best when: Neutral to slightly bullish
        
        **💰 Cash-Secured Put**
        - Income generation while waiting to buy stock
        - Max Profit: Premium received
        - Best when: Willing to own stock at strike price
        
        **📊 Probability Guidelines:**
        - 95% Confidence Interval: Price likely to stay within range
        - High Implied Volatility: Options expensive, consider selling
        - Low Implied Volatility: Options cheap, consider buying
        - Time Decay: Accelerates in final 30 days
        
        **🔢 Greeks Interpretation:**
        - **Delta**: Price sensitivity (0.50 = 50¢ move per $1 stock move)
        - **Gamma**: Delta acceleration (how fast delta changes)
        - **Theta**: Time decay (daily option value loss)
        - **Vega**: Volatility sensitivity
        - **Rho**: Interest rate sensitivity
        """)

def _generate_options_ai_analysis(ticker, current_price, expiry_date, days_to_expiry, volatility, 
                                prob_analysis, optimal_strikes, strategy_analysis, current_vix):
    """Generate AI analysis for options strategy"""
    
    try:
        llm_analyzer = get_llm_analyzer()
        
        # Prepare options data for AI
        prob_info = ""
        if prob_analysis:
            prob_info = f"""
            - Expected Price: ${prob_analysis['expected_price']:.2f}
            - 95% Confidence Range: ${prob_analysis['price_lower_95']:.2f} - ${prob_analysis['price_upper_95']:.2f}
            - Probability Above Current: {prob_analysis['prob_above_target']*100:.1f}%
            """
        
        strikes_info = ""
        if optimal_strikes:
            strikes_info = "Optimal Strikes:\n"
            for key, strike_data in optimal_strikes.items():
                strikes_info += f"- {key}: ${strike_data['strike']:.2f} (${strike_data['price']:.2f})\n"
        
        strategy_info = ""
        if strategy_analysis:
            strategy_info = f"""
            Strategy Analysis:
            - Theoretical Price: ${strategy_analysis.get('theoretical_price', 0):.2f}
            - Implied Volatility: {strategy_analysis.get('implied_volatility', 0)*100:.1f}%
            - Probability of Profit: {strategy_analysis.get('prob_profit', 0)*100:.1f}%
            - Breakeven: ${strategy_analysis.get('breakeven', 0):.2f}
            """
        
        vix_context = f"Current VIX: {current_vix:.2f}" if current_vix else "VIX data unavailable"
        
        analysis_prompt = f"""
        Provide comprehensive options trading analysis and recommendations:

        **Market Context:**
        - Ticker: {ticker}
        - Current Price: ${current_price:.2f}
        - Expiry Date: {expiry_date}
        - Days to Expiry: {days_to_expiry}
        - Historical Volatility: {volatility*100:.1f}%
        - {vix_context}
        
        **Probability Analysis:**
        {prob_info}
        
        **Options Data:**
        {strikes_info}
        
        **Strategy Analysis:**
        {strategy_info}

        Please provide:
        1. **Market Assessment** - What do current options metrics tell us about market expectations?
        2. **Strategy Recommendations** - Which options strategies are most attractive given current conditions?
        3. **Risk Analysis** - What are the key risks and how to manage them?
        4. **Volatility Analysis** - Is implied volatility high/low relative to historical? Trade implications?
        5. **Timing Considerations** - Optimal entry/exit timing given time decay and market conditions?
        6. **Alternative Strategies** - Other options strategies to consider for this setup?

        Focus on practical, actionable advice for options traders with specific entry/exit criteria.
        """
        
        # Generate AI response
        ai_response = None
        for method_name in ['analyze', 'get_analysis', 'generate_analysis', 'chat', 'query']:
            if hasattr(llm_analyzer, method_name):
                ai_response = getattr(llm_analyzer, method_name)(analysis_prompt)
                break
        
        if not ai_response and callable(llm_analyzer):
            ai_response = llm_analyzer(analysis_prompt)
        
        if ai_response:
            # Store in session state
            if 'ai_options_analysis' not in st.session_state:
                st.session_state.ai_options_analysis = {}
            
            analysis_key = f"options_analysis_{ticker}_{expiry_date}_{current_price:.2f}"
            st.session_state.ai_options_analysis[analysis_key] = ai_response
            st.success("✅ Options AI Analysis completed!")
        else:
            st.warning("⚠️ AI Analysis could not be generated. Check LLM configuration.")
            
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        st.info("💡 AI analysis requires proper LLM configuration.")

def _display_options_ai_results():
    """Display AI options analysis results"""
    
    if 'ai_options_analysis' in st.session_state and st.session_state.ai_options_analysis:
        # Get the latest analysis
        latest_key = list(st.session_state.ai_options_analysis.keys())[-1]
        ai_content = st.session_state.ai_options_analysis[latest_key]
        
        if AI_FORMATTER_AVAILABLE:
            display_ai_analysis(
                ai_content=ai_content,
                analysis_type="Options Strategy Analysis",
                tab_color=get_tab_color("options"),
                analysis_key=latest_key,
                session_key="ai_options_analysis",
                regenerate_key="regenerate_options_ai",
                clear_key="clear_options_ai",
                show_debug=True,
                show_metadata=True
            )
        else:
            st.markdown("#### 🧠 AI Options Analysis Results")
            content_text = str(ai_content.get('content', ai_content)) if isinstance(ai_content, dict) else str(ai_content)
            st.markdown(content_text)
    
    else:
        if AI_FORMATTER_AVAILABLE:
            display_ai_placeholder(
                analysis_type="Options Strategy Analysis",
                features_list=[
                    "Comprehensive options strategy recommendations based on market conditions",
                    "Volatility analysis and implied vs. historical volatility comparison",
                    "Probability-based trade recommendations with risk/reward analysis",
                    "Greeks analysis and position management guidance",
                    "Market timing recommendations and optimal entry/exit points",
                    "Alternative strategy suggestions and portfolio optimization"
                ]
            )
        else:
            st.info("👆 Click 'Generate Options AI Analysis' to get intelligent options trading insights and strategy recommendations") 