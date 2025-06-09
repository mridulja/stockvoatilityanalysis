"""
Iron Condor Playbook Tab for Streamlit Stock Analysis Application

This module provides comprehensive Iron Condor analysis including:
- Real-time options data analysis with yfinance
- Multiple probability calculation methods (Delta, Credit/Width, Black-Scholes)
- Strategy classification (Bread & Butter, Big Boy, Chicken IC, Conservative)
- IV rank calculations for market condition assessment
- Strike selection using delta targeting and price-based fallbacks
- Interactive P&L diagrams and strategy comparison charts
- AI-powered analysis and recommendations

Author: Mridul jain. 
Enhanced Stock Analysis System
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
RISK_FREE_RATE = 0.05

# Check for AI analysis availability
try:
    from llm_analysis import get_llm_analyzer, format_vix_data_for_llm
    from llm_input_formatters import format_iron_condor_data_for_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

# Check for Iron Condor modules
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

def get_next_friday():
    """Get next Friday date"""
    today = date.today()
    days_ahead = 4 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)

def get_monthly_expiry():
    """Get next monthly options expiry (3rd Friday)"""
    today = date.today()
    first_day = today.replace(day=1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    
    if third_friday <= today:
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_friday = next_month + timedelta(days=(4 - next_month.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
    
    return third_friday

@st.cache_data(ttl=300)
def get_current_price(ticker):
    """Get current/latest price for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d', interval='1m')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        else:
            hist = stock.history(period='5d')
            return hist['Close'].iloc[-1] if not hist.empty else None
    except:
        return None

@st.cache_data(ttl=300)
def validate_ticker(ticker):
    """Validate if a ticker exists and can be traded"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='5d')
        
        if hist.empty or not info:
            return False, "No data available"
        
        if 'longName' in info or 'shortName' in info:
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            return True, f"${current_price:.2f}" if current_price else "Valid"
        else:
            return False, "Invalid ticker"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

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

def _generate_iron_condor_ai_analysis(ticker, current_price, expiry_date, days_to_expiry, 
                                    strategy_focus, selected_ic, market_conditions, 
                                    historical_volatility, current_vix):
    """Generate AI analysis for Iron Condor strategy"""
    
    if not LLM_AVAILABLE:
        st.warning("⚠️ AI analysis not available - missing LLM modules")
        return
    
    try:
        # Format data for LLM
        ic_data = format_iron_condor_data_for_llm(
            ticker=ticker,
            current_price=current_price,
            expiry_date=expiry_date,
            days_to_expiry=days_to_expiry,
            strategy_focus=strategy_focus,
            iron_condor=selected_ic,
            market_conditions=market_conditions,
            historical_vol=historical_volatility,
            current_vix=current_vix
        )
        
        # Get LLM analyzer
        analyzer = get_llm_analyzer()
        
        # Generate analysis
        with st.spinner("🧠 AI analyzing Iron Condor strategy..."):
            
            analysis_prompt = f"""
            Analyze this Iron Condor trading opportunity and provide detailed insights:
            
            {ic_data}
            
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
            st.session_state['ic_ai_analysis'] = ai_response
            st.session_state['ic_ai_timestamp'] = datetime.now()
            
    except Exception as e:
        st.error(f"❌ AI analysis failed: {str(e)}")

def _display_iron_condor_ai_results():
    """Display stored AI analysis results with proper formatting"""
    
    if 'ic_ai_analysis' in st.session_state:
        ai_content = st.session_state['ic_ai_analysis']
        
        if AI_FORMATTER_AVAILABLE:
            # Use the shared AI formatter for consistent display
            display_ai_analysis(
                ai_content=ai_content,
                analysis_type="Iron Condor Analysis",
                tab_color=get_tab_color("Iron Condor"),
                analysis_key="iron_condor_analysis",
                session_key="ic_ai_analysis",
                regenerate_key="regen_ic_ai",
                clear_key="clear_ic_ai",
                show_debug=True,
                show_metadata=True
            )
        else:
            # Fallback to basic display
            timestamp = st.session_state.get('ic_ai_timestamp', datetime.now())
            
            st.markdown("#### 🤖 AI Iron Condor Analysis")
            st.markdown(f"*Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
            
            # Display AI analysis in an expandable container
            with st.container():
                st.markdown(ai_content)
            
            # Option to regenerate
            if st.button("🔄 Regenerate AI Analysis", key="regen_ic_ai"):
                if 'ic_ai_analysis' in st.session_state:
                    del st.session_state['ic_ai_analysis']
                st.rerun()
    else:
        if AI_FORMATTER_AVAILABLE:
            display_ai_placeholder(
                analysis_type="Iron Condor Analysis", 
                features_list=[
                    "Comprehensive Iron Condor strategy analysis and optimization",
                    "Multiple probability calculation methods (Delta, Black-Scholes, Credit/Width)",
                    "Strategy classification and risk/reward analysis",
                    "Market condition assessment with IV rank calculations",
                    "Entry and exit timing recommendations with Greeks analysis",
                    "Alternative strategies and portfolio management guidance"
                ]
            )
        else:
            st.info("🤖 No AI analysis available. Generate analysis using the button above.")

def render_iron_condor_playbook_tab(results, vix_data, session_tickers):
    """Render the Iron Condor Trading Playbook tab"""
    
    st.markdown("## 🦅 Iron Condor Trading Playbook")
    st.markdown("*Complete Iron Condor trading system with probability analysis*")
    
    st.info("Iron Condor tab functionality will be implemented here")

    # === TICKER SELECTION ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Universal ticker input
        ticker_input = st.text_input(
            "🎯 Enter Stock/ETF Symbol:",
            value="SPY",
            help="Enter any valid stock or ETF symbol (e.g., AAPL, SPY, QQQ, TSLA)",
            key="iron_condor_ticker"
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
        if "iron_condor_expiry_type_value" not in st.session_state:
            st.session_state.iron_condor_expiry_type_value = "Next Friday"
        
        expiry_type = st.selectbox(
            "Expiry Type:",
            ["Next Friday", "Monthly Expiry", "Custom Date"],
            index=["Next Friday", "Monthly Expiry", "Custom Date"].index(st.session_state.iron_condor_expiry_type_value),
            key="iron_condor_expiry_type",
            on_change=lambda: setattr(st.session_state, 'iron_condor_expiry_type_value', st.session_state.iron_condor_expiry_type)
        )
    
    with expiry_col2:
        if expiry_type == "Next Friday":
            if IRON_CONDOR_AVAILABLE:
                selected_expiry = ic_get_next_friday()
            else:
                selected_expiry = get_next_friday()
        elif expiry_type == "Monthly Expiry":
            if IRON_CONDOR_AVAILABLE:
                selected_expiry = get_next_monthly_expiry()
            else:
                selected_expiry = get_monthly_expiry()
        else:  # Custom Date
            min_date = date.today() + timedelta(days=1)
            max_date = date.today() + timedelta(days=365)
            selected_expiry = st.date_input(
                "Select Custom Date:",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="iron_condor_custom_date"
            )
        
        st.write(f"**Selected:** {selected_expiry}")
    
    # Convert selected_expiry to date object if it's a string (do this before using it anywhere)
    if isinstance(selected_expiry, str):
        try:
            selected_expiry = datetime.strptime(selected_expiry, '%Y-%m-%d').date()
        except ValueError:
            st.error("Invalid date format")
            return
    
    with expiry_col3:
        days_to_expiry = (selected_expiry - date.today()).days
        st.metric("Days to Expiry", f"{days_to_expiry} days")
    
    # === STRATEGY FOCUS ===
    st.markdown("#### 🎯 Strategy Focus")
    
    focus_col1, focus_col2 = st.columns(2)
    
    with focus_col1:
        # Initialize strategy focus in session state if not exists
        if "ic_strategy_focus_value" not in st.session_state:
            st.session_state.ic_strategy_focus_value = "Balanced"
        
        strategy_focus = st.selectbox(
            "Strategy Focus:",
            ["Balanced", "High Probability", "Earnings Play", "Conservative"],
            index=["Balanced", "High Probability", "Earnings Play", "Conservative"].index(st.session_state.ic_strategy_focus_value),
            help="Choose strategy focus based on market outlook and risk tolerance",
            key="ic_strategy_focus",
            on_change=lambda: setattr(st.session_state, 'ic_strategy_focus_value', st.session_state.ic_strategy_focus)
        )
    
    with focus_col2:
        # Display focus explanation
        focus_explanations = {
            "Balanced": "Equal risk/reward balance, moderate probability",
            "High Probability": "Higher probability of success, lower premium",
            "Earnings Play": "Wider spreads for earnings volatility",
            "Conservative": "Lower risk, higher probability trades"
        }
        st.info(f"📋 {focus_explanations[strategy_focus]}")
    
    # === MARKET CONDITION ASSESSMENT ===
    st.markdown("#### 📊 Market Condition Assessment")
    
    market_col1, market_col2, market_col3 = st.columns(3)
    
    # VIX Analysis
    with market_col1:
        current_vix = None
        if vix_data is not None and not vix_data.empty:
            current_vix = vix_data['VIX_Close'].iloc[-1]
            condition, condition_class, icon = get_vix_condition(current_vix)
            st.metric("Current VIX", f"{current_vix:.2f}")
            st.markdown(f"{icon} {condition}")
        else:
            st.metric("VIX", "N/A")
    
    # IV Rank Estimation
    with market_col2:
        try:
            # Calculate historical volatility
            hist_data = yf.download(ticker_input, period='252d', progress=False)
            if not hist_data.empty:
                returns = hist_data['Close'].pct_change().dropna()
                historical_volatility = returns.std() * np.sqrt(252)
                
                # Ensure we have a numeric value
                if hasattr(historical_volatility, 'iloc'):
                    historical_volatility = historical_volatility.iloc[0] if len(historical_volatility) > 0 else 0.3
                
                # Estimate IV rank based on VIX and historical vol
                if current_vix:
                    implied_vol_estimate = current_vix / 100 * 1.2  # Rough estimate
                    iv_rank = min(100, max(0, (implied_vol_estimate / historical_volatility - 0.5) * 100))
                else:
                    iv_rank = 50  # Default
                
                st.metric("Est. IV Rank", f"{iv_rank:.0f}%")
                
                if iv_rank > 70:
                    st.success("🟢 High IV - Sell premium")
                elif iv_rank > 30:
                    st.warning("🟡 Medium IV - Neutral")
                else:
                    st.error("🔴 Low IV - Avoid selling")
            else:
                st.metric("IV Rank", "N/A")
                historical_volatility = 0.3
        except:
            st.metric("IV Rank", "N/A")
            historical_volatility = 0.3
    
    # Ensure historical_volatility is a number
    if pd.isna(historical_volatility) or not isinstance(historical_volatility, (int, float)):
        historical_volatility = 0.3
    
    # Trade Recommendation
    with market_col3:
        trade_ok, trade_msg = should_trade(current_vix) if current_vix else (True, "Monitor closely")
        
        if trade_ok:
            st.success("✅ Trading Approved")
        else:
            st.error("⛔ Avoid Trading")
        
        st.write(trade_msg)
    
    # === IRON CONDOR ANALYSIS BUTTON ===
    st.markdown("#### 🔄 Iron Condor Analysis")
    
    if st.button("🚀 Analyze Iron Condor Opportunities", type="primary", key="analyze_iron_condor"):
        if not IRON_CONDOR_AVAILABLE:
            st.error("❌ Iron Condor analysis module not available")
            st.info("Install iron_condor_analysis.py and iron_condor_charts.py for full functionality")
            return
        
        with st.spinner("Analyzing Iron Condor opportunities..."):
            try:
                # Initialize Iron Condor analyzer
                analyzer = IronCondorAnalyzer()
                
                # Analyze Iron Condor strategies (this method does everything - fetches data and analyzes)
                analysis_results = analyzer.analyze_iron_condor_strategies(
                    ticker_input, 
                    selected_expiry.strftime('%Y-%m-%d')
                )
                
                if analysis_results and analysis_results['strategies']:
                    ic_strategies = analysis_results['strategies']
                    st.success(f"✅ Found {len(ic_strategies)} Iron Condor opportunities")
                    
                    # Store in session state
                    st.session_state[f'ic_strategies_{ticker_input}'] = ic_strategies
                    st.session_state[f'ic_analysis_{ticker_input}'] = analysis_results
                    st.session_state[f'ic_current_price_{ticker_input}'] = analysis_results['current_price']
                else:
                    st.warning("⚠️ No suitable Iron Condor opportunities found")
                    return
                    
            except Exception as e:
                st.error(f"❌ Error analyzing Iron Condor: {str(e)}")
                return
    
    # Check if we have Iron Condor strategies
    ic_strategies_key = f'ic_strategies_{ticker_input}'
    if ic_strategies_key not in st.session_state:
        st.info("👆 Click 'Analyze Iron Condor Opportunities' to begin")
        return
    
    ic_strategies = st.session_state[ic_strategies_key]
    
    # === STRATEGY SELECTION ===
    st.markdown("#### 🎯 Iron Condor Strategy Selection")
    
    # Display strategies in a table
    strategy_df = pd.DataFrame(ic_strategies)
    
    # Map the actual field names from the analyzer results
    display_data = []
    for strategy in ic_strategies:
        display_data.append({
            'Type': strategy.get('strategy_type', 'Standard'),
            'Call Short': strategy.get('call_short', 0),
            'Call Long': strategy.get('call_long', 0),
            'Put Short': strategy.get('put_short', 0),
            'Put Long': strategy.get('put_long', 0),
            'Net Credit': strategy.get('total_credit', 0),
            'Max Profit': strategy.get('max_profit', 0),
            'Max Loss': strategy.get('max_loss', 0),
            'POP %': strategy.get('pop_black_scholes', 0) * 100
        })
    
    display_df = pd.DataFrame(display_data)
    st.dataframe(
        display_df.round(2),
        use_container_width=True,
        height=200
    )
    
    # Select strategy for detailed analysis
    selected_strategy_idx = st.selectbox(
        "Select Iron Condor for detailed analysis:",
        range(len(ic_strategies)),
        format_func=lambda x: f"#{x+1}: {ic_strategies[x].get('strategy_type', 'Standard')} - Credit: ${ic_strategies[x].get('total_credit', 0):.2f} - POP: {ic_strategies[x].get('pop_black_scholes', 0)*100:.1f}%",
        key="selected_ic_strategy"
    )
    
    selected_ic = ic_strategies[selected_strategy_idx]
    
    # === DETAILED ANALYSIS ===
    st.markdown("#### 📈 Detailed Iron Condor Analysis")
    
    # Key metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        if IRON_CONDOR_AVAILABLE:
            st.metric("Net Credit", ic_format_currency(selected_ic.get('total_credit', 0)))
            st.metric("Max Profit", ic_format_currency(selected_ic.get('max_profit', 0)))
        else:
            st.metric("Net Credit", f"${selected_ic.get('total_credit', 0):.2f}")
            st.metric("Max Profit", f"${selected_ic.get('max_profit', 0):.2f}")
    
    with metrics_col2:
        if IRON_CONDOR_AVAILABLE:
            st.metric("Max Loss", ic_format_currency(selected_ic.get('max_loss', 0)))
        else:
            st.metric("Max Loss", f"${selected_ic.get('max_loss', 0):.2f}")
        st.metric("Risk/Reward", f"{selected_ic.get('risk_reward_ratio', 0):.2f}")
    
    with metrics_col3:
        if IRON_CONDOR_AVAILABLE:
            st.metric("Probability of Profit", ic_format_percentage(selected_ic.get('pop_black_scholes', 0)))
        else:
            st.metric("Probability of Profit", f"{selected_ic.get('pop_black_scholes', 0)*100:.1f}%")
        st.metric("Breakeven Width", f"${selected_ic.get('upper_breakeven', 0) - selected_ic.get('lower_breakeven', 0):.2f}")
    
    with metrics_col4:
        st.metric("Wing Width", f"${selected_ic.get('wing_width', 0):.0f}")
        st.metric("Days to Expiry", f"{selected_ic.get('dte', days_to_expiry)} days")
    
    # === VISUALIZATION ===
    st.markdown("#### 📊 Iron Condor Visualization")
    
    try:
        # P&L Chart
        if IRON_CONDOR_AVAILABLE:
            pnl_chart = create_iron_condor_pnl_chart(selected_ic, current_price)
            if pnl_chart:
                st.plotly_chart(pnl_chart, use_container_width=True)
        else:
            st.info("📈 P&L visualization requires iron_condor_charts.py module")
    except Exception as e:
        st.error(f"Could not create P&L chart: {str(e)}")
    
    # === AI ANALYSIS SECTION ===
    st.markdown("#### 🤖 AI-Powered Iron Condor Analysis")
    
    if LLM_AVAILABLE:
        ai_col1, ai_col2 = st.columns([1, 3])
        
        with ai_col1:
            if st.button("🧠 Generate AI Analysis", type="primary", key="generate_ic_ai"):
                market_conditions = {
                    'vix': current_vix,
                    'iv_rank': iv_rank if 'iv_rank' in locals() else None,
                    'trade_approved': trade_ok
                }
                
                _generate_iron_condor_ai_analysis(
                    ticker_input, current_price, selected_expiry, days_to_expiry,
                    strategy_focus, selected_ic, market_conditions,
                    historical_volatility, current_vix
                )
        
        with ai_col2:
            _display_iron_condor_ai_results()
    elif AI_FORMATTER_AVAILABLE:
        # Show placeholder when LLM not available but formatter is
        display_ai_placeholder("Iron Condor Analysis")
    else:
        st.warning("⚠️ AI analysis not available - install LLM modules for AI insights")
    
    # === RECOMMENDATION ===
    st.markdown("#### 💡 Iron Condor Recommendation")
    
    # Assess trade quality
    pop = selected_ic.get('pop_black_scholes', 0) * 100
    risk_reward = selected_ic.get('risk_reward_ratio', 0)
    
    if pop > 65 and risk_reward > 0.25:
        if current_vix and current_vix < 25:
            trade_quality = "Excellent"
            trade_color = "green"
        else:
            trade_quality = "Good"
            trade_color = "orange"
    elif pop > 55 and risk_reward > 0.2:
        trade_quality = "Fair"
        trade_color = "yellow"
    else:
        trade_quality = "Poor"
        trade_color = "red"
    
    st.markdown(f"""
    <div style="padding: 1.5rem; border-radius: 12px; border-left: 4px solid {trade_color}; 
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">
        <h4>🎯 {selected_ic.get('strategy_type', 'Standard')} Iron Condor: <span style="color: {trade_color};">{trade_quality}</span></h4>
        <p><strong>Strikes:</strong> {selected_ic.get('put_long', 0):.0f}/{selected_ic.get('put_short', 0):.0f}/{selected_ic.get('call_short', 0):.0f}/{selected_ic.get('call_long', 0):.0f}</p>
        <p><strong>Credit Received:</strong> ${selected_ic.get('total_credit', 0):.2f}</p>
        <p><strong>Probability of Profit:</strong> {pop:.1f}%</p>
        <p><strong>Max Risk:</strong> ${selected_ic.get('max_loss', 0):.2f} | <strong>Max Reward:</strong> ${selected_ic.get('max_profit', 0):.2f}</p>
        <p><strong>Breakeven Range:</strong> ${selected_ic.get('lower_breakeven', 0):.2f} - ${selected_ic.get('upper_breakeven', 0):.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # === RISK MANAGEMENT ===
    st.markdown("#### ⚠️ Iron Condor Risk Management")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("""
        **Entry Rules:**
        - ✅ High IV environment (IV Rank > 50%)
        - ✅ VIX < 30 for stable conditions
        - ✅ POP > 60% minimum
        - ✅ Credit > 1/3 of wing width
        - ✅ 30-45 DTE optimal
        """)
    
    with risk_col2:
        st.markdown("""
        **Exit Rules:**
        - 🎯 Take profits at 25-50% of max profit
        - ⛔ Cut losses at 200% of credit received
        - ⏰ Close at 21 DTE regardless of P&L
        - 📈 Manage early if tested (50% of wing width)
        - 🔄 Consider rolling if profitable
        """)
    
    # === ADVANCED ANALYSIS ===
    with st.expander("🔬 Advanced Iron Condor Analysis"):
        
        # Greeks Summary
        st.markdown("##### Greeks Analysis")
        greeks_col1, greeks_col2, greeks_col3, greeks_col4 = st.columns(4)
        
        with greeks_col1:
            st.metric("Net Delta", f"{selected_ic.get('net_delta', 0):.3f}")
        with greeks_col2:
            st.metric("Net Gamma", f"{selected_ic.get('net_gamma', 0):.4f}")
        with greeks_col3:
            st.metric("Net Theta", f"{selected_ic.get('net_theta', 0):.3f}")
        with greeks_col4:
            st.metric("Net Vega", f"{selected_ic.get('net_vega', 0):.3f}")
        
        # Scenario Analysis
        st.markdown("##### Scenario Analysis")
        scenarios = [
            ("Bull Move (+5%)", current_price * 1.05),
            ("Current Price", current_price),
            ("Bear Move (-5%)", current_price * 0.95),
            ("Large Move (+10%)", current_price * 1.10),
            ("Large Move (-10%)", current_price * 0.90)
        ]
        
        scenario_results = []
        for scenario_name, scenario_price in scenarios:
            # Calculate P&L at expiration for scenario price
            breakeven_lower = selected_ic.get('lower_breakeven', 0)
            breakeven_upper = selected_ic.get('upper_breakeven', 0)
            net_credit = selected_ic.get('total_credit', 0)
            wing_width = selected_ic.get('wing_width', 0)
            put_long = selected_ic.get('put_long', 0)
            put_short = selected_ic.get('put_short', 0)
            call_short = selected_ic.get('call_short', 0)
            call_long = selected_ic.get('call_long', 0)
            
            if (scenario_price > breakeven_lower and scenario_price < breakeven_upper):
                pnl = net_credit
            elif scenario_price <= put_long:
                pnl = net_credit - wing_width
            elif scenario_price >= call_long:
                pnl = net_credit - wing_width
            elif scenario_price <= put_short:
                pnl = net_credit - (put_short - scenario_price)
            elif scenario_price >= call_short:
                pnl = net_credit - (scenario_price - call_short)
            else:
                pnl = net_credit
            
            max_loss = selected_ic.get('max_loss', 1)  # Avoid division by zero
            scenario_results.append({
                'Scenario': scenario_name,
                'Stock Price': f"${scenario_price:.2f}",
                'P&L': f"${pnl:.2f}",
                'Return %': f"{(pnl / abs(max_loss) * 100):.1f}%" if max_loss != 0 else "N/A"
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        st.dataframe(scenario_df, use_container_width=True) 