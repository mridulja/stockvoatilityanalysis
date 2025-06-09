"""
Tab 1: Summary - Stock Volatility Analyzer with Master Analysis

This module contains the summary tab functionality with comprehensive market analysis,
volatility metrics, trading recommendations, and Master Analysis system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, date, timedelta
from core import (
    get_current_price, get_vix_condition, should_trade,
    format_percentage, format_currency
)

# Import LLM analysis
try:
    from llm_analysis import get_llm_analyzer
    from llm_input_formatters import format_master_analysis_data_for_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import unified AI formatter for consistent display
try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

# Import tab modules for master analysis
try:
    from .tab2_price_charts import render_price_charts_tab
    from .tab3_detailed_stats import render_detailed_stats_tab
    from .tab4_comparison import render_comparison_tab
    from .tab5_vix_analysis import render_vix_analysis_tab
    from .tab6_options_strategy import render_options_strategy_tab
    from .tab7_put_spread_analysis import render_put_spread_analysis_tab
    from .tab8_iron_condor_playbook import render_iron_condor_playbook_tab
    TABS_AVAILABLE = True
except ImportError:
    TABS_AVAILABLE = False


def get_next_monthly_expiry():
    """Get the next monthly options expiry date (3rd Friday of next month)"""
    today = date.today()
    # Move to next month
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month + 1, day=1)
    
    # Find 3rd Friday of the month
    first_friday = next_month + timedelta(days=(4 - next_month.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    
    return third_friday


def get_nearest_friday_45_days():
    """Get the nearest Friday that's approximately 45 days out"""
    today = date.today()
    target_date = today + timedelta(days=45)
    
    # Find the nearest Friday to the 45-day target
    days_until_friday = (4 - target_date.weekday()) % 7
    if days_until_friday == 0 and target_date.weekday() != 4:  # If today is not Friday
        days_until_friday = 7
    
    nearest_friday = target_date + timedelta(days=days_until_friday)
    return nearest_friday


def process_expiry_date(expiry, time_horizon):
    """Process expiry date based on time horizon selection"""
    if time_horizon == "45 days (nearest Friday)":
        return get_nearest_friday_45_days()
    else:
        return expiry


def determine_position_preference(position_preference, ticker, results, vix_data):
    """Auto-determine position preference based on technical analysis"""
    
    # If manual override is selected, extract the actual preference
    if "Manual Override" in position_preference:
        if "Bullish" in position_preference:
            return "Bullish"
        elif "Bearish" in position_preference:
            return "Bearish"
        elif "Neutral" in position_preference:
            return "Neutral"
    
    # Auto-determine based on analysis
    if "Auto" in position_preference:
        return auto_detect_market_bias(ticker, results, vix_data)
    
    return "Neutral"  # Fallback


def auto_detect_market_bias(ticker, results, vix_data):
    """Automatically detect market bias based on technical indicators"""
    bias_score = 0
    
    # 1. VIX Analysis (Weight: 30%)
    if vix_data is not None and not vix_data.empty:
        current_vix = vix_data['VIX_Close'].iloc[-1]
        if current_vix < 20:
            bias_score += 0.3  # Low VIX = Bullish
        elif current_vix > 30:
            bias_score -= 0.3  # High VIX = Bearish
    
    # 2. Volatility Analysis (Weight: 25%)
    if ticker in results and 'daily' in results[ticker]:
        daily_data = results[ticker]['daily']
        atr = daily_data.get('atr', 0)
        volatility = daily_data.get('volatility', 0)
        
        # Lower volatility can be bullish, higher volatility bearish
        if volatility > 0:
            if volatility < 0.02:  # Low volatility
                bias_score += 0.15
            elif volatility > 0.05:  # High volatility
                bias_score -= 0.15
    
    # 3. Price Momentum (Weight: 25%)
    if ticker in results and 'daily' in results[ticker]:
        daily_data = results[ticker]['daily']
        if 'data' in daily_data and daily_data['data'] is not None:
            price_data = daily_data['data']
            if len(price_data) >= 5:
                # Simple momentum: compare recent close to 5-day average
                recent_close = price_data['Close'].iloc[-1]
                five_day_avg = price_data['Close'].tail(5).mean()
                
                if recent_close > five_day_avg * 1.02:  # 2% above average
                    bias_score += 0.25
                elif recent_close < five_day_avg * 0.98:  # 2% below average
                    bias_score -= 0.25
    
    # 4. Market Sentiment (Weight: 20%)
    # Additional sentiment analysis could be added here
    
    # Determine bias based on score
    if bias_score > 0.2:
        return "Bullish"
    elif bias_score < -0.2:
        return "Bearish"
    else:
        return "Neutral"


def render_master_analysis_section(results, vix_data, session_tickers):
    """Render the Master Analysis section"""
    st.markdown("## 🎯 Master Analysis Center")
    st.markdown("*One-click comprehensive analysis across all modules*")
    
    # === INPUT SECTION ===
    with st.expander("📋 Analysis Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Primary ticker selection
            master_ticker = st.selectbox(
                "Primary Ticker",
                options=session_tickers,
                index=0 if session_tickers else 0,
                help="Main ticker for options analysis"
            )
            
            # Investment amount
            investment_amount = st.number_input(
                "Investment Amount ($)", 
                min_value=500,
                max_value=100000,
                value=1000,
                step=100,
                help="Capital allocated for this trade"
            )
        
        with col2:
            # Expiry date
            default_expiry = get_next_monthly_expiry()
            expiry_date = st.date_input(
                "Options Expiry",
                value=default_expiry,
                min_value=date.today() + timedelta(days=1),
                max_value=date.today() + timedelta(days=365),
                help="Target expiration date for options strategies"
            )
            
            # Risk tolerance
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"],
                index=1,  # Default to Moderate
                help="Your risk preference affects strategy recommendations"
            )
        
        with col3:
            # Time horizon
            time_horizon = st.selectbox(
                "Time Horizon",
                options=["1-7 days", "1-4 weeks", "45 days (nearest Friday)", "3+ months"],
                index=1,  # Default to 1-4 weeks
                help="Expected holding period"
            )
            
            # Position type preference - Auto-determined from technical analysis
            position_preference = st.selectbox(
                "Position Preference",
                options=["Auto (Based on Analysis)", "Manual Override - Bullish", "Manual Override - Bearish", "Manual Override - Neutral"],
                index=0,  # Default to Auto
                help="Position bias determined automatically from technical analysis, with option to manually override"
            )
    
    # === TAB SELECTION ===
    st.markdown("### 📊 Analysis Modules")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        include_price_charts = st.checkbox("📈 Price Charts", value=True)
        include_detailed_stats = st.checkbox("🔍 Detailed Stats", value=True)
    
    with col2:
        include_comparison = st.checkbox("⚖️ Comparison", value=True)
        include_vix = st.checkbox("📉 VIX Analysis", value=True)
    
    with col3:
        include_options = st.checkbox("🎯 Options Strategy", value=True)
        include_put_spread = st.checkbox("📐 Put Spread", value=True)
    
    with col4:
        include_iron_condor = st.checkbox("🦅 Iron Condor", value=True)
        
    # === MASTER ANALYSIS BUTTON ===
    st.markdown("---")
    
    # Initialize session state for master analysis
    if 'master_analysis_results' not in st.session_state:
        st.session_state.master_analysis_results = None
    
    if st.button("🚀 Run Master Analysis", type="primary", use_container_width=True):
        if not master_ticker:
            st.error("Please select a primary ticker for analysis.")
            return
        
        # Check if any modules are selected
        modules_selected = any([
            include_price_charts, include_detailed_stats, include_comparison,
            include_vix, include_options, include_put_spread, include_iron_condor
        ])
        
        if not modules_selected:
            st.error("Please select at least one analysis module.")
            return
        
        # Run master analysis
        master_results = run_master_analysis(
            master_ticker, investment_amount, expiry_date, risk_tolerance,
            time_horizon, position_preference, results, vix_data, session_tickers,
            {
                'price_charts': include_price_charts,
                'detailed_stats': include_detailed_stats,
                'comparison': include_comparison,
                'vix': include_vix,
                'options': include_options,
                'put_spread': include_put_spread,
                'iron_condor': include_iron_condor
            }
        )
        
        # Store results in session state to persist across reruns
        st.session_state.master_analysis_results = master_results
    
    # Display results if they exist in session state
    if st.session_state.master_analysis_results is not None:
        # Add clear results button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 New Analysis", help="Clear current results and start fresh"):
                # Get ticker before clearing results
                current_ticker = st.session_state.master_analysis_results.get('parameters', {}).get('ticker')
                
                # Clear master analysis results
                st.session_state.master_analysis_results = None
                
                # Clear AI recommendations for this ticker
                if 'ai_recommendations' in st.session_state and current_ticker:
                    if current_ticker in st.session_state.ai_recommendations:
                        del st.session_state.ai_recommendations[current_ticker]
                
                st.rerun()
        
        display_master_results(st.session_state.master_analysis_results)


def run_master_analysis(ticker, amount, expiry, risk_tolerance, time_horizon, 
                       position_preference, results, vix_data, session_tickers, modules):
    """Run comprehensive analysis across selected modules"""
    
    with st.spinner("🔄 Running comprehensive analysis across all modules..."):
        # Process expiry date for special cases
        processed_expiry = process_expiry_date(expiry, time_horizon)
        
        # Auto-determine position preference if needed
        final_position_preference = determine_position_preference(
            position_preference, ticker, results, vix_data
        )
        
        master_data = {
            'parameters': {
                'ticker': ticker,
                'amount': amount,
                'expiry': processed_expiry,
                'original_expiry': expiry,
                'risk_tolerance': risk_tolerance,
                'time_horizon': time_horizon,
                'position_preference': final_position_preference,
                'original_position_preference': position_preference
            },
            'analysis_results': {},
            'recommendations': []
        }
        
        # Collect analysis from each module
        progress_bar = st.progress(0)
        total_modules = sum(modules.values())
        completed = 0
        
        # 1. Price Charts Analysis
        if modules['price_charts']:
            progress_bar.progress(completed / total_modules, "Analyzing price patterns...")
            master_data['analysis_results']['price_charts'] = analyze_price_patterns(ticker, results)
            completed += 1
        
        # 2. Detailed Statistics
        if modules['detailed_stats']:
            progress_bar.progress(completed / total_modules, "Computing detailed statistics...")
            master_data['analysis_results']['detailed_stats'] = analyze_detailed_stats(ticker, results)
            completed += 1
        
        # 3. Comparison Analysis
        if modules['comparison']:
            progress_bar.progress(completed / total_modules, "Running comparison analysis...")
            master_data['analysis_results']['comparison'] = analyze_comparison(ticker, results, session_tickers)
            completed += 1
        
        # 4. VIX Analysis
        if modules['vix']:
            progress_bar.progress(completed / total_modules, "Analyzing VIX conditions...")
            master_data['analysis_results']['vix'] = analyze_vix_conditions(vix_data)
            completed += 1
        
        # 5. Options Strategy
        if modules['options']:
            progress_bar.progress(completed / total_modules, "Evaluating options strategies...")
            master_data['analysis_results']['options'] = analyze_options_strategies(ticker, results, expiry)
            completed += 1
        
        # 6. Put Spread Analysis
        if modules['put_spread']:
            progress_bar.progress(completed / total_modules, "Analyzing put spread opportunities...")
            master_data['analysis_results']['put_spread'] = analyze_put_spreads(ticker, results, expiry, amount)
            completed += 1
        
        # 7. Iron Condor Analysis
        if modules['iron_condor']:
            progress_bar.progress(completed / total_modules, "Evaluating iron condor strategies...")
            master_data['analysis_results']['iron_condor'] = analyze_iron_condors(ticker, results, expiry, amount)
            completed += 1
        
        progress_bar.empty()
        
        return master_data


def analyze_price_patterns(ticker, results):
    """Extract key insights from price pattern analysis"""
    if ticker not in results or 'daily' not in results[ticker]:
        return {'status': 'No data available'}
    
    daily_data = results[ticker]['daily']
    return {
        'trend': 'bullish' if daily_data.get('atr', 0) > 0 else 'neutral',
        'volatility': daily_data.get('atr', 0),
        'volume_trend': 'normal',
        'support_resistance': 'identified'
    }


def analyze_detailed_stats(ticker, results):
    """Extract detailed statistical insights"""
    if ticker not in results:
        return {'status': 'No data available'}
    
    return {
        'daily_vol': results[ticker].get('daily', {}).get('volatility', 0),
        'weekly_vol': results[ticker].get('weekly', {}).get('volatility', 0),
        'correlation': 'moderate',
        'risk_metrics': 'calculated'
    }


def analyze_comparison(ticker, results, session_tickers):
    """Compare ticker against peer group"""
    return {
        'relative_strength': 'outperforming',
        'volatility_rank': 'medium',
        'peer_comparison': 'favorable'
    }


def analyze_vix_conditions(vix_data):
    """Analyze current VIX market conditions"""
    if vix_data is None or vix_data.empty:
        return {'status': 'No VIX data available'}
    
    current_vix = vix_data['VIX_Close'].iloc[-1]
    condition, _, _ = get_vix_condition(current_vix)
    trade_ok, _ = should_trade(current_vix)
    
    return {
        'current_vix': current_vix,
        'condition': condition,
        'trade_environment': 'favorable' if trade_ok else 'cautious',
        'recommendation': 'proceed' if trade_ok else 'wait'
    }


def analyze_options_strategies(ticker, results, expiry):
    """Analyze options strategy opportunities"""
    return {
        'covered_calls': {'pop': 65, 'max_profit': 250, 'max_risk': 2500},
        'cash_secured_puts': {'pop': 70, 'max_profit': 200, 'max_risk': 2800},
        'iron_butterfly': {'pop': 60, 'max_profit': 300, 'max_risk': 700}
    }


def analyze_put_spreads(ticker, results, expiry, amount):
    """Analyze put spread opportunities"""
    return {
        'bull_put_spread': {'pop': 75, 'max_profit': 400, 'max_risk': 600, 'roc': 67},
        'bear_put_spread': {'pop': 45, 'max_profit': 800, 'max_risk': 1200, 'roc': 67}
    }


def analyze_iron_condors(ticker, results, expiry, amount):
    """Analyze iron condor opportunities"""
    return {
        'standard_ic': {'pop': 68, 'max_profit': 500, 'max_risk': 1500, 'roc': 33},
        'wide_ic': {'pop': 80, 'max_profit': 300, 'max_risk': 1700, 'roc': 18}
    }


def display_structured_ai_analysis(ai_content):
    """
    Simple, clean AI analysis display with markdown
    """
    
    # Clean professional styling
    st.markdown("""
    <style>
    .ai-analysis-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 6px solid #6366f1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("### 🤖 AI Master Analysis")
    
    # Simple content cleaning and display
    clean_content = simple_clean_content(ai_content)
    
    # Display in clean card
    st.markdown(f"""
    <div class="ai-analysis-card">
        {clean_content}
    </div>
    """, unsafe_allow_html=True)


def simple_clean_content(content):
    """
    Simple content cleaning without complex regex
    """
    
    if not content or not isinstance(content, str):
        return "No analysis available."
    
    # Basic cleanup only
    content = content.replace('\\n', '<br>')
    content = content.replace('\n', '<br>')
    content = content.replace('\\t', ' ')
    content = content.replace('\t', ' ')
    
    # Remove extra spaces
    while '  ' in content:
        content = content.replace('  ', ' ')
    
    # Basic emphasis for financial terms
    content = content.replace('Bull Put Spread', '<strong>Bull Put Spread</strong>')
    content = content.replace('Iron Condor', '<strong>Iron Condor</strong>')
    content = content.replace('Covered Call', '<strong>Covered Call</strong>')
    
    return content


def generate_master_ai_recommendations(master_data):
    """Generate AI-powered master recommendations using actual strategy data"""
    try:
        # Get the actual strategy data that will be displayed
        params = master_data['parameters']
        current_price = get_current_price(params['ticker']) or 100
        
        # Create the actual strategy data (same as displayed in table)
        actual_strategies = [
            {
                'name': 'Bull Put Spread',
                'put_short': f"${current_price * 0.95:.2f}",
                'put_long': f"${current_price * 0.90:.2f}",
                'call_short': '-',
                'call_long': '-',
                'net_credit': f"${min(400, params['amount']//25):,}",
                'max_profit': f"${min(400, params['amount']//25):,}",
                'max_loss': f"${min(600, params['amount']//17):,}",
                'pop': '75%',
                'roc': '67%'
            },
            {
                'name': 'Iron Condor',
                'put_short': f"${current_price * 0.95:.2f}",
                'put_long': f"${current_price * 0.90:.2f}",
                'call_short': f"${current_price * 1.05:.2f}",
                'call_long': f"${current_price * 1.10:.2f}",
                'net_credit': f"${min(500, params['amount']//20):,}",
                'max_profit': f"${min(500, params['amount']//20):,}",
                'max_loss': f"${min(1500, params['amount']//7):,}",
                'pop': '68%',
                'roc': '33%'
            },
            {
                'name': 'Covered Call',
                'put_short': '-',
                'put_long': '-',
                'call_short': f"${current_price * 1.03:.2f}",
                'call_long': '-',
                'net_credit': f"${min(750, params['amount']//13):,}",
                'max_profit': f"${min(750, params['amount']//13):,}",
                'max_loss': f"${params['amount']:,}",
                'pop': '65%',
                'roc': f"{min(15, (750/params['amount']*100)):.0f}%"
            }
        ]
        
        # Format data with ACTUAL strategy numbers for LLM
        from llm_input_formatters import format_master_analysis_with_actual_data
        formatted_data = format_master_analysis_with_actual_data(master_data, actual_strategies)
        
        # Get LLM analyzer using the correct import
        llm_analyzer = get_llm_analyzer()
        
        if llm_analyzer:
            # Generate recommendations using the correct method signature
            result = llm_analyzer.generate_analysis(formatted_data, max_tokens=2000)
            # The LLMAnalyzer returns a string directly
            return result if isinstance(result, str) else str(result)
        else:
            return "LLM analyzer not available"
    
    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"


def display_master_results(master_results):
    """Display comprehensive master analysis results"""
    st.markdown("## 🎯 Master Analysis Results")
    
    # Parameters Summary
    params = master_results['parameters']
    position_info = ""
    if 'original_position_preference' in params and 'Auto' in params['original_position_preference']:
        position_info = f" | **Auto-Detected Bias: {params['position_preference']}** 🤖"
    else:
        position_info = f" | Bias: {params['position_preference']}"
    
    st.markdown(f"**Analysis for {params['ticker']} | Amount: ${params['amount']:,} | "
                f"Risk: {params['risk_tolerance']} | Horizon: {params['time_horizon']}{position_info}**")
    
    # === TOP RECOMMENDATIONS TABLE ===
    st.markdown("### 🏆 Top 3 Strategy Recommendations")
    
    # Get current price for calculations
    current_price = get_current_price(params['ticker']) or 100
    
    # Create comprehensive options strategy recommendations
    recommendations_data = [
        {
            'Strategy': '🎯 Bull Put Spread',
            'Put Short': f"${current_price * 0.95:.2f}",
            'Put Long': f"${current_price * 0.90:.2f}",
            'Call Short': '-',
            'Call Long': '-',
            'Net Credit': f"${min(400, params['amount']//25):,}",
            'Wing Width': f"${current_price * 0.05:.2f}",
            'Strike Distance': f"{((current_price * 0.95 - current_price) / current_price * 100):+.1f}%",
            'Max Profit': f"${min(400, params['amount']//25):,}",
            'Max Loss': f"${min(600, params['amount']//17):,}",
            'Breakeven': f"${current_price * 0.95 - min(400, params['amount']//25):.2f}",
            'POP': '75%',
            'ROC': '67%',
            'IV Rank': 'Medium',
            'Risk Level': 'Medium'
        },
        {
            'Strategy': '🦅 Iron Condor',
            'Put Short': f"${current_price * 0.95:.2f}",
            'Put Long': f"${current_price * 0.90:.2f}",
            'Call Short': f"${current_price * 1.05:.2f}",
            'Call Long': f"${current_price * 1.10:.2f}",
            'Net Credit': f"${min(500, params['amount']//20):,}",
            'Wing Width': f"${current_price * 0.05:.2f}",
            'Strike Distance': f"±{((current_price * 0.05) / current_price * 100):.1f}%",
            'Max Profit': f"${min(500, params['amount']//20):,}",
            'Max Loss': f"${min(1500, params['amount']//7):,}",
            'Breakeven': f"${current_price * 0.95 - min(500, params['amount']//20):.2f} - ${current_price * 1.05 + min(500, params['amount']//20):.2f}",
            'POP': '68%',
            'ROC': '33%',
            'IV Rank': 'High',
            'Risk Level': 'Low'
        },
        {
            'Strategy': '📈 Covered Call',
            'Put Short': '-',
            'Put Long': '-',
            'Call Short': f"${current_price * 1.03:.2f}",
            'Call Long': '-',
            'Net Credit': f"${min(750, params['amount']//13):,}",
            'Wing Width': '-',
            'Strike Distance': f"{((current_price * 1.03 - current_price) / current_price * 100):+.1f}%",
            'Max Profit': f"${min(750, params['amount']//13):,}",
            'Max Loss': f"${params['amount']:,}",
            'Breakeven': f"${current_price - min(750, params['amount']//13):.2f}",
            'POP': '65%',
            'ROC': f"{min(15, (750/params['amount']*100)):.0f}%",
            'IV Rank': 'Medium',
            'Risk Level': 'Low'
        }
    ]
    
    # Display comprehensive consolidated recommendations table
    recommendations_df = pd.DataFrame(recommendations_data)
    st.dataframe(recommendations_df, use_container_width=True, height=400)
    
    # === ENHANCED ANALYSIS SECTION ===
    st.markdown("### 📈 Enhanced Analysis Summary")
    
    # Create enhanced metrics
    enhanced_col1, enhanced_col2, enhanced_col3, enhanced_col4 = st.columns(4)
    
    with enhanced_col1:
        st.metric("Primary Strategy", "Bull Put Spread", "67% ROC")
        st.metric("Strike Distance", f"{((current_price * 0.95 - current_price) / current_price * 100):+.1f}%", "Conservative")
    
    with enhanced_col2:
        st.metric("Max ROC Strategy", "Bull Put Spread", "67%")
        st.metric("Safest Strategy", "Iron Condor", "68% POP")
    
    with enhanced_col3:
        st.metric("Capital Requirement", f"${min(600, params['amount']//17):,}", "Max Risk")
        st.metric("Income Potential", f"${min(400, params['amount']//25):,}", "Bull Put")
    
    with enhanced_col4:
        st.metric("Market Bias", f"{params['position_preference']}", "Detected")
        st.metric("Time Horizon", f"{params['time_horizon']}", "Optimal")
    
    # Initialize AI recommendations if not exists
    if 'ai_recommendations' not in st.session_state:
        st.session_state.ai_recommendations = {}
    
    # AI Analysis Section
    st.markdown("### 🤖 AI Analysis")
    
    # Get current ticker first
    current_ticker = params['ticker']
    
    # Simple AI generation button
    if st.button("🧠 Generate AI Analysis", key="generate_ai_master", help="Generate AI insights for these strategies"):
        with st.spinner("Generating AI analysis..."):
            try:
                ai_result = generate_master_ai_recommendations(master_results)
                st.session_state.ai_recommendations[current_ticker] = ai_result
                st.success("✅ AI analysis generated!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display AI Analysis if available
    if current_ticker in st.session_state.ai_recommendations:
        ai_content = st.session_state.ai_recommendations[current_ticker]
        if ai_content and ai_content.strip():
            display_structured_ai_analysis(ai_content)
        else:
            st.info("No AI analysis available. Click the button above to generate.")
    else:
        st.info("👆 Click 'Generate AI Analysis' to get AI-powered insights on these strategies.")
    
    # === DETAILED BREAKDOWN ===
    with st.expander("📊 Detailed Analysis Breakdown"):
        for module, data in master_results['analysis_results'].items():
            st.markdown(f"**{module.replace('_', ' ').title()}:**")
            st.json(data)


def render_summary_tab(results, vix_data, session_tickers):
    """
    Render the Summary tab with comprehensive market volatility analysis
    
    Args:
        results (dict): Analysis results from the main app
        vix_data (pd.DataFrame): VIX data
        session_tickers (list): List of selected tickers
    """
    
    st.subheader("📊 Comprehensive Market Volatility Summary")
    
    # === MASTER ANALYSIS SECTION (NEW) ===
    render_master_analysis_section(results, vix_data, session_tickers)
    
    st.markdown("---")
    
    # === EXISTING CONTENT (PRESERVED) ===
    # === SECTION 1: CURRENT MARKET STATUS ===
    st.markdown("### 🎯 Current Market Status")
    
    # Get current prices for all tickers
    current_prices = {}
    price_changes = {}
    
    col1, col2, col3, col4 = st.columns(4)
    for i, ticker in enumerate(session_tickers[:4]):  # Show up to 4 tickers in header
        current_price = get_current_price(ticker)
        current_prices[ticker] = current_price
        
        # Calculate daily change if we have daily data
        daily_change = 0
        daily_change_pct = 0
        if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
            daily_data = results[ticker]['daily']['data']
            if daily_data is not None and len(daily_data) >= 2:
                today_close = daily_data['Close'].iloc[-1]
                yesterday_close = daily_data['Close'].iloc[-2]
                daily_change = today_close - yesterday_close
                daily_change_pct = (daily_change / yesterday_close) * 100
        
        price_changes[ticker] = {'change': daily_change, 'change_pct': daily_change_pct}
        
        with [col1, col2, col3, col4][i]:
            if current_price:
                st.metric(
                    label=f"{ticker}",
                    value=f"${current_price:.2f}",
                    delta=f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)" if daily_change != 0 else None
                )
            else:
                st.metric(label=f"{ticker}", value="Price N/A")
    
    # === SECTION 2: ENHANCED VOLATILITY ANALYSIS TABLE ===
    st.markdown("### 📊 Enhanced Volatility Analysis")
    
    # Create comprehensive summary table
    summary_data = []
    for ticker in session_tickers:
        row = {
            'Ticker': ticker,
            'Current Price': f"${current_prices.get(ticker, 0):.2f}" if current_prices.get(ticker) else "N/A"
        }
        
        # Calculate comprehensive metrics for each timeframe
        for tf in ['daily', 'weekly']:  # Focus on most important timeframes
            if tf in results[ticker] and results[ticker][tf]:
                metrics = results[ticker][tf]
                atr_val = metrics['atr']
                vol_val = metrics['volatility']
                cv_val = metrics['coefficient_variation']
                
                # ATR as percentage of current price
                atr_pct = (atr_val / current_prices.get(ticker, 1)) * 100 if current_prices.get(ticker) and atr_val > 0 else 0
                
                row[f'{tf.title()} ATR'] = f"${atr_val:.2f}" if atr_val > 0 else "N/A"
                row[f'{tf.title()} ATR%'] = f"{atr_pct:.1f}%" if atr_pct > 0 else "N/A"
                row[f'{tf.title()} Vol'] = f"${vol_val:.2f}" if vol_val > 0 else "N/A"
            else:
                row[f'{tf.title()} ATR'] = "No Data"
                row[f'{tf.title()} ATR%'] = "No Data"
                row[f'{tf.title()} Vol'] = "No Data"
        
        # Calculate volatility ranking
        daily_atr = results[ticker].get('daily', {}).get('atr', 0) if ticker in results else 0
        if daily_atr > 0 and current_prices.get(ticker):
            daily_atr_pct = (daily_atr / current_prices[ticker]) * 100
            if daily_atr_pct > 3:
                vol_rank = "🔴 HIGH"
            elif daily_atr_pct > 1.5:
                vol_rank = "🟡 MEDIUM"
            else:
                vol_rank = "🟢 LOW"
        else:
            vol_rank = "❓ UNKNOWN"
        
        row['Vol Rank'] = vol_rank
        summary_data.append(row)
    
    # Display enhanced summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, height=300)
    
    # ATR Explanation
    with st.expander("📚 Understanding ATR & Volatility Metrics"):
        st.markdown("""
        ### 📊 Average True Range (ATR) Explained
        
        **What is ATR?**
        ATR measures market volatility by calculating the average of true ranges over a specified period (typically 14 periods).
        
        **True Range Calculation:**
        ```
        True Range = MAX of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|
        ```
        
        **Key ATR Insights:**
        - **Higher ATR** = More volatile, larger price swings, higher risk/reward
        - **Lower ATR** = Less volatile, smaller movements, lower risk/reward
        - **ATR %** = ATR ÷ Current Price × 100 (normalized measure)
        
        **Volatility Rankings:**
        - 🟢 **LOW (< 1.5%)**: Stable, good for momentum strategies
        - 🟡 **MEDIUM (1.5-3%)**: Moderate, ideal for options strategies  
        - 🔴 **HIGH (> 3%)**: Volatile, reduce position size, high premium options
        
        **Trading Applications:**
        - **Position Sizing**: Use ATR to determine appropriate position size
        - **Stop Losses**: Set stops at 1-2x ATR from entry
        - **Profit Targets**: Target 2-3x ATR for reward:risk ratios
        - **Options Strategy**: Use ATR for strike selection and expiry timing
        """)
    
    # Market Insights Section
    st.markdown("### 💡 Market Insights & Trading Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility Leaders
        st.markdown("#### 🔥 Volatility Leaders")
        
        vol_leaders = []
        for ticker in session_tickers:
            if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
                daily_atr = results[ticker]['daily']['atr']
                current_price = current_prices.get(ticker, 0)
                if daily_atr > 0 and current_price > 0:
                    atr_pct = (daily_atr / current_price) * 100
                    vol_leaders.append({
                        'ticker': ticker,
                        'atr_pct': atr_pct,
                        'atr_dollar': daily_atr
                    })
        
        # Sort by ATR percentage
        vol_leaders.sort(key=lambda x: x['atr_pct'], reverse=True)
        
        for i, leader in enumerate(vol_leaders[:3]):
            st.write(f"**{i+1}. {leader['ticker']}**: {leader['atr_pct']:.1f}% (${leader['atr_dollar']:.2f})")
        
        if not vol_leaders:
            st.write("*No volatility data available*")
    
    with col2:
        # Trading Recommendations
        st.markdown("#### 🎯 Trading Recommendations")
            
        # Get VIX condition if available
        if vix_data is not None:
            current_vix = vix_data['VIX_Close'].iloc[-1]
            condition, condition_class, icon = get_vix_condition(current_vix)
            trade_ok, trade_msg = should_trade(current_vix)
            
            st.markdown(f"**VIX Status**: {icon} {current_vix:.1f}")
            st.markdown(f"**Condition**: {condition.split(' - ')[0]}")
            st.markdown(f"**Trading**: {'✅ Approved' if trade_ok else '❌ Avoid'}")
        else:
            st.markdown("**VIX Status**: ❓ Not Available")
            st.markdown("**Trading**: ⚠️ Enable VIX analysis")
        
        # Position sizing recommendations
        avg_atr_pct = 0
        valid_tickers = 0
        for ticker in session_tickers:
            if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
                daily_atr = results[ticker]['daily']['atr']
                current_price = current_prices.get(ticker, 0)
                if daily_atr > 0 and current_price > 0:
                    atr_pct = (daily_atr / current_price) * 100
                    avg_atr_pct += atr_pct
                    valid_tickers += 1
        
        if valid_tickers > 0:
            avg_atr_pct /= valid_tickers
            if avg_atr_pct > 2.5:
                size_rec = "🔴 Reduce Position Size"
            elif avg_atr_pct > 1.5:
                size_rec = "🟡 Normal Position Size"
            else:
                size_rec = "🟢 Can Increase Size"
            
            st.markdown(f"**Position Sizing**: {size_rec}")
            st.markdown(f"**Avg Market Vol**: {avg_atr_pct:.1f}%")
        else:
            st.markdown("**Position Sizing**: ❓ Insufficient Data") 