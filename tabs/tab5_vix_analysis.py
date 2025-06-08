"""
VIX Analysis Tab for Stock Volatility Analyzer

This module provides comprehensive VIX (Volatility Index) analysis including:
- Real-time VIX conditions and market assessment
- VIX trading strategies and recommendations
- Volatility term structure analysis
- Historical VIX patterns and regime analysis
- VIX-based options strategy recommendations
- AI-powered market condition insights

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
import warnings

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

def get_vix_condition(vix_value):
    """Determine market condition based on VIX level"""
    if pd.isna(vix_value):
        return "Unknown", "vix-normal", "🤷", "neutral"
    
    if vix_value < 15:
        return "Calm Markets - Clean Trend", "vix-calm", "🟢", "bullish"
    elif 15 <= vix_value < 19:
        return "Normal Markets - Trendy", "vix-normal", "🔵", "neutral" 
    elif 19 <= vix_value < 26:
        return "Choppy Market - Proceed with Caution", "vix-choppy", "🟡", "cautious"
    elif 26 <= vix_value < 36:
        return "High Volatility - Big Swings, Don't Trade", "vix-volatile", "🔴", "bearish"
    else:
        return "Extreme Volatility - Very Bad Day, DO NOT TRADE", "vix-extreme", "🚨", "extreme_bearish"

def get_vix_trading_strategy(vix_value, vix_percentile=None):
    """Get VIX-based trading strategy recommendations"""
    condition, _, _, regime = get_vix_condition(vix_value)
    
    strategies = {
        "bullish": {
            "primary": "Sell Puts / Cash-Secured Puts",
            "secondary": "Buy Calls / Call Spreads",
            "vix_play": "Sell VIX calls / Short volatility",
            "risk": "Low",
            "timeframe": "Medium to Long term"
        },
        "neutral": {
            "primary": "Iron Condors / Strangles",
            "secondary": "Covered Calls / Protective Puts",
            "vix_play": "VIX mean reversion trades",
            "risk": "Medium",
            "timeframe": "Short to Medium term"
        },
        "cautious": {
            "primary": "Protective Strategies / Hedging",
            "secondary": "Cash Positions / Defensive Spreads",
            "vix_play": "Long VIX calls / Volatility hedge",
            "risk": "Medium-High",
            "timeframe": "Short term"
        },
        "bearish": {
            "primary": "Cash / Protective Puts",
            "secondary": "Bear Spreads / Short Calls",
            "vix_play": "Long VIX / Volatility surge play",
            "risk": "High",
            "timeframe": "Very Short term"
        },
        "extreme_bearish": {
            "primary": "CASH ONLY - No New Positions",
            "secondary": "Close Existing Positions",
            "vix_play": "Long VIX / Crisis hedge",
            "risk": "Extreme",
            "timeframe": "Immediate"
        }
    }
    
    return strategies.get(regime, strategies["neutral"])

@st.cache_data(ttl=300)
def fetch_vix_term_structure():
    """Fetch VIX term structure data"""
    try:
        # VIX family symbols
        vix_symbols = {
            "VIX": "^VIX",      # Current month
            "VIX3M": "^VIX3M",  # 3-month
            "VIX6M": "^VIX6M",  # 6-month
            "VIX9D": "^VIX9D"   # 9-day
        }
        
        vix_data = {}
        for name, symbol in vix_symbols.items():
            try:
                data = yf.download(symbol, period='1d', interval='1d', progress=False)
                if not data.empty:
                    # Handle multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    
                    # Ensure we get a scalar value
                    close_value = data['Close'].iloc[-1]
                    if hasattr(close_value, 'item'):
                        vix_data[name] = close_value.item()  # Convert to scalar
                    else:
                        vix_data[name] = float(close_value)  # Ensure scalar
            except:
                vix_data[name] = None
        
        return vix_data
    except Exception as e:
        st.warning(f"Could not fetch VIX term structure: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def fetch_extended_vix_data(days=252):
    """Fetch extended VIX data for percentile analysis"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        vix_data = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        
        if vix_data.empty:
            return None
            
        # Handle multi-level columns
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.droplevel(1)
        
        return vix_data
    except Exception as e:
        st.warning(f"Could not fetch extended VIX data: {str(e)}")
        return None

def calculate_vix_percentiles(vix_data, current_vix):
    """Calculate VIX percentiles and regime analysis"""
    if vix_data is None or vix_data.empty:
        return {}
    
    close_prices = vix_data['Close'].dropna()
    
    if len(close_prices) == 0:
        return {}
    
    percentiles = {
        'current': current_vix,
        '1st': np.percentile(close_prices, 1),
        '5th': np.percentile(close_prices, 5),
        '10th': np.percentile(close_prices, 10),
        '25th': np.percentile(close_prices, 25),
        '50th': np.percentile(close_prices, 50),
        '75th': np.percentile(close_prices, 75),
        '90th': np.percentile(close_prices, 90),
        '95th': np.percentile(close_prices, 95),
        '99th': np.percentile(close_prices, 99),
        'mean': close_prices.mean(),
        'std': close_prices.std(),
        'current_percentile': (close_prices < current_vix).mean() * 100
    }
    
    return percentiles

def create_vix_chart(vix_data, days=90):
    """Create comprehensive VIX chart"""
    if vix_data is None or vix_data.empty:
        return None
    
    # Get recent data
    recent_data = vix_data.tail(days)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['VIX Level & Market Regimes', 'VIX Volume'],
        row_heights=[0.7, 0.3]
    )
    
    # Main VIX line
    fig.add_trace(
        go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name='VIX',
            line=dict(color='#8b5cf6', width=3),
            hovertemplate='<b>VIX</b><br>Date: %{x}<br>Level: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add regime zones
    fig.add_hline(y=15, line_dash="dash", line_color="#10b981", opacity=0.6, row=1, 
                  annotation_text="Calm Markets (< 15)")
    fig.add_hline(y=19, line_dash="dash", line_color="#06b6d4", opacity=0.6, row=1,
                  annotation_text="Normal Markets (15-19)")
    fig.add_hline(y=25, line_dash="dash", line_color="#f59e0b", opacity=0.6, row=1,
                  annotation_text="Choppy Markets (19-25)")
    fig.add_hline(y=35, line_dash="dash", line_color="#ef4444", opacity=0.6, row=1,
                  annotation_text="High Volatility (25-35)")
    
    # Fill regime zones
    fig.add_hrect(y0=0, y1=15, fillcolor="#10b981", opacity=0.1, row=1)
    fig.add_hrect(y0=15, y1=19, fillcolor="#06b6d4", opacity=0.1, row=1)
    fig.add_hrect(y0=19, y1=25, fillcolor="#f59e0b", opacity=0.1, row=1)
    fig.add_hrect(y0=25, y1=35, fillcolor="#ef4444", opacity=0.1, row=1)
    fig.add_hrect(y0=35, y1=100, fillcolor="#7f1d1d", opacity=0.15, row=1)
    
    # Volume chart
    if 'Volume' in recent_data.columns:
        fig.add_trace(
            go.Bar(
                x=recent_data.index,
                y=recent_data['Volume'],
                name='Volume',
                marker_color='rgba(99, 102, 241, 0.6)',
                hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title='📊 VIX Analysis with Market Regime Zones',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="VIX Level", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig

def create_vix_distribution_chart(vix_data):
    """Create VIX distribution and percentile chart"""
    if vix_data is None or vix_data.empty:
        return None
    
    close_prices = vix_data['Close'].dropna()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['VIX Distribution', 'VIX Percentiles'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=close_prices,
            nbinsx=30,
            name='VIX Distribution',
            marker_color='rgba(139, 92, 246, 0.7)',
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Box plot for percentiles
    fig.add_trace(
        go.Box(
            y=close_prices,
            name='VIX Range',
            marker_color='#6366f1',
            boxpoints='outliers'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='📈 VIX Statistical Analysis',
        height=400,
        showlegend=False
    )
    
    return fig

def create_vix_term_structure_chart(term_structure):
    """Create VIX term structure chart"""
    if not term_structure or len(term_structure) < 2:
        return None
    
    # Map to approximate days
    term_mapping = {
        "VIX9D": 9,
        "VIX": 30,
        "VIX3M": 90,
        "VIX6M": 180
    }
    
    valid_data = [(term_mapping.get(k), v) for k, v in term_structure.items() 
                  if k in term_mapping and v is not None]
    
    if len(valid_data) < 2:
        return None
    
    valid_data.sort()  # Sort by days
    days, values = zip(*valid_data)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=days,
            y=values,
            mode='lines+markers',
            name='VIX Term Structure',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=10, color='#6366f1'),
            hovertemplate='<b>%{text}</b><br>Days: %{x}<br>VIX: %{y:.2f}<extra></extra>',
            text=[f"VIX{d}D" if d != 30 else "VIX" for d in days]
        )
    )
    
    # Determine contango vs backwardation
    if len(values) >= 2:
        slope = (values[-1] - values[0]) / (days[-1] - days[0])
        structure_type = "Contango" if slope > 0 else "Backwardation"
        color = "#10b981" if slope > 0 else "#ef4444"
        
        fig.add_annotation(
            x=max(days) * 0.7,
            y=max(values) * 0.9,
            text=f"Structure: {structure_type}",
            showarrow=False,
            font=dict(size=14, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1
        )
    
    fig.update_layout(
        title='📊 VIX Term Structure',
        xaxis_title='Days to Expiration',
        yaxis_title='Implied Volatility (%)',
        height=400,
        showlegend=False
    )
    
    return fig

def render_vix_analysis_tab(results, vix_data, session_tickers):
    """Render the complete VIX Analysis tab"""
    
    st.markdown("### 📉 Comprehensive VIX Market Analysis")
    
    # Fetch current VIX data
    if vix_data is not None and not vix_data.empty:
        current_vix = vix_data['VIX_Close'].iloc[-1]
    else:
        # Fallback: fetch current VIX
        try:
            vix_ticker = yf.Ticker("^VIX")
            current_data = vix_ticker.history(period='1d')
            current_vix = current_data['Close'].iloc[-1] if not current_data.empty else None
        except:
            current_vix = None
    
    if current_vix is None:
        st.error("❌ Unable to fetch current VIX data. Please check your internet connection.")
        return
    
    # === CURRENT VIX CONDITION ===
    condition, condition_class, icon, regime = get_vix_condition(current_vix)
    trading_strategy = get_vix_trading_strategy(current_vix)
    
    st.markdown(f"""
    <div class="{condition_class}">
        <h2>{icon} Current VIX: {current_vix:.2f}</h2>
        <p><strong>Market Condition:</strong> {condition}</p>
        <p><strong>Trading Regime:</strong> {regime.replace('_', ' ').title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # === TRADING STRATEGY RECOMMENDATIONS ===
    st.markdown("#### 🎯 VIX-Based Trading Strategy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        **Primary Strategy:**  
        {trading_strategy['primary']}
        
        **Risk Level:** {trading_strategy['risk']}
        """)
    
    with col2:
        st.markdown(f"""
        **Secondary Strategy:**  
        {trading_strategy['secondary']}
        
        **Timeframe:** {trading_strategy['timeframe']}
        """)
    
    with col3:
        st.markdown(f"""
        **VIX Play:**  
        {trading_strategy['vix_play']}
        """)
    
    # === EXTENDED VIX ANALYSIS ===
    st.markdown("#### 📊 Extended VIX Analysis")
    
    # Fetch extended data for percentile analysis
    extended_vix_data = fetch_extended_vix_data(days=252)  # 1 year
    
    if extended_vix_data is not None:
        percentiles = calculate_vix_percentiles(extended_vix_data, current_vix)
        
        if percentiles:
            # Display percentiles
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Percentile", f"{percentiles['current_percentile']:.1f}%")
            with col2:
                st.metric("52W Mean", f"{percentiles['mean']:.2f}")
            with col3:
                st.metric("52W Median", f"{percentiles['50th']:.2f}")
            with col4:
                st.metric("52W Range", f"{percentiles['5th']:.1f} - {percentiles['95th']:.1f}")
            with col5:
                volatility_regime = "High" if percentiles['current_percentile'] > 75 else "Low" if percentiles['current_percentile'] < 25 else "Normal"
                st.metric("Volatility Regime", volatility_regime)
        
        # Create charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            vix_chart = create_vix_chart(extended_vix_data, days=90)
            if vix_chart:
                st.plotly_chart(vix_chart, use_container_width=True)
        
        with chart_col2:
            dist_chart = create_vix_distribution_chart(extended_vix_data)
            if dist_chart:
                st.plotly_chart(dist_chart, use_container_width=True)
    
    # === VIX TERM STRUCTURE ===
    st.markdown("#### ⏳ VIX Term Structure Analysis")
    
    term_structure = fetch_vix_term_structure()
    
    if term_structure:
        # Display term structure metrics
        ts_col1, ts_col2 = st.columns(2)
        
        with ts_col1:
            st.markdown("**Current Term Structure:**")
            for name, value in term_structure.items():
                if value is not None:
                    # Ensure value is a scalar for formatting
                    if hasattr(value, 'item'):
                        display_value = value.item()
                    elif hasattr(value, '__iter__') and not isinstance(value, str):
                        display_value = float(value)
                    else:
                        display_value = float(value)
                    st.markdown(f"- **{name}**: {display_value:.2f}")
        
        with ts_col2:
            ts_chart = create_vix_term_structure_chart(term_structure)
            if ts_chart:
                st.plotly_chart(ts_chart, use_container_width=True)
    else:
        st.info("📊 VIX term structure data not available")
    
    # === VIX HISTORICAL PATTERNS ===
    if vix_data is not None:
        st.markdown("#### 📈 VIX Statistics (Analysis Period)")
        
        vix_stats = vix_data['VIX_Close'].describe()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Mean VIX", f"{vix_stats['mean']:.2f}")
        with stat_col2:
            st.metric("Std Dev", f"{vix_stats['std']:.2f}")
        with stat_col3:
            st.metric("Min VIX", f"{vix_stats['min']:.2f}")
        with stat_col4:
            st.metric("Max VIX", f"{vix_stats['max']:.2f}")
    
    # === AI ANALYSIS SECTION ===
    st.markdown("### 🤖 AI-Powered VIX Analysis")
    
    if LLM_AVAILABLE and AI_FORMATTER_AVAILABLE:
        ai_col1, ai_col2 = st.columns([1, 3])
        
        with ai_col1:
            if st.button("🧠 Generate VIX AI Analysis", type="primary", key="generate_vix_ai"):
                _generate_vix_ai_analysis(current_vix, percentiles if 'percentiles' in locals() else {}, 
                                        term_structure, trading_strategy, session_tickers)
        
        with ai_col2:
            _display_vix_ai_results()
    
    elif AI_FORMATTER_AVAILABLE:
        st.info("🤖 AI analysis requires the LLM analyzer. Please ensure llm_analysis.py is available.")
    else:
        st.info("🤖 AI analysis requires the unified AI formatter. Please ensure shared/ai_formatter.py is available.")
    
    # === VIX TRADING RECOMMENDATIONS ===
    with st.expander("📋 Detailed VIX Trading Guide"):
        st.markdown("""
        #### 🎯 VIX-Based Trading Strategies
        
        **Low VIX (< 15) - Calm Markets:**
        - ✅ Sell puts and put spreads
        - ✅ Buy calls and call spreads  
        - ✅ Short volatility (VXX, UVXY)
        - ❌ Avoid buying protection
        
        **Normal VIX (15-19) - Trending Markets:**
        - ✅ Iron condors and strangles
        - ✅ Covered calls
        - ✅ Trend-following strategies
        - ⚠️ Moderate position sizing
        
        **Elevated VIX (19-25) - Choppy Markets:**
        - ✅ Protective strategies
        - ✅ Shorter-term trades
        - ✅ Hedge existing positions
        - ❌ Reduce position size
        
        **High VIX (25-35) - Volatile Markets:**
        - ✅ Cash positions
        - ✅ Protective puts
        - ✅ Long volatility trades
        - ❌ No new aggressive positions
        
        **Extreme VIX (> 35) - Crisis Mode:**
        - 🚨 **CASH ONLY**
        - 🚨 Close existing positions
        - 🚨 Wait for stabilization
        - 🚨 Prepare for opportunities
        """)

def _generate_vix_ai_analysis(current_vix, percentiles, term_structure, trading_strategy, session_tickers):
    """Generate AI analysis for VIX conditions"""
    
    try:
        llm_analyzer = get_llm_analyzer()
        
        # Prepare VIX data for AI
        condition, _, _, regime = get_vix_condition(current_vix)
        
        percentile_info = f"Current percentile: {percentiles.get('current_percentile', 'N/A'):.1f}%" if percentiles else "Percentile data unavailable"
        
        # Format term structure with safe value extraction
        def safe_format_value(v):
            if hasattr(v, 'item'):
                return v.item()
            elif hasattr(v, '__iter__') and not isinstance(v, str):
                return float(v)
            else:
                return float(v)
        
        term_structure_info = "Term structure: " + ", ".join([f"{k}: {safe_format_value(v):.2f}" for k, v in term_structure.items() if v is not None]) if term_structure else "Term structure unavailable"
        
        tickers_context = f"Current portfolio tickers: {', '.join(session_tickers)}" if session_tickers else "No specific tickers selected"
        
        analysis_prompt = f"""
        Provide professional VIX analysis and trading guidance:

        **Current Market Condition:**
        - VIX Level: {current_vix:.2f}
        - Market Regime: {condition}
        - Trading Classification: {regime.replace('_', ' ').title()}
        - {percentile_info}
        - {term_structure_info}
        
        **Recommended Strategy:**
        - Primary: {trading_strategy['primary']}
        - Secondary: {trading_strategy['secondary']}
        - VIX Play: {trading_strategy['vix_play']}
        - Risk Level: {trading_strategy['risk']}
        
        **Portfolio Context:**
        {tickers_context}

        Please provide:
        1. **Market Assessment** - What does current VIX tell us about market sentiment?
        2. **Trading Implications** - How should traders position given this VIX level?
        3. **Risk Management** - What precautions are needed in this environment?
        4. **Opportunity Analysis** - What opportunities exist in this volatility regime?
        5. **Timeline Outlook** - How long might this regime persist?
        6. **Portfolio Adjustments** - Specific recommendations for the current ticker portfolio

        Keep analysis practical and actionable for active traders and portfolio managers.
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
            if 'ai_vix_analysis' not in st.session_state:
                st.session_state.ai_vix_analysis = {}
            
            analysis_key = f"vix_analysis_{current_vix:.2f}_{regime}"
            st.session_state.ai_vix_analysis[analysis_key] = ai_response
            st.success("✅ VIX AI Analysis completed!")
        else:
            st.warning("⚠️ AI Analysis could not be generated. Check LLM configuration.")
            
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        st.info("💡 AI analysis requires proper LLM configuration.")

def _display_vix_ai_results():
    """Display AI VIX analysis results"""
    
    if 'ai_vix_analysis' in st.session_state and st.session_state.ai_vix_analysis:
        # Get the latest analysis
        latest_key = list(st.session_state.ai_vix_analysis.keys())[-1]
        ai_content = st.session_state.ai_vix_analysis[latest_key]
        
        if AI_FORMATTER_AVAILABLE:
            display_ai_analysis(
                ai_content=ai_content,
                analysis_type="VIX Market Analysis",
                tab_color=get_tab_color("macro"),
                analysis_key=latest_key,
                session_key="ai_vix_analysis",
                regenerate_key="regenerate_vix_ai",
                clear_key="clear_vix_ai",
                show_debug=True,
                show_metadata=True
            )
        else:
            st.markdown("#### 🧠 AI VIX Analysis Results")
            content_text = str(ai_content.get('content', ai_content)) if isinstance(ai_content, dict) else str(ai_content)
            st.markdown(content_text)
    
    else:
        if AI_FORMATTER_AVAILABLE:
            display_ai_placeholder(
                analysis_type="VIX Market Analysis",
                features_list=[
                    "Current market regime assessment and implications",
                    "VIX-based trading strategy recommendations",
                    "Risk management guidance for current volatility environment",
                    "Volatility term structure analysis and opportunities",
                    "Market timing and regime persistence forecasting",
                    "Portfolio-specific adjustments based on VIX conditions"
                ]
            )
        else:
            st.info("👆 Click 'Generate VIX AI Analysis' to get intelligent market condition insights and trading recommendations") 