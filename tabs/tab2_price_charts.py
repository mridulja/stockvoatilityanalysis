"""
Tab 2: Price Charts - Stock Volatility Analyzer

This module contains the price charts tab functionality with comprehensive chart analysis,
technical indicators, AI analysis, and trading signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from core import (
    get_current_price, get_vix_condition, create_enhanced_price_chart,
    format_percentage, format_currency
)

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


def render_price_charts_tab(results, vix_data, session_tickers):
    """
    Render the Price Charts tab with comprehensive chart analysis
    
    Args:
        results (dict): Analysis results from the main app
        vix_data (pd.DataFrame): VIX data
        session_tickers (list): List of selected tickers
    """
    
    st.subheader("📈 Advanced Interactive Price Charts")
    
    # === CHART CONFIGURATION SECTION ===
    st.markdown("### ⚙️ Chart Configuration")
    
    config_col1, config_col2, config_col3, config_col4 = st.columns(4)
    
    with config_col1:
        chart_ticker = st.selectbox(
            "📊 Select Ticker:",
            session_tickers,
            help="Choose ticker for detailed chart analysis",
            key="chart_ticker"
        )
    
    with config_col2:
        chart_timeframe = st.selectbox(
            "⏰ Timeframe:",
            ['hourly', 'daily', 'weekly'],
            index=1,  # Default to daily
            help="Select data timeframe for analysis",
            key="chart_timeframe"
        )
    
    with config_col3:
        chart_type = st.selectbox(
            "📈 Chart Type:",
            ['Candlestick', 'Line', 'OHLC', 'Area'],
            help="Choose chart visualization style",
            key="chart_type"
        )
    
    with config_col4:
        show_volume = st.checkbox(
            "📊 Show Volume",
            value=True,
            help="Display volume subplot",
            key="show_volume"
        )
    
    # === TECHNICAL INDICATORS SECTION ===
    st.markdown("### 🔧 Technical Indicators")
    
    indicators_col1, indicators_col2, indicators_col3 = st.columns(3)
    
    with indicators_col1:
        st.markdown("**📈 Trend Indicators**")
        show_sma_20 = st.checkbox("SMA 20", value=True, key="sma_20")
        show_sma_50 = st.checkbox("SMA 50", value=False, key="sma_50")
        show_ema_12 = st.checkbox("EMA 12", value=False, key="ema_12")
        show_ema_26 = st.checkbox("EMA 26", value=False, key="ema_26")
    
    with indicators_col2:
        st.markdown("**📊 Volatility Indicators**")
        show_bb = st.checkbox("Bollinger Bands", value=True, key="bollinger_bands")
        show_atr_bands = st.checkbox("ATR Bands", value=True, key="atr_bands")
        show_support_resistance = st.checkbox("Support/Resistance", value=False, key="support_resistance")
    
    with indicators_col3:
        st.markdown("**⚡ Market Indicators**")
        show_vix_overlay = st.checkbox("VIX Overlay", value=False, key="vix_overlay")
        show_volatility_regime = st.checkbox("Volatility Regime", value=True, key="vol_regime")
        show_price_levels = st.checkbox("Key Price Levels", value=False, key="price_levels")
    
    # === MAIN CHART GENERATION ===
    if chart_ticker in results and chart_timeframe in results[chart_ticker] and results[chart_ticker][chart_timeframe]:
        chart_data = results[chart_ticker][chart_timeframe]['data']
        
        if chart_data is not None and not chart_data.empty:
            st.markdown("### 📊 Enhanced Price Chart")
            
            # Calculate additional indicators
            chart_data_enhanced = chart_data.copy()
            
            # Simple Moving Averages
            if show_sma_20 and len(chart_data_enhanced) >= 20:
                chart_data_enhanced['SMA_20'] = chart_data_enhanced['Close'].rolling(window=20).mean()
            if show_sma_50 and len(chart_data_enhanced) >= 50:
                chart_data_enhanced['SMA_50'] = chart_data_enhanced['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            if show_ema_12:
                chart_data_enhanced['EMA_12'] = chart_data_enhanced['Close'].ewm(span=12).mean()
            if show_ema_26:
                chart_data_enhanced['EMA_26'] = chart_data_enhanced['Close'].ewm(span=26).mean()
            
            # Bollinger Bands
            if show_bb and len(chart_data_enhanced) >= 20:
                sma_20 = chart_data_enhanced['Close'].rolling(window=20).mean()
                std_20 = chart_data_enhanced['Close'].rolling(window=20).std()
                chart_data_enhanced['BB_Upper'] = sma_20 + (2 * std_20)
                chart_data_enhanced['BB_Lower'] = sma_20 - (2 * std_20)
                chart_data_enhanced['BB_Middle'] = sma_20
            
            # ATR Bands
            if show_atr_bands and 'true_range' in chart_data_enhanced.columns:
                atr_14 = chart_data_enhanced['true_range'].rolling(window=min(14, len(chart_data_enhanced))).mean()
                chart_data_enhanced['ATR_Upper'] = chart_data_enhanced['Close'] + (1.5 * atr_14)
                chart_data_enhanced['ATR_Lower'] = chart_data_enhanced['Close'] - (1.5 * atr_14)
            
            # Create enhanced price chart
            enhanced_chart = create_enhanced_price_chart(
                chart_data_enhanced, 
                chart_ticker, 
                chart_timeframe,
                chart_type,
                show_volume,
                {
                    'sma_20': show_sma_20,
                    'sma_50': show_sma_50,
                    'ema_12': show_ema_12,
                    'ema_26': show_ema_26,
                    'bollinger_bands': show_bb,
                    'atr_bands': show_atr_bands,
                    'vix_overlay': show_vix_overlay and vix_data is not None,
                    'volatility_regime': show_volatility_regime
                },
                vix_data
            )
            
            if enhanced_chart:
                st.plotly_chart(enhanced_chart, use_container_width=True)
            
            # === CHART STATISTICS & ANALYSIS ===
            st.markdown("### 📊 Chart Statistics & Analysis")
            
            # Get current price for context
            current_price = get_current_price(chart_ticker)
            chart_stats = results[chart_ticker][chart_timeframe]['stats']
            
            stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
            
            with stats_col1:
                st.metric(
                    "Current Price", 
                    f"${current_price:.2f}" if current_price else "N/A",
                    f"{((current_price - chart_data_enhanced['Close'].iloc[-1]) / chart_data_enhanced['Close'].iloc[-1] * 100):+.2f}%" if current_price else None
                )
            
            with stats_col2:
                atr_val = results[chart_ticker][chart_timeframe]['atr']
                atr_pct = (atr_val / current_price * 100) if current_price and atr_val > 0 else 0
                st.metric(
                    "ATR", 
                    f"${atr_val:.2f}" if atr_val > 0 else "N/A",
                    f"{atr_pct:.1f}% of price" if atr_pct > 0 else None
                )
            
            with stats_col3:
                period_high = chart_data_enhanced['High'].max()
                period_low = chart_data_enhanced['Low'].min()
                st.metric("Period High", f"${period_high:.2f}")
                st.metric("Period Low", f"${period_low:.2f}")
            
            with stats_col4:
                if 'Volume' in chart_data_enhanced.columns:
                    avg_volume = chart_data_enhanced['Volume'].mean()
                    recent_volume = chart_data_enhanced['Volume'].tail(5).mean()
                    volume_change = ((recent_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
                    st.metric(
                        "Avg Volume", 
                        f"{avg_volume:,.0f}" if avg_volume > 0 else "N/A",
                        f"{volume_change:+.1f}% vs avg" if volume_change != 0 else None
                    )
            
            with stats_col5:
                volatility = results[chart_ticker][chart_timeframe]['volatility']
                price_range_pct = ((period_high - period_low) / period_low * 100) if period_low > 0 else 0
                st.metric("Price Range", f"{price_range_pct:.1f}%")
                st.metric("Volatility", f"${volatility:.2f}" if volatility > 0 else "N/A")
            
            # === TECHNICAL ANALYSIS INSIGHTS ===
            st.markdown("### 🔍 Technical Analysis Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("#### 📈 Price Action Analysis")
                
                # Recent trend analysis
                if len(chart_data_enhanced) >= 5:
                    recent_closes = chart_data_enhanced['Close'].tail(5)
                    trend_strength = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0] * 100
                    
                    if trend_strength > 2:
                        trend_status = "🟢 Strong Uptrend"
                    elif trend_strength > 0.5:
                        trend_status = "📈 Mild Uptrend"
                    elif trend_strength < -2:
                        trend_status = "🔴 Strong Downtrend"
                    elif trend_strength < -0.5:
                        trend_status = "📉 Mild Downtrend"
                    else:
                        trend_status = "➡️ Sideways/Neutral"
                    
                    st.write(f"**5-Period Trend**: {trend_status} ({trend_strength:+.2f}%)")
                
                # Support/Resistance levels
                if len(chart_data_enhanced) >= 20:
                    recent_data = chart_data_enhanced.tail(20)
                    resistance = recent_data['High'].max()
                    support = recent_data['Low'].min()
                    
                    current_close = chart_data_enhanced['Close'].iloc[-1]
                    resistance_distance = ((resistance - current_close) / current_close * 100)
                    support_distance = ((current_close - support) / current_close * 100)
                    
                    st.write(f"**Resistance**: ${resistance:.2f} (+{resistance_distance:.1f}%)")
                    st.write(f"**Support**: ${support:.2f} (-{support_distance:.1f}%)")
                
                # Bollinger Band position
                if show_bb and 'BB_Upper' in chart_data_enhanced.columns:
                    latest_close = chart_data_enhanced['Close'].iloc[-1]
                    bb_upper = chart_data_enhanced['BB_Upper'].iloc[-1]
                    bb_lower = chart_data_enhanced['BB_Lower'].iloc[-1]
                    
                    if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                        bb_position = (latest_close - bb_lower) / (bb_upper - bb_lower)
                        
                        if bb_position > 0.8:
                            bb_status = "🔴 Near Upper Band (Overbought)"
                        elif bb_position < 0.2:
                            bb_status = "🟢 Near Lower Band (Oversold)"
                        else:
                            bb_status = f"🟡 Mid-Band ({bb_position:.1%} position)"
                        
                        st.write(f"**Bollinger Position**: {bb_status}")
            
            with insights_col2:
                st.markdown("#### ⚡ Volatility Analysis")
                
                # ATR trend analysis
                if 'true_range' in chart_data_enhanced.columns and len(chart_data_enhanced) >= 14:
                    atr_series = chart_data_enhanced['true_range'].rolling(window=14).mean()
                    recent_atr = atr_series.tail(5).mean()
                    historical_atr = atr_series.mean()
                    
                    if recent_atr > historical_atr * 1.2:
                        atr_trend = "🔥 Volatility Expanding"
                    elif recent_atr < historical_atr * 0.8:
                        atr_trend = "❄️ Volatility Contracting"
                    else:
                        atr_trend = "🔄 Volatility Stable"
                    
                    st.write(f"**ATR Trend**: {atr_trend}")
                    st.write(f"**Recent ATR**: ${recent_atr:.2f}")
                    st.write(f"**Historical ATR**: ${historical_atr:.2f}")
                
                # Volume analysis
                if 'Volume' in chart_data_enhanced.columns:
                    recent_volume = chart_data_enhanced['Volume'].tail(5).mean()
                    avg_volume = chart_data_enhanced['Volume'].mean()
                    
                    if recent_volume > avg_volume * 1.5:
                        volume_status = "🔥 High Volume Activity"
                    elif recent_volume < avg_volume * 0.5:
                        volume_status = "😴 Low Volume Activity"
                    else:
                        volume_status = "🔄 Normal Volume"
                    
                    st.write(f"**Volume Status**: {volume_status}")
                
                # VIX correlation (if available)
                if vix_data is not None and show_vix_overlay:
                    current_vix = vix_data['VIX_Close'].iloc[-1]
                    condition, _, icon = get_vix_condition(current_vix)
                    st.write(f"**VIX Context**: {icon} {current_vix:.1f}")
                    st.write(f"**Market Condition**: {condition.split(' - ')[0]}")
            
            # === TRADING SIGNALS SECTION ===
            with st.expander("🎯 Trading Signals & Recommendations"):
                st.markdown("#### 📊 Technical Signals Summary")
                
                signals = []
                
                # Moving average signals
                if show_sma_20 and 'SMA_20' in chart_data_enhanced.columns:
                    current_price_chart = chart_data_enhanced['Close'].iloc[-1]
                    sma_20_current = chart_data_enhanced['SMA_20'].iloc[-1]
                    
                    if not pd.isna(sma_20_current):
                        if current_price_chart > sma_20_current:
                            signals.append("📈 Price above SMA(20) - Bullish")
                        else:
                            signals.append("📉 Price below SMA(20) - Bearish")
                
                # ATR-based signals
                if atr_val > 0 and current_price:
                    atr_pct = (atr_val / current_price) * 100
                    if atr_pct > 3:
                        signals.append("⚠️ High volatility - Reduce position size")
                    elif atr_pct < 1:
                        signals.append("😴 Low volatility - Consider breakout strategies")
                    else:
                        signals.append("🎯 Normal volatility - Good for trend following")
                
                # Volume signals
                if 'Volume' in chart_data_enhanced.columns:
                    recent_volume = chart_data_enhanced['Volume'].tail(3).mean()
                    avg_volume = chart_data_enhanced['Volume'].mean()
                    
                    if recent_volume > avg_volume * 1.5:
                        signals.append("📊 High volume confirms price movement")
                    elif recent_volume < avg_volume * 0.5:
                        signals.append("⚠️ Low volume - Weak price movement")
                
                # Display signals
                if signals:
                    for signal in signals:
                        st.write(f"• {signal}")
                else:
                    st.write("• No clear signals detected at current levels")
                
                # Risk management recommendations
                st.markdown("#### 🛡️ Risk Management")
                
                if atr_val > 0:
                    stop_loss_distance = atr_val * 1.5
                    profit_target_distance = atr_val * 2.5
                    
                    st.write(f"**Suggested Stop Loss**: ±${stop_loss_distance:.2f} ({atr_val * 1.5 / current_price * 100:.1f}%)" if current_price else f"±${stop_loss_distance:.2f}")
                    st.write(f"**Profit Target**: ±${profit_target_distance:.2f} ({atr_val * 2.5 / current_price * 100:.1f}%)" if current_price else f"±${profit_target_distance:.2f}")
                    st.write(f"**Risk:Reward Ratio**: 1:1.67 (based on 1.5x ATR stop, 2.5x ATR target)")
            
            # === AI ANALYSIS SECTION ===
            _render_ai_analysis_section(chart_ticker, chart_timeframe, current_price, atr_val, atr_pct, 
                                      period_high, period_low, price_range_pct, volatility, 
                                      chart_data_enhanced, show_sma_20, show_bb, show_atr_bands, 
                                      show_vix_overlay, vix_data, locals())
        
        else:
            st.warning(f"No chart data available for {chart_ticker} - {chart_timeframe}")
            
    else:
        st.info("Please run the Enhanced Analysis first to generate chart data.")
        
        # Show sample chart instructions
        st.markdown("""
        ### 📋 Chart Features Available After Analysis:
        
        **📈 Chart Types:**
        - Candlestick (OHLC visualization)
        - Line charts (clean price action)
        - OHLC bars (detailed price info)
        - Area charts (filled price movement)
        
        **🔧 Technical Indicators:**
        - Moving Averages (SMA 20/50, EMA 12/26)
        - Bollinger Bands (volatility bands)
        - ATR Bands (volatility-based support/resistance)
        - Volume analysis and overlays
        
        **📊 Advanced Features:**
        - Real-time price updates
        - Support/resistance detection
        - Volatility regime analysis
        - VIX correlation overlay
        - Trading signal generation
        - Risk management calculations
        """)


def _render_ai_analysis_section(chart_ticker, chart_timeframe, current_price, atr_val, atr_pct, 
                               period_high, period_low, price_range_pct, volatility, 
                               chart_data_enhanced, show_sma_20, show_bb, show_atr_bands, 
                               show_vix_overlay, vix_data, local_vars):
    """
    Render the AI Analysis section for the Price Charts tab
    
    Args:
        chart_ticker (str): Selected ticker symbol
        chart_timeframe (str): Selected timeframe
        current_price (float): Current price
        atr_val (float): ATR value
        atr_pct (float): ATR percentage
        period_high (float): Period high price
        period_low (float): Period low price
        price_range_pct (float): Price range percentage
        volatility (float): Volatility value
        chart_data_enhanced (pd.DataFrame): Enhanced chart data
        show_sma_20 (bool): Whether SMA 20 is enabled
        show_bb (bool): Whether Bollinger Bands are enabled
        show_atr_bands (bool): Whether ATR Bands are enabled
        show_vix_overlay (bool): Whether VIX overlay is enabled
        vix_data (pd.DataFrame): VIX data
        local_vars (dict): Local variables from the calling function
    """
    
    st.markdown("### 🤖 AI-Powered Chart Analysis")
    
    # Check if LLM is available by trying to import
    try:
        from llm_analysis import get_llm_analyzer
        LLM_AVAILABLE = True
    except ImportError:
        LLM_AVAILABLE = False
    
    if LLM_AVAILABLE:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Use form to prevent page refresh and tab switching
            with st.form("ai_analysis_form"):
                ai_button_clicked = st.form_submit_button("🤖 Generate AI Analysis", type="primary", help="Get AI-powered technical analysis and recommendations")
            
            if ai_button_clicked:
                # Prepare data for AI analysis
                ai_analysis_data = {
                    'ticker': chart_ticker,
                    'timeframe': chart_timeframe,
                    'current_price': current_price,
                    'atr': atr_val,
                    'atr_percentage': atr_pct,
                    'period_high': period_high,
                    'period_low': period_low,
                    'price_range_pct': price_range_pct,
                    'volatility': volatility,
                    'trend_strength': local_vars.get('trend_strength', 0),
                    'trend_status': local_vars.get('trend_status', "Unknown"),
                    'bb_position': local_vars.get('bb_position', None),
                    'bb_status': local_vars.get('bb_status', None),
                    'atr_trend': local_vars.get('atr_trend', "Unknown"),
                    'volume_status': local_vars.get('volume_status', "Unknown"),
                    'recent_closes': chart_data_enhanced['Close'].tail(10).tolist(),
                    'technical_indicators': {
                        'sma_20_enabled': show_sma_20,
                        'sma_50_enabled': False,  # Default to False since not passed
                        'bollinger_bands': show_bb,
                        'atr_bands': show_atr_bands,
                        'vix_overlay': show_vix_overlay
                    },
                    'vix_data': {
                        'current_vix': vix_data['VIX_Close'].iloc[-1] if vix_data is not None else None,
                        'vix_condition': get_vix_condition(vix_data['VIX_Close'].iloc[-1])[0] if vix_data is not None else "Unknown"
                    }
                }
                
                # Generate AI analysis
                with st.spinner("🤖 AI is analyzing the chart data..."):
                    try:
                        llm_analyzer = get_llm_analyzer()
                        
                        # Prepare VIX formatting
                        vix_display = f"{ai_analysis_data['vix_data']['current_vix']:.1f}" if ai_analysis_data['vix_data']['current_vix'] is not None else 'N/A'
                        
                        # Create comprehensive prompt for chart analysis
                        analysis_prompt = f"""
                        Analyze the following stock chart data and provide professional trading insights:

                        **Stock**: {chart_ticker} ({chart_timeframe} timeframe)
                        **Current Price**: ${current_price:.2f}
                        **ATR**: ${atr_val:.2f} ({atr_pct:.1f}% of price)
                        **Price Range**: {price_range_pct:.1f}% (High: ${period_high:.2f}, Low: ${period_low:.2f})
                        **Volatility**: ${volatility:.2f}
                        **Trend**: {local_vars.get('trend_status', 'Unknown')} ({local_vars.get('trend_strength', 0):+.2f}%)
                        **ATR Trend**: {local_vars.get('atr_trend', 'Unknown')}
                        **Volume**: {local_vars.get('volume_status', 'Unknown')}
                        **VIX Context**: {ai_analysis_data['vix_data']['vix_condition']} (VIX: {vix_display})
                        
                        **Technical Indicators Active**: 
                        - SMA 20: {show_sma_20}
                        - Bollinger Bands: {show_bb} {('- ' + local_vars.get('bb_status', '')) if local_vars.get('bb_status') else ''}
                        - ATR Bands: {show_atr_bands}
                        - VIX Overlay: {show_vix_overlay}

                        **Recent Price Action**: {', '.join([f'${p:.2f}' for p in chart_data_enhanced['Close'].tail(5).tolist()])}

                        Please provide:
                        1. **Market Assessment** (2-3 sentences on current market condition)
                        2. **Technical Setup** (Key levels, patterns, indicator signals)
                        3. **Trading Recommendation** (Bullish/Bearish/Neutral with reasoning)
                        4. **Risk Management** (Entry, stop-loss, profit targets)
                        5. **Time Horizon** (Best timeframe for this setup)

                        Keep the analysis concise, professional, and actionable for traders.
                        """
                        
                        # Try different method names that might be available
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
                            # Last resort - try to call it directly
                            ai_response = "AI Analysis functionality needs to be configured. Please check llm_analysis.py for the correct method name."
                        
                        if ai_response:
                            # Store in session state for persistence
                            if 'ai_analysis' not in st.session_state:
                                st.session_state.ai_analysis = {}
                            st.session_state.ai_analysis[f"{chart_ticker}_{chart_timeframe}"] = ai_response
                            st.success("✅ AI Analysis completed!")
                        else:
                            st.warning("⚠️ AI Analysis could not be generated. Check LLM configuration.")
                        
                    except Exception as e:
                        st.error(f"AI Analysis Error: {str(e)}")
                        st.info("💡 AI analysis requires proper LLM configuration. Check llm_analysis.py setup.")
                        
                        # Show available methods for debugging
                        try:
                            llm_analyzer = get_llm_analyzer()
                            available_methods = [method for method in dir(llm_analyzer) if not method.startswith('_')]
                            st.write(f"Available methods: {available_methods}")
                        except:
                            pass
        
        with col2:
            # Display AI analysis using unified formatter
            analysis_key = f"{chart_ticker}_{chart_timeframe}"
            
            if 'ai_analysis' in st.session_state and analysis_key in st.session_state.ai_analysis:
                ai_analysis = st.session_state.ai_analysis[analysis_key]
                
                if AI_FORMATTER_AVAILABLE:
                    # Use unified AI formatter with technical analysis styling
                    display_ai_analysis(
                        ai_content=ai_analysis,
                        analysis_type="Technical Analysis",
                        tab_color=get_tab_color("technical"),
                        analysis_key=analysis_key,
                        session_key="ai_analysis",
                        regenerate_key="regenerate_tech_ai",
                        clear_key="clear_tech_ai",
                        show_debug=True,
                        show_metadata=True
                    )
                else:
                    # Fallback to simple display
                    st.markdown("#### 🧠 AI Technical Analysis Results")
                    content_text = str(ai_analysis.get('content', ai_analysis)) if isinstance(ai_analysis, dict) else str(ai_analysis)
                    st.markdown(content_text)
                    
                    # Simple action buttons
                    if st.button("🔄 Regenerate", key="regenerate_fallback"):
                        if analysis_key in st.session_state.ai_analysis:
                            del st.session_state.ai_analysis[analysis_key]
                            st.rerun()
            
            else:
                if AI_FORMATTER_AVAILABLE:
                    # Use unified placeholder
                    display_ai_placeholder(
                        analysis_type="Technical Analysis",
                        features_list=[
                            "Market sentiment analysis and trend identification",
                            "Support/resistance level detection",
                            "Chart pattern recognition (triangles, flags, etc.)",
                            "Technical indicator confluence analysis",
                            "Risk-adjusted entry and exit recommendations",
                            "Volatility regime assessment and trading implications"
                        ]
                    )
                else:
                    # Fallback placeholder
                    st.info("👆 Click 'Generate AI Analysis' to get intelligent chart insights")
    
    else:
        if AI_FORMATTER_AVAILABLE:
            # Use unified setup instructions
            display_ai_setup_instructions("Technical Analysis")
        else:
            # Fallback setup instructions
            st.info("🤖 AI Analysis not available. Install and configure `llm_analysis.py` for intelligent chart insights.") 