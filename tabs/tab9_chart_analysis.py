"""
Chart Analysis Tab for Stock Volatility Analyzer

This module provides AI-powered chart analysis capabilities including:
- Image upload for stock charts (PNG, JPG)
- OpenAI o3-mini vision model integration
- Comprehensive pattern recognition and technical analysis
- Price action and volume analysis
- Support/resistance identification
- Options strategy recommendations
- Risk management guidelines

Author: Enhanced by AI Assistant
Date: 2025
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Import shared functions
try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

# Import chart analyzer
try:
    from core.chart_analyzer import ChartAnalyzer, validate_image_file
    CHART_ANALYZER_AVAILABLE = True
except ImportError:
    CHART_ANALYZER_AVAILABLE = False

def render_chart_analysis_tab(results, vix_data, session_tickers):
    """Render the comprehensive Chart Analysis tab"""
    
    # Initialize session state for tab stability
    if 'chart_tab_initialized' not in st.session_state:
        st.session_state.chart_tab_initialized = True
    
    st.markdown("### 📊 AI-Powered Chart Analysis")
    st.markdown("Upload your stock chart images for professional technical analysis and options strategy recommendations.")
    
    # Check if chart analyzer is available
    if not CHART_ANALYZER_AVAILABLE:
        st.error("❌ Chart Analyzer not available. Please ensure core/chart_analyzer.py is properly installed.")
        st.info("💡 This feature requires OpenAI API access and the chart analyzer module.")
        return
    
    # === IMAGE UPLOAD SECTION ===
    st.markdown("#### 📤 Upload Chart Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a chart image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a PNG or JPG image of a stock chart for analysis",
            key="chart_image_uploader"
        )
    
    with col2:
        st.markdown("**Supported formats:**")
        st.markdown("- PNG files")
        st.markdown("- JPG/JPEG files")
        st.markdown("- Max size: 10MB")
    
    # === ANALYSIS OPTIONS ===
    if uploaded_file is not None:
        # Validate file
        is_valid, validation_message = validate_image_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {validation_message}")
            return
        
        st.success(f"✅ {validation_message}")
        
        # Display uploaded image
        st.markdown("#### 🖼️ Uploaded Chart")
        st.image(uploaded_file, caption="Chart for Analysis", use_container_width=True)
        
        # Analysis options
        st.markdown("#### ⚙️ Analysis Options")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            analysis_mode = st.radio(
                "Select Analysis Depth:",
                ["Quick Analysis", "Deep Technical Analysis"],
                help="Quick: Fast pattern recognition and key strategies. Deep: Comprehensive multi-factor analysis."
            )
        
        with analysis_col2:
            additional_context = st.text_area(
                "Additional Context (Optional):",
                placeholder="e.g., Recent earnings, market conditions, specific concerns...",
                height=100,
                help="Provide any additional context about the stock or market conditions"
            )
        
        # Analysis button
        analyze_button = st.button(
            "🧠 Analyze Chart with AI",
            type="primary",
            help="Generate comprehensive chart analysis using OpenAI o3-mini"
        )
        
        if analyze_button:
            # Reset the uploaded file in session state to prevent re-uploads
            if uploaded_file is not None:
                _perform_chart_analysis(uploaded_file, analysis_mode, additional_context)
            else:
                st.error("Please upload a chart image first.")
    
    else:
        # Show placeholder when no image uploaded
        st.markdown("#### 🎯 What You'll Get:")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **📊 Technical Analysis:**
            - Chart pattern recognition
            - Trend analysis and strength
            - Support & resistance levels
            - Price action insights
            """)
            
            st.markdown("""
            **📈 Volume Analysis:**
            - Volume confirmation patterns
            - Accumulation/distribution signals
            - Volume-price relationships
            """)
        
        with feature_col2:
            st.markdown("""
            **🎯 Options Strategies:**
            - Put selling opportunities
            - Call buying/selling strategies
            - Spread recommendations
            - Risk management guidelines
            """)
            
            st.markdown("""
            **⚠️ Risk Management:**
            - Stop-loss recommendations
            - Position sizing guidance
            - Time horizon considerations
            """)
    
    # === ANALYSIS RESULTS SECTION ===
    st.markdown("### 🤖 Analysis Results")
    
    if AI_FORMATTER_AVAILABLE:
        _display_chart_analysis_results()
    else:
        _display_chart_analysis_fallback()
    
    # === STRATEGY IMPLEMENTATION GUIDE ===
    with st.expander("📋 Options Strategy Implementation Guide"):
        st.markdown("""
        #### 🎯 Strategy Selection Based on Chart Patterns
        
        **🟢 Bullish Patterns (Uptrend/Breakouts):**
        - **Call Buying**: For strong breakouts above resistance
        - **Put Selling**: Cash-secured puts near support levels
        - **Bull Call Spreads**: Limited risk upside participation
        
        **🔴 Bearish Patterns (Downtrend/Breakdowns):**
        - **Put Buying**: For breakdown below support
        - **Call Selling**: Covered calls or credit spreads
        - **Bear Put Spreads**: Limited risk downside participation
        
        **🟡 Neutral Patterns (Consolidation/Range):**
        - **Iron Condors**: Range-bound premium collection
        - **Straddles/Strangles**: Volatility expansion plays
        - **Calendar Spreads**: Time decay strategies
        
        #### ⚠️ Risk Management Principles
        
        1. **Position Sizing**: Never risk more than 2-5% of portfolio on single trade
        2. **Stop Losses**: Set clear exit points before entering trades
        3. **Time Decay**: Be aware of theta impact on options positions
        4. **Volatility**: Consider IV levels when selecting strategies
        5. **Earnings**: Avoid holding through earnings unless specifically trading the event
        """)

def _perform_chart_analysis(uploaded_file, analysis_mode, additional_context):
    """Perform chart analysis using AI"""
    
    try:
        # Show progress
        with st.spinner("🧠 Analyzing chart with AI... This may take 30-60 seconds..."):
            
            # Initialize analyzer
            analyzer = ChartAnalyzer()
            
            # Determine analysis type
            analysis_type = "quick" if analysis_mode == "Quick Analysis" else "deep"
            
            # Perform analysis
            analysis_result = analyzer.analyze_chart(
                image_file=uploaded_file,
                analysis_type=analysis_type,
                additional_context=additional_context
            )
            
            # Store results in session state
            if 'chart_analysis_results' not in st.session_state:
                st.session_state.chart_analysis_results = {}
            
            # Create unique key for this analysis
            analysis_key = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.chart_analysis_results[analysis_key] = {
                'result': analysis_result,
                'mode': analysis_mode,
                'context': additional_context,
                'filename': uploaded_file.name
            }
            
            st.success("✅ Chart analysis completed successfully!")
            # Force a rerun to show the results immediately
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if "max_tokens" in str(e):
            st.info("💡 API parameter issue detected. Please ensure you're using the latest OpenAI library.")
        elif "401" in str(e) or "API key" in str(e):
            st.info("💡 Please check your OpenAI API key configuration.")
        else:
            st.info("💡 Please check your OpenAI API key and model access.")

def _display_chart_analysis_results():
    """Display chart analysis results using AI formatter"""
    
    if 'chart_analysis_results' in st.session_state and st.session_state.chart_analysis_results:
        # Get the latest analysis
        latest_key = list(st.session_state.chart_analysis_results.keys())[-1]
        analysis_data = st.session_state.chart_analysis_results[latest_key]
        
        # Extract content
        analysis_result = analysis_data['result']
        analysis_content = analysis_result.get('analysis_content', 'No analysis content available')
        
        # Display using AI formatter
        if AI_FORMATTER_AVAILABLE:
            display_ai_analysis(
                ai_content=analysis_content,
                analysis_type=f"Chart Analysis ({analysis_data['mode']})",
                tab_color=get_tab_color("chart"),
                analysis_key=latest_key,
                session_key="chart_analysis_results",
                regenerate_key="regenerate_chart_analysis",
                clear_key="clear_chart_analysis",
                show_debug=True,
                show_metadata=True
            )
        else:
            st.markdown("#### 🧠 AI Chart Analysis Results")
            st.markdown(analysis_content)
        
        # Display metadata
        metadata = analysis_result.get('metadata', {})
        if metadata:
            with st.expander("ℹ️ Analysis Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Analysis Type", metadata.get('analysis_type', 'N/A'))
                with col2:
                    st.metric("Model Used", metadata.get('model', 'N/A'))
                with col3:
                    timestamp = metadata.get('timestamp', '')
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        st.metric("Generated", dt.strftime('%H:%M:%S'))
    
    else:
        if AI_FORMATTER_AVAILABLE:
            display_ai_placeholder(
                analysis_type="Chart Analysis",
                features_list=[
                    "🔍 Advanced pattern recognition (wedges, triangles, channels)",
                    "📈 Comprehensive trend and momentum analysis",
                    "📊 Price action and candlestick pattern identification",
                    "📉 Volume analysis and confirmation signals",
                    "🎯 Support and resistance level mapping",
                    "⚡ Gap analysis and breakout potential assessment",
                    "💰 Tailored options strategy recommendations",
                    "⚠️ Risk management and position sizing guidance"
                ]
            )
        else:
            st.info("👆 Upload a chart image and click 'Analyze Chart with AI' to get comprehensive technical analysis")

def _display_chart_analysis_fallback():
    """Fallback display when AI formatter is not available"""
    
    if 'chart_analysis_results' in st.session_state and st.session_state.chart_analysis_results:
        latest_key = list(st.session_state.chart_analysis_results.keys())[-1]
        analysis_data = st.session_state.chart_analysis_results[latest_key]
        analysis_result = analysis_data['result']
        
        st.markdown("#### 🧠 AI Chart Analysis Results")
        
        # Display analysis content
        analysis_content = analysis_result.get('analysis_content', 'No analysis available')
        st.markdown(analysis_content)
        
        # Clear button
        if st.button("🗑️ Clear Analysis"):
            st.session_state.chart_analysis_results = {}
            st.rerun()
    
    else:
        st.info("👆 Upload a chart image and click 'Analyze Chart with AI' to get comprehensive technical analysis")

def _create_sample_analysis_demo():
    """Create a demo section showing sample analysis"""
    
    st.markdown("#### 📚 Sample Analysis Examples")
    
    with st.expander("🔍 See Sample Chart Analysis"):
        st.markdown("""
        **Sample Analysis Output:**
        
        ## 📊 Pattern Analysis
        **Pattern**: Falling Wedge
        **Confidence**: 85%
        **Description**: Classic falling wedge pattern with converging trendlines showing decreasing volume on declines.
        
        ## 📈 Trend Analysis  
        **Primary Trend**: Bullish (Long-term)
        **Short-term**: Consolidation phase
        **Strength**: Moderate with potential for breakout
        
        ## 🎯 Options Strategies
        
        ### Strategy 1: Bull Call Spread
        **Type**: Call Buy/Call Sell
        **Rationale**: Breakout above wedge resistance expected
        **Strikes**: Buy $25 Call, Sell $35 Call
        **Risk Level**: Medium
        
        ### Strategy 2: Cash-Secured Put
        **Type**: Put Sell
        **Rationale**: Strong support at wedge lower boundary
        **Strike**: $22 Put (near support)
        **Risk Level**: Low-Medium
        
        ## ⚠️ Risk Management
        - **Stop Loss**: Below $21 (wedge breakdown)
        - **Position Size**: Conservative (2-3% of portfolio)
        - **Time Horizon**: 2-6 weeks for pattern completion
        """)

# Additional utility functions for the tab
def get_analysis_summary_stats():
    """Get summary statistics of analysis results"""
    if 'chart_analysis_results' not in st.session_state:
        return {}
    
    results = st.session_state.chart_analysis_results
    return {
        'total_analyses': len(results),
        'quick_analyses': sum(1 for r in results.values() if r['mode'] == 'Quick Analysis'),
        'deep_analyses': sum(1 for r in results.values() if r['mode'] == 'Deep Technical Analysis'),
        'latest_analysis': max(results.keys()) if results else None
    }

def export_analysis_results():
    """Export analysis results to text format"""
    if 'chart_analysis_results' not in st.session_state:
        return None
    
    export_data = []
    for key, data in st.session_state.chart_analysis_results.items():
        export_data.append(f"""
Analysis ID: {key}
Filename: {data['filename']}
Mode: {data['mode']}
Context: {data['context']}
Results:
{data['result'].get('analysis_content', 'No content')}
{'='*50}
        """)
    
    return '\n'.join(export_data) 