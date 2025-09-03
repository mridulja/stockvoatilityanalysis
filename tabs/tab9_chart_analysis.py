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
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage
import io

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
    st.info("💡 **New Features**: AI automatically detects asset info during analysis! Download professional PDF reports with high-quality images, proper markdown formatting, comprehensive metadata, and technical information!")
    
    # Check if chart analyzer is available
    if not CHART_ANALYZER_AVAILABLE:
        st.error("❌ Chart Analyzer not available. Please ensure core/chart_analyzer.py is properly installed.")
        st.info("💡 This feature requires OpenAI API access and the chart analyzer module.")
        return
    
    # === DEBUG & MODEL INFO SECTION ===
    with st.expander("🔧 Debug & Model Information"):
        if CHART_ANALYZER_AVAILABLE:
            try:
                analyzer = ChartAnalyzer()
                model_info = analyzer.get_model_info()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🤖 Model Configuration:**")
                    st.code(f"Primary: {model_info['primary_model']}")
                    st.code(f"Fallback: {model_info['fallback_model']}")
                    
                    # Show vision support
                    vision_support = model_info['vision_support']
                    st.markdown("**👁️ Vision Support:**")
                    if vision_support['primary_supports_vision']:
                        st.success(f"✅ {model_info['primary_model']} supports vision")
                    else:
                        st.warning(f"⚠️ {model_info['primary_model']} does NOT support vision")
                    
                    if vision_support['fallback_supports_vision']:
                        st.success(f"✅ {model_info['fallback_model']} supports vision")
                    else:
                        st.warning(f"⚠️ {model_info['fallback_model']} does NOT support vision")
                
                with col2:
                    st.markdown("**📊 Client Information:**")
                    st.code(f"Client Type: {model_info['openai_client_type']}")
                    
                    # Add model switching options
                    st.markdown("**🔄 Model Management:**")
                    if st.button("📋 List Available Models"):
                        analyzer.list_available_models()
                        st.info("Check console for detailed model information")
                    
                    # Quick model switch
                    st.markdown("**Quick Model Switch:**")
                    if st.button("🔄 Switch to GPT-4o"):
                        analyzer.set_model("gpt-4o")
                        st.success("Switched to GPT-4o (vision supported)")
                        st.rerun()
                    
                    if st.button("🔄 Switch to GPT-4o-mini"):
                        analyzer.set_model("gpt-4o-mini")
                        st.success("Switched to GPT-4o-mini (vision supported)")
                        st.rerun()
                    
                    # Test ChartAnalyzer functionality
                    st.markdown("---")
                    st.markdown("**🧪 ChartAnalyzer Test:**")
                    if st.button("🔍 Test ChartAnalyzer"):
                        try:
                            test_analyzer = ChartAnalyzer()
                            st.success("✅ ChartAnalyzer initialized successfully")
                            
                            # Test model info
                            model_info = test_analyzer.get_model_info()
                            st.info(f"Primary Model: {model_info['primary_model']}")
                            st.info(f"Fallback Model: {model_info['fallback_model']}")
                            
                            # Test vision support
                            vision_support = model_info['vision_support']
                            if vision_support['primary_supports_vision']:
                                st.success(f"✅ {model_info['primary_model']} supports vision")
                            else:
                                st.warning(f"⚠️ {model_info['primary_model']} does NOT support vision")
                                
                        except Exception as e:
                            st.error(f"❌ ChartAnalyzer test failed: {str(e)}")
                            st.error(f"Error type: {type(e).__name__}")
                    
                    # Test OpenAI API connection
                    if st.button("🔌 Test OpenAI API"):
                        try:
                            test_analyzer = ChartAnalyzer()
                            # Try to list models to test API connection
                            test_analyzer.list_available_models()
                            st.success("✅ OpenAI API connection successful")
                            st.info("Check console for model information")
                        except Exception as e:
                            st.error(f"❌ OpenAI API test failed: {str(e)}")
                            st.error(f"Error type: {type(e).__name__}")
                            if "api_key" in str(e).lower():
                                st.warning("💡 Check your OPENAI_API_KEY environment variable")
                            elif "401" in str(e):
                                st.warning("💡 Invalid API key - check your OpenAI account")
                            elif "quota" in str(e).lower():
                                st.warning("💡 API quota exceeded - check your OpenAI billing")
                
                # Important note about GPT-5-mini
                if model_info['primary_model'] == 'gpt-5-mini':
                    st.success("""
                    🎉 **Great News**: GPT-5-mini now supports vision and is multimodal!
                    This means you get the latest reasoning capabilities PLUS image analysis.
                    """)
                
                # Show current model capabilities
                if model_info['vision_support']['primary_supports_vision']:
                    if 'gpt-5' in model_info['primary_model']:
                        st.success(f"🚀 Current primary model ({model_info['primary_model']}) is the latest GPT-5 with vision support!")
                    else:
                        st.success(f"✅ Current primary model ({model_info['primary_model']}) supports vision - perfect for chart analysis!")
                else:
                    st.error(f"❌ Current primary model ({model_info['primary_model']}) does NOT support vision - cannot analyze charts!")
                
                # Add text-only analysis option
                st.markdown("---")
                st.markdown("**📝 Text-Only Analysis (GPT-5-mini):**")
                st.markdown("You can also use GPT-5-mini for text analysis (though it now supports images too!):")
                
                text_content = st.text_area(
                    "Text to Analyze:",
                    placeholder="Paste text content here for analysis (e.g., earnings report, news article, etc.)",
                    height=100,
                    key="text_analysis_input"
                )
                
                if st.button("🧠 Analyze Text with GPT-5-mini"):
                    if text_content.strip():
                        try:
                            with st.spinner("Analyzing text with GPT-5-mini..."):
                                analysis_result = analyzer.analyze_text_only(
                                    text_content=text_content,
                                    analysis_type="deep",
                                    additional_context="",
                                    system_prompt=""
                                )
                                
                                # Store results
                                if 'chart_analysis_results' not in st.session_state:
                                    st.session_state.chart_analysis_results = {}
                                
                                analysis_key = f"text_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                st.session_state.chart_analysis_results[analysis_key] = {
                                    'result': analysis_result,
                                    'mode': 'Text Analysis',
                                    'context': '',
                                    'system_prompt': '',
                                    'filename': 'Text Content',
                                    'content_type': 'text'
                                }
                                
                                st.success("✅ Text analysis completed!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Text analysis failed: {str(e)}")
                    else:
                        st.warning("Please enter some text to analyze.")
            except Exception as e:
                st.error(f"Error getting model info: {str(e)}")
        else:
            st.error("Chart Analyzer not available")
    
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
                placeholder="e.g., Recent earnings, market conditions, specific concerns. Always think critically, before you answer.",
                height=100,
                help="Provide any additional context about the stock or market conditions"
            )
        
        # System prompt input
        st.markdown("#### 🎯 Custom System Prompt (Optional)")
        
        # Add button to load default system prompt
        col1, col2 = st.columns([3, 1])
        with col1:
            system_prompt = st.text_area(
                "System Prompt:",
                placeholder="Customize how the AI should approach the analysis (e.g., 'Focus on swing trading patterns' or 'Emphasize risk management')",
                height=80,
                help="Customize the AI's analysis approach and focus areas"
            )
        with col2:
            if st.button("📋 Load Default", help="Load the professional default system prompt"):
                if CHART_ANALYZER_AVAILABLE:
                    analyzer = ChartAnalyzer()
                    default_prompt = analyzer.get_default_system_prompt()
                    st.session_state.default_system_prompt = default_prompt
                    st.rerun()
        
        # Show default prompt if loaded
        if 'default_system_prompt' in st.session_state:
            with st.expander("📋 Default System Prompt (Click to copy)"):
                st.code(st.session_state.default_system_prompt, language="text")
                if st.button("📋 Copy to Input"):
                    st.session_state.system_prompt_input = st.session_state.default_system_prompt
                    st.rerun()
        
        # Show example system prompts
        with st.expander("💡 Example System Prompts"):
            st.markdown("""
            **🎯 Swing Trading Focus:**
            ```
            Focus on swing trading patterns with 1-4 week timeframes. Emphasize support/resistance levels and breakout setups. Prioritize risk/reward ratios above 2:1.
            ```
            
            **📊 Day Trading Focus:**
            ```
            Analyze for intraday trading opportunities. Focus on momentum indicators, volume spikes, and short-term support/resistance. Emphasize quick entry/exit strategies.
            ```
            
            **💰 Income Generation:**
            ```
            Prioritize options selling strategies for income generation. Focus on high-probability setups with defined risk. Emphasize cash-secured puts and covered calls.
            ```
            
            **⚠️ Conservative Approach:**
            ```
            Provide conservative, risk-averse analysis. Emphasize capital preservation over aggressive gains. Focus on high-probability setups with tight risk management.
            ```
            """)
        
        # Use the loaded prompt if available
        if 'system_prompt_input' in st.session_state:
            system_prompt = st.session_state.system_prompt_input
        
        # Analysis button
        analyze_button = st.button(
            "🧠 Analyze Chart with AI",
            type="primary",
            help="Generate comprehensive chart analysis using OpenAI o3-mini"
        )
        
        if analyze_button:
            # Reset the uploaded file in session state to prevent re-uploads
            if uploaded_file is not None:
                _perform_chart_analysis(uploaded_file, analysis_mode, additional_context, system_prompt)
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
    
    # === CONSOLE LOG VIEWER ===
    with st.expander("📋 Console Log Viewer (Debug Output)"):
        st.markdown("""
        **Console logs from the ChartAnalyzer will appear here during analysis.**
        This shows which model is being used, API calls, and processing details.
        """)
        
        # Create a placeholder for console logs
        if 'console_logs' not in st.session_state:
            st.session_state.console_logs = []
        
        # Display existing logs
        if st.session_state.console_logs:
            st.markdown("**Recent Logs:**")
            for log in st.session_state.console_logs[-10:]:  # Show last 10 logs
                st.code(log, language="text")
        else:
            st.info("No logs yet. Run a chart analysis to see debug output.")
        
        # Clear logs button
        if st.button("🗑️ Clear Logs"):
            st.session_state.console_logs = []
            st.rerun()
    
    # === SESSION STATE DEBUG ===
    with st.expander("🔍 Session State Debug"):
        st.markdown("**Current Session State for Chart Analysis:**")
        
        if 'chart_analysis_results' in st.session_state:
            results = st.session_state.chart_analysis_results
            st.info(f"📊 Chart Analysis Results: {len(results)} analyses stored")
            
            if results:
                # Show the latest analysis key
                latest_key = list(results.keys())[-1]
                st.success(f"✅ Latest Analysis Key: {latest_key}")
                
                # Show basic info about the latest analysis
                latest_data = results[latest_key]
                st.markdown("**Latest Analysis Data:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mode", latest_data.get('mode', 'N/A'))
                    st.metric("Filename", latest_data.get('filename', 'N/A'))
                with col2:
                    st.metric("Company", latest_data.get('company_name', 'N/A'))
                    st.metric("Ticker", latest_data.get('ticker_symbol', 'N/A'))
                
                # Show if result exists
                if 'result' in latest_data:
                    result = latest_data['result']
                    if result:
                        st.success("✅ Analysis result exists")
                        if 'analysis_content' in result:
                            content_length = len(result.get('analysis_content', ''))
                            st.info(f"📝 Analysis content length: {content_length} characters")
                            if content_length > 0:
                                st.success("✅ Analysis content is not empty")
                            else:
                                st.warning("⚠️ Analysis content is empty")
                        else:
                            st.warning("⚠️ No 'analysis_content' key in result")
                    else:
                        st.error("❌ Analysis result is None or empty")
                else:
                    st.error("❌ No 'result' key in analysis data")
                
                # Show all keys in the latest data
                st.markdown("**All Keys in Latest Analysis Data:**")
                st.code(list(latest_data.keys()), language="text")
                
                # Show all keys in the result
                if 'result' in latest_data and latest_data['result']:
                    st.markdown("**All Keys in Analysis Result:**")
                    st.code(list(latest_data['result'].keys()), language="text")
            else:
                st.warning("⚠️ No analyses stored yet")
        else:
            st.warning("⚠️ 'chart_analysis_results' not in session state")
        
        # Show all session state keys
        st.markdown("**All Session State Keys:**")
        all_keys = list(st.session_state.keys())
        st.code(all_keys, language="text")
    
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

def _perform_chart_analysis(uploaded_file, analysis_mode, additional_context, system_prompt):
    """Perform chart analysis using AI"""
    
    try:
        # Show progress
        with st.spinner("🧠 Analyzing chart with AI... This may take 30-60 seconds..."):
            
            # Initialize analyzer
            try:
                analyzer = ChartAnalyzer()
            except Exception as e:
                st.error(f"❌ Failed to initialize ChartAnalyzer: {str(e)}")
                return
            
            # Determine analysis type
            analysis_type = "quick" if analysis_mode == "Quick Analysis" else "deep"
            
            # Perform analysis
            with st.spinner("🧠 Analyzing chart with AI..."):
                try:
                    analysis_result = analyzer.analyze_chart(
                        image_file=uploaded_file,
                        analysis_type=analysis_type,
                        additional_context=additional_context,
                        system_prompt=system_prompt
                    )
                except Exception as e:
                    st.error(f"❌ analyze_chart() failed: {str(e)}")
                    return
            
            # Check if we got a valid result
            if not analysis_result:
                st.error("❌ Analysis result is None or empty")
                return
            
            # Extract asset info from the analysis content
            analysis_content = analysis_result.get('analysis_content', '')
            
            if not analysis_content:
                st.error("❌ Analysis content is empty - no content received from AI")
                return
            
            # Parse asset info
            try:
                asset_info = analyzer._parse_asset_info_from_analysis(analysis_content)
                company_name = asset_info.get('company_name', 'Unknown')
                ticker_from_chart = asset_info.get('ticker_symbol', 'Unknown')
            except Exception as e:
                company_name = 'Unknown'
                ticker_from_chart = 'Unknown'
            
            # Store results in session state
            if 'chart_analysis_results' not in st.session_state:
                st.session_state.chart_analysis_results = {}
            
            # Create unique key for this analysis
            analysis_key = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.chart_analysis_results[analysis_key] = {
                'result': analysis_result,
                'mode': analysis_mode,
                'context': additional_context,
                'system_prompt': system_prompt,
                'filename': uploaded_file.name,
                'image_file': uploaded_file,  # Store the image file for PDF generation
                'company_name': company_name,  # Store extracted company name
                'ticker_symbol': ticker_from_chart  # Store extracted ticker
            }
            
            st.success("✅ Chart analysis completed successfully!")
            st.info(f"📊 Detected: {company_name} ({ticker_from_chart})")
            # Force a rerun to show the results immediately
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        
        if "max_tokens" in str(e):
            st.info("💡 API parameter issue detected. Please ensure you're using the latest OpenAI library.")
        elif "401" in str(e) or "API key" in str(e):
            st.info("💡 Please check your OpenAI API key configuration.")
        elif "vision" in str(e).lower():
            st.info("💡 Vision model issue detected. Check if the model supports image analysis.")
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
                
                # Show extracted company info
                company_name = analysis_data.get('company_name', 'Unknown')
                ticker_from_chart = analysis_data.get('ticker_symbol', 'Unknown')
                
                if company_name != 'Unknown' or ticker_from_chart != 'Unknown':
                    st.markdown("**🏢 Extracted from Chart:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Company", company_name)
                    with col2:
                        st.metric("Ticker", ticker_from_chart)
                
                # Show system prompt if used
                if analysis_data.get('system_prompt'):
                    st.markdown("**🔧 System Prompt Used:**")
                    st.code(analysis_data['system_prompt'], language="text")
                
                # PDF Download Section
                st.markdown("---")
                st.markdown("#### 📄 Export Analysis Report")
                
                # Use extracted ticker from chart or fallback to filename
                ticker_symbol = analysis_data.get('ticker_symbol', 'Unknown')
                if ticker_symbol == 'Unknown':
                    ticker_symbol = get_ticker_from_filename(analysis_data.get('filename', ''))
                
                # Get company name
                company_name = analysis_data.get('company_name', 'Unknown')
                
                # Generate and download PDF button
                if st.button("📥 Generate & Download PDF Report", type="secondary", help="Generate and download a professional PDF report with metadata and disclaimer"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            pdf_bytes, pdf_filename = generate_chart_analysis_pdf(
                                analysis_data, 
                                analysis_result, 
                                ticker_symbol
                            )
                            
                            if pdf_bytes and pdf_filename:
                                # Create download button
                                st.download_button(
                                    label="💾 Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=pdf_filename,
                                    mime="application/pdf",
                                    help="Click to download your AI chart analysis report"
                                )
                                
                                st.success(f"✅ PDF report generated successfully! Filename: {pdf_filename}")
                                st.info("📋 Report includes: Chart image, analysis results, metadata, context, technical information (tokens used), and legal disclaimer")
                            else:
                                st.error("❌ Failed to generate PDF report")
                                
                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
                            st.info("💡 Make sure you have the required PDF libraries installed")
    
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
        
        # Show system prompt if used
        if analysis_data.get('system_prompt'):
            with st.expander("🔧 System Prompt Used"):
                st.code(analysis_data['system_prompt'], language="text")
        
        # PDF Download Section
        st.markdown("---")
        st.markdown("#### 📄 Export Analysis Report")
        
        # Use extracted ticker from chart or fallback to filename
        ticker_symbol = analysis_data.get('ticker_symbol', 'Unknown')
        if ticker_symbol == 'Unknown':
            ticker_symbol = get_ticker_from_filename(analysis_data.get('filename', ''))
        
        # Get company name
        company_name = analysis_data.get('company_name', 'Unknown')
        
        # Generate and download PDF button
        if st.button("📥 Generate & Download PDF Report", type="secondary", help="Generate and download a professional PDF report with metadata and disclaimer"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_bytes, pdf_filename = generate_chart_analysis_pdf(
                        analysis_data, 
                        analysis_result, 
                        ticker_symbol
                    )
                    
                    if pdf_bytes and pdf_filename:
                        # Create download button
                        st.download_button(
                            label="💾 Download PDF Report",
                            data=pdf_bytes,
                            file_name=pdf_filename,
                            mime="application/pdf",
                            help="Click to download your AI chart analysis report"
                        )
                        
                        st.success(f"✅ PDF report generated successfully! Filename: {pdf_filename}")
                        st.info("📋 Report includes: Chart image, analysis results, metadata, context, technical information (tokens used), and legal disclaimer")
                    else:
                        st.error("❌ Failed to generate PDF report")
                        
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.info("💡 Make sure you have the required PDF libraries installed")
        
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
System Prompt: {data.get('system_prompt', 'None')}
Results:
{data['result'].get('analysis_content', 'No content')}
{'='*50}
        """)
    
    return '\n'.join(export_data)

def generate_chart_analysis_pdf(analysis_data, analysis_result, ticker_symbol="UNKNOWN"):
    """Generate a professional PDF report for chart analysis results"""
    
    try:
        # Get company info
        company_name = analysis_data.get('company_name', 'Unknown Company')
        
        # Create a better PDF filename
        safe_ticker = ticker_symbol.replace('/', '_').replace('\\', '_')
        pdf_filename = f"chart_analysis_{safe_ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Add custom style for analysis headers
        analysis_header_style = ParagraphStyle(
            'AnalysisHeader',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=16,
            textColor=colors.darkred,
            fontName='Helvetica-Bold'
        )
        
        # Title
        story.append(Paragraph("AI-Powered Chart Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata Table
        metadata_data = [
            ['Analysis Date', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Time', datetime.now().strftime('%I:%M %p')],
            ['Company Name', company_name],
            ['Ticker Symbol', ticker_symbol],
            ['Analysis Mode', analysis_data.get('mode', 'N/A')],
            ['AI Model', analysis_result.get('metadata', {}).get('model', 'GPT-5-mini')],
            ['Analysis Type', analysis_result.get('metadata', {}).get('analysis_type', 'N/A')],
            ['Analysis Timeframe', 'Quick Analysis' if analysis_data.get('mode') == 'Quick Analysis' else 'Deep Technical Analysis']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(Paragraph("Report Metadata", heading_style))
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Chart Image
        story.append(Paragraph("Chart Analysis", heading_style))
        try:
            # Convert uploaded file to PIL Image for PDF
            image_file = analysis_data.get('image_file')
            if image_file:
                # Reset file pointer to beginning
                image_file.seek(0)
                
                # Convert to PIL Image
                pil_image = PILImage.open(image_file)
                
                # Convert to RGB if necessary (for better PDF compatibility)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Calculate optimal size for PDF (higher quality)
                # Use 300 DPI for better quality
                dpi = 300
                max_width_inches = 6.0
                max_height_inches = 8.0
                
                # Calculate dimensions in pixels at 300 DPI
                max_width_pixels = int(max_width_inches * dpi)
                max_height_pixels = int(max_height_inches * dpi)
                
                # Get original dimensions
                img_width, img_height = pil_image.size
                
                # Calculate scaling factor to fit within bounds while maintaining aspect ratio
                width_scale = max_width_pixels / img_width
                height_scale = max_height_pixels / img_height
                scale_factor = min(width_scale, height_scale, 1.0)  # Don't upscale
                
                # Apply scaling if needed
                if scale_factor < 1.0:
                    new_width = int(img_width * scale_factor)
                    new_height = int(img_height * scale_factor)
                    pil_image = pil_image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                
                # Save to BytesIO with high quality
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="PNG", optimize=False, quality=95)
                img_buffer.seek(0)
                
                # Calculate PDF dimensions (convert pixels to points at 72 DPI)
                pdf_width = (pil_image.size[0] / dpi) * 72
                pdf_height = (pil_image.size[1] / dpi) * 72
                
                # Add image to PDF with calculated dimensions
                pdf_image = Image(img_buffer, width=pdf_width, height=pdf_height)
                story.append(pdf_image)
                story.append(Spacer(1, 10))
                
                # Add image caption
                caption_style = ParagraphStyle(
                    'Caption',
                    parent=styles['Normal'],
                    fontSize=9,
                    spaceAfter=6,
                    alignment=TA_CENTER,
                    textColor=colors.grey
                )
                story.append(Paragraph(f"Chart Image: {company_name} ({ticker_symbol}) - {analysis_data.get('mode', 'Analysis')} - High Quality (300 DPI)", caption_style))
                
            else:
                story.append(Paragraph("Chart image not available for this analysis", normal_style))
        except Exception as e:
            story.append(Paragraph("Chart image could not be included in the report", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Analysis Content
        story.append(Paragraph("Technical Analysis Results", heading_style))
        
        # Clean and format the analysis content for PDF
        analysis_content = analysis_result.get('analysis_content', 'No analysis content available')
        
        # Simple markdown parsing for better formatting
        lines = analysis_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line - add small spacer
                story.append(Spacer(1, 6))
                continue
            
            # Parse markdown elements
            if line.startswith('### '):
                # H3 header
                header_text = line.replace('### ', '').strip()
                story.append(Paragraph(header_text, analysis_header_style))
                story.append(Spacer(1, 8))
            
            elif line.startswith('## '):
                # H2 header
                header_text = line.replace('## ', '').strip()
                story.append(Paragraph(header_text, heading_style))
                story.append(Spacer(1, 10))
            
            elif line.startswith('# '):
                # H1 header
                header_text = line.replace('# ', '').strip()
                story.append(Paragraph(header_text, title_style))
                story.append(Spacer(1, 12))
            
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet point
                bullet_text = line.replace('- ', '').replace('* ', '').strip()
                # Clean up markdown formatting in bullet text
                bullet_text = bullet_text.replace('**', '').replace('*', '').replace('`', '')
                story.append(Paragraph(f"• {bullet_text}", normal_style))
                story.append(Spacer(1, 4))
            
            elif line.startswith('**') and line.endswith('**'):
                # Bold text as small header
                bold_text = line.replace('**', '').strip()
                bold_style = ParagraphStyle(
                    'BoldText',
                    parent=normal_style,
                    fontName='Helvetica-Bold',
                    fontSize=11,
                    spaceAfter=4,
                    spaceBefore=8
                )
                story.append(Paragraph(bold_text, bold_style))
                story.append(Spacer(1, 4))
            
            else:
                # Regular paragraph
                # Clean up markdown formatting
                clean_line = line.replace('**', '').replace('*', '').replace('`', '')
                if clean_line.strip():
                    story.append(Paragraph(clean_line, normal_style))
                    story.append(Spacer(1, 6))
        
        # Additional Context (if provided)
        if analysis_data.get('context'):
            story.append(Spacer(1, 20))
            story.append(Paragraph("Additional Context", heading_style))
            story.append(Paragraph(analysis_data['context'], normal_style))
        
        # Tokens Used Information
        story.append(Spacer(1, 20))
        story.append(Paragraph("Technical Information", heading_style))
        
        # Get tokens information
        tokens_used = analysis_result.get('metadata', {}).get('tokens_used', 'N/A')
        api_duration = analysis_result.get('metadata', {}).get('api_duration_seconds', 'N/A')
        
        tokens_info = f"""
        <b>AI Model Usage:</b><br/>
        • Tokens Used: {tokens_used}<br/>
        • API Response Time: {api_duration} seconds<br/>
        • Model: {analysis_result.get('metadata', {}).get('model', 'GPT-5-mini')}<br/>
        • Analysis Timestamp: {analysis_result.get('metadata', {}).get('timestamp', 'N/A')}
        """
        
        story.append(Paragraph(tokens_info, normal_style))
        
        # Disclaimer
        story.append(Spacer(1, 30))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
        
        disclaimer_text = """
        <b>DISCLAIMER:</b><br/>
        This analysis is generated by artificial intelligence and is for educational and informational purposes only. 
        It should not be considered as financial advice, investment recommendations, or a solicitation to buy or sell any security. 
        The analysis is based on technical patterns and historical data, which may not accurately predict future price movements. 
        Always conduct your own research, consider your risk tolerance, and consult with a qualified financial advisor before making investment decisions. 
        Past performance does not guarantee future results. Trading options involves substantial risk and may result in the loss of your entire investment.
        """
        
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build the PDF
        doc.build(story)
        
        # Read the generated PDF
        with open(pdf_filename, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Clean up temporary files
        # The BytesIO object is automatically garbage collected, so no explicit cleanup needed here
        # if 'img_buffer' in locals() and img_buffer:
        #     img_buffer.close()
        
        # Clean up the PDF file
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        
        return pdf_bytes, f"chart_analysis_{ticker_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
    except Exception as e:
        # Clean up any temporary files on error
        if 'pdf_filename' in locals() and os.path.exists(pdf_filename):
            try:
                os.remove(pdf_filename)
            except:
                pass
        
        return None, None

def get_ticker_from_filename(filename):
    """Extract ticker symbol from filename or return default"""
    if not filename:
        return "UNKNOWN"
    
    # Try to extract ticker from filename
    filename_lower = filename.lower()
    common_tickers = ['spy', 'qqq', 'aapl', 'msft', 'tsla', 'nvda', 'googl', 'amzn', 'meta', 'nflx']
    
    for ticker in common_tickers:
        if ticker in filename_lower:
            return ticker.upper()
    
    # If no common ticker found, try to extract from filename
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) > 0:
            potential_ticker = parts[0].upper()
            if len(potential_ticker) <= 5:  # Most tickers are 1-5 characters
                return potential_ticker
    
    return "UNKNOWN" 