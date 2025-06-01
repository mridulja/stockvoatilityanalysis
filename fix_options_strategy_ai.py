#!/usr/bin/env python3
"""
Fix Options Strategy tab to use optional AI analysis instead of automatic
"""

with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the automatic LLM analysis section in Options Strategy tab
old_auto_ai = '''                        # Automatic LLM Analysis - runs immediately after strategy completion
                        if LLM_AVAILABLE:
                            # Check for OpenAI API key
                            api_key_available = bool(os.getenv('OPENAI_API_KEY'))
                            
                            if api_key_available:
                                st.markdown("---")
                                st.markdown("### 🤖 AI-Powered Professional Analysis")
                                
                                with st.spinner("🤖 AI is analyzing your options strategy..."):
                                    try:
                                        # Get LLM analyzer
                                        llm_analyzer = get_llm_analyzer()
                                        
                                        if llm_analyzer is not None:
                                            # Prepare VIX data for LLM
                                            vix_data_for_llm = None
                                            if 'condition' in locals() and 'latest_vix' in locals():
                                                vix_data_for_llm = format_vix_data_for_llm(
                                                    latest_vix, condition, trade_approved
                                                )
                                            
                                            # Prepare confidence levels data
                                            confidence_data_for_llm = format_confidence_levels_for_llm(conf_data)
                                            
                                            # Generate LLM analysis automatically
                                            llm_result = llm_analyzer.analyze_options_strategy(
                                                ticker=strategy_ticker,
                                                current_price=current_price,
                                                strategy_timeframe=strategy_timeframe,
                                                recommendations=recommendations,
                                                prob_dist=prob_dist,
                                                vix_data=vix_data_for_llm,
                                                atr=atr,
                                                confidence_levels=confidence_data_for_llm
                                            )
                                            
                                            if llm_result['llm_analysis']['success']:
                                                # Display AI analysis with modern styling at the top
                                                st.markdown("""
                                                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                                            padding: 2rem; border-radius: 16px; margin: 1rem 0; 
                                                            border-left: 6px solid var(--secondary-color); box-shadow: var(--shadow-lg);">
                                                    <h3 style="color: var(--secondary-color); margin: 0 0 1rem 0; font-weight: 700;">
                                                        🤖 Professional AI Trading Analysis
                                                    </h3>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Display the AI analysis
                                                ai_analysis = llm_result['llm_analysis']['analysis']
                                                st.markdown(ai_analysis)
                                                
                                                # Show analysis metadata in an expander
                                                with st.expander("📊 Analysis Metadata"):
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("Tokens Used", llm_result['llm_analysis']['tokens_used'])
                                                    with col2:
                                                        st.metric("Model", llm_result['llm_analysis']['model'])
                                                    with col3:
                                                        st.metric("Analysis Time", 
                                                                datetime.fromisoformat(llm_result['llm_analysis']['timestamp']).strftime("%H:%M:%S"))
                                                
                                                st.success("✅ AI analysis complete!")
                                            
                                            else:
                                                st.warning(f"⚠️ AI analysis failed: {llm_result['llm_analysis']['error']}")
                                        
                                        else:
                                            st.warning("⚠️ Failed to initialize AI analyzer")
                                            
                                    except Exception as e:
                                        st.warning(f"⚠️ AI analysis error: {str(e)}")
                            
                            else:
                                # Show setup instructions for missing API key
                                st.markdown("---")
                                st.markdown("### 🤖 AI Analysis Available")
                                st.info("💡 **Set up your OpenAI API key to unlock professional AI trading analysis!**")
                                
                                with st.expander("🔑 Quick Setup Instructions"):
                                    st.markdown("""
                                    **To enable automatic AI analysis:**
                                    1. Get an OpenAI API key from https://platform.openai.com/api-keys
                                    2. Set it as an environment variable: `OPENAI_API_KEY=your-key-here`
                                    3. **Or** create a `.env` file with: `OPENAI_API_KEY=your-key-here`
                                    4. Restart the application
                                    
                                    **Features you'll unlock:**
                                    - 🧠 **Executive Summary**: Overall strategy assessment
                                    - 📋 **Risk Management**: Position sizing and stop-loss guidance  
                                    - 🎯 **Entry/Exit Criteria**: Specific trading actions
                                    - 📊 **Market Conditions**: VIX impact and timing analysis
                                    """)
                        
                        else:
                            st.markdown("---")
                            st.info("💡 **Note**: Install OpenAI package to enable automatic AI-powered analysis and recommendations.")'''

# Replace with optional AI analysis
new_optional_ai = '''                        # Optional AI Analysis for Options Strategy
                        if enable_ai_analysis and LLM_AVAILABLE:
                            st.markdown("---")
                            st.markdown("### 🤖 AI Options Strategy Analysis")
                            
                            if st.button("🧠 Generate AI Options Analysis", key="options_ai_btn"):
                                with st.spinner("🤖 AI analyzing options strategy..."):
                                    try:
                                        llm_analyzer = get_llm_analyzer()
                                        
                                        if llm_analyzer is not None:
                                            # Prepare VIX data for LLM
                                            vix_data_for_llm = None
                                            if 'condition' in locals() and 'latest_vix' in locals():
                                                vix_data_for_llm = format_vix_data_for_llm(
                                                    latest_vix, condition, trade_approved
                                                )
                                            
                                            # Prepare confidence levels data  
                                            confidence_data_for_llm = format_confidence_levels_for_llm(conf_data)
                                            
                                            # Generate LLM analysis
                                            llm_result = llm_analyzer.analyze_options_strategy(
                                                ticker=strategy_ticker,
                                                current_price=current_price,
                                                strategy_timeframe=strategy_timeframe,
                                                recommendations=recommendations,
                                                prob_dist=prob_dist,
                                                vix_data=vix_data_for_llm,
                                                atr=atr,
                                                confidence_levels=confidence_data_for_llm
                                            )
                                            
                                            if llm_result['llm_analysis']['success']:
                                                st.markdown("""
                                                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                                            padding: 2rem; border-radius: 16px; margin: 1rem 0; 
                                                            border-left: 6px solid var(--secondary-color); box-shadow: var(--shadow-lg);">
                                                    <h3 style="color: var(--secondary-color); margin: 0 0 1rem 0; font-weight: 700;">
                                                        🤖 Professional AI Trading Analysis
                                                    </h3>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                st.markdown(llm_result['llm_analysis']['analysis'])
                                                
                                                with st.expander("📊 Analysis Details"):
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("Tokens Used", llm_result['llm_analysis']['tokens_used'])
                                                    with col2:
                                                        st.metric("Model", llm_result['llm_analysis']['model'])
                                                    with col3:
                                                        st.metric("Analysis Time", 
                                                                datetime.fromisoformat(llm_result['llm_analysis']['timestamp']).strftime("%H:%M:%S"))
                                                
                                                st.success("✅ AI analysis complete!")
                                            else:
                                                st.warning(f"⚠️ AI analysis failed: {llm_result['llm_analysis']['error']}")
                                        else:
                                            st.warning("⚠️ Failed to initialize AI analyzer")
                                    except Exception as e:
                                        st.error(f"❌ AI analysis error: {str(e)}")'''

content = content.replace(old_auto_ai, new_optional_ai)

# Write back the fixed content
with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed Options Strategy tab to use optional AI analysis")
print("🤖 AI analysis is now consistent across all tabs - only runs when checkbox is enabled") 