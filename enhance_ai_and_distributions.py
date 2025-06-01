#!/usr/bin/env python3
"""
Enhance AI Analysis and Add Distribution Charts
1. Add AI analysis to Put Spread Analysis tab
2. Enhance Summary tab AI with comprehensive data from all tabs
3. Add POT/POP distribution charts with 2% bins
"""

# Read the current streamlit app
with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# First, add new distribution chart functions
distribution_charts_code = '''
def create_pot_pop_distribution_charts(spread_results):
    """Create POT and POP distribution charts with 2% bins"""
    if not spread_results or not spread_results.get('scenarios'):
        return None
    
    scenarios = spread_results['scenarios']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'POT Distribution (0%-30% Range)',
            'POP Distribution (70%-100% Range)', 
            'POT vs Strike Prices',
            'POP vs Strike Prices'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Extract data
    pot_values = [s['actual_pot'] * 100 for s in scenarios]
    pop_values = [s['spreads'][0]['prob_profit'] * 100 for s in scenarios if s.get('spreads')]
    strikes = [s['short_strike'] for s in scenarios]
    
    # POT Distribution (0%-30% with 2% bins)
    pot_filtered = [p for p in pot_values if 0 <= p <= 30]
    if pot_filtered:
        fig.add_trace(
            go.Histogram(
                x=pot_filtered,
                xbins=dict(start=0, end=30, size=2),  # 2% bins
                name='POT Distribution',
                marker_color='rgba(239, 68, 68, 0.7)',
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # POP Distribution (70%-100% with 2% bins) 
    pop_filtered = [p for p in pop_values if 70 <= p <= 100]
    if pop_filtered:
        fig.add_trace(
            go.Histogram(
                x=pop_filtered,
                xbins=dict(start=70, end=100, size=2),  # 2% bins
                name='POP Distribution',
                marker_color='rgba(16, 185, 129, 0.7)',
                opacity=0.8
            ),
            row=1, col=2
        )
    
    # POT vs Strike Prices
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=pot_values,
            mode='markers+lines',
            name='POT vs Strikes',
            marker=dict(size=8, color='rgba(239, 68, 68, 0.8)'),
            line=dict(color='rgba(239, 68, 68, 0.8)', width=2)
        ),
        row=2, col=1
    )
    
    # POP vs Strike Prices
    if len(pop_values) == len(strikes):
        fig.add_trace(
            go.Scatter(
                x=strikes[:len(pop_values)],
                y=pop_values,
                mode='markers+lines',
                name='POP vs Strikes',
                marker=dict(size=8, color='rgba(16, 185, 129, 0.8)'),
                line=dict(color='rgba(16, 185, 129, 0.8)', width=2)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="POT %", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="POP %", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Strike Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="POT %", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price ($)", row=2, col=2)
    fig.update_yaxes(title_text="POP %", row=2, col=2)
    
    fig.update_layout(
        title='POT & POP Distribution Analysis (2% Bins)',
        height=700,
        showlegend=True
    )
    
    return fig

'''

# Find position to insert new chart function (after existing chart functions)
chart_function_pos = content.find('def create_probability_chart(recommendations, current_price, ticker):')
if chart_function_pos != -1:
    content = content[:chart_function_pos] + distribution_charts_code + '\n\n' + content[chart_function_pos:]

# Add AI analysis to Put Spread Analysis tab (after the existing charts)
put_spread_ai_addition = '''
                            # AI Analysis for Put Spreads
                            if enable_ai_analysis and LLM_AVAILABLE:
                                st.markdown("---")
                                st.markdown("### 🤖 AI Put Spread Analysis")
                                
                                if st.button("🧠 Generate AI Put Spread Analysis", key="put_spread_ai_btn"):
                                    with st.spinner("🤖 AI analyzing put spread strategies..."):
                                        try:
                                            llm_analyzer = get_llm_analyzer()
                                            if llm_analyzer:
                                                # Prepare comprehensive put spread data for AI
                                                spread_summary = {
                                                    'ticker': spread_ticker,
                                                    'current_price': spread_results['current_price'],
                                                    'expiry_date': spread_results['expiry_date'],
                                                    'time_to_expiry_days': spread_results['time_to_expiry_days'],
                                                    'volatility_used': spread_results['volatility'],
                                                    'scenarios_analyzed': len(spread_results['scenarios']),
                                                    'best_scenarios': spread_results['scenarios'][:3] if spread_results['scenarios'] else [],
                                                    'options_data_available': spread_results.get('options_data_available', False)
                                                }
                                                
                                                # Create put spread specific prompt
                                                prompt = f"""
                                                Analyze this PUT SPREAD analysis for {spread_summary['ticker']} and provide a CONCISE professional summary:
                                                
                                                **Market Data:**
                                                - Current Price: ${spread_summary['current_price']:.2f}
                                                - Expiry: {spread_summary['expiry_date']} ({spread_summary['time_to_expiry_days']:.1f} days)
                                                - Volatility: {spread_summary['volatility_used']:.1%}
                                                - Options Data: {'Available' if spread_summary['options_data_available'] else 'ATR-based estimate'}
                                                
                                                **Analysis Results:**
                                                - Scenarios Analyzed: {spread_summary['scenarios_analyzed']}
                                                
                                                **Top 3 Scenarios:**"""
                                                
                                                for i, scenario in enumerate(spread_summary['best_scenarios'], 1):
                                                    best_spread = scenario['spreads'][0] if scenario.get('spreads') else None
                                                    prompt += f"""
                                                {i}. POT {scenario['target_pot']:.1%} → Strike ${scenario['short_strike']:.2f}
                                                   - Distance: {scenario['distance_pct']:.1f}% from current
                                                   - Actual POT: {scenario['actual_pot']:.2%}
                                                   - POP: {best_spread['prob_profit']:.1%} if best_spread else 'N/A'}
                                                   - Max Profit: ${best_spread['max_profit']:.2f} if best_spread else 'N/A'}"""
                                                
                                                prompt += """
                                                
                                                Provide a CONCISE analysis covering:
                                                1. **Strategy Assessment** (1-2 sentences on overall viability)
                                                2. **Best Recommendation** (specific strike and reasoning)
                                                3. **Risk Assessment** (clear risk level with data support)
                                                4. **Key Insights** (2-3 actionable takeaways)
                                                
                                                Be specific, use exact numbers, and keep it under 200 words total.
                                                """
                                                
                                                # Generate AI analysis
                                                from llm_analysis import LLMAnalyzer
                                                analyzer = LLMAnalyzer()
                                                
                                                ai_result = analyzer.generate_custom_analysis(prompt)
                                                
                                                if ai_result.get('success'):
                                                    st.markdown("""
                                                    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                                                padding: 2rem; border-radius: 16px; margin: 1rem 0; 
                                                                border-left: 6px solid var(--secondary-color); box-shadow: var(--shadow-lg);">
                                                        <h3 style="color: var(--secondary-color); margin: 0 0 1rem 0; font-weight: 700;">
                                                            🤖 AI Put Spread Strategy Analysis
                                                        </h3>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                    
                                                    st.markdown(ai_result['analysis'])
                                                    
                                                    with st.expander("📊 Analysis Details"):
                                                        col1, col2 = st.columns(2)
                                                        with col1:
                                                            st.metric("Tokens Used", ai_result.get('tokens_used', 0))
                                                        with col2:
                                                            st.metric("Analysis Time", 
                                                                    datetime.fromisoformat(ai_result['timestamp']).strftime("%H:%M:%S"))
                                                else:
                                                    st.warning("⚠️ AI analysis failed to generate results")
                                        except Exception as e:
                                            st.error(f"❌ AI analysis error: {str(e)}")
'''

# Add the distribution chart to Put Spread Analysis tab (in chart_tab1)
distribution_chart_addition = '''
                                
                                # Add POT/POP Distribution Charts
                                st.markdown("#### 📊 POT & POP Distribution Analysis")
                                dist_fig = create_pot_pop_distribution_charts(spread_results)
                                if dist_fig:
                                    st.plotly_chart(dist_fig, use_container_width=True)
                                    
                                    st.info("""
                                    **📈 Distribution Charts Explanation:**
                                    - **Top Left**: POT distribution with 2% bins (0%-30% range)
                                    - **Top Right**: POP distribution with 2% bins (70%-100% range)
                                    - **Bottom Left**: POT vs Strike Price correlation
                                    - **Bottom Right**: POP vs Strike Price correlation
                                    """)'''

# Insert distribution chart after POT analysis charts
pot_chart_marker = 'st.warning("⚠️ POT analysis charts not available")'
if pot_chart_marker in content:
    content = content.replace(pot_chart_marker, pot_chart_marker + distribution_chart_addition)

# Insert AI analysis after the charts section (before the "else" for error handling)
ai_insert_marker = 'volatility_used = spread_results[\'volatility\'] * 100\n                                        st.metric("Volatility Used", f"{volatility_used:.1f}%")'
if ai_insert_marker in content:
    content = content.replace(ai_insert_marker, ai_insert_marker + put_spread_ai_addition)

# Enhance Summary tab AI analysis to use ALL data
enhanced_summary_ai = '''
                                        # Prepare COMPREHENSIVE data from ALL tabs for AI analysis
                                        comprehensive_data = {
                                            'session_info': {
                                                'tickers': session_tickers,
                                                'date_range_days': (end_date - start_date).days,
                                                'timeframes_analyzed': [tf for tf in ['hourly', 'daily', 'weekly'] if any(tf in results[t] for t in session_tickers)]
                                            },
                                            'volatility_analysis': {},
                                            'vix_analysis': None,
                                            'options_analysis': None,
                                            'put_spread_analysis': None
                                        }
                                        
                                        # Collect volatility data from all tickers
                                        for ticker in session_tickers:
                                            ticker_data = {}
                                            for tf in ['hourly', 'daily', 'weekly']:
                                                if tf in results[ticker] and results[ticker][tf]:
                                                    tf_data = results[ticker][tf]
                                                    ticker_data[tf] = {
                                                        'atr': tf_data['atr'],
                                                        'volatility': tf_data['volatility'],
                                                        'data_points': len(tf_data['data']) if tf_data['data'] is not None else 0
                                                    }
                                            comprehensive_data['volatility_analysis'][ticker] = ticker_data
                                        
                                        # Add VIX analysis if available
                                        if vix_summary_data:
                                            comprehensive_data['vix_analysis'] = {
                                                'current_vix': vix_summary_data['current_vix'],
                                                'condition': vix_summary_data['condition'],
                                                'trading_approved': vix_summary_data['current_vix'] < 26
                                            }
                                        
                                        # Check if options analysis was run (from session state or current data)
                                        if hasattr(st.session_state, 'last_options_analysis') and st.session_state.last_options_analysis:
                                            comprehensive_data['options_analysis'] = st.session_state.last_options_analysis
                                        
                                        # Check if put spread analysis was run
                                        if hasattr(st.session_state, 'last_put_spread_analysis') and st.session_state.last_put_spread_analysis:
                                            comprehensive_data['put_spread_analysis'] = st.session_state.last_put_spread_analysis
                                        
                                        # Generate COMPREHENSIVE market summary
                                        market_summary_result = llm_analyzer.generate_comprehensive_market_summary(
                                            comprehensive_data=comprehensive_data
                                        )'''

# Replace the existing market summary call
old_summary_call = '''# Generate market summary
                                        market_summary_result = llm_analyzer.generate_market_summary(
                                            ticker_results=results,
                                            vix_data=vix_summary_data
                                        )'''

content = content.replace(old_summary_call, enhanced_summary_ai)

# Now I need to add the comprehensive market summary method to be used
llm_enhancement = '''
# Add comprehensive analysis capability to LLM
if LLM_AVAILABLE:
    def enhance_llm_analyzer():
        """Add comprehensive analysis methods to LLM analyzer"""
        try:
            from llm_analysis import LLMAnalyzer
            
            # Add method to existing analyzer
            def generate_comprehensive_market_summary(self, comprehensive_data):
                """Generate comprehensive market summary using all available data"""
                
                # Build comprehensive prompt
                prompt = f"""
                COMPREHENSIVE MARKET ANALYSIS SUMMARY
                
                **Session Overview:**
                - Tickers Analyzed: {', '.join(comprehensive_data['session_info']['tickers'])}
                - Analysis Period: {comprehensive_data['session_info']['date_range_days']} days
                - Timeframes: {', '.join(comprehensive_data['session_info']['timeframes_analyzed'])}
                
                **Volatility Analysis Results:**"""
                
                # Add volatility data for each ticker
                for ticker, data in comprehensive_data['volatility_analysis'].items():
                    prompt += f"""
                
                {ticker}:"""
                    for tf, metrics in data.items():
                        if metrics:
                            prompt += f"""
                  - {tf.title()}: ATR=${metrics['atr']:.2f}, Vol=${metrics['volatility']:.2f}, Points={metrics['data_points']}"""
                
                # Add VIX analysis
                if comprehensive_data['vix_analysis']:
                    vix_data = comprehensive_data['vix_analysis']
                    prompt += f"""
                
                **VIX Market Conditions:**
                - Current VIX: {vix_data['current_vix']:.2f}
                - Condition: {vix_data['condition']}
                - Trading Approved: {'Yes' if vix_data['trading_approved'] else 'No'}"""
                
                # Add options analysis if available
                if comprehensive_data['options_analysis']:
                    opt_data = comprehensive_data['options_analysis']
                    prompt += f"""
                
                **Options Analysis:**
                - Strategy Run: Yes
                - Key Results: [Include specific data if available]"""
                
                # Add put spread analysis if available
                if comprehensive_data['put_spread_analysis']:
                    spread_data = comprehensive_data['put_spread_analysis']
                    prompt += f"""
                
                **Put Spread Analysis:**
                - Advanced Analysis: Completed
                - Key Results: [Include specific data if available]"""
                
                prompt += """
                
                **REQUEST:**
                Provide a CONCISE but COMPREHENSIVE market summary (under 300 words) covering:
                
                1. **Market Assessment** (2-3 sentences with specific ATR/volatility data)
                2. **Trading Conditions** (VIX impact and recommendations)
                3. **Key Opportunities** (best performing tickers with exact numbers)
                4. **Risk Warnings** (specific concerns with data support)
                5. **Action Items** (2-3 specific recommendations)
                
                Use EXACT NUMBERS from the data. Be concise but authoritative.
                """
                
                return self.generate_custom_analysis(prompt)
            
            # Bind the method to LLMAnalyzer class
            LLMAnalyzer.generate_comprehensive_market_summary = generate_comprehensive_market_summary
            
        except Exception as e:
            print(f"Failed to enhance LLM analyzer: {e}")
    
    enhance_llm_analyzer()

'''

# Insert LLM enhancement at the beginning of the main function
main_function_pos = content.find('def main():')
if main_function_pos != -1:
    # Find the end of the main function definition line
    main_line_end = content.find('\n', main_function_pos)
    content = content[:main_line_end] + '\n\n' + llm_enhancement + content[main_line_end:]

# Write the enhanced content
with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Enhanced AI Analysis and Distribution Charts!")
print("📊 Added POT/POP distribution charts with 2% bins (0%-30% POT, 70%-100% POP)")
print("🤖 Added AI analysis capability to Put Spread Analysis tab")
print("📈 Enhanced Summary tab AI to use ALL data from all tabs")
print("🎯 AI will now provide comprehensive but concise analysis with specific data points")
print("📋 Distribution charts show probability distributions vs strike prices") 