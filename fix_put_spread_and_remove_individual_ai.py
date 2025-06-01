#!/usr/bin/env python3
"""
Fix Put Spread Analysis module availability and remove individual AI buttons
Keep only Summary AI that uses ALL data from all tabs
"""

# Read the current streamlit app
with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("🔧 Fixing Put Spread Analysis module availability...")

# Fix the PUT_SPREAD_AVAILABLE check - the issue is that it's checking at module level
# but the module is available, so let's fix the logic
put_spread_fix = '''
        with tab7:
            st.subheader("📐 Advanced Put Spread Analysis")
            
            st.subheader("🏗️ Black-Scholes Put Spread Probability Analysis")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                        border-left: 4px solid var(--secondary-color);">
                <h4 style="color: var(--secondary-color); margin: 0 0 1rem 0;">
                    🎯 Advanced Options Analysis with Black-Scholes Formulas
                </h4>
                <p style="margin: 0; color: var(--text-secondary);">
                    This analysis uses precise Black-Scholes calculations for Probability of Profit (POP) and 
                    Probability of Touching (POT) for vertical put spreads with real options chain data.
                </p>
            </div>
            """, unsafe_allow_html=True)'''

# Replace the entire tab7 section to remove the PUT_SPREAD_AVAILABLE check and individual AI
old_tab7_start = 'with tab7:'
old_tab7_end = 'if __name__ == "__main__":'

# Find the tab7 section
tab7_start_pos = content.find(old_tab7_start)
if tab7_start_pos != -1:
    # Find the end of tab7 (start of main check)
    tab7_end_pos = content.find(old_tab7_end)
    
    if tab7_end_pos != -1:
        # Extract everything before tab7 and after main check
        before_tab7 = content[:tab7_start_pos]
        after_main = content[tab7_end_pos:]
        
        # Create new tab7 content without PUT_SPREAD_AVAILABLE check and without individual AI
        new_tab7_content = '''        with tab7:
            st.subheader("📐 Advanced Put Spread Analysis")
            
            st.subheader("🏗️ Black-Scholes Put Spread Probability Analysis")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                        border-left: 4px solid var(--secondary-color);">
                <h4 style="color: var(--secondary-color); margin: 0 0 1rem 0;">
                    🎯 Advanced Options Analysis with Black-Scholes Formulas
                </h4>
                <p style="margin: 0; color: var(--text-secondary);">
                    This analysis uses precise Black-Scholes calculations for Probability of Profit (POP) and 
                    Probability of Touching (POT) for vertical put spreads with real options chain data.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Put Spread Configuration
            st.markdown("### 📅 Put Spread Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                spread_ticker = st.selectbox(
                    "Select ticker for put spread analysis:",
                    session_tickers,
                    help="Choose the ticker for comprehensive put spread analysis",
                    key="spread_ticker"
                )
            
            with col2:
                # Expiry date selection with presets
                expiry_preset = st.selectbox(
                    "Expiry preset:",
                    ["Same Day", "Next Friday", "Custom Date"],
                    help="Quick presets for common expiry dates",
                    key="expiry_preset"
                )
                
                if expiry_preset == "Same Day":
                    spread_expiry = get_same_day_expiry()
                elif expiry_preset == "Next Friday":
                    spread_expiry = get_next_friday()
                else:  # Custom Date
                    custom_expiry = st.date_input(
                        "Custom expiry date:",
                        value=date.today() + timedelta(days=1),
                        min_value=date.today(),
                        max_value=date.today() + timedelta(days=60),
                        key="custom_expiry"
                    )
                    spread_expiry = custom_expiry.strftime('%Y-%m-%d')
            
            with col3:
                st.info(f"**Selected Expiry**: {spread_expiry}")
                
                # POT Target levels
                pot_target_preset = st.selectbox(
                    "POT analysis levels:",
                    ["Standard (20%, 10%, 5%, 2%, 1%)", 
                     "Conservative (10%, 5%, 2%, 1%, 0.5%)",
                     "Aggressive (20%, 15%, 10%, 5%, 2%)",
                     "Ultra-Safe (5%, 2%, 1%, 0.5%, 0.25%)"],
                    help="Preset POT levels to analyze",
                    key="pot_preset"
                )
            
            # Custom POT levels
            st.markdown("#### 🎯 Custom POT Levels")
            custom_pot_input = st.text_input(
                "Custom POT levels (% - comma separated):",
                placeholder="e.g., 20, 10, 5, 2, 1, 0.5",
                help="Enter custom POT percentages to analyze",
                key="custom_pot"
            )
            
            # Generate analysis button
            if st.button("🚀 Generate Advanced Put Spread Analysis", type="primary", key="spread_analysis_btn"):
                try:
                    st.info("🔄 Generating comprehensive put spread analysis with Black-Scholes formulas...")
                    
                    # Initialize analyzer
                    try:
                        from put_spread_analysis import (
                            PutSpreadAnalyzer, 
                            format_percentage, 
                            format_currency, 
                            get_next_friday, 
                            get_same_day_expiry
                        )
                        spread_analyzer = PutSpreadAnalyzer()
                    except ImportError:
                        st.error("❌ Put Spread Analysis module not found. Please ensure put_spread_analysis.py is in the same directory.")
                        return
                    
                    # Get current price
                    with st.spinner("Fetching current market data..."):
                        current_price = get_current_price(spread_ticker)
                    
                    if current_price is None:
                        st.error(f"❌ Could not fetch current price for {spread_ticker}")
                        return
                    
                    st.success(f"✅ Current price: ${current_price:.2f}")
                    
                    # Determine POT levels to analyze
                    if custom_pot_input.strip():
                        try:
                            pot_levels = [float(x.strip())/100 for x in custom_pot_input.split(',')]
                        except ValueError:
                            st.error("❌ Invalid POT format. Using standard levels.")
                            pot_levels = [0.20, 0.10, 0.05, 0.02, 0.01]
                    else:
                        if pot_target_preset == "Standard (20%, 10%, 5%, 2%, 1%)":
                            pot_levels = [0.20, 0.10, 0.05, 0.02, 0.01]
                        elif pot_target_preset == "Conservative (10%, 5%, 2%, 1%, 0.5%)":
                            pot_levels = [0.10, 0.05, 0.02, 0.01, 0.005]
                        elif pot_target_preset == "Aggressive (20%, 15%, 10%, 5%, 2%)":
                            pot_levels = [0.20, 0.15, 0.10, 0.05, 0.02]
                        else:  # Ultra-Safe
                            pot_levels = [0.05, 0.02, 0.01, 0.005, 0.0025]
                    
                    # Get ATR for volatility estimation
                    atr_value = None
                    if spread_ticker in results and 'daily' in results[spread_ticker]:
                        atr_value = results[spread_ticker]['daily']['atr']
                    
                    # Run comprehensive analysis
                    with st.spinner("🧮 Running Black-Scholes calculations..."):
                        spread_results = spread_analyzer.analyze_put_spread_scenarios(
                            ticker=spread_ticker,
                            current_price=current_price,
                            expiry_date=spread_expiry,
                            atr=atr_value,
                            target_pot_levels=pot_levels
                        )
                    
                    if spread_results and spread_results['scenarios']:
                        st.success("✅ Put spread analysis complete!")
                        
                        # Store put spread analysis in session state for Summary AI
                        if spread_results and spread_results['scenarios']:
                            best_scenario = spread_results['scenarios'][0]
                            best_spread = best_scenario['spreads'][0] if best_scenario.get('spreads') else None
                            
                            st.session_state.last_put_spread_analysis = {
                                'ticker': spread_ticker,
                                'expiry_date': spread_results['expiry_date'],
                                'current_price': spread_results['current_price'],
                                'scenarios_count': len(spread_results['scenarios']),
                                'best_pot_target': best_scenario['target_pot'],
                                'best_pot_actual': best_scenario['actual_pot'],
                                'best_strike': best_scenario['short_strike'],
                                'best_distance_pct': best_scenario['distance_pct'],
                                'best_pop': best_spread['prob_profit'] if best_spread else None,
                                'volatility_used': spread_results['volatility'],
                                'options_data_available': spread_results.get('options_data_available', False),
                                'timestamp': datetime.now().isoformat()
                            }
                        
                        # Display comprehensive Black-Scholes parameters
                        st.markdown("### 📊 Black-Scholes Analysis Parameters")
                        
                        # Create a beautiful parameters display with mathematical notation
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                    border-left: 4px solid var(--primary-color); box-shadow: var(--shadow-sm);">
                            <h4 style="color: var(--primary-color); margin: 0 0 1rem 0; font-weight: 700;">
                                🧮 Mathematical Variables Used in Black-Scholes Calculations
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Extract values for current spread analysis
                        current_price = spread_results['current_price']
                        time_to_expiry_years = spread_results['time_to_expiry_days'] / 365.25
                        volatility = spread_results['volatility']
                        risk_free_rate = spread_results['risk_free_rate']
                        dividend_yield = spread_results.get('dividend_yield', 0.0)
                        
                        # Get example strikes from first scenario
                        if spread_results['scenarios']:
                            example_scenario = spread_results['scenarios'][0]
                            short_strike = example_scenario['short_strike']
                            if example_scenario['spreads']:
                                long_strike = example_scenario['spreads'][0]['long_strike']
                            else:
                                long_strike = short_strike - 5
                        else:
                            short_strike = current_price * 0.95
                            long_strike = current_price * 0.90
                        
                        # Display parameters in organized sections
                        st.markdown("#### 🎯 Core Market Variables")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            **S = Current Stock Price**
                            - **Value**: {format_currency(current_price)}
                            - **Description**: Underlying asset price
                            - **Usage**: Base for all calculations
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **σ = Implied Volatility**
                            - **Value**: {format_percentage(volatility)}
                            - **Description**: Expected price volatility
                            - **Usage**: Measures uncertainty in price movement
                            """)
                        
                        with col3:
                            st.markdown(f"""
                            **T = Time to Expiration**
                            - **Value**: {time_to_expiry_years:.4f} years ({spread_results['time_to_expiry_days']:.1f} days)
                            - **Description**: Time until option expires
                            - **Usage**: Time decay factor in pricing
                            """)
                        
                        st.markdown("#### 📈 Strike Price Variables")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            **A = Lower Strike Price (Long PUT)**
                            - **Value**: {format_currency(long_strike)}
                            - **Description**: Strike of PUT we BUY
                            - **Usage**: Provides downside protection
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **B = Higher Strike Price (Short PUT)**
                            - **Value**: {format_currency(short_strike)}
                            - **Description**: Strike of PUT we SELL
                            - **Usage**: Generates premium income
                            """)
                        
                        st.markdown("#### 💰 Economic Variables")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            **r = Risk-free Interest Rate**
                            - **Value**: {format_percentage(risk_free_rate)}
                            - **Description**: Treasury rate for discounting
                            - **Usage**: Present value calculations
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **q = Dividend Yield**
                            - **Value**: {format_percentage(dividend_yield)}
                            - **Description**: Expected dividend payments
                            - **Usage**: Adjusts for dividend impact
                            """)
                        
                        st.markdown("#### 🔢 Mathematical Functions")
                        st.markdown("""
                        **N() = Cumulative Normal Distribution Function**
                        - **Description**: Standard normal cumulative distribution
                        - **Usage**: Converts Z-scores to probabilities
                        - **Formula**: ∫_{-∞}^{x} (1/√2π) * e^(-t²/2) dt
                        """)
                        
                        # Show the actual Black-Scholes formulas being used
                        st.markdown("#### 📝 Black-Scholes Formulas Used")
                        
                        st.markdown("""
                        **Probability of Profit Formula:**
                        ```
                        P(profit) = N((ln(S/B) + (r-q+σ²/2)T) / (σ√T)) - N((ln(S/A) + (r-q+σ²/2)T) / (σ√T))
                        ```
                        
                        **Probability of Touching (POT) Formula:**
                        ```
                        POT = 2 × N((|ln(S/B)| - (r-q-σ²/2)T) / (σ√T))
                        ```
                        
                        Where:
                        - **S** = Current stock price
                        - **A** = Lower strike (long put)
                        - **B** = Higher strike (short put)  
                        - **r** = Risk-free rate
                        - **q** = Dividend yield
                        - **σ** = Implied volatility
                        - **T** = Time to expiration
                        - **N()** = Cumulative normal distribution
                        """)
                        
                        # Options data source information
                        if spread_results['options_data_available']:
                            st.success("✅ **Data Source**: Real options chain data used for implied volatility")
                        else:
                            st.warning("⚠️ **Data Source**: Using ATR-based volatility estimate (options chain unavailable)")
                        
                        # Summary metrics in a clean format
                        st.markdown("#### 📊 Quick Reference Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📈 Current Price (S)", format_currency(current_price))
                        with col2:
                            st.metric("⏰ Time to Expiry (T)", f"{spread_results['time_to_expiry_days']:.1f} days")
                        with col3:
                            st.metric("📊 Volatility (σ)", format_percentage(volatility))
                        with col4:
                            st.metric("💰 Risk-free Rate (r)", format_percentage(risk_free_rate))
                        
                        # Enhanced POT Strike Price Analysis
                        st.markdown("### 🎯 POT Strike Price Analysis - Key Probability Levels")
                        
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                    border-left: 4px solid var(--secondary-color);">
                            <h4 style="color: var(--secondary-color); margin: 0 0 1rem 0; font-weight: 700;">
                                🎯 Strike Prices for Specific POT Levels
                            </h4>
                            <p style="margin: 0; color: var(--text-secondary); font-size: 1rem;">
                                The table below shows the exact strike prices needed to achieve your target POT percentages.
                                This is the core probability analysis for PUT selling strategies.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create enhanced POT table
                        pot_data = []
                        for scenario in spread_results['scenarios']:
                            target_pot_pct = scenario['target_pot'] * 100
                            actual_pot_pct = scenario['actual_pot'] * 100
                            strike_price = scenario['short_strike']
                            distance_dollars = scenario['distance_from_current']
                            distance_pct = scenario['distance_pct']
                            
                            # Determine safety level based on distance
                            if distance_pct > 5:
                                safety = "🟢 VERY SAFE"
                            elif distance_pct > 3:
                                safety = "🟡 SAFE"
                            elif distance_pct > 2:
                                safety = "🟠 MODERATE"
                            else:
                                safety = "🔴 RISKY"
                            
                            pot_data.append({
                                'Target POT %': f"{target_pot_pct:.1f}%",
                                'Strike Price': format_currency(strike_price),
                                'Actual POT %': f"{actual_pot_pct:.2f}%",
                                'Distance from Current': format_currency(distance_dollars),
                                'Distance %': f"{distance_pct:.1f}%",
                                'Safety Level': safety,
                                'Recommendation': 'SELL PUT' if distance_pct > 2 else 'AVOID'
                            })
                        
                        pot_df = pd.DataFrame(pot_data)
                        st.dataframe(pot_df, use_container_width=True)
                        
                        # Add summary of key POT levels
                        st.markdown("#### 📊 Key POT Levels Summary")
                        
                        # Create columns for key POT levels
                        if len(spread_results['scenarios']) >= 5:
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            for i, scenario in enumerate(spread_results['scenarios'][:5]):
                                target_pct = scenario['target_pot'] * 100
                                strike = scenario['short_strike']
                                distance_pct = scenario['distance_pct']
                                
                                with [col1, col2, col3, col4, col5][i]:
                                    st.metric(
                                        f"{target_pct:.1f}% POT",
                                        format_currency(strike),
                                        f"{distance_pct:.1f}% away"
                                    )
                        
                        # Explanation of POT significance
                        st.markdown("""
                        **💡 POT Analysis Explanation:**
                        - **POT %** = Probability the stock will touch this strike price before expiration
                        - **Lower POT %** = Safer strike (less likely to be touched)
                        - **Higher Distance %** = More conservative strike selection
                        - **Recommendation** = Based on probability and distance analysis
                        """)
                        
                        # Spread Analysis for Best Scenarios
                        st.markdown("### 📈 Put Spread Probability of Profit Analysis")
                        
                        # Show detailed spread analysis for the most conservative scenarios
                        best_scenarios = spread_results['scenarios'][:3]  # Top 3 most conservative
                        
                        for i, scenario in enumerate(best_scenarios):
                            if scenario['spreads']:
                                st.markdown(f"#### 🎯 POT {format_percentage(scenario['target_pot'])} - Strike ${scenario['short_strike']:.2f}")
                                
                                spread_data = []
                                for spread in scenario['spreads']:
                                    spread_data.append({
                                        'Spread Width': f"${spread['width']:.0f}",
                                        'Long Strike': format_currency(spread['long_strike']),
                                        'Short Strike': format_currency(spread['short_strike']),
                                        'Probability of Profit': format_percentage(spread['prob_profit']),
                                        'Max Profit': format_currency(spread['max_profit']),
                                        'Distance %': f"{spread['distance_pct']:.1f}%"
                                    })
                                
                                spread_df = pd.DataFrame(spread_data)
                                st.dataframe(spread_df, use_container_width=True)
                        
                        # Best Recommendation
                        if spread_results['scenarios']:
                            best_scenario = spread_results['scenarios'][0]
                            if best_scenario['spreads']:
                                best_spread = best_scenario['spreads'][0]
                                
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                                            padding: 2rem; border-radius: 16px; margin: 1rem 0; 
                                            border: 2px solid var(--success-color); box-shadow: var(--shadow-lg);">
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                ### 🏆 OPTIMAL PUT SPREAD RECOMMENDATION
                                
                                **Strategy**: Buy ${best_spread['long_strike']:.2f} PUT / Sell ${best_spread['short_strike']:.2f} PUT
                                
                                **Black-Scholes Analysis**:
                                - **Probability of Profit**: {format_percentage(best_spread['prob_profit'])}
                                - **Probability of Touching**: {format_percentage(best_scenario['actual_pot'])}
                                - **Distance from Current**: {format_currency(best_scenario['distance_from_current'])} ({best_scenario['distance_pct']:.1f}%)
                                - **Max Profit**: {format_currency(best_spread['max_profit'])}
                                - **Spread Width**: ${best_spread['width']:.0f}
                                
                                **Risk Assessment**: {"🟢 LOW RISK" if best_scenario['distance_pct'] > 5 else "🟡 MEDIUM RISK" if best_scenario['distance_pct'] > 2 else "🔴 HIGH RISK"}
                                
                                **Expiry**: {spread_expiry}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Statistical Analysis Section - POT & POP Charts
                        st.markdown("---")
                        st.markdown("### 📊 Put Spread Statistical Analysis & Charts")
                        
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                    border-left: 4px solid var(--secondary-color);">
                            <h4 style="color: var(--secondary-color); margin: 0 0 1rem 0;">
                                📈 Interactive Statistical Analysis
                            </h4>
                            <p style="margin: 0; color: var(--text-secondary);">
                                Comprehensive visual analysis of POT levels, POP distributions, safety assessments, and strategy comparisons.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create statistical charts in tabs
                        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 POT & POP Analysis", "📈 Strategy Comparison", "📋 Statistical Summary"])
                        
                        with chart_tab1:
                            # POT Analysis Charts
                            pot_fig = create_pot_analysis_charts(spread_results)
                            if pot_fig:
                                st.plotly_chart(pot_fig, use_container_width=True)
                                
                                st.info("""
                                **📊 POT Analysis Charts:**
                                - **Target vs Actual POT**: Shows accuracy of POT calculations
                                - **Strike Distance Distribution**: Histogram of distances from current price
                                - **Safety Level Breakdown**: Pie chart of risk categories
                                - **POT vs Distance Correlation**: Relationship between POT and safety
                                """)
                            else:
                                st.warning("⚠️ POT analysis charts not available")
                                
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
                                """)
                        
                        with chart_tab2:
                            # Strategy Comparison Charts
                            comp_fig = create_strategy_comparison_chart(spread_results)
                            if comp_fig:
                                st.plotly_chart(comp_fig, use_container_width=True)
                                
                                st.info("""
                                **📈 Strategy Comparison:**
                                - **Risk vs Reward**: Distance % vs Probability of Profit
                                - **Strike Price Levels**: Comparison across POT targets
                                """)
                            
                            # Add POP vs POT scatter plot
                            if spread_results['scenarios']:
                                pot_data = [s['actual_pot'] * 100 for s in spread_results['scenarios']]
                                pop_data = [s['spreads'][0]['prob_profit'] * 100 for s in spread_results['scenarios'] if s.get('spreads')]
                                distance_data = [s['distance_pct'] for s in spread_results['scenarios']]
                                target_pot_data = [s['target_pot'] * 100 for s in spread_results['scenarios']]
                                
                                if pot_data and pop_data and len(pot_data) == len(pop_data):
                                    fig_pop_pot = go.Figure()
                                    
                                    fig_pop_pot.add_trace(go.Scatter(
                                        x=pot_data,
                                        y=pop_data,
                                        mode='markers+text',
                                        text=[f"{t:.1f}%" for t in target_pot_data[:len(pop_data)]],
                                        textposition='top center',
                                        marker=dict(
                                            size=[d/2 + 8 for d in distance_data[:len(pop_data)]],
                                            color=distance_data[:len(pop_data)],
                                            colorscale='RdYlGn',
                                            showscale=True,
                                            colorbar=dict(title="Distance %")
                                        ),
                                        name='POP vs POT'
                                    ))
                                    
                                    fig_pop_pot.update_layout(
                                        title='Probability of Profit vs Probability of Touching',
                                        xaxis_title='POT (Probability of Touching) %',
                                        yaxis_title='POP (Probability of Profit) %',
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig_pop_pot, use_container_width=True)
                                    
                                    st.success("""
                                    **🎯 Key Insight**: Lower POT = Safer strikes, Higher POP = Better profit odds
                                    """)
                        
                        with chart_tab3:
                            # Statistical Summary
                            st.markdown("#### 📋 Comprehensive Statistical Summary")
                            
                            if spread_results['scenarios']:
                                summary_data = []
                                for scenario in spread_results['scenarios']:
                                    best_spread = scenario['spreads'][0] if scenario.get('spreads') else None
                                    
                                    summary_data.append({
                                        'POT Target': f"{scenario['target_pot']:.1%}",
                                        'POT Actual': f"{scenario['actual_pot']:.2%}",
                                        'Strike Price': f"${scenario['short_strike']:.2f}",
                                        'Distance %': f"{scenario['distance_pct']:.1f}%",
                                        'POP %': f"{best_spread['prob_profit']*100:.1f}%" if best_spread else "N/A",
                                        'Max Profit': f"${best_spread['max_profit']:.2f}" if best_spread else "N/A",
                                        'Safety': 'High' if scenario['distance_pct'] > 5 else 'Medium' if scenario['distance_pct'] > 2 else 'Low'
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df, use_container_width=True)
                                
                                # Key metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_distance = np.mean([s['distance_pct'] for s in spread_results['scenarios']])
                                    st.metric("Avg Distance %", f"{avg_distance:.1f}%")
                                
                                with col2:
                                    safe_count = sum(1 for s in spread_results['scenarios'] if s['distance_pct'] > 5)
                                    st.metric("Safe Strikes (>5%)", f"{safe_count}/{len(spread_results['scenarios'])}")
                                
                                with col3:
                                    volatility_used = spread_results['volatility'] * 100
                                    st.metric("Volatility Used", f"{volatility_used:.1f}%")
                    
                    else:
                        st.error("❌ Could not generate put spread analysis. Check data availability.")
                    
                except Exception as e:
                    st.error(f"❌ Put spread analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        # No analysis run yet - show standalone Options
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
                    padding: 2rem; border-radius: 16px; margin: 2rem 0; 
                    border-left: 6px solid var(--warning-color); box-shadow: var(--shadow-md);">
            <h2 style="color: var(--warning-color); margin: 0; font-weight: 700;">
                📊 Enhanced Stock Volatility Analyzer
            </h2>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                ℹ️ Run the Enhanced Analysis first to see all features, or use Basic Options Strategy below.
            </p>
        </div>
        """, unsafe_allow_html=True)

'''
        
        # Replace the tab7 content
        content = before_tab7 + new_tab7_content + after_main

print("✅ Fixed Put Spread Analysis module availability")
print("🚫 Removing individual AI analysis buttons from all tabs...")

# Remove AI analysis from Price Charts tab
content = content.replace('''# AI Analysis for Price Charts
                if enable_ai_analysis and LLM_AVAILABLE:
                    st.markdown("---")
                    st.markdown("### 🤖 AI Price Analysis")
                    
                    if st.button("🧠 Generate AI Price Analysis", key="price_ai_btn"):
                        with st.spinner("🤖 AI analyzing price patterns..."):
                            try:
                                llm_analyzer = get_llm_analyzer()
                                if llm_analyzer:
                                    # Prepare price data for AI
                                    price_summary = {
                                        'ticker': chart_ticker,
                                        'timeframe': chart_timeframe,
                                        'current_price': chart_data['Close'].iloc[-1],
                                        'price_change': chart_data['Close'].iloc[-1] - chart_data['Close'].iloc[0],
                                        'volatility': chart_data['true_range'].mean() if 'true_range' in chart_data.columns else None,
                                        'trend': 'Bullish' if chart_data['Close'].iloc[-1] > chart_data['Close'].iloc[0] else 'Bearish'
                                    }
                                    
                                    # Generate AI analysis (you'll need to add this method to llm_analysis.py)
                                    # For now, show a placeholder
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                                padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                                border-left: 4px solid var(--secondary-color);">
                                        <h4 style="color: var(--secondary-color); margin: 0 0 1rem 0;">
                                            🤖 AI Price Pattern Analysis
                                        </h4>
                                        <p>AI analysis for price patterns will be implemented here.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"AI analysis failed: {str(e)}")''', '')

# Remove AI analysis from Options Strategy tab
ai_options_section = '''                        # Optional AI Analysis for Options Strategy
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

content = content.replace(ai_options_section, '')

# Enhance Summary tab AI to use enhanced data from put spread analysis
enhanced_summary_data = '''                                        # Add put spread analysis data to comprehensive summary
                                        if comprehensive_data['put_spread_analysis']:
                                            spread_data = comprehensive_data['put_spread_analysis']
                                            prompt += f"""
                
                **Put Spread Analysis:**
                - Ticker: {spread_data['ticker']}
                - Current Price: ${spread_data['current_price']:.2f}
                - Expiry: {spread_data['expiry_date']}
                - Scenarios Analyzed: {spread_data['scenarios_count']}
                - Best POT Target: {spread_data['best_pot_target']:.1%}
                - Best Strike: ${spread_data['best_strike']:.2f}
                - Distance: {spread_data['best_distance_pct']:.1f}%
                - Best POP: {spread_data['best_pop']:.1%} if spread_data['best_pop'] else 'N/A'
                - Volatility Used: {spread_data['volatility_used']:.1%}
                - Data Source: {'Options Chain' if spread_data['options_data_available'] else 'ATR-based'}"""
                
                # Add options analysis data to comprehensive summary
                if comprehensive_data['options_analysis']:
                    opt_data = comprehensive_data['options_analysis']
                    prompt += f"""
                
                **Options Analysis:**
                - Ticker: {opt_data['ticker']}
                - Timeframe: {opt_data['timeframe']}
                - Current Price: ${opt_data['current_price']:.2f}
                - Best Strike: ${opt_data['best_strike']:.2f}
                - Distance: {opt_data['best_distance_pct']:.1f}%
                - Safety Score: {opt_data['best_safety_score']:.1%}
                - VIX Approved: {'Yes' if opt_data['vix_approved'] else 'No'}"""'''

# Find and replace the options/put spread analysis placeholders with real data
old_placeholder = '''                # Add options analysis if available
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
                - Key Results: [Include specific data if available]"""'''

content = content.replace(old_placeholder, enhanced_summary_data)

# Write the fixed content
with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("🎯 Summary of fixes:")
print("1. ✅ Fixed Put Spread Analysis module availability - removed PUT_SPREAD_AVAILABLE check")
print("2. 🚫 Removed AI analysis buttons from Price Charts tab")
print("3. 🚫 Removed AI analysis buttons from Options Strategy tab") 
print("4. 🚫 Removed AI analysis buttons from Put Spread Analysis tab")
print("5. 🤖 Enhanced Summary AI to use detailed data from all analyses")
print("6. 📊 Kept all distribution charts and statistical analysis")
print("7. 🎯 AI analysis now ONLY available in Summary tab with comprehensive data") 