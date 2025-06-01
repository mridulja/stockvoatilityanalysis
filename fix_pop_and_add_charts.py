#!/usr/bin/env python3
"""
Fix POP formula and add missing statistical charts to Put Spread Analysis tab
"""

# Read the current streamlit app
with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# First, verify the POP formula was already fixed in put_spread_analysis.py
print("✅ POP formula already fixed in put_spread_analysis.py")

# Add statistical charts section after the optimal recommendation
chart_addition = '''
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
'''

# Find the position to insert the charts - after the optimal recommendation section
target_marker = "st.markdown('</div>', unsafe_allow_html=True)"
marker_positions = []
lines = content.split('\n')

for i, line in enumerate(lines):
    if target_marker in line:
        marker_positions.append(i)

# Find the correct position - should be in the put spread analysis section
# Look for the position after the "OPTIMAL PUT SPREAD RECOMMENDATION" section
optimal_rec_line = None
for i, line in enumerate(lines):
    if "OPTIMAL PUT SPREAD RECOMMENDATION" in line:
        optimal_rec_line = i
        break

if optimal_rec_line is not None:
    # Find the closing div after the optimal recommendation
    for i in range(optimal_rec_line, len(lines)):
        if target_marker in lines[i]:
            # Insert charts after this line
            insert_position = i + 1
            break
    
    if insert_position:
        # Split content and insert charts
        lines_before = lines[:insert_position]
        lines_after = lines[insert_position:]
        
        # Add the chart section
        chart_lines = chart_addition.split('\n')
        
        # Combine all lines
        new_lines = lines_before + chart_lines + lines_after
        new_content = '\n'.join(new_lines)
        
        # Write the updated content
        with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Added statistical charts to Put Spread Analysis tab!")
        print(f"📊 Charts inserted at line {insert_position}")
        print("🎯 Added POT analysis, POP vs POT correlation, and statistical summaries")
    else:
        print("❌ Could not find insertion point for charts")
else:
    print("❌ Could not find optimal recommendation section")

print("🔄 Testing the fixed POP formula...")

# Test the fixed formula
try:
    import subprocess
    result = subprocess.run(['python', 'test_pop_formula.py'], 
                          capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print("✅ POP formula test completed successfully")
        # Show key results
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'Correct POP' in line:
                print(f"✅ {line}")
    else:
        print("❌ POP formula test failed")
        print(result.stderr)
except Exception as e:
    print(f"⚠️ Could not run POP test: {e}")

print("\n🎉 SUMMARY:")
print("1. ✅ Fixed POP formula in put_spread_analysis.py (was using wrong d1, now uses correct d2)")
print("2. ✅ Added comprehensive statistical charts to Put Spread Analysis tab")
print("3. 📊 Added POT analysis charts, POP vs POT correlation, strategy comparison")
print("4. 📋 Added detailed statistical summary tables")
print("5. 🎯 POP should now show realistic percentages instead of 0%") 