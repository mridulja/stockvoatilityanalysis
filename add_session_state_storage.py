#!/usr/bin/env python3
"""
Add session state storage for analysis results so Summary AI can access all data
"""

# Read the current streamlit app
with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add session state storage after successful options analysis
options_storage = '''
                        
                        # Store options analysis in session state for Summary AI
                        best_rec_for_storage = recommendations[0] if recommendations else None
                        if best_rec_for_storage:
                            st.session_state.last_options_analysis = {
                                'ticker': strategy_ticker,
                                'timeframe': strategy_timeframe,
                                'current_price': current_price,
                                'best_strike': best_rec_for_storage['strike'],
                                'best_distance_pct': best_rec_for_storage['distance_pct'],
                                'best_prob_below': best_rec_for_storage['prob_below'],
                                'best_safety_score': best_rec_for_storage['safety_score'],
                                'confidence_levels': conf_data if 'conf_data' in locals() else None,
                                'vix_approved': trade_approved if 'trade_approved' in locals() else True,
                                'timestamp': datetime.now().isoformat()
                            }'''

# Insert after successful options strategy generation
options_marker = 'st.success("✅ Enhanced options strategy analysis complete!")'
if options_marker in content:
    content = content.replace(options_marker, options_marker + options_storage)

# Add session state storage after successful put spread analysis
put_spread_storage = '''
                        
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
                            }'''

# Insert after successful put spread analysis
put_spread_marker = 'st.success("✅ Put spread analysis complete!")'
if put_spread_marker in content:
    content = content.replace(put_spread_marker, put_spread_marker + put_spread_storage)

# Write the enhanced content
with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Added session state storage for analysis results!")
print("🎯 Options analysis results now stored in st.session_state.last_options_analysis")
print("📐 Put spread analysis results now stored in st.session_state.last_put_spread_analysis")
print("🤖 Summary AI can now access comprehensive data from all tabs") 