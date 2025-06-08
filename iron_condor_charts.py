"""
Iron Condor Charts Module - Enhanced with Time Decay Simulation

This module provides comprehensive visualization capabilities for Iron Condor strategies
including P&L diagrams, time decay simulation, exit strategy analysis, and technical metrics.

Features:
- Interactive P&L diagrams
- Time decay simulation with multiple price scenarios
- Exit strategy comparison (21 DTE vs Hold to Expiry)
- Technical metrics visualization (Greeks, ROC, POPrem)
- Educational charts with decision analysis
- Profit/loss simulation over time

Author: AI Assistant
Date: 2025
Version: 2.0 - Enhanced with Simulation and Technical Analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_iron_condor_pnl_chart(strategy, current_price, price_range_pct=0.30):
    """Create clean Iron Condor P&L diagram with minimal text overlap"""
    try:
        # Extract strategy parameters
        call_short = strategy.get('call_short', 0)
        call_long = strategy.get('call_long', 0)
        put_short = strategy.get('put_short', 0)
        put_long = strategy.get('put_long', 0)
        total_credit = strategy.get('total_credit', 0)
        
        if not all([call_short, call_long, put_short, put_long, total_credit]):
            return None
        
        # Calculate key levels
        max_profit = total_credit * 100
        max_loss = (strategy.get('wing_width', 0) - total_credit) * 100
        breakeven_lower = put_short - total_credit
        breakeven_upper = call_short + total_credit
        
        # Create price range for P&L calculation
        price_range_dollars = current_price * price_range_pct
        price_min = current_price - price_range_dollars
        price_max = current_price + price_range_dollars
        
        # Ensure we cover all critical points
        critical_points = [put_long, put_short, breakeven_lower, current_price, 
                          breakeven_upper, call_short, call_long]
        price_min = min(price_min, min(critical_points) - 10)
        price_max = max(price_max, max(critical_points) + 10)
        
        # Generate price points with higher density around critical areas
        price_points = []
        
        # Add dense points around critical areas
        for critical in critical_points:
            local_range = np.linspace(critical - 5, critical + 5, 20)
            price_points.extend(local_range)
        
        # Add broader range points
        broad_range = np.linspace(price_min, price_max, 100)
        price_points.extend(broad_range)
        
        # Remove duplicates and sort
        price_points = sorted(list(set(price_points)))
        price_array = np.array(price_points)
        
        # Calculate P&L for each price point
        pnl_values = []
        
        for price in price_array:
            # Call spread P&L: short call - long call
            if price <= call_short:
                call_pnl = 0  # Both expire worthless
            elif price <= call_long:
                call_pnl = (price - call_short) * 100  # Short call ITM, long OTM
            else:
                call_pnl = (call_long - call_short) * 100  # Both ITM, spread at max loss
            
            # Put spread P&L: short put - long put
            if price >= put_short:
                put_pnl = 0  # Both expire worthless
            elif price >= put_long:
                put_pnl = (put_short - price) * 100  # Short put ITM, long OTM
            else:
                put_pnl = (put_short - put_long) * 100  # Both ITM, spread at max loss
            
            # Total P&L = Credit received + Call spread P&L + Put spread P&L
            total_pnl = max_profit + call_pnl + put_pnl
            pnl_values.append(total_pnl)
        
        # Create the plot with better layout
        fig = go.Figure()
        
        # Add P&L line with profit/loss color coding
        profit_mask = np.array(pnl_values) > 0
        loss_mask = np.array(pnl_values) < 0
        
        # Profit zone (green)
        profit_prices = price_array[profit_mask]
        profit_pnl = np.array(pnl_values)[profit_mask]
        
        if len(profit_prices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=profit_prices,
                    y=profit_pnl,
                    mode='lines',
                    name='Profit Zone',
                    line=dict(color='green', width=4),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.2)'
                ))
        
        # Loss zone (red)
        loss_prices = price_array[loss_mask]
        loss_pnl = np.array(pnl_values)[loss_mask]
        
        if len(loss_prices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=loss_prices,
                    y=loss_pnl,
                    mode='lines',
                    name='Loss Zone',
                    line=dict(color='red', width=4),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)'
                ))
        
        # Add full P&L line for continuity
        fig.add_trace(
            go.Scatter(
                x=price_array,
                y=pnl_values,
                mode='lines',
                name='Iron Condor P&L',
                line=dict(color='blue', width=3),
                showlegend=False
            ))
        
        # Add key vertical lines with minimal annotations
        # Current price
        fig.add_vline(
            x=current_price,
            line_dash="solid",
            line_color="black",
            line_width=3,
            annotation_text=f"Current: ${current_price:.0f}",
            annotation_position="top"
        )
        
        # Breakevens
        fig.add_vline(
            x=breakeven_lower,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"BE: ${breakeven_lower:.0f}",
            annotation_position="bottom left"
        )
        
        fig.add_vline(
            x=breakeven_upper,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"BE: ${breakeven_upper:.0f}",
            annotation_position="bottom right"
        )
        
        # Strike prices - minimal annotations
        fig.add_vline(x=put_long, line_dash="dot", line_color="gray", opacity=0.7)
        fig.add_vline(x=put_short, line_dash="dot", line_color="red", opacity=0.7)
        fig.add_vline(x=call_short, line_dash="dot", line_color="red", opacity=0.7)
        fig.add_vline(x=call_long, line_dash="dot", line_color="gray", opacity=0.7)
        
        # Add horizontal lines for max profit/loss
        fig.add_hline(
            y=max_profit,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Max Profit: ${max_profit:.0f}",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=-max_loss,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max Loss: ${max_loss:.0f}",
            annotation_position="right"
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.5)
        
        # Clean annotations for strikes (positioned to avoid overlap)
        fig.add_annotation(
            x=put_long, y=max_profit * 0.8,
            text=f"Long Put<br>${put_long:.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray",
            font=dict(size=10, color="gray")
        )
        
        fig.add_annotation(
            x=put_short, y=max_profit * 0.5,
            text=f"Short Put<br>${put_short:.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            font=dict(size=11, color="red")
        )
        
        fig.add_annotation(
            x=call_short, y=max_profit * 0.5,
            text=f"Short Call<br>${call_short:.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            font=dict(size=11, color="red")
        )
        
        fig.add_annotation(
            x=call_long, y=max_profit * 0.8,
            text=f"Long Call<br>${call_long:.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray",
            font=dict(size=10, color="gray")
        )
        
        # Add profit zone highlight
        profit_zone_width = breakeven_upper - breakeven_lower
        profit_zone_pct = (profit_zone_width / current_price) * 100
        
        fig.add_annotation(
            x=(breakeven_lower + breakeven_upper) / 2,
            y=max_profit * 0.3,
            text=f"PROFIT ZONE<br>${profit_zone_width:.0f} wide<br>({profit_zone_pct:.1f}% of stock price)",
            showarrow=False,
            bgcolor="rgba(0, 255, 0, 0.1)",
            bordercolor="green",
            borderwidth=2,
            font=dict(size=12, color="green")
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Iron Condor P&L Diagram - {strategy.get('strategy_type', 'Standard')}",
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            height=600,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            font=dict(size=12),
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1
            ),
            yaxis=dict(
                gridcolor='lightgray',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=2
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating Iron Condor P&L chart: {e}")
        return None

def create_time_decay_simulation_chart(simulation_df, strategy):
    """Create comprehensive time decay simulation chart with improved clarity"""
    try:
        if simulation_df.empty:
            return None
        
        # Create subplots with better spacing and cleaner layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'P&L by Price Scenario Over Time',
                'Theta Decay Analysis', 
                'POPrem Evolution',
                'Exit Rule Comparison'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )
        
        # Get unique scenarios and create cleaner color scheme
        scenarios = simulation_df['price_scenario'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 1. P&L Evolution Chart (Top Left) - Simplified
        for i, scenario in enumerate(scenarios):
            scenario_data = simulation_df[simulation_df['price_scenario'] == scenario]
            
            fig.add_trace(
                go.Scatter(
                    x=scenario_data['dte'],
                    y=scenario_data['profit_loss'],
                    mode='lines',
                    name=f'{scenario}',
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add 21 DTE exit line - cleaner
        fig.add_vline(x=21, line_dash="dash", line_color="red", 
                      annotation_text="Rule A Exit", annotation_position="top",
                      row=1, col=1)
        
        # 2. Theta Analysis (Top Right) - Simplified
        at_money_data = simulation_df[simulation_df['price_scenario'] == '+0.0%']
        if not at_money_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=at_money_data['dte'],
                    y=at_money_data['theta_total'],
                    mode='lines+markers',
                    name='Net Theta',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. POPrem Evolution (Bottom Left) - Simplified  
        if not at_money_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=at_money_data['dte'],
                    y=at_money_data['pop_remaining'] * 100,
                    mode='lines+markers',
                    name='POPrem %',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Exit Strategy Analysis (Bottom Right) - Simplified
        rule_a_scenarios = []
        rule_b_scenarios = []
        
        for scenario in scenarios:
            scenario_data = simulation_df[simulation_df['price_scenario'] == scenario]
            
            # Rule A: Exit at 21 DTE or 50% profit
            rule_a_data = scenario_data[scenario_data['dte'] >= 21]
            if not rule_a_data.empty:
                rule_a_exit = rule_a_data.iloc[-1]  # Last day before 21 DTE
                rule_a_scenarios.append({
                    'scenario': scenario,
                    'profit': rule_a_exit['profit_loss'],
                    'dte_exit': rule_a_exit['dte']
                })
            
            # Rule B: Hold to expiry
            rule_b_data = scenario_data[scenario_data['dte'] == 0]
            if not rule_b_data.empty:
                rule_b_exit = rule_b_data.iloc[0]
                rule_b_scenarios.append({
                    'scenario': scenario,
                    'profit': rule_b_exit['profit_loss'],
                    'dte_exit': 0
                })
        
        # Plot exit comparison - cleaner bars
        if rule_a_scenarios and rule_b_scenarios:
            scenarios_list = [s['scenario'] for s in rule_a_scenarios]
            rule_a_profits = [s['profit'] for s in rule_a_scenarios]
            rule_b_profits = [s['profit'] for s in rule_b_scenarios]
            
            fig.add_trace(
                go.Bar(
                    x=scenarios_list,
                    y=rule_a_profits,
                    name='Rule A (21 DTE)',
                    marker_color='orange',
                    opacity=0.8,
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=scenarios_list,
                    y=rule_b_profits,
                    name='Rule B (Expiry)',
                    marker_color='lightblue',
                    opacity=0.8,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout with better spacing and cleaner formatting
        fig.update_layout(
            title={
                'text': f"Iron Condor Time Decay Simulation - {strategy.get('strategy_type', 'Standard')}",
                'x': 0.5,
                'font': {'size': 18}
            },
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            font=dict(size=11)
        )
        
        # Update axes with cleaner labels and better formatting
        fig.update_xaxes(title_text="Days to Expiry", row=1, col=1, title_font_size=12)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1, title_font_size=12)
        
        fig.update_xaxes(title_text="Days to Expiry", row=1, col=2, title_font_size=12)
        fig.update_yaxes(title_text="Theta ($)", row=1, col=2, title_font_size=12)
        
        fig.update_xaxes(title_text="Days to Expiry", row=2, col=1, title_font_size=12)
        fig.update_yaxes(title_text="POPrem (%)", row=2, col=1, title_font_size=12)
        
        fig.update_xaxes(title_text="Price Scenario", row=2, col=2, title_font_size=12)
        fig.update_yaxes(title_text="Exit P&L ($)", row=2, col=2, title_font_size=12)
        
        return fig
        
    except Exception as e:
        print(f"Error creating time decay simulation chart: {e}")
        return None

def create_exit_strategy_analysis_chart(exit_analysis):
    """Create comprehensive exit strategy analysis chart"""
    try:
        if not exit_analysis:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Profit Comparison by Price Scenario',
                'Exit Strategy Performance Summary',
                'Risk vs Reward Analysis',
                'Theta Capture Efficiency'
            ]
        )
        
        # Extract data
        rule_a_scenarios = exit_analysis['rule_a_21_dte']['scenarios']
        rule_b_scenarios = exit_analysis['rule_b_hold_expiry']['scenarios']
        
        scenarios = [s['price_scenario'] for s in rule_a_scenarios]
        rule_a_profits = [s['profit_loss'] for s in rule_a_scenarios]
        rule_b_profits = [s['profit_loss'] for s in rule_b_scenarios]
        
        # 1. Profit Comparison
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=rule_a_profits,
                name='21 DTE Exit',
                marker_color='#f59e0b',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=rule_b_profits,
                name='Hold to Expiry',
                marker_color='#06b6d4',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 2. Performance Summary
        summary = exit_analysis['summary']
        metrics = ['Avg Profit', 'Win Rate %', 'Theta Captured %']
        rule_a_values = [
            summary['rule_a_avg_profit'],
            summary['rule_a_win_rate'],
            summary['theta_captured_rule_a']
        ]
        rule_b_values = [
            summary['rule_b_avg_profit'],
            summary['rule_b_win_rate'],
            100  # Hold to expiry captures 100% potential theta
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=rule_a_values,
                name='21 DTE Rule',
                marker_color='#f59e0b',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=rule_b_values,
                name='Hold to Expiry',
                marker_color='#06b6d4',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Risk vs Reward Scatter
        rule_a_theta = [s['profit_captured_pct'] for s in rule_a_scenarios]
        rule_a_pop_rem = [s['pop_remaining'] * 100 for s in rule_a_scenarios]
        
        fig.add_trace(
            go.Scatter(
                x=rule_a_pop_rem,
                y=rule_a_theta,
                mode='markers',
                name='21 DTE Risk/Reward',
                marker=dict(
                    size=12,
                    color=rule_a_profits,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Profit ($)")
                ),
                text=scenarios,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Theta Capture Efficiency
        exit_days = [s['exit_day'] for s in rule_a_scenarios]
        theta_captured = [s['profit_captured_pct'] for s in rule_a_scenarios]
        
        fig.add_trace(
            go.Scatter(
                x=exit_days,
                y=theta_captured,
                mode='markers+lines',
                name='Theta Efficiency',
                line=dict(color='#10b981', width=3),
                marker=dict(size=8, color='#10b981'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Price Scenario", row=1, col=1)
        fig.update_yaxes(title_text="Profit ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        
        fig.update_xaxes(title_text="POP Remaining (%)", row=2, col=1)
        fig.update_yaxes(title_text="Theta Captured (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Exit Day", row=2, col=2)
        fig.update_yaxes(title_text="Theta Captured (%)", row=2, col=2)
        
        fig.update_layout(
            title="Iron Condor Exit Strategy Analysis - Rule A vs Rule B",
            height=700,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating exit strategy analysis chart: {e}")
        return None

def create_technical_metrics_dashboard(strategy, current_price):
    """Create comprehensive technical metrics dashboard with key performance indicators"""
    try:
        print(f"Creating technical dashboard...")
        print(f"Strategy keys available: {list(strategy.keys())}")
        
        # Extract key values with fallbacks and debugging
        call_short = strategy.get('call_short', 0)
        total_credit = strategy.get('total_credit', 0)
        wing_width = strategy.get('wing_width', 5)
        
        print(f"Extracted values: call_short={call_short}, total_credit={total_credit}, wing_width={wing_width}")
        
        # Calculate metrics with fallbacks
        try:
            credit_ratio = (total_credit / wing_width) if wing_width > 0 else 0
            max_profit = total_credit * 100
            roc = (max_profit / (wing_width * 100 - max_profit)) * 100 if wing_width > 0 else 0
            
            print(f"Calculated: credit_ratio={credit_ratio:.1%}, max_profit=${max_profit}, roc={roc:.1f}%")
        except Exception as calc_error:
            print(f"Calculation error: {calc_error}")
            credit_ratio = 0
            max_profit = 0
            roc = 0
        
        # Create subplots with mixed chart types
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Credit Efficiency Gauge',
                'Strategy Breakdown',
                'Probability Metrics',
                'Greeks Analysis',
                'Risk Metrics',
                'P&L Profile'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "xy"}, {"type": "indicator"}],
                [{"type": "xy"}, {"type": "indicator"}, {"type": "xy"}]
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )
        
        # 1. Credit Efficiency Gauge (Top Left)
        try:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=credit_ratio * 100,  # Convert to percentage for display
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Credit Efficiency<br>{credit_ratio*100:.1f}¢ per $1"},
                    delta={'reference': 33, 'suffix': '¢'},
                    gauge={
                        'axis': {'range': [None, 50], 'ticksuffix': '¢'},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 15], 'color': "#ffcccc"},
                            {'range': [15, 33], 'color': "#ffffcc"},
                            {'range': [33, 40], 'color': "#ccffcc"},
                            {'range': [40, 50], 'color': "#ccffff"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 33
                        }
                    }
                ),
                row=1, col=1
            )
        except Exception as gauge_error:
            print(f"Gauge creation error: {gauge_error}")
            # Fallback: Add a simple text annotation
            fig.add_annotation(
                x=0.16, y=0.72,
                text=f"Credit Efficiency<br>{credit_ratio*100:.1f}¢ per $1",
                showarrow=False,
                font=dict(size=14),
                xref="paper", yref="paper"
            )
        
        # 2. Strategy Breakdown (Top Center)
        try:
            components = ['Max Profit', 'Max Loss', 'Credit', 'Wing Width']
            values = [
                strategy.get('max_profit', 0),
                abs(strategy.get('max_loss', 0)),
                strategy.get('total_credit', 0) * 100,
                strategy.get('wing_width', 0) * 10  # Scale for visibility
            ]
            
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=values,
                    marker_color=['#10b981', '#ef4444', '#3b82f6', '#8b5cf6'],
                    name='Strategy Components',
                    showlegend=False
                ),
                row=1, col=2
            )
        except Exception as bar_error:
            print(f"Bar chart error: {bar_error}")
        
        # 3. POP Gauge (Top Right)
        try:
            pop_value = strategy.get('pop_black_scholes', 0) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=pop_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probability of Profit"},
                    gauge={
                        'axis': {'range': [None, 100], 'ticksuffix': '%'},
                        'bar': {'color': "#10b981"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcccc"},
                            {'range': [50, 70], 'color': "#ffffcc"},
                            {'range': [70, 85], 'color': "#ccffcc"},
                            {'range': [85, 100], 'color': "#ccffff"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ),
                row=1, col=3
            )
        except Exception as pop_error:
            print(f"POP gauge error: {pop_error}")
        
        # 4. Greeks Analysis (Bottom Left)
        try:
            greeks = ['Theta', 'Gamma', 'Vega', 'Delta']
            greek_values = [
                abs(strategy.get('net_theta', 0)),
                abs(strategy.get('net_gamma', 0)) * 1000,  # Scale gamma
                abs(strategy.get('net_vega', 0)),
                abs(strategy.get('net_delta', 0))
            ]
            
            fig.add_trace(
                go.Bar(
                    x=greeks,
                    y=greek_values,
                    marker_color='#f59e0b',
                    name='Greeks',
                    showlegend=False
                ),
                row=2, col=1
            )
        except Exception as greeks_error:
            print(f"Greeks chart error: {greeks_error}")
        
        # 5. ROC Gauge (Bottom Center)
        try:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=roc,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Return on Capital"},
                    gauge={
                        'axis': {'range': [None, 50], 'ticksuffix': '%'},
                        'bar': {'color': "#8b5cf6"},
                        'steps': [
                            {'range': [0, 10], 'color': "#ffcccc"},
                            {'range': [10, 15], 'color': "#ffffcc"},
                            {'range': [15, 25], 'color': "#ccffcc"},
                            {'range': [25, 50], 'color': "#ccffff"}
                        ],
                        'threshold': {
                            'line': {'color': "blue", 'width': 4},
                            'thickness': 0.75,
                            'value': 15
                        }
                    }
                ),
                row=2, col=2
            )
        except Exception as roc_error:
            print(f"ROC gauge error: {roc_error}")
        
        # 6. P&L Profile (Bottom Right) - WITHOUT add_vline to avoid the error
        try:
            # Calculate P&L profile
            max_profit = strategy.get('max_profit', 0)
            max_loss = strategy.get('max_loss', 0)
            breakeven_lower = strategy.get('lower_breakeven', current_price * 0.9)
            breakeven_upper = strategy.get('upper_breakeven', current_price * 1.1)
            
            # Create price range around current price
            price_range = np.linspace(current_price * 0.85, current_price * 1.15, 100)
            pnl_values = []
            
            for price in price_range:
                if breakeven_lower <= price <= breakeven_upper:
                    pnl = max_profit
                elif price < breakeven_lower:
                    pnl = max_profit - (breakeven_lower - price) * 100
                else:  # price > breakeven_upper
                    pnl = max_profit - (price - breakeven_upper) * 100
                
                # Cap at max loss
                pnl = max(pnl, -abs(max_loss))
            
                pnl_values.append(pnl)
            
            fig.add_trace(
                go.Scatter(
                    x=price_range,
                    y=pnl_values,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0,255,0,0.3)',
                    line=dict(color='blue', width=3),
                    name='P&L Profile',
                    showlegend=False
                ),
                row=2, col=3
            )
            
            # Add current price as a scatter point instead of vline
            current_pnl = max_profit if breakeven_lower <= current_price <= breakeven_upper else 0
            fig.add_trace(
                go.Scatter(
                    x=[current_price],
                    y=[current_pnl],
                    mode='markers+text',
                    text=[f"Current: ${current_price:.2f}"],
                    textposition='top center',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name='Current Price',
                    showlegend=False
                ),
                row=2, col=3
            )
            
        except Exception as pnl_error:
            print(f"P&L chart error: {pnl_error}")
            # Fallback: show a simple message
            fig.add_annotation(
                x=0.83, y=0.25,
                text="P&L Profile<br>Insufficient Data",
                showarrow=False,
                font=dict(size=14),
                xref="paper", yref="paper"
            )
        
        # Update layout with explanations
        fig.update_layout(
            title={
                'text': f"Iron Condor Technical Dashboard - {strategy.get('strategy_type', 'Standard')}",
                'x': 0.5,
                'font': {'size': 18}
            },
            height=800,
            showlegend=False,
            font=dict(size=11),
            annotations=[
                # Add explanations as annotations
                dict(
                    text="Credit Efficiency: Higher is better<br>33¢+ per $1 = Bread & Butter Rule",
                    x=0.16, y=0.45,
                    xref="paper", yref="paper",
                    font=dict(size=10, color="gray"),
                    showarrow=False
                ),
                dict(
                    text="POP: Probability of Profit<br>Higher % = Better odds",
                    x=0.83, y=0.72,
                    xref="paper", yref="paper",
                    font=dict(size=10, color="gray"),
                    showarrow=False
                ),
                dict(
                    text="ROC: Return on Capital<br>15%+ is good target",
                    x=0.83, y=0.28,
                    xref="paper", yref="paper",
                    font=dict(size=10, color="gray"),
                    showarrow=False
                )
            ]
        )
        
        # Update axes
        fig.update_xaxes(title_text="Strategy Component", row=1, col=2, title_font_size=12)
        fig.update_yaxes(title_text="Value ($)", row=1, col=2, title_font_size=12)
        
        fig.update_xaxes(title_text="Greek", row=2, col=1, title_font_size=12)
        fig.update_yaxes(title_text="Value", row=2, col=1, title_font_size=12)
        
        fig.update_xaxes(title_text="Stock Price ($)", row=2, col=3, title_font_size=12)
        fig.update_yaxes(title_text="P&L ($)", row=2, col=3, title_font_size=12)
        
        print(f"Dashboard created successfully!")
        return fig
        
    except Exception as e:
        print(f"Error creating technical metrics dashboard: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a simple fallback chart
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Technical Dashboard Error:<br>{str(e)}<br><br>Check console for details",
            showarrow=False,
            font=dict(size=16),
            xref="paper", yref="paper"
        )
        fig.update_layout(
            title="Technical Metrics Dashboard - Error",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

def create_strategy_comparison_chart(analysis_results):
    """Create strategy comparison chart with enhanced metrics"""
    try:
        if not analysis_results or not analysis_results.get('strategies'):
            return None
        
        strategies = analysis_results['strategies'][:5]  # Top 5 strategies
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'POP vs ROC Analysis',
                'Risk vs Reward Comparison',
                'Strategy Type Distribution',
                'Greeks Risk Profile'
            ]
        )
        
        # Extract data
        strategy_names = [f"#{i+1} {s['strategy_type']}" for i, s in enumerate(strategies)]
        pop_values = [s.get('pop_black_scholes', 0) * 100 for s in strategies]
        roc_values = [s.get('roc_percent', 0) for s in strategies]
        max_profits = [s.get('max_profit', 0) for s in strategies]
        max_losses = [s.get('max_loss', 0) for s in strategies]
        theta_values = [abs(s.get('net_theta', 0)) for s in strategies]
        gamma_values = [abs(s.get('net_gamma', 0)) for s in strategies]
        
        # 1. POP vs ROC Analysis
        fig.add_trace(
            go.Scatter(
                x=pop_values,
                y=roc_values,
                mode='markers+text',
                text=[f"#{i+1}" for i in range(len(strategies))],
                textposition='top center',
                marker=dict(
                    size=[p/5 for p in max_profits],  # Size by max profit
                    color=pop_values,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="POP %")
                ),
                name='Strategies',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Risk vs Reward
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=max_profits,
                name='Max Profit',
                marker_color='#10b981',
                opacity=0.8
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=[-loss for loss in max_losses],  # Negative for visual
                name='Max Loss',
                marker_color='#ef4444',
                opacity=0.8
            ),
            row=1, col=2
        )
        
        # 3. Strategy Type Distribution
        strategy_types = [s['strategy_type'] for s in strategies]
        type_counts = {}
        for st in strategy_types:
            type_counts[st] = type_counts.get(st, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                name="Strategy Types",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Greeks Risk Profile
        fig.add_trace(
            go.Scatter(
                x=theta_values,
                y=gamma_values,
                mode='markers+text',
                text=[f"#{i+1}" for i in range(len(strategies))],
                textposition='top center',
                marker=dict(
                    size=12,
                    color=roc_values,
                    colorscale='Viridis',
                    showscale=False
                ),
                name='Greeks Profile',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="POP (%)", row=1, col=1)
        fig.update_yaxes(title_text="ROC (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Profit/Loss ($)", row=1, col=2)
        
        fig.update_xaxes(title_text="Daily Theta ($)", row=2, col=2)
        fig.update_yaxes(title_text="Gamma Risk", row=2, col=2)
        
        fig.update_layout(
            title="Iron Condor Strategy Comparison Analysis",
            height=700,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating strategy comparison chart: {e}")
        return None

def create_pop_distribution_chart(analysis_results):
    """Create POP distribution analysis chart"""
    try:
        if not analysis_results or not analysis_results.get('strategies'):
            return None
        
        strategies = analysis_results['strategies']
        
        # Extract POP data
        pop_delta = [s.get('pop_delta_method', 0) * 100 for s in strategies]
        pop_credit = [s.get('pop_credit_method', 0) * 100 for s in strategies]
        pop_bs = [s.get('pop_black_scholes', 0) * 100 for s in strategies]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['POP Method Comparison', 'POP Distribution by Strategy Type']
        )
        
        # 1. POP Method Comparison
        strategy_names = [f"#{i+1}" for i in range(len(strategies))]
        
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=pop_delta,
                name='Delta Method',
                marker_color='#ef4444',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=pop_credit,
                name='Credit/Width Method',
                marker_color='#f59e0b',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=pop_bs,
                name='Black-Scholes Method',
                marker_color='#10b981',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 2. POP by Strategy Type
        strategy_types = list(set([s['strategy_type'] for s in strategies]))
        avg_pop_by_type = {}
        
        for st in strategy_types:
            type_strategies = [s for s in strategies if s['strategy_type'] == st]
            avg_pop = np.mean([s.get('pop_black_scholes', 0) * 100 for s in type_strategies])
            avg_pop_by_type[st] = avg_pop
        
        fig.add_trace(
            go.Bar(
                x=list(avg_pop_by_type.keys()),
                y=list(avg_pop_by_type.values()),
                marker_color='#8b5cf6',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Strategy Rank", row=1, col=1)
        fig.update_yaxes(title_text="POP (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Strategy Type", row=1, col=2)
        fig.update_yaxes(title_text="Average POP (%)", row=1, col=2)
        
        fig.update_layout(
            title="Probability of Profit (POP) Analysis",
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating POP distribution chart: {e}")
        return None

def create_trade_management_dashboard(strategy, current_price):
    """Create trade management dashboard with decision support"""
    try:
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Current Trade Status',
                'Exit Decision Matrix',
                'Risk Assessment',
                'Profit Targets',
                'Time Decay Forecast',
                'Action Recommendations'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # Calculate current metrics
        dte = strategy.get('dte', 30)
        max_profit = strategy.get('max_profit', 0)
        current_pop = strategy.get('pop_black_scholes', 0) * 100
        
        # 1. Current Trade Status Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=dte,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Days to Expiry"},
                gauge={
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 21], 'color': "red"},
                        {'range': [21, 45], 'color': "yellow"},
                        {'range': [45, 60], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 21
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Exit Decision Matrix
        exit_options = ['Hold', 'Close 25%', 'Close 50%', 'Close 100%']
        exit_scores = [
            75 if dte > 21 else 25,  # Hold score
            60,  # Close 25%
            80 if dte <= 21 else 40,  # Close 50%
            90 if dte <= 14 else 30   # Close 100%
        ]
        
        colors = ['#10b981' if score > 70 else '#f59e0b' if score > 50 else '#ef4444' for score in exit_scores]
        
        fig.add_trace(
            go.Bar(
                x=exit_options,
                y=exit_scores,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Risk Assessment
        risk_factors = ['Gamma Risk', 'Time Risk', 'Volatility Risk']
        risk_levels = [
            strategy.get('gamma_risk', 0) * 100,
            (45 - dte) / 45 * 100 if dte < 45 else 0,
            50  # Default vol risk
        ]
        
        fig.add_trace(
            go.Scatter(
                x=risk_factors,
                y=risk_levels,
                mode='markers',
                marker=dict(
                    size=[r/2 + 10 for r in risk_levels],
                    color=risk_levels,
                    colorscale='Reds',
                    showscale=False
                ),
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. Profit Targets
        profit_targets = ['25%', '50%', '75%', '100%']
        target_values = [max_profit * p for p in [0.25, 0.5, 0.75, 1.0]]
        
        fig.add_trace(
            go.Bar(
                x=profit_targets,
                y=target_values,
                marker_color='#06b6d4',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Time Decay Forecast
        days_ahead = list(range(0, min(dte, 30), 2))
        theta_forecast = [strategy.get('net_theta', 0) * day for day in days_ahead]
        
        fig.add_trace(
            go.Scatter(
                x=days_ahead,
                y=theta_forecast,
                mode='lines+markers',
                line=dict(color='#ef4444', width=3),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. Action Recommendations (as text table)
        recommendations = [
            ['DTE > 21', 'Monitor daily, look for 50% profit'],
            ['DTE = 21', 'Close position regardless of P&L'],
            ['Profit > 50%', 'Consider taking profits'],
            ['High Gamma', 'Reduce position size'],
            ['Low POP', 'Consider early exit']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Condition', 'Recommendation']),
                cells=dict(values=[[r[0] for r in recommendations], [r[1] for r in recommendations]]),
                fill_color='lightgray'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f"Trade Management Dashboard - Current POP: {current_pop:.1f}%",
            height=700,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating trade management dashboard: {e}")
        return None

def create_volatility_impact_chart(analysis_results):
    """Create volatility impact analysis chart"""
    try:
        if not analysis_results or not analysis_results.get('strategies'):
            return None
        
        strategies = analysis_results['strategies']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Volatility vs POP', 'Volatility vs Profit Potential']
        )
        
        # Extract volatility data
        volatilities = [s.get('volatility', 0.25) * 100 for s in strategies]
        pop_values = [s.get('pop_black_scholes', 0) * 100 for s in strategies]
        max_profits = [s.get('max_profit', 0) for s in strategies]
        strategy_types = [s.get('strategy_type', 'Unknown') for s in strategies]
        
        # 1. Volatility vs POP
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=pop_values,
                mode='markers',
                text=strategy_types,
                marker=dict(
                    size=12,
                    color=max_profits,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Max Profit ($)")
                ),
                name='Strategies'
            ),
            row=1, col=1
        )
        
        # 2. Volatility vs Profit Potential
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=max_profits,
                mode='markers',
                text=strategy_types,
                marker=dict(
                    size=12,
                    color=pop_values,
                    colorscale='RdYlGn',
                    showscale=False
                ),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="POP (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Max Profit ($)", row=1, col=2)
        
        fig.update_layout(
            title="Volatility Impact Analysis",
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating volatility impact chart: {e}")
        return None

def create_earnings_impact_analysis(analysis_results, earnings_date=None):
    """Create earnings impact analysis for Iron Condor strategies"""
    try:
        if not analysis_results or not analysis_results.get('strategies'):
            return None
        
        strategies = analysis_results['strategies']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Earnings Play Suitability', 'DTE vs Strategy Type']
        )
        
        # Filter for short DTE strategies (earnings plays)
        short_dte_strategies = [s for s in strategies if s.get('dte', 30) <= 14]
        
        if short_dte_strategies:
            # 1. Earnings Play Analysis
            wing_widths = [s.get('wing_width', 5) for s in short_dte_strategies]
            pop_values = [s.get('pop_black_scholes', 0) * 100 for s in short_dte_strategies]
            
            fig.add_trace(
                go.Scatter(
                    x=wing_widths,
                    y=pop_values,
                    mode='markers',
                    text=[s.get('strategy_type', '') for s in short_dte_strategies],
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star'
                    ),
                    name='Earnings Plays'
                ),
                row=1, col=1
            )
        
        # 2. DTE vs Strategy Type
        dte_values = [s.get('dte', 30) for s in strategies]
        strategy_types = [s.get('strategy_type', 'Unknown') for s in strategies]
        
        # Create scatter plot with different symbols for strategy types
        type_symbols = {'Bread & Butter': 'circle', 'Big Boy': 'square', 'Chicken IC': 'star', 'Conservative': 'diamond'}
        
        for st in set(strategy_types):
            st_strategies = [s for s in strategies if s.get('strategy_type') == st]
            st_dte = [s.get('dte', 30) for s in st_strategies]
            st_pop = [s.get('pop_black_scholes', 0) * 100 for s in st_strategies]
            
            fig.add_trace(
                go.Scatter(
                    x=st_dte,
                    y=st_pop,
                    mode='markers',
                    name=st,
                    marker=dict(
                        size=10,
                        symbol=type_symbols.get(st, 'circle')
                    )
                ),
                row=1, col=2
            )
        
        # Add earnings date line if provided
        if earnings_date:
            fig.add_vline(
                x=earnings_date,
                line_dash="dash",
                line_color="red",
                annotation_text="Earnings",
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Wing Width ($)", row=1, col=1)
        fig.update_yaxes(title_text="POP (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="DTE", row=1, col=2)
        fig.update_yaxes(title_text="POP (%)", row=1, col=2)
        
        fig.update_layout(
            title="Earnings Impact Analysis for Iron Condor Strategies",
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating earnings impact analysis: {e}")
        return None 