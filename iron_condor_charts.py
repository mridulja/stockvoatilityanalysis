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
    """Create interactive Iron Condor P&L diagram"""
    try:
        # Price range for analysis
        price_min = current_price * (1 - price_range_pct)
        price_max = current_price * (1 + price_range_pct)
        prices = np.linspace(price_min, price_max, 100)
        
        # Calculate P&L for each price
        pnl_values = []
        for price in prices:
            if price <= strategy['put_long']:
                # Max loss on put side
                pnl = strategy['total_credit'] - strategy['put_width']
            elif price <= strategy['put_short']:
                # Put spread in the money
                pnl = strategy['total_credit'] - (strategy['put_short'] - price)
            elif price <= strategy['call_short']:
                # Between short strikes - max profit zone
                pnl = strategy['total_credit']
            elif price <= strategy['call_long']:
                # Call spread in the money
                pnl = strategy['total_credit'] - (price - strategy['call_short'])
            else:
                # Max loss on call side
                pnl = strategy['total_credit'] - strategy['call_width']
            
            pnl_values.append(pnl * 100)  # Convert to dollars
        
        fig = go.Figure()
        
        # Main P&L line
        fig.add_trace(go.Scatter(
            x=prices,
            y=pnl_values,
            mode='lines',
            name='Iron Condor P&L',
            line=dict(color='#6366f1', width=4),
            hovertemplate='Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
        ))
        
        # Add breakeven lines
        fig.add_vline(
            x=strategy['lower_breakeven'], 
            line_dash="dash", 
            line_color="#ef4444",
            annotation_text=f"Lower BE: ${strategy['lower_breakeven']:.2f}"
        )
        fig.add_vline(
            x=strategy['upper_breakeven'], 
            line_dash="dash", 
            line_color="#ef4444",
            annotation_text=f"Upper BE: ${strategy['upper_breakeven']:.2f}"
        )
        
        # Current price
        fig.add_vline(
            x=current_price, 
            line_dash="dot", 
            line_color="#10b981",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        # Strike prices
        for strike, label in [
            (strategy['put_long'], 'Put Long'),
            (strategy['put_short'], 'Put Short'),
            (strategy['call_short'], 'Call Short'),
            (strategy['call_long'], 'Call Long')
        ]:
            fig.add_vline(
                x=strike,
                line_dash="dashdot",
                line_color="#8b5cf6",
                opacity=0.7,
                annotation_text=f"{label}: ${strike:.0f}"
            )
        
        # Profit zone shading
        fig.add_vrect(
            x0=strategy['lower_breakeven'],
            x1=strategy['upper_breakeven'],
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text="Profit Zone"
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color="#64748b", opacity=0.5)
        
        fig.update_layout(
            title=f"Iron Condor P&L Diagram - {strategy['strategy_type']}",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating Iron Condor P&L chart: {e}")
        return None

def create_time_decay_simulation_chart(simulation_df, strategy):
    """Create comprehensive time decay simulation chart"""
    try:
        if simulation_df.empty:
            return None
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Profit/Loss Over Time by Price Scenario',
                'Theta Decay Analysis',
                'Probability of Profit Remaining (POPrem)',
                'Gamma Risk Over Time',
                'Exit Strategy Analysis',
                'Greeks Summary'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08
        )
        
        # Color mapping for price scenarios
        scenario_colors = {
            '-15.0%': '#ef4444',
            '-5.0%': '#f97316', 
            '+0.0%': '#10b981',
            '+5.0%': '#06b6d4',
            '+15.0%': '#8b5cf6'
        }
        
        # 1. Profit/Loss Over Time
        for scenario in simulation_df['price_scenario'].unique():
            scenario_data = simulation_df[simulation_df['price_scenario'] == scenario]
            
            fig.add_trace(
                go.Scatter(
                    x=scenario_data['day'],
                    y=scenario_data['profit_loss'],
                    mode='lines+markers',
                    name=f'P&L {scenario}',
                    line=dict(color=scenario_colors.get(scenario, '#64748b'), width=3),
                    marker=dict(size=4),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add 21 DTE line
        fig.add_vline(
            x=strategy['dte'] - 21,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="21 DTE Exit",
            row=1, col=1
        )
        
        # 2. Theta Decay Analysis
        at_the_money_data = simulation_df[simulation_df['price_scenario'] == '+0.0%']
        if not at_the_money_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=at_the_money_data['day'],
                    y=at_the_money_data['theta_total'],
                    mode='lines',
                    name='Daily Theta',
                    line=dict(color='#ef4444', width=3),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. POPrem Analysis
        if not at_the_money_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=at_the_money_data['day'],
                    y=at_the_money_data['pop_remaining'] * 100,
                    mode='lines+markers',
                    name='POP Remaining %',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Gamma Risk
        if not at_the_money_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=at_the_money_data['day'],
                    y=at_the_money_data['gamma_total'],
                    mode='lines',
                    name='Total Gamma',
                    line=dict(color='#8b5cf6', width=3),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 5. Exit Strategy Comparison
        rule_a_data = simulation_df[simulation_df['should_close_21_dte'] == True].groupby('price_scenario').first()
        rule_b_data = simulation_df[simulation_df['dte'] == 0]
        
        exit_scenarios = rule_a_data.index.tolist()
        rule_a_profits = rule_a_data['profit_loss'].tolist()
        rule_b_profits = rule_b_data.groupby('price_scenario')['profit_loss'].first().reindex(exit_scenarios).tolist()
        
        fig.add_trace(
            go.Bar(
                x=exit_scenarios,
                y=rule_a_profits,
                name='21 DTE Exit',
                marker_color='#f59e0b',
                opacity=0.8,
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=exit_scenarios,
                y=rule_b_profits,
                name='Hold to Expiry',
                marker_color='#06b6d4',
                opacity=0.8,
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Greeks Summary (current values)
        current_data = simulation_df[simulation_df['day'] == 0]
        if not current_data.empty:
            greeks_summary = current_data.groupby('price_scenario').first()
            
            fig.add_trace(
                go.Scatter(
                    x=greeks_summary.index,
                    y=greeks_summary['theta_total'],
                    mode='markers',
                    name='Current Theta',
                    marker=dict(size=12, color='#ef4444'),
                    showlegend=False
                ),
                row=3, col=2
            )
        
        # Update subplot titles and axes
        fig.update_xaxes(title_text="Days from Entry", row=1, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Days from Entry", row=1, col=2)
        fig.update_yaxes(title_text="Theta ($/day)", row=1, col=2)
        
        fig.update_xaxes(title_text="Days from Entry", row=2, col=1)
        fig.update_yaxes(title_text="POP Remaining (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Days from Entry", row=2, col=2)
        fig.update_yaxes(title_text="Gamma", row=2, col=2)
        
        fig.update_xaxes(title_text="Price Scenario", row=3, col=1)
        fig.update_yaxes(title_text="Exit P&L ($)", row=3, col=1)
        
        fig.update_xaxes(title_text="Price Scenario", row=3, col=2)
        fig.update_yaxes(title_text="Current Theta", row=3, col=2)
        
        fig.update_layout(
            title=f"Iron Condor Time Decay Simulation - {strategy['strategy_type']}",
            height=900,
            showlegend=True
        )
        
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
    """Create comprehensive technical metrics dashboard"""
    try:
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'ROC & Risk Metrics',
                'Greeks Analysis',
                'Profit Zone Analysis',
                'Time Decay Progression',
                'Credit Efficiency',
                'Strategy Classification'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "indicator"}, {"type": "pie"}]
            ]
        )
        
        # 1. ROC & Risk Metrics
        metrics_names = ['ROC %', 'Risk/Reward', 'Margin Req ($)']
        metrics_values = [
            strategy.get('roc_percent', 0),
            strategy.get('risk_reward_ratio', 0),
            strategy.get('margin_required', 0) / 100  # Scale down for display
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                marker_color=['#10b981', '#06b6d4', '#f59e0b'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Greeks Analysis
        greeks_names = ['Theta/Day', 'Gamma', 'Vega']
        greeks_values = [
            abs(strategy.get('net_theta', 0)),
            abs(strategy.get('net_gamma', 0)),
            abs(strategy.get('net_vega', 0))
        ]
        
        fig.add_trace(
            go.Bar(
                x=greeks_names,
                y=greeks_values,
                marker_color=['#ef4444', '#8b5cf6', '#06b6d4'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Profit Zone Analysis
        distance_from_current = [
            strategy.get('breakeven_lower', 0) - current_price,
            0,  # Current price
            strategy.get('breakeven_upper', 0) - current_price
        ]
        price_labels = ['Lower BE', 'Current', 'Upper BE']
        
        fig.add_trace(
            go.Scatter(
                x=distance_from_current,
                y=[1, 1, 1],
                mode='markers+text',
                text=price_labels,
                textposition='top center',
                marker=dict(size=[15, 20, 15], color=['red', 'green', 'red']),
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. Time Decay Progression (estimate)
        days = list(range(0, strategy.get('dte', 30) + 1, 5))
        theta_decay = [strategy.get('theta_decay_daily', 0) * day for day in days]
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=theta_decay,
                mode='lines+markers',
                name='Cumulative Theta',
                line=dict(color='#ef4444', width=3),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Credit Efficiency Gauge
        credit_efficiency = strategy.get('credit_to_width_ratio', 0) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=credit_efficiency,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Credit/Width %"},
                delta={'reference': 33},  # 1/3 rule reference
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 33], 'color': "yellow"},
                        {'range': [33, 50], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 33
                    }
                }
            ),
            row=2, col=2
        )
        
        # 6. Strategy Classification
        strategy_breakdown = {
            'Quality Score': 85 if strategy.get('pop_black_scholes', 0) > 0.7 else 65,
            'Risk Level': 25 if strategy.get('gamma_risk', 0) < 0.1 else 75,
            'Time Sensitivity': 60 if strategy.get('dte', 30) > 21 else 90
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(strategy_breakdown.keys()),
                values=list(strategy_breakdown.values()),
                hole=0.3,
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_xaxes(title_text="Metric", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        
        fig.update_xaxes(title_text="Greek", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        
        fig.update_xaxes(title_text="Distance from Current ($)", row=1, col=3)
        fig.update_yaxes(title_text="", row=1, col=3, showticklabels=False)
        
        fig.update_xaxes(title_text="Days", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Theta ($)", row=2, col=1)
        
        fig.update_layout(
            title=f"Technical Metrics Dashboard - {strategy.get('strategy_type', 'Iron Condor')}",
            height=700,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating technical metrics dashboard: {e}")
        return None

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