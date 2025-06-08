"""
Chart creation functions for Stock Volatility Analyzer

This module contains all functions for creating interactive charts and visualizations
using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_price_chart(data, ticker, timeframe, show_vix=False):
    """Create interactive price chart with ranges and VIX overlay"""
    if data is None or data.empty:
        return None
    
    # Determine number of subplots
    rows = 3 if (show_vix and 'VIX_Close' in data.columns) else 2
    subplot_titles = [f'{ticker} Price Action ({timeframe.title()})', 'Range Analysis']
    if show_vix and 'VIX_Close' in data.columns:
        subplot_titles.append('VIX Analysis')
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=[0.5, 0.25, 0.25] if rows == 3 else [0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'] if 'Open' in data.columns else data['Close'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f'{ticker} Price'
        ),
        row=1, col=1
    )
    
    # Range bar chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['range'],
            name='Range',
            marker_color='rgba(99, 102, 241, 0.7)',
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Add ATR line
    if len(data) > 1 and 'true_range' in data.columns:
        atr_window = min(14, len(data))
        atr_line = data['true_range'].rolling(window=atr_window).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=atr_line,
                mode='lines',
                name=f'{atr_window}-period ATR',
                line=dict(color='#ef4444', width=3)
            ),
            row=2, col=1
        )
    
    # Add VIX chart if available
    if show_vix and 'VIX_Close' in data.columns and rows == 3:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['VIX_Close'],
                mode='lines+markers',
                name='VIX',
                line=dict(color='#8b5cf6', width=3),
                marker=dict(size=5, color='#8b5cf6')
            ),
            row=3, col=1
        )
        
        # Add VIX level zones
        fig.add_hline(y=15, line_dash="dash", line_color="#10b981", opacity=0.6, row=3)
        fig.add_hline(y=19, line_dash="dash", line_color="#06b6d4", opacity=0.6, row=3)
        fig.add_hline(y=25, line_dash="dash", line_color="#f59e0b", opacity=0.6, row=3)
        fig.add_hline(y=35, line_dash="dash", line_color="#ef4444", opacity=0.6, row=3)
    
    fig.update_layout(
        title=f'{ticker} - {timeframe.title()} Analysis with Enhanced ATR',
        xaxis_rangeslider_visible=False,
        height=800 if rows == 3 else 600,
        showlegend=True
    )
    
    return fig


def create_comparison_chart(results_dict, metric='atr'):
    """Create comparison chart between multiple tickers"""
    tickers = list(results_dict.keys())
    timeframes = ['hourly', 'daily', 'weekly']
    
    fig = go.Figure()
    
    for ticker in tickers:
        values = []
        for tf in timeframes:
            if tf in results_dict[ticker] and results_dict[ticker][tf] and metric in results_dict[ticker][tf]:
                val = results_dict[ticker][tf][metric]
                values.append(val if not pd.isna(val) and val > 0 else 0)
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=ticker,
            x=timeframes,
            y=values,
            text=[f'${v:.2f}' if v > 0 else 'N/A' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f'{metric.upper()} Comparison Across Timeframes (Enhanced Calculation)',
        xaxis_title='Timeframe',
        yaxis_title=f'{metric.upper()} Value ($)',
        barmode='group',
        height=400
    )
    
    return fig


def create_enhanced_price_chart(data, ticker, timeframe, chart_type='Candlestick', show_volume=True, indicators=None, vix_data=None):
    """Create enhanced interactive price chart with technical indicators"""
    if data is None or data.empty:
        return None
    
    # Determine subplot configuration
    rows = 3 if show_volume else 2
    subplot_titles = [f'{ticker} - {chart_type} Chart ({timeframe.title()})', 'Technical Indicators']
    if show_volume:
        subplot_titles.append('Volume')
    
    # Configure row heights
    if show_volume:
        row_heights = [0.6, 0.25, 0.15]
    else:
        row_heights = [0.7, 0.3]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}]] * rows
    )
    
    # Main price chart based on type
    if chart_type == 'Candlestick':
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'] if 'Open' in data.columns else data['Close'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f'{ticker} Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
    elif chart_type == 'Line':
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{ticker} Close',
                line=dict(color='#2196f3', width=2)
            ),
            row=1, col=1
        )
    elif chart_type == 'Area':
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                fill='tonexty',
                mode='lines',
                name=f'{ticker} Close',
                line=dict(color='#2196f3', width=1),
                fillcolor='rgba(33, 150, 243, 0.1)'
            ),
            row=1, col=1
        )
    elif chart_type == 'OHLC':
        fig.add_trace(
            go.Ohlc(
                x=data.index,
                open=data['Open'] if 'Open' in data.columns else data['Close'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f'{ticker} OHLC'
            ),
            row=1, col=1
        )
    
    # Add technical indicators if specified
    if indicators:
        # Moving Averages
        if indicators.get('sma_20') and 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#ff9800', width=2, dash='solid')
                ),
                row=1, col=1
            )
        
        if indicators.get('sma_50') and 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#e91e63', width=2, dash='solid')
                ),
                row=1, col=1
            )
        
        if indicators.get('ema_12') and 'EMA_12' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_12'],
                    mode='lines',
                    name='EMA 12',
                    line=dict(color='#4caf50', width=2, dash='dot')
                ),
                row=1, col=1
            )
        
        if indicators.get('ema_26') and 'EMA_26' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_26'],
                    mode='lines',
                    name='EMA 26',
                    line=dict(color='#9c27b0', width=2, dash='dot')
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if indicators.get('bollinger_bands') and all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='Bollinger Bands',
                    line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='#607d8b', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # ATR Bands
        if indicators.get('atr_bands') and all(col in data.columns for col in ['ATR_Upper', 'ATR_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ATR_Upper'],
                    mode='lines',
                    name='ATR Upper',
                    line=dict(color='rgba(255, 152, 0, 0.6)', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ATR_Lower'],
                    mode='lines',
                    name='ATR Bands',
                    line=dict(color='rgba(255, 152, 0, 0.6)', width=1, dash='dash')
                ),
                row=1, col=1
            )
    
    # Technical indicators subplot (ATR)
    if 'true_range' in data.columns:
        atr_window = min(14, len(data))
        atr_line = data['true_range'].rolling(window=atr_window).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=atr_line,
                mode='lines',
                name=f'ATR ({atr_window})',
                line=dict(color='#f44336', width=2)
            ),
            row=2, col=1
        )
        
        # Add ATR average line
        atr_avg = atr_line.mean()
        fig.add_hline(
            y=atr_avg,
            line_dash="dash",
            line_color="#f44336",
            opacity=0.5,
            row=2,
            annotation_text=f"ATR Avg: ${atr_avg:.2f}"
        )
    
    # Volume subplot
    if show_volume and 'Volume' in data.columns:
        colors = ['#26a69a' if close >= open else '#ef5350' 
                 for close, open in zip(data['Close'], data['Open'] if 'Open' in data.columns else data['Close'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=3 if show_volume else 2, col=1
        )
        
        # Add volume moving average
        if len(data) >= 20:
            vol_ma = data['Volume'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=vol_ma,
                    mode='lines',
                    name='Volume MA(20)',
                    line=dict(color='#ff9800', width=2)
                ),
                row=3 if show_volume else 2, col=1
            )
    
    # VIX overlay if requested
    if indicators and indicators.get('vix_overlay') and vix_data is not None:
        # Add VIX to a secondary y-axis on the main chart
        fig.add_trace(
            go.Scatter(
                x=vix_data.index,
                y=vix_data['VIX_Close'],
                mode='lines',
                name='VIX',
                line=dict(color='#9c27b0', width=2, dash='dot'),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Update layout for secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title='VIX',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - Enhanced {chart_type} Analysis ({timeframe.title()})',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=rows, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="ATR ($)", row=2, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig 