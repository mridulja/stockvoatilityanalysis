"""
Calculation functions for Stock Volatility Analyzer

This module contains all mathematical calculations including ATR, volatility metrics,
VIX analysis, and statistical computations.
"""

import pandas as pd
import numpy as np


def get_vix_condition(vix_value):
    """Determine market condition based on VIX level"""
    if pd.isna(vix_value):
        return "Unknown", "vix-normal", "🤷"
    
    if vix_value < 15:
        return "Calm Markets - Clean Trend", "vix-calm", "🟢"
    elif 15 <= vix_value < 19:
        return "Normal Markets - Trendy", "vix-normal", "🔵"
    elif 19 <= vix_value < 26:
        return "Choppy Market - Proceed with Caution", "vix-choppy", "🟡"
    elif 26 <= vix_value < 36:
        return "High Volatility - Big Swings, Don't Trade", "vix-volatile", "🔴"
    else:
        return "Extreme Volatility - Very Bad Day, DO NOT TRADE", "vix-extreme", "🚨"


def should_trade(vix_value):
    """Determine if trading is recommended based on VIX"""
    if pd.isna(vix_value):
        return False, "Unknown VIX - Cannot assess"
    
    if vix_value < 26:
        return True, "Trading conditions acceptable"
    else:
        return False, "VIX too high - Avoid trading"


def calculate_true_range(data):
    """Calculate True Range for ATR calculation"""
    if data is None or len(data) < 2:
        return None
    
    data = data.copy()
    
    # True Range calculation: max of (H-L, |H-C_prev|, |L-C_prev|)
    data['prev_close'] = data['Close'].shift(1)
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = abs(data['High'] - data['prev_close'])
    data['tr3'] = abs(data['Low'] - data['prev_close'])
    
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # For the first row, use simple range since no previous close
    data.loc[data.index[0], 'true_range'] = data.loc[data.index[0], 'tr1']
    
    return data['true_range']


def calculate_volatility_metrics(data, timeframe='hourly', vix_data=None):
    """Calculate volatility metrics for given timeframe"""
    if data is None or data.empty:
        return None
    
    # Calculate ranges and true range
    data = data.copy()
    data['range'] = data['High'] - data['Low']
    
    # Calculate true range for better ATR
    true_range = calculate_true_range(data)
    if true_range is not None:
        data['true_range'] = true_range
    else:
        data['true_range'] = data['range']  # Fallback to simple range
    
    # Resample based on timeframe
    if timeframe == 'daily':
        resampled = data.resample('D').agg({
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last', 
            'Volume': 'sum',
            'range': 'max',
            'true_range': 'max'
        })
        resampled['range'] = resampled['High'] - resampled['Low']
        daily_tr = calculate_true_range(resampled)
        if daily_tr is not None:
            resampled['true_range'] = daily_tr
    elif timeframe == 'weekly':
        resampled = data.resample('W').agg({
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last', 
            'Volume': 'sum',
            'range': 'max',
            'true_range': 'max'
        })
        resampled['range'] = resampled['High'] - resampled['Low']
        weekly_tr = calculate_true_range(resampled)
        if weekly_tr is not None:
            resampled['true_range'] = weekly_tr
    else:  # hourly
        resampled = data
    
    # Remove any rows with NaN values
    resampled = resampled.dropna()
    
    if len(resampled) == 0:
        return None
    
    # Calculate statistics
    range_stats = resampled['range'].describe(percentiles=[.25, .5, .75])
    
    # Calculate ATR (Average True Range)
    atr_window = min(14, len(resampled))
    if atr_window > 0 and not resampled['true_range'].isna().all():
        atr = resampled['true_range'].rolling(window=atr_window).mean().iloc[-1]
        if pd.isna(atr) and len(resampled) >= 1:
            atr = resampled['true_range'].mean()
    else:
        atr = 0
    
    # Calculate additional metrics
    volatility = resampled['range'].std()
    cv = volatility / range_stats['mean'] if range_stats['mean'] > 0 else 0
    
    # Add VIX data if available for daily analysis
    if timeframe == 'daily' and vix_data is not None:
        resampled = resampled.join(vix_data, how='left')
    
    return {
        'stats': range_stats,
        'atr': atr if not pd.isna(atr) else 0,
        'volatility': volatility if not pd.isna(volatility) else 0,
        'coefficient_variation': cv,
        'data': resampled,
        'atr_window': atr_window
    } 