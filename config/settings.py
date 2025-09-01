"""
Configuration settings for Stock Volatility Analyzer

This module contains all application settings, constants, and default values.
"""

from datetime import date, timedelta

# Default stock tickers
DEFAULT_TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'GLD', 'NVDA', 
'GOOGL', 'AMZN', 'META', 'NFLX', 'IWM', 'DIA', 'VRT', 'HOOD', 
'GDX', 'OUST','IBIT', 'SMCI', 'FTNT', 'EW', 'LLY', 'T', 'BABA', 
'JD']

# Application configuration
APP_CONFIG = {
    'page_title': 'Stock Volatility Analyzer',
    'page_icon': '🎯',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'version': '2.0.0',
    'author': 'Mridul Jain'
}

# Chart configuration
CHART_CONFIG = {
    'default_height': 800,
    'colors': {
        'primary': '#6366f1',
        'secondary': '#06b6d4',
        'success': '#10b981',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'bullish': '#26a69a',
        'bearish': '#ef5350'
    },
    'atr_window': 14,
    'volume_ma_window': 20,
    'bollinger_period': 20,
    'bollinger_std': 2
}

# VIX thresholds for market conditions
VIX_THRESHOLDS = {
    'calm': 15,
    'normal': 19,
    'choppy': 26,
    'volatile': 36
}

# Volatility ranking thresholds (ATR as percentage of price)
VOLATILITY_THRESHOLDS = {
    'low': 1.5,
    'medium': 3.0
}

# Date range settings
DATE_CONFIG = {
    'min_days': 90,
    'max_years': 2,
    'default_days': 90,
    'cache_ttl': 300  # 5 minutes
}

# Analysis options
ANALYSIS_OPTIONS = {
    'timeframes': ['hourly', 'daily', 'weekly'],
    'intervals': {
        'hourly': '1h',
        'daily': '1d',
        'weekly': '1wk'
    },
    'chart_types': ['Candlestick', 'Line', 'OHLC', 'Area'],
    'technical_indicators': [
        'SMA 20', 'SMA 50', 'EMA 12', 'EMA 26',
        'Bollinger Bands', 'ATR Bands', 'VIX Overlay'
    ]
}

# Tab configuration
TAB_CONFIG = {
    'tabs': [
        {'name': '📊 Summary', 'key': 'summary'},
        {'name': '📈 Price Charts', 'key': 'price_charts'},
        {'name': '🔍 Detailed Stats', 'key': 'detailed_stats'},
        {'name': '⚖️ Comparison', 'key': 'comparison'},
        {'name': '📉 VIX Analysis', 'key': 'vix_analysis'},
        {'name': '🎯 Options Strategy', 'key': 'options_strategy'},
        {'name': '📐 Put Spread Analysis', 'key': 'put_spread'},
        {'name': '🦅 Iron Condor Playbook', 'key': 'iron_condor'}
    ]
}

# Colors for different elements
COLORS = {
    'vix_conditions': {
        'calm': '#10b981',
        'normal': '#06b6d4',
        'choppy': '#f59e0b',
        'volatile': '#ef4444',
        'extreme': '#dc2626'
    },
    'volatility_ranks': {
        'low': '#10b981',
        'medium': '#f59e0b',
        'high': '#ef4444',
        'unknown': '#64748b'
    }
}

# Risk management settings
RISK_MANAGEMENT = {
    'stop_loss_multiplier': 1.5,  # ATR multiplier for stop loss
    'profit_target_multiplier': 2.5,  # ATR multiplier for profit target
    'position_size_thresholds': {
        'reduce': 2.5,  # Reduce position if ATR% > this
        'normal': 1.5,  # Normal position if ATR% between normal and reduce
        'increase': 1.0  # Can increase if ATR% < this
    }
}

# Data validation settings
DATA_VALIDATION = {
    'min_data_points': 20,
    'required_columns': ['High', 'Low', 'Close'],
    'optional_columns': ['Open', 'Volume'],
    'max_missing_percentage': 10  # Max % of missing data allowed
}

# Export/import settings
EXPORT_CONFIG = {
    'formats': ['CSV', 'Excel', 'JSON'],
    'compression': True,
    'include_metadata': True
}

# Performance settings
PERFORMANCE = {
    'max_tickers': 10,
    'max_data_points': 10000,
    'cache_size': 100,
    'parallel_processing': True
}

# Feature flags
FEATURES = {
    'ai_analysis': True,
    'put_spread_analysis': True,
    'iron_condor_analysis': True,
    'export_functionality': True,
    'advanced_charts': True,
    'real_time_data': True
} 