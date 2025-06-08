"""
Core functionality for Stock Volatility Analyzer

This package contains the core data fetching, calculation, and utility functions
used throughout the application.
"""

# Version info
__version__ = "2.0.0"
__author__ = "Mridul Jain"

# Import main functions for easy access
from .data_fetchers import (
    fetch_stock_data,
    fetch_vix_data,
    get_current_price,
    validate_ticker
)

from .calculations import (
    get_vix_condition,
    should_trade,
    calculate_true_range,
    calculate_volatility_metrics
)

from .charts import (
    create_price_chart,
    create_comparison_chart,
    create_enhanced_price_chart
)

from .styling import (
    get_custom_css,
    format_percentage,
    format_currency
)

__all__ = [
    'fetch_stock_data',
    'fetch_vix_data', 
    'get_current_price',
    'validate_ticker',
    'get_vix_condition',
    'should_trade',
    'calculate_true_range',
    'calculate_volatility_metrics',
    'create_price_chart',
    'create_comparison_chart',
    'create_enhanced_price_chart',
    'get_custom_css',
    'format_percentage',
    'format_currency'
] 