"""
Tab modules for Stock Volatility Analyzer

This package contains individual tab implementations for the Streamlit application.
Each tab is self-contained with its own logic and display functions.
"""

# Import all available tab modules
from .tab1_summary import render_summary_tab
from .tab2_price_charts import render_price_charts_tab
from .tab3_detailed_stats import render_detailed_stats_tab
from .tab4_comparison import render_comparison_tab
from .tab5_vix_analysis import render_vix_analysis_tab
from .tab6_options_strategy import render_options_strategy_tab

__all__ = [
    'render_summary_tab',
    'render_price_charts_tab',
    'render_detailed_stats_tab',
    'render_comparison_tab',
    'render_vix_analysis_tab',
    'render_options_strategy_tab'
] 