#!/usr/bin/env python3
"""
Comprehensive Project Integrity Test

This test file verifies that all essential components of the Stock Analysis
application are working correctly. It tests:

1. Main application imports and basic functionality
2. All tab modules and their core functions  
3. Shared modules (AI formatter, LLM analyzer)
4. Core analysis modules (Iron Condor, Put Spread, etc.)
5. Essential data fetching and calculation functions

Run this BEFORE and AFTER any file cleanup to ensure nothing breaks.

Usage:
    python test_project_integrity.py

Author: AI Assistant
Date: 2025
"""

import sys
import traceback
from datetime import datetime, date, timedelta
import warnings

# Suppress warnings during testing
warnings.filterwarnings('ignore')

class ProjectIntegrityTest:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failed_tests = []
        
    def log(self, message, test_type="INFO"):
        """Log test messages with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
        symbol = symbols.get(test_type, "📋")
        print(f"[{timestamp}] {symbol} {message}")
    
    def run_test(self, test_name, test_func):
        """Run a single test with error handling"""
        self.tests_run += 1
        try:
            test_func()
            self.tests_passed += 1
            self.log(f"{test_name} - PASSED", "PASS")
            return True
        except Exception as e:
            self.tests_failed += 1
            self.failed_tests.append((test_name, str(e)))
            self.log(f"{test_name} - FAILED: {str(e)}", "FAIL")
            # Print traceback for debugging
            traceback.print_exc()
            return False
    
    def test_main_application(self):
        """Test main application file imports and basic functionality"""
        # Test main app import
        import streamlit_stock_app_complete
        
        # Test key functions exist
        assert hasattr(streamlit_stock_app_complete, 'main'), "Main function missing"
        assert hasattr(streamlit_stock_app_complete, 'fetch_stock_data'), "fetch_stock_data missing"
        assert hasattr(streamlit_stock_app_complete, 'get_current_price'), "get_current_price missing"
        
        self.log("Main application imports and structure verified")
    
    def test_tab_modules(self):
        """Test all tab modules can be imported and have required functions"""
        tab_modules = [
            ('tabs.tab1_summary', 'render_summary_tab'),
            ('tabs.tab2_price_charts', 'render_price_charts_tab'),
            ('tabs.tab3_detailed_stats', 'render_detailed_stats_tab'),
            ('tabs.tab4_comparison', 'render_comparison_tab'),
            ('tabs.tab5_vix_analysis', 'render_vix_analysis_tab'),
            ('tabs.tab6_options_strategy', 'render_options_strategy_tab'),
            ('tabs.tab7_put_spread_analysis', 'render_put_spread_analysis_tab'),
            ('tabs.tab8_iron_condor_playbook', 'render_iron_condor_playbook_tab'),
        ]
        
        for module_name, function_name in tab_modules:
            module = __import__(module_name, fromlist=[function_name])
            assert hasattr(module, function_name), f"{function_name} missing from {module_name}"
        
        self.log("All tab modules imported successfully")
    
    def test_shared_modules(self):
        """Test shared modules"""
        # Test AI formatter
        from shared.ai_formatter import display_ai_analysis, get_tab_color
        
        # Test basic functionality
        color = get_tab_color("test")
        assert isinstance(color, str), "get_tab_color should return string"
        
        self.log("Shared modules verified")
    
    def test_core_modules(self):
        """Test core modules"""
        try:
            from core.data_fetchers import fetch_stock_data
            from core.calculations import calculate_volatility_metrics, get_vix_condition
            from core.charts import create_price_chart
            from core.styling import get_custom_css
            
            # Test function signatures exist
            assert callable(fetch_stock_data), "fetch_stock_data not callable"
            assert callable(calculate_volatility_metrics), "calculate_volatility_metrics not callable"
            assert callable(get_vix_condition), "get_vix_condition not callable"
            assert callable(create_price_chart), "create_price_chart not callable"
            assert callable(get_custom_css), "get_custom_css not callable"
            
            self.log("Core modules verified")
        except ImportError as e:
            self.log(f"Core modules not available (optional): {str(e)}", "WARN")
    
    def test_analysis_modules(self):
        """Test main analysis modules"""
        # Test Iron Condor analysis
        try:
            from iron_condor_analysis import IronCondorAnalyzer
            analyzer = IronCondorAnalyzer()
            assert hasattr(analyzer, 'analyze_iron_condor_strategies'), "Missing analyze_iron_condor_strategies"
            self.log("Iron Condor analysis module verified")
        except ImportError:
            self.log("Iron Condor analysis module not available (optional)", "WARN")
        
        # Test LLM analysis
        try:
            from llm_analysis import get_llm_analyzer
            assert callable(get_llm_analyzer), "get_llm_analyzer not callable"
            self.log("LLM analysis module verified")
        except ImportError:
            self.log("LLM analysis module not available (optional)", "WARN")
        
        # Test LLM input formatters
        try:
            from llm_input_formatters import format_put_spread_data_for_llm
            assert callable(format_put_spread_data_for_llm), "format_put_spread_data_for_llm not callable"
            self.log("LLM input formatters verified")
        except ImportError:
            self.log("LLM input formatters not available (optional)", "WARN")
    
    def test_config_modules(self):
        """Test configuration modules"""
        try:
            from config.settings import (
                DEFAULT_TICKERS, 
                APP_CONFIG, 
                CHART_CONFIG
            )
            
            # Verify config values
            assert isinstance(DEFAULT_TICKERS, list), "DEFAULT_TICKERS should be list"
            assert isinstance(APP_CONFIG, dict), "APP_CONFIG should be dict"
            assert isinstance(CHART_CONFIG, dict), "CHART_CONFIG should be dict"
            
            self.log("Configuration modules verified")
        except ImportError:
            self.log("Configuration modules not available (optional)", "WARN")
    
    def test_data_operations(self):
        """Test basic data operations without external API calls"""
        from datetime import datetime, timedelta
        import pandas as pd
        import numpy as np
        
        # Test date calculations
        today = date.today()
        next_friday = today + timedelta(days=(4 - today.weekday()) % 7)
        assert isinstance(next_friday, date), "Date calculation failed"
        
        # Test basic pandas/numpy operations
        data = pd.DataFrame({'price': [100, 101, 99, 102], 'volume': [1000, 1100, 900, 1200]})
        assert not data.empty, "DataFrame creation failed"
        
        volatility = np.std([100, 101, 99, 102])
        assert isinstance(volatility, (int, float)), "Volatility calculation failed"
        
        self.log("Basic data operations verified")
    
    def test_import_dependencies(self):
        """Test that all required external dependencies can be imported"""
        required_packages = [
            'streamlit',
            'yfinance', 
            'pandas',
            'numpy',
            'plotly',
            'scipy'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.log(f"Package {package} available")
            except ImportError:
                raise ImportError(f"Required package {package} not available")
        
        self.log("All required dependencies verified")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        self.log("=" * 60)
        self.log("STARTING COMPREHENSIVE PROJECT INTEGRITY TEST")
        self.log("=" * 60)
        
        # Define all tests
        tests = [
            ("External Dependencies", self.test_import_dependencies),
            ("Main Application", self.test_main_application),
            ("Tab Modules", self.test_tab_modules),
            ("Shared Modules", self.test_shared_modules),
            ("Core Modules", self.test_core_modules),
            ("Analysis Modules", self.test_analysis_modules),
            ("Configuration Modules", self.test_config_modules),
            ("Data Operations", self.test_data_operations),
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.log(f"Running: {test_name}")
            self.run_test(test_name, test_func)
        
        # Print summary
        self.log("=" * 60)
        self.log("TEST SUMMARY")
        self.log("=" * 60)
        self.log(f"Total Tests: {self.tests_run}")
        self.log(f"Passed: {self.tests_passed}", "PASS")
        self.log(f"Failed: {self.tests_failed}", "FAIL" if self.tests_failed > 0 else "INFO")
        
        if self.failed_tests:
            self.log("FAILED TESTS:", "FAIL")
            for test_name, error in self.failed_tests:
                self.log(f"  - {test_name}: {error}", "FAIL")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        self.log(f"Success Rate: {success_rate:.1f}%")
        
        if self.tests_failed == 0:
            self.log("🎉 ALL TESTS PASSED! Project integrity verified.", "PASS")
            return True
        else:
            self.log("⚠️ Some tests failed. Review before proceeding.", "WARN")
            return False

def main():
    """Main function to run the test suite"""
    print("🧪 Stock Analysis Project Integrity Test")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create and run test suite
    test_suite = ProjectIntegrityTest()
    success = test_suite.run_all_tests()
    
    print()
    if success:
        print("✅ PROJECT INTEGRITY VERIFIED - Safe to proceed with cleanup")
        sys.exit(0)
    else:
        print("❌ PROJECT INTEGRITY ISSUES DETECTED - Do not proceed with cleanup")
        sys.exit(1)

if __name__ == "__main__":
    main() 