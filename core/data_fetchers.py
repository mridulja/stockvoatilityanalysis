"""
Data fetching functions for Stock Volatility Analyzer

This module contains all functions related to fetching data from Yahoo Finance
and other data sources.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@st.cache_data(ttl=300)
def fetch_stock_data(ticker, start_date, end_date, interval='1h'):
    """Fetch stock data with caching"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            return None
            
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure consistent column names
        data.columns = data.columns.str.title()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=300)
def fetch_vix_data(start_date, end_date):
    """Fetch VIX data with caching"""
    try:
        from .calculations import get_vix_condition
        
        vix_data = yf.download("^VIX", start=start_date, end=end_date, interval='1d', progress=False)
        
        if vix_data.empty:
            return None
            
        # Handle multi-level columns
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.droplevel(1)
        
        # Ensure consistent column names
        vix_data.columns = vix_data.columns.str.title()
        
        # Add market condition analysis
        vix_data['VIX_Close'] = vix_data['Close']
        vix_data['Market_Condition'] = vix_data['VIX_Close'].apply(lambda x: get_vix_condition(x)[0])
        vix_data['Condition_Color'] = vix_data['VIX_Close'].apply(lambda x: get_vix_condition(x)[1])
        vix_data['Condition_Icon'] = vix_data['VIX_Close'].apply(lambda x: get_vix_condition(x)[2])
        
        return vix_data[['VIX_Close', 'Market_Condition', 'Condition_Color', 'Condition_Icon']]
    except Exception as e:
        st.warning(f"Could not fetch VIX data: {str(e)}")
        return None


@st.cache_data(ttl=300)
def get_current_price(ticker):
    """Get current/latest price for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d', interval='1m')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        else:
            # Fallback to daily data
            hist = stock.history(period='5d')
            return hist['Close'].iloc[-1] if not hist.empty else None
    except:
        return None


@st.cache_data(ttl=300)
def validate_ticker(ticker):
    """Validate if a ticker exists and can be traded"""
    try:
        stock = yf.Ticker(ticker)
        # Try to get basic info and recent price
        info = stock.info
        hist = stock.history(period='5d')
        
        if hist.empty or not info:
            return False, "No data available"
        
        # Check if it's a valid stock with basic info
        if 'longName' in info or 'shortName' in info:
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            return True, f"${current_price:.2f}" if current_price else "Valid"
        else:
            return False, "Invalid ticker"
            
    except Exception as e:
        return False, f"Error: {str(e)}" 