import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def analyze_stock_volatility(ticker: str):
    """Fetch and analyze hourly data for a stock"""
    print(f"\nAnalyzing {ticker}...")
    
    # Fetch 90 days of hourly data
    data = yf.download(ticker, period='90d', interval='1h')
    
    if data.empty:
        return f"No data found for {ticker}"
    
    # Handle potential multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Ensure we have the right column names
    data.columns = data.columns.str.title()  # Convert to title case
    
    # Calculate hourly ranges
    data['hourly_range'] = data['High'] - data['Low']
    
    # Calculate daily ranges
    daily = data.resample('D').agg({'High':'max', 'Low':'min', 'Close':'last'})
    daily['daily_range'] = daily['High'] - daily['Low']
    
    # Calculate weekly ranges  
    weekly = data.resample('W').agg({'High':'max', 'Low':'min', 'Close':'last'})
    weekly['weekly_range'] = weekly['High'] - weekly['Low']
    
    # Calculate statistics
    hourly_stats = data['hourly_range'].describe(percentiles=[.25, .5, .75])
    daily_stats = daily['daily_range'].describe(percentiles=[.25, .5, .75])
    weekly_stats = weekly['weekly_range'].describe(percentiles=[.25, .5, .75])
    
    # Calculate Average True Range (ATR) approximation
    hourly_atr = data['hourly_range'].rolling(window=14).mean().iloc[-1]
    daily_atr = daily['daily_range'].rolling(window=14).mean().iloc[-1]
    weekly_atr = weekly['weekly_range'].rolling(window=14).mean().iloc[-1]
    
    results = {
        'ticker': ticker,
        'hourly_stats': hourly_stats,
        'daily_stats': daily_stats, 
        'weekly_stats': weekly_stats,
        'hourly_atr': hourly_atr,
        'daily_atr': daily_atr,
        'weekly_atr': weekly_atr
    }
    
    return results

def print_analysis(results):
    """Print formatted analysis results"""
    ticker = results['ticker']
    print(f"\n{'='*50}")
    print(f"VOLATILITY ANALYSIS FOR {ticker}")
    print(f"{'='*50}")
    
    for timeframe in ['hourly', 'daily', 'weekly']:
        stats = results[f'{timeframe}_stats']
        atr = results[f'{timeframe}_atr']
        
        print(f"\n{timeframe.upper()} RANGE ANALYSIS:")
        print(f"  Min Range:     ${stats['min']:.2f}")
        print(f"  25th Percentile: ${stats['25%']:.2f}")
        print(f"  Median Range:   ${stats['50%']:.2f}")
        print(f"  Mean Range:     ${stats['mean']:.2f}")
        print(f"  75th Percentile: ${stats['75%']:.2f}")
        print(f"  Max Range:      ${stats['max']:.2f}")
        print(f"  Std Deviation:  ${stats['std']:.2f}")
        print(f"  ATR (14-period): ${atr:.2f}")

# Main analysis
def main():
    tickers = ['SPY', 'QQQ']
    results = {}
    
    for ticker in tickers:
        results[ticker] = analyze_stock_volatility(ticker)
        print_analysis(results[ticker])
    
    # Comparative analysis
    print(f"\n{'='*50}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*50}")
    
    spy_results = results['SPY']
    qqq_results = results['QQQ']
    
    for timeframe in ['hourly', 'daily', 'weekly']:
        spy_atr = spy_results[f'{timeframe}_atr']
        qqq_atr = qqq_results[f'{timeframe}_atr']
        
        print(f"\n{timeframe.upper()} ATR Comparison:")
        print(f"  SPY: ${spy_atr:.2f}")
        print(f"  QQQ: ${qqq_atr:.2f}")
        print(f"  QQQ/SPY Ratio: {qqq_atr/spy_atr:.2f}x")

if __name__ == "__main__":
    main()