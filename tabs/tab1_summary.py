"""
Tab 1: Summary - Stock Volatility Analyzer

This module contains the summary tab functionality with comprehensive market analysis,
volatility metrics, and trading recommendations.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from core import (
    get_current_price, get_vix_condition, should_trade,
    format_percentage, format_currency
)


def render_summary_tab(results, vix_data, session_tickers):
    """
    Render the Summary tab with comprehensive market volatility analysis
    
    Args:
        results (dict): Analysis results from the main app
        vix_data (pd.DataFrame): VIX data
        session_tickers (list): List of selected tickers
    """
    
    st.subheader("📊 Comprehensive Market Volatility Summary")
    
    # === SECTION 1: CURRENT MARKET STATUS ===
    st.markdown("### 🎯 Current Market Status")
    
    # Get current prices for all tickers
    current_prices = {}
    price_changes = {}
    
    col1, col2, col3, col4 = st.columns(4)
    for i, ticker in enumerate(session_tickers[:4]):  # Show up to 4 tickers in header
        current_price = get_current_price(ticker)
        current_prices[ticker] = current_price
        
        # Calculate daily change if we have daily data
        daily_change = 0
        daily_change_pct = 0
        if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
            daily_data = results[ticker]['daily']['data']
            if daily_data is not None and len(daily_data) >= 2:
                today_close = daily_data['Close'].iloc[-1]
                yesterday_close = daily_data['Close'].iloc[-2]
                daily_change = today_close - yesterday_close
                daily_change_pct = (daily_change / yesterday_close) * 100
        
        price_changes[ticker] = {'change': daily_change, 'change_pct': daily_change_pct}
        
        with [col1, col2, col3, col4][i]:
            if current_price:
                st.metric(
                    label=f"{ticker}",
                    value=f"${current_price:.2f}",
                    delta=f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)" if daily_change != 0 else None
                )
            else:
                st.metric(label=f"{ticker}", value="Price N/A")
    
    # === SECTION 2: ENHANCED VOLATILITY ANALYSIS TABLE ===
    st.markdown("### 📊 Enhanced Volatility Analysis")
    
    # Create comprehensive summary table
    summary_data = []
    for ticker in session_tickers:
        row = {
            'Ticker': ticker,
            'Current Price': f"${current_prices.get(ticker, 0):.2f}" if current_prices.get(ticker) else "N/A"
        }
        
        # Calculate comprehensive metrics for each timeframe
        for tf in ['daily', 'weekly']:  # Focus on most important timeframes
            if tf in results[ticker] and results[ticker][tf]:
                metrics = results[ticker][tf]
                atr_val = metrics['atr']
                vol_val = metrics['volatility']
                cv_val = metrics['coefficient_variation']
                
                # ATR as percentage of current price
                atr_pct = (atr_val / current_prices.get(ticker, 1)) * 100 if current_prices.get(ticker) and atr_val > 0 else 0
                
                row[f'{tf.title()} ATR'] = f"${atr_val:.2f}" if atr_val > 0 else "N/A"
                row[f'{tf.title()} ATR%'] = f"{atr_pct:.1f}%" if atr_pct > 0 else "N/A"
                row[f'{tf.title()} Vol'] = f"${vol_val:.2f}" if vol_val > 0 else "N/A"
            else:
                row[f'{tf.title()} ATR'] = "No Data"
                row[f'{tf.title()} ATR%'] = "No Data"
                row[f'{tf.title()} Vol'] = "No Data"
        
        # Calculate volatility ranking
        daily_atr = results[ticker].get('daily', {}).get('atr', 0) if ticker in results else 0
        if daily_atr > 0 and current_prices.get(ticker):
            daily_atr_pct = (daily_atr / current_prices[ticker]) * 100
            if daily_atr_pct > 3:
                vol_rank = "🔴 HIGH"
            elif daily_atr_pct > 1.5:
                vol_rank = "🟡 MEDIUM"
            else:
                vol_rank = "🟢 LOW"
        else:
            vol_rank = "❓ UNKNOWN"
        
        row['Vol Rank'] = vol_rank
        summary_data.append(row)
    
    # Display enhanced summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, height=300)
    
    # ATR Explanation
    with st.expander("📚 Understanding ATR & Volatility Metrics"):
        st.markdown("""
        ### 📊 Average True Range (ATR) Explained
        
        **What is ATR?**
        ATR measures market volatility by calculating the average of true ranges over a specified period (typically 14 periods).
        
        **True Range Calculation:**
        ```
        True Range = MAX of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|
        ```
        
        **Key ATR Insights:**
        - **Higher ATR** = More volatile, larger price swings, higher risk/reward
        - **Lower ATR** = Less volatile, smaller movements, lower risk/reward
        - **ATR %** = ATR ÷ Current Price × 100 (normalized measure)
        
        **Volatility Rankings:**
        - 🟢 **LOW (< 1.5%)**: Stable, good for momentum strategies
        - 🟡 **MEDIUM (1.5-3%)**: Moderate, ideal for options strategies  
        - 🔴 **HIGH (> 3%)**: Volatile, reduce position size, high premium options
        
        **Trading Applications:**
        - **Position Sizing**: Use ATR to determine appropriate position size
        - **Stop Losses**: Set stops at 1-2x ATR from entry
        - **Profit Targets**: Target 2-3x ATR for reward:risk ratios
        - **Options Strategy**: Use ATR for strike selection and expiry timing
        """)
    
    # Market Insights Section
    st.markdown("### 💡 Market Insights & Trading Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility Leaders
        st.markdown("#### 🔥 Volatility Leaders")
        
        vol_leaders = []
        for ticker in session_tickers:
            if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
                daily_atr = results[ticker]['daily']['atr']
                current_price = current_prices.get(ticker, 0)
                if daily_atr > 0 and current_price > 0:
                    atr_pct = (daily_atr / current_price) * 100
                    vol_leaders.append({
                        'ticker': ticker,
                        'atr_pct': atr_pct,
                        'atr_dollar': daily_atr
                    })
        
        # Sort by ATR percentage
        vol_leaders.sort(key=lambda x: x['atr_pct'], reverse=True)
        
        for i, leader in enumerate(vol_leaders[:3]):
            st.write(f"**{i+1}. {leader['ticker']}**: {leader['atr_pct']:.1f}% (${leader['atr_dollar']:.2f})")
        
        if not vol_leaders:
            st.write("*No volatility data available*")
    
    with col2:
        # Trading Recommendations
        st.markdown("#### 🎯 Trading Recommendations")
            
        # Get VIX condition if available
        if vix_data is not None:
            current_vix = vix_data['VIX_Close'].iloc[-1]
            condition, condition_class, icon = get_vix_condition(current_vix)
            trade_ok, trade_msg = should_trade(current_vix)
            
            st.markdown(f"**VIX Status**: {icon} {current_vix:.1f}")
            st.markdown(f"**Condition**: {condition.split(' - ')[0]}")
            st.markdown(f"**Trading**: {'✅ Approved' if trade_ok else '❌ Avoid'}")
        else:
            st.markdown("**VIX Status**: ❓ Not Available")
            st.markdown("**Trading**: ⚠️ Enable VIX analysis")
        
        # Position sizing recommendations
        avg_atr_pct = 0
        valid_tickers = 0
        for ticker in session_tickers:
            if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
                daily_atr = results[ticker]['daily']['atr']
                current_price = current_prices.get(ticker, 0)
                if daily_atr > 0 and current_price > 0:
                    atr_pct = (daily_atr / current_price) * 100
                    avg_atr_pct += atr_pct
                    valid_tickers += 1
        
        if valid_tickers > 0:
            avg_atr_pct /= valid_tickers
            if avg_atr_pct > 2.5:
                size_rec = "🔴 Reduce Position Size"
            elif avg_atr_pct > 1.5:
                size_rec = "🟡 Normal Position Size"
            else:
                size_rec = "🟢 Can Increase Size"
            
            st.markdown(f"**Position Sizing**: {size_rec}")
            st.markdown(f"**Avg Market Vol**: {avg_atr_pct:.1f}%")
        else:
            st.markdown("**Position Sizing**: ❓ Insufficient Data") 