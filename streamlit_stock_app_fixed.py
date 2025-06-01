import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Volatility Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .vix-calm { background-color: #d4edda; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-normal { background-color: #cce5ff; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-choppy { background-color: #fff3cd; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-volatile { background-color: #f8d7da; padding: 0.5rem; border-radius: 0.25rem; }
    .vix-extreme { background-color: #f5c6cb; padding: 0.5rem; border-radius: 0.25rem; }
    .trade-recommend { background-color: #e7f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; }
    .strike-recommend { background-color: #f0fff4; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #28a745; }
</style>
""", unsafe_allow_html=True)

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

def main():
    # Header
    st.markdown('<div class="main-header">📊 Enhanced Stock Volatility Analyzer with Options Strategy</div>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("Analysis Parameters")
    
    # Ticker selection
    default_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
    selected_tickers = st.sidebar.multiselect(
        "Select Stock Tickers",
        options=default_tickers + ['NVDA', 'GOOGL', 'AMZN', 'META', 'NFLX', 'IWM', 'DIA'],
        default=['SPY', 'QQQ'],
        help="Choose up to 5 tickers for comparison"
    )
    
    # ALWAYS SHOW THE OPTIONS STRATEGY SECTION
    st.subheader("🎯 Options Trading Strategy Recommendations")
    
    # Strategy configuration
    st.markdown("### 📅 Trade Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_ticker = st.selectbox(
            "Select ticker for options strategy:",
            selected_tickers if selected_tickers else ['SPY'],
            help="Choose the ticker you want to trade options on"
        )
        
        trade_date = st.date_input(
            "Select trade date:",
            value=date.today(),
            min_value=date.today() - timedelta(days=7),
            max_value=date.today() + timedelta(days=30),
            help="Date when you plan to enter the trade"
        )
    
    with col2:
        strategy_timeframe = st.selectbox(
            "Options expiry timeframe:",
            ['daily', 'weekly'],
            help="Daily = same day expiry, Weekly = end of week expiry"
        )
        
        target_probability = st.slider(
            "Target probability threshold (%):",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum acceptable probability of strike being hit"
        ) / 100
    
    # Custom strike testing
    st.markdown("### 🎯 Custom Strike Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        custom_strikes_input = st.text_input(
            "Enter custom strikes (comma-separated):",
            placeholder="e.g., 580, 575, 570",
            help="Enter specific strike prices you want to analyze"
        )
    
    with col2:
        num_recommendations = st.slider(
            "Number of strike recommendations:",
            min_value=3,
            max_value=10,
            value=5,
            help="How many strike prices to recommend"
        )
    
    if st.button("🚀 Generate Options Strategy", type="primary"):
        # IMMEDIATE DEBUG - Show we entered the button handler
        st.write("🔧 DEBUG: Button clicked! Starting options strategy generation...")
        st.write(f"🔧 DEBUG: Button pressed at {datetime.now()}")
        
        try:
            st.write("🔧 DEBUG: Entered try block successfully")
            st.info("🔄 Generating options strategy...")
            
            # Debug: Check basic variables
            st.write(f"🔧 DEBUG: strategy_ticker = {strategy_ticker}")
            st.write(f"🔧 DEBUG: strategy_timeframe = {strategy_timeframe}")
            st.write(f"🔧 DEBUG: target_probability = {target_probability}")
            st.write(f"🔧 DEBUG: trade_date = {trade_date}")
            
            st.write("🔧 DEBUG: About to fetch current price...")
            
            # Get current price
            with st.spinner("Fetching current price..."):
                st.write("🔧 DEBUG: Inside spinner for current price...")
                current_price = get_current_price(strategy_ticker)
                st.write(f"🔧 DEBUG: get_current_price returned: {current_price}")
            
            if current_price is None:
                st.error(f"❌ Could not fetch current price for {strategy_ticker}")
                st.error("Please try again or select a different ticker")
                st.warning("⚠️ Try selecting a different ticker or check your internet connection")
                st.write("🔧 DEBUG: Exiting due to current_price = None")
                return
            
            st.success(f"✅ Current price fetched: ${current_price:.2f}")
            st.write(f"🔧 DEBUG: Successfully got current price: ${current_price:.2f}")
            
            st.markdown(f"### 📊 Analysis for {strategy_ticker}")
            st.info(f"**Current Price**: ${current_price:.2f} | **Trade Date**: {trade_date} | **Timeframe**: {strategy_timeframe}")
            
            # Simple demonstration strategy
            st.markdown("### 🎯 Basic PUT Strategy Demonstration")
            
            # Calculate some basic strikes
            strikes_below = []
            for i in range(num_recommendations):
                pct_below = (i + 1) * 2  # 2%, 4%, 6%, etc.
                strike = current_price * (1 - pct_below/100)
                strikes_below.append(round(strike, 2))
            
            st.markdown("### 📋 Suggested PUT Strike Prices")
            strike_data = []
            for i, strike in enumerate(strikes_below):
                distance = current_price - strike
                distance_pct = (distance / current_price) * 100
                
                strike_data.append({
                    'Rank': i + 1,
                    'Strike Price': f"${strike:.2f}",
                    'Distance': f"${distance:.2f}",
                    'Distance %': f"{distance_pct:.1f}%",
                    'Recommendation': 'CONSERVATIVE' if distance_pct > 5 else 'AGGRESSIVE'
                })
            
            strike_df = pd.DataFrame(strike_data)
            st.dataframe(strike_df, use_container_width=True)
            
            # Best recommendation
            best_strike = strikes_below[0]
            st.markdown('<div class="strike-recommend">', unsafe_allow_html=True)
            st.markdown(f"""
            ### 🏆 RECOMMENDED STRATEGY: ${best_strike:.2f} PUT
            - **Current Price**: ${current_price:.2f}
            - **Strike Distance**: ${current_price - best_strike:.2f} ({((current_price - best_strike)/current_price)*100:.1f}%)
            - **Strategy**: Sell PUT at ${best_strike:.2f}
            - **Timeframe**: {strategy_timeframe.title()}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Custom strikes analysis
            if custom_strikes_input:
                try:
                    custom_strikes = [float(x.strip()) for x in custom_strikes_input.split(',')]
                    st.info(f"Analyzing custom strikes: {custom_strikes}")
                    
                    st.markdown("### 🔍 Custom Strike Analysis")
                    
                    custom_data = []
                    for strike in custom_strikes:
                        distance = current_price - strike
                        distance_pct = (distance / current_price) * 100
                        
                        custom_data.append({
                            'Strike': f"${strike:.2f}",
                            'Distance': f"${distance:.2f}",
                            'Distance %': f"{distance_pct:.1f}%",
                            'Risk Level': 'LOW' if distance_pct > 5 else 'MEDIUM' if distance_pct > 2 else 'HIGH'
                        })
                    
                    custom_df = pd.DataFrame(custom_data)
                    st.dataframe(custom_df, use_container_width=True)
                
                except ValueError:
                    st.error("❌ Invalid strike format. Please use comma-separated numbers (e.g., 580, 575, 570)")
                except Exception as e:
                    st.error(f"❌ Custom strikes analysis failed: {str(e)}")
            
            st.success("✅ Options strategy analysis complete!")
            st.info("Note: This is a simplified demonstration. For full analysis with probability calculations, run the Enhanced Analysis first.")
            
        except Exception as e:
            st.write("🔧 DEBUG: CAUGHT EXCEPTION!")
            st.write(f"🔧 DEBUG: Exception type: {type(e).__name__}")
            st.write(f"🔧 DEBUG: Exception message: {str(e)}")
            st.error(f"❌ Unexpected error: {str(e)}")
            st.error("Please try again or contact support")
            # Show the full traceback for debugging
            import traceback
            st.write("🔧 DEBUG: Full traceback:")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 