"""
Put Spread Probability Analysis Module

This module implements comprehensive probability calculations for vertical put spreads using:
- Black-Scholes formulas for probability of profit
- Probability of Touching (POT) calculations
- Options chain data integration
- Robust time and volatility handling

Features:
- Probability of Profit calculations
- Probability of Touching (POT) analysis
- Strike distance optimization for target POT levels
- Same-day, weekly, and custom expiry support
- Robust data handling and logical assumptions

Author: Mridul Jain
Date: 2025
Version: 1.0 - Put Spread Analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import warnings

class PutSpreadAnalyzer:
    """
    Comprehensive put spread probability analysis using Black-Scholes methodology
    """
    
    def __init__(self):
        """Initialize the Put Spread Analyzer"""
        self.risk_free_rate = 0.05  # Default 5% risk-free rate (can be updated)
        self.min_time_fraction = 1/365/24  # Minimum time: 1 hour to avoid division by zero
    
    def fetch_options_chain(self, ticker: str, expiry_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch options chain data from yfinance
        
        Args:
            ticker: Stock ticker symbol
            expiry_date: Options expiry date in YYYY-MM-DD format
            
        Returns:
            DataFrame with options data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get all available expiry dates
            options_dates = stock.options
            
            if not options_dates:
                return None
            
            # Find closest expiry date if exact match not found
            target_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            available_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in options_dates]
            
            # Find exact match or closest date
            if target_date in available_dates:
                selected_date = expiry_date
            else:
                # Find closest date
                closest_date = min(available_dates, key=lambda x: abs((x - target_date).days))
                selected_date = closest_date.strftime('%Y-%m-%d')
            
            # Fetch options chain
            options_chain = stock.option_chain(selected_date)
            puts = options_chain.puts
            
            if puts.empty:
                return None
            
            # Clean and prepare data
            puts = puts.copy()
            puts['strike'] = puts['strike'].astype(float)
            puts['impliedVolatility'] = puts['impliedVolatility'].astype(float)
            puts['bid'] = puts['bid'].astype(float)
            puts['ask'] = puts['ask'].astype(float)
            puts['volume'] = puts['volume'].fillna(0).astype(int)
            puts['openInterest'] = puts['openInterest'].fillna(0).astype(int)
            
            # Calculate mid price
            puts['mid_price'] = (puts['bid'] + puts['ask']) / 2
            
            # Filter out options with zero bid/ask or extreme IV
            puts = puts[
                (puts['bid'] > 0) & 
                (puts['ask'] > 0) & 
                (puts['impliedVolatility'] > 0) &
                (puts['impliedVolatility'] < 5.0)  # Filter extreme IV
            ]
            
            return puts
            
        except Exception as e:
            warnings.warn(f"Failed to fetch options chain: {str(e)}")
            return None
    
    def get_dividend_yield(self, ticker: str) -> float:
        """
        Get dividend yield for the ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Annual dividend yield as decimal
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try to get dividend yield
            dividend_yield = info.get('dividendYield', 0.0)
            
            if dividend_yield is None:
                dividend_yield = 0.0
            
            return float(dividend_yield)
            
        except:
            return 0.0  # Default to 0% dividend yield
    
    def calculate_time_to_expiry(self, expiry_date: str, current_time: Optional[datetime] = None) -> float:
        """
        Calculate time to expiry in years, handling edge cases
        
        Args:
            expiry_date: Expiry date in YYYY-MM-DD format
            current_time: Current time (defaults to now)
            
        Returns:
            Time to expiry in years (minimum: 1 hour)
        """
        if current_time is None:
            current_time = datetime.now()
        
        expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        
        # For same-day expiry, assume market close (4 PM ET)
        if expiry_dt.date() == current_time.date():
            expiry_dt = expiry_dt.replace(hour=16, minute=0, second=0)
        else:
            # For future dates, assume end of trading day
            expiry_dt = expiry_dt.replace(hour=16, minute=0, second=0)
        
        time_diff = expiry_dt - current_time
        time_years = time_diff.total_seconds() / (365.25 * 24 * 3600)
        
        # Ensure minimum time to avoid division by zero
        return max(time_years, self.min_time_fraction)
    
    def calculate_implied_volatility_proxy(self, current_price: float, atr: float, time_to_expiry: float) -> float:
        """
        Calculate implied volatility proxy using ATR when options IV not available
        
        Args:
            current_price: Current stock price
            atr: Average True Range
            time_to_expiry: Time to expiry in years
            
        Returns:
            Annualized volatility estimate
        """
        if atr <= 0 or current_price <= 0:
            return 0.3  # Default 30% volatility
        
        # Convert ATR to annualized volatility
        daily_volatility = atr / current_price
        annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
        
        # Apply reasonable bounds
        return max(0.1, min(annualized_volatility, 2.0))  # Between 10% and 200%
    
    def probability_of_profit_spread(self, 
                                   current_price: float,
                                   long_strike: float,
                                   short_strike: float,
                                   time_to_expiry: float,
                                   volatility: float,
                                   risk_free_rate: float = None,
                                   dividend_yield: float = 0.0) -> float:
        """
        Calculate probability of profit for vertical put spread using Black-Scholes
        
        Args:
            current_price: Current stock price (S)
            long_strike: Long put strike price (A)
            short_strike: Short put strike price (B)
            time_to_expiry: Time to expiry in years (T)
            volatility: Implied volatility (σ)
            risk_free_rate: Risk-free rate (r)
            dividend_yield: Dividend yield (q)
            
        Returns:
            Probability of profit as decimal (0-1)
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if time_to_expiry <= 0 or volatility <= 0 or current_price <= 0:
            return 0.0
        
        S = current_price
        A = long_strike
        B = short_strike
        T = time_to_expiry
        sigma = volatility
        r = risk_free_rate
        q = dividend_yield
        
        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T
        
        # Calculate d1 and d2 for both strikes
        d1_B = (np.log(S / B) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d1_A = (np.log(S / A) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        
        # Probability of profit = N(d1_B) - N(d1_A)
        prob_profit = stats.norm.cdf(d1_B) - stats.norm.cdf(d1_A)
        
        return max(0.0, min(1.0, prob_profit))  # Bound between 0 and 1
    
    def probability_of_touching(self,
                              current_price: float,
                              strike_price: float,
                              time_to_expiry: float,
                              volatility: float,
                              risk_free_rate: float = None,
                              dividend_yield: float = 0.0) -> float:
        """
        Calculate probability of touching (POT) for a strike price
        
        Args:
            current_price: Current stock price (S)
            strike_price: Strike price to analyze (K)
            time_to_expiry: Time to expiry in years (T)
            volatility: Implied volatility (σ)
            risk_free_rate: Risk-free rate (r)
            dividend_yield: Dividend yield (q)
            
        Returns:
            Probability of touching as decimal (0-1)
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if time_to_expiry <= 0 or volatility <= 0 or current_price <= 0:
            return 0.0
        
        S = current_price
        K = strike_price
        T = time_to_expiry
        sigma = volatility
        r = risk_free_rate
        q = dividend_yield
        
        sqrt_T = np.sqrt(T)
        
        # Correct POT formula for barrier options
        # For a down-and-in barrier (PUT strikes below current price):
        # POT = 2 * N(-d) where d = (ln(S/K) + (r - q - σ²/2)T) / (σ√T)
        
        # If strike is above current price, POT = 1 (already touched)
        if K >= S:
            return 1.0
        
        # Calculate d parameter
        d = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        
        # POT for down barrier (PUT strikes below current price)
        pot = 2 * stats.norm.cdf(-d)
        
        # Alternative formula check (should give similar results):
        # mu = r - q - 0.5 * sigma**2
        # lambda_val = 1 + (2 * mu) / (sigma**2)
        # pot_alt = stats.norm.cdf(-d) + (K/S)**lambda_val * stats.norm.cdf(-d + 2*np.log(K/S)/(sigma*sqrt_T))
        
        return max(0.0, min(1.0, pot))  # Bound between 0 and 1
    
    def find_strike_for_target_pot(self,
                                 current_price: float,
                                 target_pot: float,
                                 time_to_expiry: float,
                                 volatility: float,
                                 risk_free_rate: float = None,
                                 dividend_yield: float = 0.0,
                                 search_range: float = 0.5) -> Optional[float]:
        """
        Find strike price that achieves target POT using binary search
        
        Args:
            current_price: Current stock price
            target_pot: Target probability of touching (0-1)
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            search_range: Search range as fraction of current price
            
        Returns:
            Strike price that achieves target POT, or None if not found
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Validate inputs
        if target_pot <= 0 or target_pot >= 1:
            return None
        
        # Search below current price for put strikes
        # Lower strikes = lower POT, Higher strikes = higher POT
        low_strike = current_price * (1 - search_range)  # Far below = low POT
        high_strike = current_price * 0.995  # Close below = high POT
        
        tolerance = 0.005  # 0.5% tolerance
        max_iterations = 100
        
        # Test boundaries first
        low_pot = self.probability_of_touching(current_price, low_strike, time_to_expiry, volatility, risk_free_rate, dividend_yield)
        high_pot = self.probability_of_touching(current_price, high_strike, time_to_expiry, volatility, risk_free_rate, dividend_yield)
        
        # If target is outside bounds, extend search
        if target_pot < low_pot:
            low_strike = current_price * (1 - 0.8)  # Search further
        elif target_pot > high_pot:
            high_strike = current_price * 0.999  # Search closer
        
        for iteration in range(max_iterations):
            mid_strike = (low_strike + high_strike) / 2
            
            pot = self.probability_of_touching(
                current_price, mid_strike, time_to_expiry, 
                volatility, risk_free_rate, dividend_yield
            )
            
            if abs(pot - target_pot) < tolerance:
                return mid_strike
            
            # Binary search logic: 
            # If calculated POT > target POT, we need LOWER strike (further from current)
            # If calculated POT < target POT, we need HIGHER strike (closer to current)
            if pot > target_pot:
                high_strike = mid_strike
            else:
                low_strike = mid_strike
        
        # Return best approximation
        final_strike = (low_strike + high_strike) / 2
        return final_strike
    
    def analyze_put_spread_scenarios(self,
                                   ticker: str,
                                   current_price: float,
                                   expiry_date: str,
                                   volatility: float = None,
                                   atr: float = None,
                                   target_pot_levels: List[float] = None) -> Dict[str, Any]:
        """
        Comprehensive put spread analysis for various POT levels
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            expiry_date: Options expiry date
            volatility: Implied volatility (if None, will estimate from ATR)
            atr: Average True Range for volatility estimation
            target_pot_levels: List of target POT levels to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if target_pot_levels is None:
            target_pot_levels = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.0025]
        
        # Calculate time to expiry
        time_to_expiry = self.calculate_time_to_expiry(expiry_date)
        
        # Ensure minimum time for meaningful analysis (at least 2 hours)
        min_time_hours = 2 / (365.25 * 24)  # 2 hours in years
        time_to_expiry = max(time_to_expiry, min_time_hours)
        
        # Get dividend yield
        dividend_yield = self.get_dividend_yield(ticker)
        
        # Determine volatility with better defaults
        if volatility is None or volatility <= 0:
            if atr and atr > 0:
                volatility = self.calculate_implied_volatility_proxy(current_price, atr, time_to_expiry)
            else:
                # Use smarter default based on ticker
                if ticker.upper() in ['SPY', 'QQQ', 'IWM', 'DIA']:
                    volatility = 0.18  # ETFs typically ~18%
                elif ticker.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                    volatility = 0.25  # Large cap tech ~25%
                else:
                    volatility = 0.30  # Default 30%
        
        # Try to fetch options chain for real implied volatility
        options_data = self.fetch_options_chain(ticker, expiry_date)
        if options_data is not None and not options_data.empty:
            # Use median IV from available options, but filter reasonable values
            filtered_iv = options_data[
                (options_data['impliedVolatility'] > 0.05) & 
                (options_data['impliedVolatility'] < 2.0)
            ]['impliedVolatility']
            
            if len(filtered_iv) > 0:
                median_iv = filtered_iv.median()
                volatility = median_iv
        
        # Analyze each POT level
        results = {
            'ticker': ticker,
            'current_price': current_price,
            'expiry_date': expiry_date,
            'time_to_expiry_days': time_to_expiry * 365.25,
            'time_to_expiry_years': time_to_expiry,
            'volatility': volatility,
            'dividend_yield': dividend_yield,
            'risk_free_rate': self.risk_free_rate,
            'scenarios': [],
            'options_data_available': options_data is not None
        }
        
        # Sort target POT levels from lowest to highest (safest to riskiest)
        target_pot_levels_sorted = sorted(target_pot_levels)
        
        for target_pot in target_pot_levels_sorted:
            # Find optimal short strike for target POT
            short_strike = self.find_strike_for_target_pot(
                current_price, target_pot, time_to_expiry, 
                volatility, self.risk_free_rate, dividend_yield
            )
            
            if short_strike is not None and short_strike > 0:
                # Calculate actual POT for verification
                actual_pot = self.probability_of_touching(
                    current_price, short_strike, time_to_expiry,
                    volatility, self.risk_free_rate, dividend_yield
                )
                
                # Analyze different spread widths (scaled based on current price)
                base_width = max(1, current_price * 0.005)  # 0.5% of stock price as base
                spread_widths = [base_width * i for i in [1, 2, 3, 4, 5]]  # Progressive widths
                spread_scenarios = []
                
                for width in spread_widths:
                    long_strike = short_strike - width
                    
                    if long_strike > 0:
                        prob_profit = self.probability_of_profit_spread(
                            current_price, long_strike, short_strike,
                            time_to_expiry, volatility, 
                            self.risk_free_rate, dividend_yield
                        )
                        
                        spread_scenarios.append({
                            'width': width,
                            'long_strike': long_strike,
                            'short_strike': short_strike,
                            'prob_profit': prob_profit,
                            'max_profit': width,  # Max profit = width for credit spreads
                            'max_loss': 0,  # Simplified - actual calculation would be more complex
                            'distance_from_current': current_price - short_strike,
                            'distance_pct': ((current_price - short_strike) / current_price) * 100
                        })
                
                scenario = {
                    'target_pot': target_pot,
                    'target_pot_pct': target_pot * 100,
                    'short_strike': short_strike,
                    'actual_pot': actual_pot,
                    'actual_pot_pct': actual_pot * 100,
                    'distance_from_current': current_price - short_strike,
                    'distance_pct': ((current_price - short_strike) / current_price) * 100,
                    'spreads': spread_scenarios
                }
                
                results['scenarios'].append(scenario)
        
        return results

# Utility functions for Streamlit integration
def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage string"""
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string"""
    return f"${value:.{decimals}f}"

def get_next_friday() -> str:
    """Get next Friday's date for weekly options"""
    today = date.today()
    days_ahead = 4 - today.weekday()  # Friday is 4
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    next_friday = today + timedelta(days_ahead)
    return next_friday.strftime('%Y-%m-%d')

def get_same_day_expiry() -> str:
    """Get today's date for same-day expiry"""
    return date.today().strftime('%Y-%m-%d') 