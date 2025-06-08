"""
Iron Condor Trading Analysis Module - Enhanced with Technical Reference

This module provides comprehensive Iron Condor strategy analysis with real-time options data,
probability calculations, time decay simulation, exit rule analysis, and strategy variations 
based on the Iron Condor Trading Playbook.

Features:
- Real-time options data from yfinance
- Black-Scholes probability calculations with Greeks
- Time decay simulation and profit/loss modeling
- Exit rule analysis (21 DTE vs Hold to Expiry)
- Comprehensive technical metrics (Theta, Gamma, ROC, POPrem)
- Multiple strategy variations (Bread & Butter, Skewed, Chicken IC, etc.)
- Trade management recommendations with decision analysis

Author: AI Assistant
Date: 2025
Version: 2.0 - Enhanced with Technical Reference and Simulation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from scipy import stats
import warnings
import math

# Suppress warnings
warnings.filterwarnings('ignore')

class IronCondorAnalyzer:
    """Comprehensive Iron Condor analysis with real-time options data and simulation"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # Default 5% risk-free rate
        
    def get_options_data(self, ticker, expiry_date=None):
        """Fetch options data from yfinance with enhanced error handling"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get current stock data
            hist = stock.history(period='1d')
            if hist.empty:
                print(f"No stock data available for {ticker}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Get expiry dates
            try:
                options_dates = stock.options
                if not options_dates:
                    print(f"No options available for {ticker}")
                    return None
            except Exception as e:
                print(f"Error getting options dates for {ticker}: {e}")
                return None
            
            # Select expiry date
            if expiry_date:
                # Use provided expiry date if available in options chain
                if expiry_date in options_dates:
                    selected_expiry = expiry_date
                else:
                    print(f"Expiry {expiry_date} not available. Using closest available.")
                    selected_expiry = min(options_dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(expiry_date, '%Y-%m-%d')).days))
            else:
                # Default to first available expiry (usually shortest term)
                selected_expiry = options_dates[0]
            
            # Fetch options chain for selected expiry
            try:
                options_chain = stock.option_chain(selected_expiry)
                calls_data = options_chain.calls
                puts_data = options_chain.puts
                
                if calls_data.empty or puts_data.empty:
                    print(f"No options data for {ticker} on {selected_expiry}")
                    return None
                
                print(f"Successfully fetched options for {ticker} expiry {selected_expiry}")
                print(f"Calls: {len(calls_data)}, Puts: {len(puts_data)}")
                
                return {
                    'calls': calls_data,
                    'puts': puts_data,
                    'expiry': selected_expiry,
                    'current_price': current_price
                }
                
            except Exception as e:
                print(f"Error fetching options chain for {ticker}: {e}")
                return None
                
        except Exception as e:
            print(f"Error in get_options_data for {ticker}: {e}")
            return None

    def get_option_premium(self, options_df, strike_price, tolerance=0.5):
        """Get option premium for a specific strike price"""
        try:
            # Find exact match first
            exact_match = options_df[options_df['strike'] == strike_price]
            
            if not exact_match.empty:
                # Use mid price if bid/ask available, otherwise use lastPrice
                row = exact_match.iloc[0]
                if 'bid' in row and 'ask' in row and pd.notna(row['bid']) and pd.notna(row['ask']) and row['bid'] > 0:
                    return (row['bid'] + row['ask']) / 2
                elif 'lastPrice' in row and pd.notna(row['lastPrice']):
                    return row['lastPrice']
                else:
                    return None
            
            # If no exact match, find closest within tolerance
            else:
                options_df['strike_diff'] = abs(options_df['strike'] - strike_price)
                closest = options_df[options_df['strike_diff'] <= tolerance]
                
                if not closest.empty:
                    closest = closest.sort_values('strike_diff').iloc[0]
                    if 'bid' in closest and 'ask' in closest and pd.notna(closest['bid']) and pd.notna(closest['ask']) and closest['bid'] > 0:
                        return (closest['bid'] + closest['ask']) / 2
                    elif 'lastPrice' in closest and pd.notna(closest['lastPrice']):
                        return closest['lastPrice']
                
                return None
                
        except Exception as e:
            print(f"Error getting option premium for strike {strike_price}: {e}")
            return None
    
    def get_iv_rank(self, ticker, lookback_days=252):
        """Calculate Implied Volatility Rank approximation using historical volatility"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{lookback_days + 30}d")
            
            if hist.empty:
                return None
            
            # Calculate historical volatility
            returns = hist['Close'].pct_change().dropna()
            hist_vol = returns.rolling(window=30).std() * np.sqrt(252)
            
            if len(hist_vol) < 2:
                return None
            
            current_vol = hist_vol.iloc[-1]
            vol_percentile = (hist_vol <= current_vol).mean() * 100
            
            return vol_percentile
            
        except Exception as e:
            print(f"Error calculating IV rank for {ticker}: {e}")
            return None
    
    def calculate_greeks(self, strike, current_price, time_to_expiry, volatility, option_type='call'):
        """Calculate comprehensive option Greeks using Black-Scholes"""
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            # Calculate d1 and d2
            d1 = (np.log(current_price / strike) + 
                  (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                 (volatility * np.sqrt(time_to_expiry))
            
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Standard normal PDF and CDF
            n_d1 = stats.norm.pdf(d1)
            n_d2 = stats.norm.pdf(d2)
            N_d1 = stats.norm.cdf(d1)
            N_d2 = stats.norm.cdf(d2)
            
            # Delta calculation
            if option_type == 'call':
                delta = N_d1
            else:  # put
                delta = N_d1 - 1
            
            # Gamma (same for calls and puts)
            gamma = n_d1 / (current_price * volatility * np.sqrt(time_to_expiry))
            
            # Theta calculation
            if option_type == 'call':
                theta = (-current_price * n_d1 * volatility / (2 * np.sqrt(time_to_expiry)) - 
                        self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * N_d2) / 365
            else:  # put
                theta = (-current_price * n_d1 * volatility / (2 * np.sqrt(time_to_expiry)) + 
                        self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * (1 - N_d2)) / 365
            
            # Vega (same for calls and puts)
            vega = current_price * n_d1 * np.sqrt(time_to_expiry) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def calculate_delta_approximation(self, strike, current_price, time_to_expiry, volatility, option_type='call'):
        """Calculate delta using Black-Scholes approximation"""
        greeks = self.calculate_greeks(strike, current_price, time_to_expiry, volatility, option_type)
        return abs(greeks['delta'])
    
    def calculate_pop_delta_method(self, call_delta, put_delta):
        """Calculate Probability of Profit using delta method"""
        return 1 - (abs(call_delta) + abs(put_delta))
    
    def calculate_pop_credit_width_method(self, credit, wing_width):
        """Calculate Probability of Profit using credit/width method"""
        if wing_width == 0:
            return 0
        return 1 - (credit / wing_width)
    
    def calculate_pop_black_scholes(self, current_price, call_strike, put_strike, 
                                  time_to_expiry, volatility, credit, wing_width):
        """Calculate POP using Black-Scholes model - FIXED VERSION"""
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return 0.0
            
            # Calculate breakeven points
            lower_breakeven = put_strike - credit
            upper_breakeven = call_strike + credit
            
            # Ensure logical breakeven points
            if lower_breakeven >= upper_breakeven:
                return 0.0
            
            # Black-Scholes probability calculation - CORRECTED FORMULA
            # We want P(lower_breakeven < S_T < upper_breakeven)
            
            # Calculate d2 values (use d2 for terminal stock price probabilities)
            sqrt_t = np.sqrt(time_to_expiry)
            vol_sqrt_t = volatility * sqrt_t
            
            # For lower breakeven
            d2_lower = (np.log(current_price / lower_breakeven) + 
                       (self.risk_free_rate - 0.5 * volatility**2) * time_to_expiry) / vol_sqrt_t
            
            # For upper breakeven  
            d2_upper = (np.log(current_price / upper_breakeven) + 
                       (self.risk_free_rate - 0.5 * volatility**2) * time_to_expiry) / vol_sqrt_t
            
            # Probability of finishing above lower breakeven
            prob_above_lower = stats.norm.cdf(d2_lower)
            
            # Probability of finishing above upper breakeven
            prob_above_upper = stats.norm.cdf(d2_upper)
            
            # POP = P(above lower) - P(above upper) = P(between breakevens)
            pop = prob_above_lower - prob_above_upper
            
            # Ensure result is between 0 and 1
            pop = max(0, min(1, pop))
            
            # Debug output
            if pop == 0:
                print(f"DEBUG POP=0: S={current_price:.2f}, BE_L={lower_breakeven:.2f}, BE_U={upper_breakeven:.2f}")
                print(f"DEBUG: T={time_to_expiry:.4f}, Vol={volatility:.4f}, d2_L={d2_lower:.4f}, d2_U={d2_upper:.4f}")
                print(f"DEBUG: P(>L)={prob_above_lower:.4f}, P(>U)={prob_above_upper:.4f}")
            
            return pop
            
        except Exception as e:
            print(f"Error in Black-Scholes POP calculation: {e}")
            return 0.0
    
    def calculate_pop_remaining(self, current_price, call_strike, put_strike, 
                               time_to_expiry, volatility, current_dte):
        """Calculate POPrem - Probability of Profit remaining from current time to expiry"""
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return 0
            
            # Calculate current Greeks for short strikes
            call_greeks = self.calculate_greeks(call_strike, current_price, time_to_expiry, volatility, 'call')
            put_greeks = self.calculate_greeks(put_strike, current_price, time_to_expiry, volatility, 'put')
            
            # POPrem approximation using delta method
            pop_rem = 1 - (abs(call_greeks['delta']) + abs(put_greeks['delta']))
            
            return max(0, min(1, pop_rem))
            
        except Exception as e:
            return 0
    
    def simulate_time_decay(self, strategy, current_price, price_scenarios=None):
        """Simulate Iron Condor time decay over multiple price scenarios"""
        try:
            if price_scenarios is None:
                price_scenarios = [-0.15, -0.05, 0.0, 0.05, 0.15]  # Price change percentages
            
            dte = strategy.get('dte', 30)
            wing_width = strategy.get('wing_width', 5)
            total_credit = strategy.get('total_credit', 0)
            call_short = strategy['call_short']
            call_long = strategy['call_long']
            put_short = strategy['put_short']
            put_long = strategy['put_long']
            
            simulation_data = []
            
            for scenario_pct in price_scenarios:
                scenario_price = current_price * (1 + scenario_pct)
                scenario_label = f"{scenario_pct:+.1%}"
                
                for day in range(dte + 1):
                    current_dte = dte - day
                    
                    # Calculate P&L at this point in time
                    profit_loss = self._calculate_scenario_pnl(
                        scenario_price, call_short, call_long, put_short, put_long, 
                        total_credit, current_dte
                    )
                    
                    # Estimate Greeks at this time point
                    theta_total = self._estimate_theta(current_dte, total_credit)
                    gamma_total = self._estimate_gamma(current_dte, scenario_price, call_short, put_short)
                    
                    # Calculate POPrem (remaining probability)
                    pop_remaining = self._estimate_pop_remaining(current_dte, scenario_price, call_short, put_short)
                    
                    # Exit rule evaluation
                    should_close_21_dte = current_dte <= 21 or profit_loss >= (total_credit * 100 * 0.5)
                    
                    simulation_data.append({
                        'day': day,
                        'dte': current_dte,
                        'price_scenario': scenario_label,
                        'scenario_price': scenario_price,
                        'profit_loss': profit_loss,
                        'theta_total': theta_total,
                        'gamma_total': gamma_total,
                        'pop_remaining': pop_remaining,
                        'should_close_21_dte': should_close_21_dte
                    })
            
            return pd.DataFrame(simulation_data)
            
        except Exception as e:
            print(f"Error in time decay simulation: {e}")
            return pd.DataFrame()
    
    def _calculate_scenario_pnl(self, price, call_short, call_long, put_short, put_long, credit, dte):
        """Calculate P&L for a specific price and time scenario"""
        try:
            # Simplified P&L calculation based on intrinsic value and time decay
            time_factor = max(0, dte / 30.0)  # Time value factor (0-1)
            
            # Call spread P&L
            if price <= call_short:
                call_pnl = credit / 2  # Keep call credit
            elif price <= call_long:
                call_pnl = (credit / 2) - (price - call_short)
            else:
                call_pnl = (credit / 2) - (call_long - call_short)
            
            # Put spread P&L
            if price >= put_short:
                put_pnl = credit / 2  # Keep put credit
            elif price >= put_long:
                put_pnl = (credit / 2) - (put_short - price)
            else:
                put_pnl = (credit / 2) - (put_short - put_long)
            
            # Add time decay benefit
            time_decay_benefit = credit * 0.5 * (1 - time_factor)
            
            total_pnl = (call_pnl + put_pnl + time_decay_benefit) * 100
            
            return total_pnl
            
        except Exception:
            return 0
    
    def _estimate_theta(self, dte, credit):
        """Estimate theta (time decay) for current DTE"""
        try:
            if dte <= 0:
                return 0
            
            # Theta increases as expiration approaches
            theta_factor = 1 / max(1, dte / 7)  # Accelerates in final week
            base_theta = credit * 0.1  # Base daily theta
            
            return base_theta * theta_factor
            
        except Exception:
            return 0
    
    def _estimate_gamma(self, dte, price, call_short, put_short):
        """Estimate gamma risk for current scenario"""
        try:
            if dte <= 0:
                return 0
            
            # Gamma risk highest when price near short strikes
            distance_to_call = abs(price - call_short)
            distance_to_put = abs(price - put_short)
            min_distance = min(distance_to_call, distance_to_put)
            
            # Gamma explodes as DTE approaches 0 and price near strikes
            gamma_risk = 0.1 / (max(1, dte / 7) * max(1, min_distance / 10))
            
            return min(gamma_risk, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0
    
    def _estimate_pop_remaining(self, dte, price, call_short, put_short):
        """Estimate remaining probability of profit"""
        try:
            if dte <= 0:
                return 0 if price < put_short or price > call_short else 1
            
            # Distance from short strikes
            distance_to_call = abs(price - call_short)
            distance_to_put = abs(price - put_short)
            
            # POP decreases as we get closer to strikes and expiration
            base_pop = 0.7  # Starting POP
            strike_factor = min(distance_to_call, distance_to_put) / 20  # Distance factor
            time_factor = dte / 30.0  # Time factor
            
            pop_remaining = base_pop * strike_factor * time_factor
            
            return max(0, min(1, pop_remaining))
            
        except Exception:
            return 0.5
    
    def calculate_technical_metrics(self, strategy, current_price):
        """Calculate comprehensive technical metrics for Iron Condor strategy"""
        try:
            dte = strategy.get('dte', 30)
            wing_width = strategy.get('wing_width', 5)
            total_credit = strategy.get('total_credit', 0)
            max_profit = strategy.get('max_profit', 0)
            max_loss = strategy.get('max_loss', 0)
            
            # Basic financial metrics
            credit_per_share = total_credit
            max_profit_dollars = max_profit * 100
            max_loss_dollars = max_loss * 100
            remaining_reward = max_profit_dollars  # At entry, full reward remains
            margin_required = max_loss_dollars
            roc_percent = (max_profit_dollars / margin_required * 100) if margin_required > 0 else 0
            
            # Greeks approximation
            net_theta = strategy.get('net_theta', -0.05)  # Positive for IC (time decay benefit)
            net_gamma = strategy.get('net_gamma', 0.02)   # Gamma risk
            net_vega = strategy.get('net_vega', -0.1)     # Negative vega (volatility risk)
            
            # Time decay estimates
            theta_decay_daily = abs(net_theta)
            days_to_50_pct_est = (max_profit_dollars * 0.5) / theta_decay_daily if theta_decay_daily > 0 else 30
            
            # Efficiency metrics
            credit_to_width_ratio = total_credit / wing_width if wing_width > 0 else 0
            gamma_risk = abs(net_gamma)
            
            # Profit zone metrics
            breakeven_lower = strategy.get('lower_breakeven', current_price - wing_width)
            breakeven_upper = strategy.get('upper_breakeven', current_price + wing_width)
            profit_zone_width = breakeven_upper - breakeven_lower
            
            # POPrem calculation (simplified)
            pop_remaining = strategy.get('pop_black_scholes', 0.7)  # At entry, equals full POP
            
            return {
                'dte': dte,
                'wing_width': wing_width,
                'credit_per_share': credit_per_share,
                'max_profit_dollars': max_profit_dollars,
                'max_loss_dollars': max_loss_dollars,
                'remaining_reward': remaining_reward,
                'margin_required': margin_required,
                'roc_percent': roc_percent,
                'net_theta': net_theta,
                'net_gamma': net_gamma,
                'net_vega': net_vega,
                'theta_decay_daily': theta_decay_daily,
                'days_to_50_pct_est': days_to_50_pct_est,
                'credit_to_width_ratio': credit_to_width_ratio,
                'gamma_risk': gamma_risk,
                'breakeven_lower': breakeven_lower,
                'breakeven_upper': breakeven_upper,
                'profit_zone_width': profit_zone_width,
                'pop_remaining': pop_remaining
            }
            
        except Exception as e:
            print(f"Error calculating technical metrics: {e}")
            return {}
    
    def find_target_strikes(self, current_price, calls, puts, target_delta=0.20, 
                           dte=None, volatility=None):
        """Find suitable strike prices for Iron Condor based on target delta"""
        try:
            if calls.empty or puts.empty:
                return None
            
            # Filter options near the money for better delta estimates
            price_range = current_price * 0.20  # ±20% from current price
            
            # Filter calls (OTM calls for short strikes)
            otm_calls = calls[
                (calls['strike'] > current_price) & 
                (calls['strike'] <= current_price + price_range)
            ].copy()
            
            # Filter puts (OTM puts for short strikes)
            otm_puts = puts[
                (puts['strike'] < current_price) & 
                (puts['strike'] >= current_price - price_range)
            ].copy()
            
            if otm_calls.empty or otm_puts.empty:
                print("No suitable OTM options found")
                return None
            
            # Calculate deltas for available strikes
            time_to_expiry = max((dte or 30) / 365.25, 1/365.25)
            vol = volatility or 0.20
            
            # Calculate deltas for calls
            call_deltas = []
            for _, row in otm_calls.iterrows():
                delta = self.calculate_delta_approximation(
                    row['strike'], current_price, time_to_expiry, vol, 'call'
                )
                call_deltas.append({
                    'strike': row['strike'],
                    'delta': abs(delta),
                    'premium': row.get('lastPrice', 0)
                })
            
            # Calculate deltas for puts
            put_deltas = []
            for _, row in otm_puts.iterrows():
                delta = self.calculate_delta_approximation(
                    row['strike'], current_price, time_to_expiry, vol, 'put'
                )
                put_deltas.append({
                    'strike': row['strike'],
                    'delta': abs(delta),
                    'premium': row.get('lastPrice', 0)
                })
            
            if not call_deltas or not put_deltas:
                print("Could not calculate deltas for options")
                return None
            
            # Find strikes closest to target delta
            call_deltas.sort(key=lambda x: abs(x['delta'] - target_delta))
            put_deltas.sort(key=lambda x: abs(x['delta'] - target_delta))
            
            # Return the best strike for each side
            best_call_strike = call_deltas[0]['strike']
            best_put_strike = put_deltas[0]['strike']
            
            print(f"Target delta {target_delta:.2f}: Call strike ${best_call_strike:.2f} (Δ={call_deltas[0]['delta']:.3f}), Put strike ${best_put_strike:.2f} (Δ={put_deltas[0]['delta']:.3f})")
            
            return {
                'call_short': best_call_strike,
                'put_short': best_put_strike,
                'call_delta': call_deltas[0]['delta'],
                'put_delta': put_deltas[0]['delta']
            }
            
        except Exception as e:
            print(f"Error finding target strikes: {e}")
            return None
    
    def calculate_iron_condor_metrics(self, current_price, call_short, call_long, 
                                    put_short, put_long, call_credit, put_credit, 
                                    time_to_expiry, volatility):
        """Calculate comprehensive Iron Condor metrics"""
        try:
            # Basic metrics
            call_width = call_long - call_short
            put_width = put_short - put_long
            total_credit = call_credit + put_credit
            
            # Max profit/loss
            max_profit = total_credit
            max_loss_call = call_width - call_credit
            max_loss_put = put_width - put_credit
            max_loss = max(max_loss_call, max_loss_put)
            
            # Breakevens
            upper_breakeven = call_short + total_credit
            lower_breakeven = put_short - total_credit
            
            # Profit zone
            profit_zone_width = upper_breakeven - lower_breakeven
            profit_zone_pct = (profit_zone_width / current_price) * 100
            
            # Distance metrics
            call_distance = ((call_short - current_price) / current_price) * 100
            put_distance = ((current_price - put_short) / current_price) * 100
            
            # Calculate deltas for POP estimation
            call_delta = self.calculate_delta_approximation(
                call_short, current_price, time_to_expiry, volatility, 'call'
            )
            put_delta = self.calculate_delta_approximation(
                put_short, current_price, time_to_expiry, volatility, 'put'
            )
            
            # POP calculations
            pop_delta = self.calculate_pop_delta_method(call_delta, put_delta)
            pop_credit = self.calculate_pop_credit_width_method(total_credit, 
                                                              max(call_width, put_width))
            pop_bs = self.calculate_pop_black_scholes(
                current_price, call_short, put_short, time_to_expiry, 
                volatility, total_credit, max(call_width, put_width)
            )
            
            # Risk/reward metrics
            risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0
            return_on_risk = (max_profit / max_loss) * 100 if max_loss > 0 else 0
            
            return {
                'max_profit': max_profit,
                'max_loss': max_loss,
                'total_credit': total_credit,
                'call_width': call_width,
                'put_width': put_width,
                'upper_breakeven': upper_breakeven,
                'lower_breakeven': lower_breakeven,
                'profit_zone_width': profit_zone_width,
                'profit_zone_pct': profit_zone_pct,
                'call_distance_pct': call_distance,
                'put_distance_pct': put_distance,
                'call_delta': call_delta,
                'put_delta': put_delta,
                'pop_delta_method': pop_delta,
                'pop_credit_method': pop_credit,
                'pop_black_scholes': pop_bs,
                'risk_reward_ratio': risk_reward_ratio,
                'return_on_risk': return_on_risk,
                'call_short': call_short,
                'call_long': call_long,
                'put_short': put_short,
                'put_long': put_long
            }
            
        except Exception as e:
            print(f"Error calculating Iron Condor metrics: {e}")
            return None
    
    def analyze_iron_condor_strategies(self, ticker, expiry_date=None, 
                                     wing_widths=[2.5, 5, 10], target_deltas=[0.15, 0.20, 0.25]):
        """Analyze Iron Condor strategies with credit requirements and optimization"""
        try:
            # Get options data
            options_data = self.get_options_data(ticker, expiry_date)
            if not options_data:
                return None
            
            calls = options_data['calls']
            puts = options_data['puts']
            expiry = options_data['expiry']
            
            # Get current price and market data
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            # Calculate time to expiry
            expiry_dt = datetime.strptime(expiry, '%Y-%m-%d')
            dte = (expiry_dt.date() - datetime.now().date()).days
            time_to_expiry = max(dte / 365.25, 1/365.25)  # Minimum 1 day
            
            # Get volatility data
            iv_rank = self.get_iv_rank(ticker)
            volatility = self.estimate_volatility(ticker)
            
            print(f"Analyzing {ticker}: Price=${current_price:.2f}, DTE={dte}, Vol={volatility:.3f}")
            
            strategies = []
            
            # Analyze each combination of wing width and delta
            for wing_width in wing_widths:
                for target_delta in target_deltas:
                    
                    # Find target strikes
                    strikes = self.find_target_strikes(
                        current_price, calls, puts, target_delta, dte, volatility
                    )
                    
                    if not strikes:
                        continue
                    
                    call_short_strike = strikes['call_short']
                    put_short_strike = strikes['put_short']
                    
                    # Calculate long strikes based on wing width
                    call_long_strike = call_short_strike + wing_width
                    put_long_strike = put_short_strike - wing_width
                    
                    # Get option premiums
                    call_short_premium = self.get_option_premium(calls, call_short_strike)
                    call_long_premium = self.get_option_premium(calls, call_long_strike)
                    put_short_premium = self.get_option_premium(puts, put_short_strike)
                    put_long_premium = self.get_option_premium(puts, put_long_strike)
                    
                    if any(p is None for p in [call_short_premium, call_long_premium, 
                                             put_short_premium, put_long_premium]):
                        continue
                    
                    # Calculate credits
                    call_credit = call_short_premium - call_long_premium
                    put_credit = put_short_premium - put_long_premium
                    total_credit = call_credit + put_credit
                    
                    # Skip if negative credit
                    if total_credit <= 0 or call_credit <= 0 or put_credit <= 0:
                        continue
                    
                    # Classify strategy type
                    metrics_temp = {
                        'max_profit': total_credit * 100,
                        'max_loss': (wing_width - total_credit) * 100,
                        'credit_to_width_ratio': total_credit / wing_width,
                        'wing_width': wing_width,
                        'dte': dte
                    }
                    
                    strategy_type = self.classify_strategy(metrics_temp, wing_width, target_delta, dte)
                    
                    # CHECK CREDIT REQUIREMENTS
                    credit_check = self.meets_credit_requirements(total_credit, wing_width, strategy_type)
                    
                    # Filter based on strategy type
                    if strategy_type == "Bread & Butter":
                        # Strict credit requirements for Bread & Butter
                        if not credit_check['meets_requirement']:
                            print(f"FILTERED OUT {strategy_type}: {credit_check['message']}")
                            continue
                    
                    # Calculate comprehensive metrics
                    metrics = self.calculate_iron_condor_metrics(
                        current_price, call_short_strike, call_long_strike,
                        put_short_strike, put_long_strike, call_credit, put_credit,
                        time_to_expiry, volatility
                    )
                    
                    if not metrics:
                        continue
                    
                    # Add credit requirement info to metrics
                    metrics.update({
                        'credit_check': credit_check,
                        'strategy_type': strategy_type,
                        'target_delta': target_delta,
                        'wing_width': wing_width,
                        'call_short': call_short_strike,
                        'call_long': call_long_strike,
                        'put_short': put_short_strike,
                        'put_long': put_long_strike,
                        'call_credit': call_credit,
                        'put_credit': put_credit,
                        'total_credit': total_credit,
                        'dte': dte,
                        'expiry_date': expiry
                    })
                    
                    strategies.append(metrics)
            
            if not strategies:
                print("No strategies met the credit requirements")
                return None
            
            # SORT AND OPTIMIZE BASED ON STRATEGY TYPE
            bread_butter_strategies = [s for s in strategies if s['strategy_type'] == "Bread & Butter"]
            other_strategies = [s for s in strategies if s['strategy_type'] != "Bread & Butter"]
            
            # Sort Bread & Butter by credit efficiency (highest credit/width ratio first)
            bread_butter_strategies.sort(key=lambda x: x['credit_check']['credit_ratio'], reverse=True)
            
            # Sort other strategies by combined R/R and POP score
            for strategy in other_strategies:
                # Calculate combined score: 70% POP + 30% R/R
                pop_score = strategy.get('pop_black_scholes', 0) * 0.7
                rr_score = min(strategy.get('risk_reward_ratio', 0) / 1.0, 1.0) * 0.3  # Normalize R/R
                strategy['combined_score'] = pop_score + rr_score
            
            other_strategies.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            # Combine results with Bread & Butter first
            final_strategies = bread_butter_strategies + other_strategies
            
            print(f"Found {len(final_strategies)} strategies ({len(bread_butter_strategies)} Bread & Butter, {len(other_strategies)} Others)")
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'expiry_date': expiry,
                'dte': dte,
                'time_to_expiry': time_to_expiry,
                'volatility': volatility,
                'iv_rank': iv_rank,
                'strategies': final_strategies,
                'bread_butter_count': len(bread_butter_strategies),
                'other_strategies_count': len(other_strategies)
            }
            
        except Exception as e:
            print(f"Error analyzing Iron Condor strategies: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def classify_strategy(self, metrics, wing_width, target_delta, dte):
        """Classify strategy type based on playbook rules"""
        credit_width_ratio = metrics['total_credit'] / wing_width
        
        if credit_width_ratio >= 0.33:
            if wing_width <= 5:
                return "Bread & Butter"
            elif wing_width > 10:
                return "Big Boy"
            else:
                return "Standard"
        elif credit_width_ratio >= 0.25:
            if dte <= 14:
                return "Chicken IC"
            else:
                return "Conservative"
        else:
            return "Low Credit"
    
    def estimate_volatility(self, ticker, lookback_days=30):
        """Estimate volatility using historical data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{lookback_days + 10}d")
            
            if hist.empty:
                return 0.25  # Default 25% volatility
            
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            return max(0.1, min(2.0, volatility))  # Cap between 10% and 200%
            
        except Exception:
            return 0.25
    
    def create_strategy_comparison_dataframe(self, analysis_results):
        """Create a comprehensive comparison DataFrame with enhanced metrics"""
        if not analysis_results or not analysis_results.get('strategies'):
            return pd.DataFrame()
        
        strategies = analysis_results['strategies']
        
        data = []
        for i, strategy in enumerate(strategies):
            data.append({
                'Rank': i + 1,
                'Strategy Type': strategy['strategy_type'],
                'Wing Width': f"${strategy['wing_width']:.1f}",
                'Call Structure': f"{strategy['call_short']:.0f}/{strategy['call_long']:.0f}",
                'Put Structure': f"{strategy['put_long']:.0f}/{strategy['put_short']:.0f}",
                'Credit': f"${strategy['total_credit']:.2f}",
                'Max Profit': f"${strategy['max_profit']:.2f}",
                'Max Loss': f"${strategy['max_loss']:.2f}",
                'DTE': f"{strategy['dte']} days",
                'POP (B-S)': f"{strategy['pop_black_scholes']:.1%}",
                'ROC': f"{strategy.get('roc_percent', 0):.1f}%",
                'Theta/Day': f"${strategy.get('net_theta', 0):.2f}",
                'Profit Zone': f"{strategy['profit_zone_pct']:.1f}%",
                'Breakevens': f"${strategy['lower_breakeven']:.2f}-${strategy['upper_breakeven']:.2f}"
            })
        
        return pd.DataFrame(data)

    def analyze_exit_strategies(self, strategy, current_price):
        """Analyze exit strategies (Rule A vs Rule B) with comprehensive comparison"""
        try:
            # Run simulation for both exit strategies
            simulation_df = self.simulate_time_decay(strategy, current_price)
            
            if simulation_df.empty:
                return None
            
            # Rule A: Exit at 21 DTE or 50% profit
            rule_a_scenarios = []
            for scenario in simulation_df['price_scenario'].unique():
                scenario_data = simulation_df[simulation_df['price_scenario'] == scenario]
                
                # Find exit point for Rule A
                exit_row = scenario_data[scenario_data['should_close_21_dte'] == True]
                if not exit_row.empty:
                    exit_data = exit_row.iloc[0]
                    rule_a_scenarios.append({
                        'price_scenario': scenario,
                        'exit_day': exit_data['day'],
                        'profit_loss': exit_data['profit_loss'],
                        'pop_remaining': exit_data['pop_remaining'],
                        'profit_captured_pct': (exit_data['profit_loss'] / (strategy.get('max_profit', 1) * 100)) * 100
                    })
            
            # Rule B: Hold to expiry
            rule_b_scenarios = []
            for scenario in simulation_df['price_scenario'].unique():
                scenario_data = simulation_df[simulation_df['price_scenario'] == scenario]
                
                # Get final expiry result
                final_row = scenario_data[scenario_data['dte'] == 0]
                if not final_row.empty:
                    final_data = final_row.iloc[0]
                    rule_b_scenarios.append({
                        'price_scenario': scenario,
                        'exit_day': strategy.get('dte', 30),
                        'profit_loss': final_data['profit_loss'],
                        'pop_remaining': 0,  # Expired
                        'profit_captured_pct': (final_data['profit_loss'] / (strategy.get('max_profit', 1) * 100)) * 100
                    })
            
            # Calculate summary statistics
            rule_a_profits = [s['profit_loss'] for s in rule_a_scenarios]
            rule_b_profits = [s['profit_loss'] for s in rule_b_scenarios]
            
            rule_a_avg_profit = np.mean(rule_a_profits) if rule_a_profits else 0
            rule_b_avg_profit = np.mean(rule_b_profits) if rule_b_profits else 0
            
            rule_a_win_rate = (len([p for p in rule_a_profits if p > 0]) / len(rule_a_profits) * 100) if rule_a_profits else 0
            rule_b_win_rate = (len([p for p in rule_b_profits if p > 0]) / len(rule_b_profits) * 100) if rule_b_profits else 0
            
            # Theta capture analysis
            max_possible_theta = strategy.get('max_profit', 1) * 100
            theta_captured_rule_a = np.mean([s['profit_captured_pct'] for s in rule_a_scenarios]) if rule_a_scenarios else 0
            
            # Risk reduction analysis
            avg_pop_remaining_a = np.mean([s['pop_remaining'] for s in rule_a_scenarios]) if rule_a_scenarios else 0
            
            return {
                'rule_a_21_dte': {
                    'scenarios': rule_a_scenarios,
                    'avg_profit': rule_a_avg_profit,
                    'win_rate': rule_a_win_rate
                },
                'rule_b_hold_expiry': {
                    'scenarios': rule_b_scenarios,
                    'avg_profit': rule_b_avg_profit,
                    'win_rate': rule_b_win_rate
                },
                'summary': {
                    'rule_a_avg_profit': rule_a_avg_profit,
                    'rule_b_avg_profit': rule_b_avg_profit,
                    'rule_a_win_rate': rule_a_win_rate,
                    'rule_b_win_rate': rule_b_win_rate,
                    'theta_captured_rule_a': theta_captured_rule_a,
                    'risk_reduction_rule_a': avg_pop_remaining_a * 100,
                    'recommendation': 'Rule A' if rule_a_avg_profit > rule_b_avg_profit else 'Rule B'
                }
            }
            
        except Exception as e:
            print(f"Error analyzing exit strategies: {e}")
            return None

    def meets_credit_requirements(self, credit, wing_width, strategy_type):
        """Check if strategy meets minimum credit requirements"""
        try:
            # Calculate credit per dollar of width
            credit_per_width = credit / wing_width if wing_width > 0 else 0
            
            # For Bread & Butter strategies (1/3rd width credit rule)
            if strategy_type == "Bread & Butter":
                # Require 33-40 cents per $1 width (0.33 - 0.40 ratio)
                min_credit_ratio = 0.33
                max_credit_ratio = 0.50  # Upper bound for practical purposes
                
                meets_requirement = min_credit_ratio <= credit_per_width <= max_credit_ratio
                
                return {
                    'meets_requirement': meets_requirement,
                    'credit_ratio': credit_per_width,
                    'min_required': min_credit_ratio,
                    'strategy_type': strategy_type,
                    'message': f"Credit/Width: {credit_per_width:.3f} (Required: {min_credit_ratio:.3f}-{max_credit_ratio:.3f})"
                }
            
            # For other strategies, optimize for R/R and POP
            else:
                # More flexible - just ensure reasonable credit
                min_credit_ratio = 0.15  # At least 15 cents per dollar
                meets_requirement = credit_per_width >= min_credit_ratio
                
                return {
                    'meets_requirement': meets_requirement,
                    'credit_ratio': credit_per_width,
                    'min_required': min_credit_ratio,
                    'strategy_type': strategy_type,
                    'message': f"Credit/Width: {credit_per_width:.3f} (Min: {min_credit_ratio:.3f})"
                }
                
        except Exception as e:
            print(f"Error checking credit requirements: {e}")
            return {
                'meets_requirement': False,
                'credit_ratio': 0,
                'min_required': 0.33,
                'strategy_type': 'Unknown',
                'message': f"Error: {e}"
            }

def format_currency(value):
    """Format currency values"""
    return f"${value:.2f}"

def format_percentage(value):
    """Format percentage values"""
    return f"{value*100:.1f}%"

def get_next_friday():
    """Get next Friday's date"""
    today = date.today()
    days_ahead = 4 - today.weekday()  # Friday is weekday 4
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

def get_next_monthly_expiry():
    """Get next monthly options expiry (3rd Friday)"""
    today = date.today()
    
    # Find 3rd Friday of current month
    first_day = today.replace(day=1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    
    # If 3rd Friday has passed, get next month's
    if third_friday <= today:
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        first_friday = next_month + timedelta(days=(4 - next_month.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
    
    return third_friday.strftime('%Y-%m-%d') 