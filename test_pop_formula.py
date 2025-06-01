#!/usr/bin/env python3
"""
Test script to verify and fix the POP (Probability of Profit) formula for put spreads
"""

import numpy as np
from scipy import stats
import pandas as pd

def test_pop_formulas():
    """Test different POP formulas to find the correct one"""
    
    # Test parameters
    S = 590.0  # Current price
    long_strike = 575.0  # Long put (A)
    short_strike = 580.0  # Short put (B)
    T = 1/365  # 1 day to expiry
    sigma = 0.25  # 25% volatility
    r = 0.05  # 5% risk-free rate
    q = 0.0  # No dividend
    
    print(f"Test Parameters:")
    print(f"Current Price (S): ${S}")
    print(f"Long Strike (A): ${long_strike}")
    print(f"Short Strike (B): ${short_strike}")
    print(f"Time to Expiry (T): {T:.4f} years")
    print(f"Volatility (σ): {sigma:.2%}")
    print(f"Risk-free Rate (r): {r:.2%}")
    print(f"Dividend Yield (q): {q:.2%}")
    print()
    
    sqrt_T = np.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    
    # Current WRONG formula in the code
    print("=== CURRENT (WRONG) FORMULA ===")
    d1_B = (np.log(S / short_strike) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d1_A = (np.log(S / long_strike) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
    
    wrong_pop = stats.norm.cdf(d1_B) - stats.norm.cdf(d1_A)
    print(f"d1_B = {d1_B:.4f}")
    print(f"d1_A = {d1_A:.4f}")
    print(f"N(d1_B) = {stats.norm.cdf(d1_B):.4f}")
    print(f"N(d1_A) = {stats.norm.cdf(d1_A):.4f}")
    print(f"Wrong POP = {wrong_pop:.4f} ({wrong_pop:.2%})")
    print()
    
    # CORRECT formula for PUT SPREAD PROBABILITY OF PROFIT
    print("=== CORRECT FORMULA FOR PUT SPREAD ===")
    
    # For a short put spread (sell higher strike, buy lower strike):
    # We profit if stock stays above the short strike minus the net credit received
    # Simplified: we profit if stock stays above short strike
    # POP = P(S_T > short_strike) = N(d2_short)
    
    # Calculate d2 for short strike
    d1_short = (np.log(S / short_strike) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2_short = d1_short - sigma_sqrt_T
    
    # For PUT spread, probability of profit is probability stock stays above short strike
    correct_pop = stats.norm.cdf(d2_short)
    
    print(f"d1_short = {d1_short:.4f}")
    print(f"d2_short = {d2_short:.4f}")
    print(f"Correct POP = N(d2_short) = {correct_pop:.4f} ({correct_pop:.2%})")
    print()
    
    # Alternative verification using put option pricing
    print("=== VERIFICATION USING PUT PRICING ===")
    
    # Calculate individual put values
    def black_scholes_put(S, K, T, r, sigma, q=0):
        """Calculate Black-Scholes put option price"""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
        return put_price, d1, d2
    
    short_put_price, d1_short_calc, d2_short_calc = black_scholes_put(S, short_strike, T, r, sigma, q)
    long_put_price, d1_long_calc, d2_long_calc = black_scholes_put(S, long_strike, T, r, sigma, q)
    
    net_credit = short_put_price - long_put_price
    breakeven = short_strike - net_credit
    
    print(f"Short PUT price ({short_strike}): ${short_put_price:.4f}")
    print(f"Long PUT price ({long_strike}): ${long_put_price:.4f}")
    print(f"Net Credit Received: ${net_credit:.4f}")
    print(f"Breakeven Price: ${breakeven:.4f}")
    print()
    
    # POP is probability that stock stays above breakeven
    d2_breakeven = (np.log(S / breakeven) + (r - q - 0.5 * sigma**2) * T) / sigma_sqrt_T
    pop_breakeven = stats.norm.cdf(d2_breakeven)
    
    print(f"POP (above breakeven) = {pop_breakeven:.4f} ({pop_breakeven:.2%})")
    print()
    
    # Test with different scenarios
    print("=== TESTING DIFFERENT SCENARIOS ===")
    scenarios = [
        (590, 580, 575, 1/365, 0.25),  # Current
        (590, 570, 565, 1/365, 0.25),  # Further OTM
        (590, 585, 580, 7/365, 0.25),  # 1 week expiry
        (100, 95, 90, 1/365, 0.30),   # Different stock
    ]
    
    for s, b, a, t, vol in scenarios:
        sqrt_t = np.sqrt(t)
        d2_b = (np.log(s / b) + (r - q - 0.5 * vol**2) * t) / (vol * sqrt_t)
        pop = stats.norm.cdf(d2_b)
        
        distance_pct = ((s - b) / s) * 100
        print(f"S=${s}, B=${b}, A=${a}, T={t*365:.1f}d, σ={vol:.0%} -> POP={pop:.2%}, Distance={distance_pct:.1f}%")

if __name__ == "__main__":
    test_pop_formulas() 