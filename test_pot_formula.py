#!/usr/bin/env python3
"""
Test script to verify POT formula correctness
"""

import sys
sys.path.append('.')

from put_spread_analysis import PutSpreadAnalyzer

def test_pot_formula():
    """Test the corrected POT formula with realistic scenarios"""
    
    analyzer = PutSpreadAnalyzer()
    
    print("🧪 Testing POT Formula Correctness")
    print("=" * 50)
    
    # Test scenario: SPY at $583, various strikes
    current_price = 583.0
    time_to_expiry = 1/365.25  # 1 day
    volatility = 0.15  # 15% annual volatility
    
    print(f"Current Price: ${current_price}")
    print(f"Time to Expiry: {time_to_expiry:.4f} years (1 day)")
    print(f"Volatility: {volatility:.1%}")
    print()
    
    # Test various strikes to see if POT behaves correctly
    test_strikes = [580, 575, 570, 565, 560, 555, 550]
    
    print("Strike Price | Distance | POT %")
    print("-" * 35)
    
    for strike in test_strikes:
        distance = current_price - strike
        distance_pct = (distance / current_price) * 100
        
        pot = analyzer.probability_of_touching(
            current_price=current_price,
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility
        )
        
        print(f"${strike:6.2f}     | ${distance:5.2f} ({distance_pct:4.1f}%) | {pot:6.1%}")
    
    print()
    print("Expected behavior:")
    print("- Strikes closer to current price should have HIGHER POT")
    print("- Strikes further from current price should have LOWER POT")
    print("- POT should be between 0% and 100%, not always 100%")
    print()
    
    # Test target POT finding
    print("🎯 Testing Target POT Strike Finding")
    print("=" * 40)
    
    target_pots = [0.20, 0.10, 0.05, 0.02, 0.01]
    
    print("Target POT | Strike Price | Actual POT | Distance")
    print("-" * 50)
    
    for target_pot in target_pots:
        strike = analyzer.find_strike_for_target_pot(
            current_price=current_price,
            target_pot=target_pot,
            time_to_expiry=time_to_expiry,
            volatility=volatility
        )
        
        if strike:
            actual_pot = analyzer.probability_of_touching(
                current_price=current_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility
            )
            distance = current_price - strike
            distance_pct = (distance / current_price) * 100
            
            print(f"{target_pot:8.1%}   | ${strike:9.2f}   | {actual_pot:8.1%}   | ${distance:5.2f} ({distance_pct:4.1f}%)")
        else:
            print(f"{target_pot:8.1%}   | Not Found     | N/A        | N/A")
    
    print()
    print("Expected behavior:")
    print("- Lower target POT should give strikes FURTHER from current price")
    print("- Higher target POT should give strikes CLOSER to current price")
    print("- Actual POT should be close to target POT")

if __name__ == "__main__":
    test_pot_formula() 