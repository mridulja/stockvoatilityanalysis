#!/usr/bin/env python3
"""
Test with realistic future dates to verify POT and strategy
"""

import sys
sys.path.append('.')

from put_spread_analysis import PutSpreadAnalyzer
from datetime import datetime, timedelta

def test_realistic_dates():
    """Test with proper future dates"""
    
    print("🎯 Testing with REALISTIC Future Dates")
    print("=" * 50)
    
    analyzer = PutSpreadAnalyzer()
    
    # Current date for reference
    today = datetime.now()
    print(f"Today: {today.strftime('%Y-%m-%d %H:%M')}")
    
    # Test with realistic parameters - SPY at $583
    current_price = 583.0
    
    # Create realistic future dates
    future_dates = [
        (today + timedelta(days=7), "1 week"),
        (today + timedelta(days=14), "2 weeks"), 
        (today + timedelta(days=30), "1 month"),
        (today + timedelta(days=60), "2 months")
    ]
    
    for future_date, timeframe_desc in future_dates:
        expiry_date = future_date.strftime('%Y-%m-%d')
        
        print(f"📅 {timeframe_desc} Expiry ({expiry_date})")
        print("-" * 50)
        
        # Calculate time to expiry
        time_to_expiry = analyzer.calculate_time_to_expiry(expiry_date)
        time_days = time_to_expiry * 365.25
        
        print(f"Time to Expiry: {time_days:.1f} days ({time_to_expiry:.4f} years)")
        
        # Test with typical SPY volatility
        volatility = 0.18  # 18% for SPY
        
        # Test key POT targets  
        target_pot_levels = [0.20, 0.10, 0.05, 0.02, 0.01]
        
        print(f"Using Volatility: {volatility:.1%}")
        print()
        print("Target POT | Strike Price | Actual POT | Distance | Safety Level")
        print("-" * 70)
        
        for target_pot in target_pot_levels:
            # Find strike for this POT level
            strike = analyzer.find_strike_for_target_pot(
                current_price=current_price,
                target_pot=target_pot,
                time_to_expiry=time_to_expiry,
                volatility=volatility
            )
            
            if strike:
                # Verify actual POT
                actual_pot = analyzer.probability_of_touching(
                    current_price=current_price,
                    strike_price=strike,
                    time_to_expiry=time_to_expiry,
                    volatility=volatility
                )
                
                distance = current_price - strike
                distance_pct = (distance / current_price) * 100
                
                # Determine safety level
                if distance_pct > 5:
                    safety = "🟢 VERY SAFE"
                elif distance_pct > 3:
                    safety = "🟡 SAFE"
                elif distance_pct > 2:
                    safety = "🟠 MODERATE"
                else:
                    safety = "🔴 RISKY"
                
                print(f"{target_pot:8.1%}   | ${strike:9.2f}   | {actual_pot:8.1%}   | ${distance:5.2f} ({distance_pct:4.1f}%) | {safety}")
            else:
                print(f"{target_pot:8.1%}   | NOT FOUND   | N/A        | N/A")
        
        print()
        
        # Find the best (lowest POT) recommendation
        best_pot = 0.01  # 1% POT
        best_strike = analyzer.find_strike_for_target_pot(
            current_price=current_price,
            target_pot=best_pot,
            time_to_expiry=time_to_expiry,
            volatility=volatility
        )
        
        if best_strike:
            actual_pot = analyzer.probability_of_touching(
                current_price, best_strike, time_to_expiry, volatility
            )
            distance_pct = ((current_price - best_strike) / current_price) * 100
            
            if distance_pct >= 3.0:
                assessment = "✅ EXCELLENT"
            elif distance_pct >= 2.0:
                assessment = "✅ GOOD"
            elif distance_pct >= 1.0:
                assessment = "⚠️ MODERATE"
            else:
                assessment = "❌ RISKY"
            
            print(f"RECOMMENDATION: Sell ${best_strike:.2f} PUT")
            print(f"POT: {actual_pot:.1%} | Distance: {distance_pct:.1f}% | Assessment: {assessment}")
        
        print("\n" + "="*70 + "\n")
    
    print("🎯 Expected Results:")
    print("- Longer timeframes should give MORE conservative strikes")
    print("- 1% POT should be SAFEST (furthest from current price)")
    print("- 20% POT should be CLOSEST to current price")
    print("- ALL strikes should be below $583 for PUT selling")
    print("- Distance should increase with longer timeframes")

if __name__ == "__main__":
    test_realistic_dates() 