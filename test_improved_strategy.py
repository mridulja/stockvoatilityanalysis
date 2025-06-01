#!/usr/bin/env python3
"""
Test the improved PUT selling strategy with longer timeframes
"""

import sys
sys.path.append('.')

from put_spread_analysis import PutSpreadAnalyzer

def test_improved_strategy():
    """Test the improved strategy with various timeframes"""
    
    print("🎯 Testing IMPROVED PUT Selling Strategy")
    print("=" * 50)
    
    analyzer = PutSpreadAnalyzer()
    
    # Test with realistic parameters - SPY at $583
    current_price = 583.0
    
    # Test different expiry timeframes
    expiry_scenarios = [
        ("2025-01-25", "7 days"),
        ("2025-02-01", "2 weeks"),
        ("2025-02-15", "1 month"),
        ("2025-03-15", "2 months")
    ]
    
    for expiry_date, timeframe_desc in expiry_scenarios:
        print(f"📅 {timeframe_desc} Expiry ({expiry_date})")
        print("-" * 50)
        
        # Calculate time to expiry
        time_to_expiry = analyzer.calculate_time_to_expiry(expiry_date)
        time_days = time_to_expiry * 365.25
        
        print(f"Time to Expiry: {time_days:.1f} days ({time_to_expiry:.4f} years)")
        
        # Test with medium volatility (typical for SPY)
        volatility = 0.18  # 18% for SPY
        
        # Test the strategy for key POT targets
        target_pot_levels = [0.20, 0.10, 0.05, 0.02, 0.01]
        
        results = analyzer.analyze_put_spread_scenarios(
            ticker="SPY",
            current_price=current_price,
            expiry_date=expiry_date,
            volatility=volatility,
            target_pot_levels=target_pot_levels
        )
        
        if results and results['scenarios']:
            print(f"Using Volatility: {results['volatility']:.1%}")
            print()
            print("Target POT | Strike Price | Actual POT | Distance | Safety Level")
            print("-" * 70)
            
            for scenario in results['scenarios']:
                target_pot = scenario['target_pot']
                strike = scenario['short_strike']
                actual_pot = scenario['actual_pot']
                distance = scenario['distance_from_current']
                distance_pct = scenario['distance_pct']
                
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
            
            print()
            
            # Highlight best recommendation
            best_scenario = results['scenarios'][0]  # Lowest POT (safest)
            
            if best_scenario['distance_pct'] >= 3.0:
                safety_assessment = "✅ EXCELLENT"
            elif best_scenario['distance_pct'] >= 2.0:
                safety_assessment = "✅ GOOD"
            elif best_scenario['distance_pct'] >= 1.0:
                safety_assessment = "⚠️ MODERATE"
            else:
                safety_assessment = "❌ RISKY"
            
            print(f"BEST: {best_scenario['target_pot']:.1%} POT → ${best_scenario['short_strike']:.2f} "
                  f"({best_scenario['distance_pct']:.1f}% away) - {safety_assessment}")
            
            # Show spread recommendation if available
            if best_scenario['spreads']:
                best_spread = best_scenario['spreads'][0]
                print(f"SPREAD: Sell ${best_spread['short_strike']:.2f} / Buy ${best_spread['long_strike']:.2f} "
                      f"(${best_spread['width']:.2f} wide, {best_spread['prob_profit']:.1%} POP)")
        
        print("\n" + "="*70 + "\n")
    
    print("🎯 What we're looking for:")
    print("- Lower POT% should give strikes FURTHER from current price")
    print("- Longer timeframes should give MORE REASONABLE distances")
    print("- 1-5% POT should give 2-5% distance from current price")
    print("- ALL distances should be at least 1-2% for practical trading")

if __name__ == "__main__":
    test_improved_strategy() 