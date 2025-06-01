#!/usr/bin/env python3
"""
Test the PUT selling strategy logic to ensure it recommends LOW POT strikes
"""

import sys
sys.path.append('.')

from put_spread_analysis import PutSpreadAnalyzer

def test_strategy_logic():
    """Test that the strategy recommends strikes with LOW POT for safe PUT selling"""
    
    print("🎯 Testing PUT Selling Strategy Logic")
    print("=" * 50)
    
    analyzer = PutSpreadAnalyzer()
    
    # Test with realistic parameters - SPY at $583
    current_price = 583.0
    expiry_date = "2025-01-25"  # Example expiry
    
    # Calculate time to expiry
    time_to_expiry = analyzer.calculate_time_to_expiry(expiry_date)
    
    print(f"Current Price: ${current_price}")
    print(f"Expiry Date: {expiry_date}")
    print(f"Time to Expiry: {time_to_expiry:.4f} years")
    print()
    
    # Test with different volatility scenarios
    volatility_scenarios = [
        (0.10, "Low Vol (10%)"),
        (0.15, "Medium Vol (15%)"),
        (0.25, "High Vol (25%)")
    ]
    
    for volatility, scenario_name in volatility_scenarios:
        print(f"📊 {scenario_name}")
        print("-" * 30)
        
        # Test the strategy for various POT targets
        target_pot_levels = [0.20, 0.10, 0.05, 0.02, 0.01]
        
        results = analyzer.analyze_put_spread_scenarios(
            ticker="SPY",
            current_price=current_price,
            expiry_date=expiry_date,
            volatility=volatility,
            target_pot_levels=target_pot_levels
        )
        
        if results and results['scenarios']:
            print("Target POT | Strike Price | Actual POT | Distance | Safety")
            print("-" * 60)
            
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
            
            # Check if strategy is working correctly
            best_scenario = results['scenarios'][0]  # Lowest POT target
            if best_scenario['distance_pct'] >= 2.0:
                print(f"✅ Strategy CORRECT: {best_scenario['target_pot']:.1%} POT gives ${best_scenario['distance_from_current']:.2f} ({best_scenario['distance_pct']:.1f}%) distance")
            else:
                print(f"❌ Strategy RISKY: {best_scenario['target_pot']:.1%} POT only gives ${best_scenario['distance_from_current']:.2f} ({best_scenario['distance_pct']:.1f}%) distance")
        
        print()
    
    print("🎯 Expected Results:")
    print("- Lower POT% should give strikes FURTHER from current price")
    print("- 10% POT should be SAFER than 20% POT")
    print("- 1% POT should be SAFEST of all")
    print("- Distance should INCREASE as POT% DECREASES")
    print("- ALL strikes should be BELOW current price for PUT selling")

if __name__ == "__main__":
    test_strategy_logic() 