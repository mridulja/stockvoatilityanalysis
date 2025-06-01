# Improve the put spread strategy for more realistic timeframes and better safety

with open('put_spread_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update the analyze_put_spread_scenarios method to use more realistic parameters
old_analyze_method = '''    def analyze_put_spread_scenarios(self,
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
        
        # Get dividend yield
        dividend_yield = self.get_dividend_yield(ticker)
        
        # Determine volatility
        if volatility is None or volatility <= 0:
            if atr and atr > 0:
                volatility = self.calculate_implied_volatility_proxy(current_price, atr, time_to_expiry)
            else:
                volatility = 0.3  # Default 30%
        
        # Try to fetch options chain for real implied volatility
        options_data = self.fetch_options_chain(ticker, expiry_date)
        if options_data is not None and not options_data.empty:
            # Use median IV from available options
            median_iv = options_data['impliedVolatility'].median()
            if median_iv > 0:
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
        
        for target_pot in target_pot_levels:
            # Find optimal short strike for target POT
            short_strike = self.find_strike_for_target_pot(
                current_price, target_pot, time_to_expiry, 
                volatility, self.risk_free_rate, dividend_yield
            )
            
            if short_strike is not None:
                # Calculate actual POT for verification
                actual_pot = self.probability_of_touching(
                    current_price, short_strike, time_to_expiry,
                    volatility, self.risk_free_rate, dividend_yield
                )
                
                # Analyze different spread widths
                spread_widths = [5, 10, 15, 20, 25]  # Dollar widths
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
        
        return results'''

# Enhanced method with better parameters
new_analyze_method = '''    def analyze_put_spread_scenarios(self,
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
        
        return results'''

content = content.replace(old_analyze_method, new_analyze_method)

# Write back
with open('put_spread_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully improved put spread strategy logic') 