# Fix POT formula and search logic
with open('put_spread_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the incorrect POT formula
old_pot_method = '''    def probability_of_touching(self,
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
            strike_price: Strike price to analyze (B)
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
        B = strike_price
        T = time_to_expiry
        sigma = volatility
        r = risk_free_rate
        q = dividend_yield
        
        sqrt_T = np.sqrt(T)
        
        # POT formula: 2 × N(|ln(S/B)| - (r - q - σ²/2)T / σ√T)
        numerator = abs(np.log(S / B)) - (r - q - 0.5 * sigma**2) * T
        denominator = sigma * sqrt_T
        
        if denominator <= 0:
            return 0.0
        
        d = numerator / denominator
        pot = 2 * stats.norm.cdf(d)
        
        return max(0.0, min(1.0, pot))  # Bound between 0 and 1'''

# Correct POT formula
new_pot_method = '''    def probability_of_touching(self,
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
        
        return max(0.0, min(1.0, pot))  # Bound between 0 and 1'''

content = content.replace(old_pot_method, new_pot_method)

# Fix the search logic in find_strike_for_target_pot
old_search_method = '''    def find_strike_for_target_pot(self,
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
        
        # Search below current price for put strikes
        low_strike = current_price * (1 - search_range)
        high_strike = current_price * 0.99  # Just below current price
        
        tolerance = 0.001  # 0.1% tolerance
        max_iterations = 50
        
        for _ in range(max_iterations):
            mid_strike = (low_strike + high_strike) / 2
            
            pot = self.probability_of_touching(
                current_price, mid_strike, time_to_expiry, 
                volatility, risk_free_rate, dividend_yield
            )
            
            if abs(pot - target_pot) < tolerance:
                return mid_strike
            
            if pot < target_pot:
                high_strike = mid_strike
            else:
                low_strike = mid_strike
        
        # Return best approximation
        return (low_strike + high_strike) / 2'''

# Corrected search logic
new_search_method = '''    def find_strike_for_target_pot(self,
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
        return final_strike'''

content = content.replace(old_search_method, new_search_method)

# Write back
with open('put_spread_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully fixed POT formula and search logic') 