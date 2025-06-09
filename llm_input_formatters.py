"""
LLM Input Data Formatters for Stock Analysis Application

This module provides data formatting functions for preparing data to send TO LLMs.
For displaying AI responses, see shared/ai_formatter.py

Functions:
- format_put_spread_data_for_llm(): Formats put spread data for LLM input
- format_iron_condor_data_for_llm(): Formats iron condor data for LLM input  
- format_options_strategy_data_for_llm(): Formats general options data for LLM input

Author: Enhanced Stock Analysis System
Date: 2025
"""

def format_put_spread_data_for_llm(ticker, current_price, expiry_date, days_to_expiry, 
                                  spread_type, spread_analysis, volatility, current_vix):
    """Format put spread data for LLM analysis"""
    
    formatted_data = f"""
    PUT SPREAD ANALYSIS DATA:
    ========================
    
    Ticker: {ticker}
    Current Price: ${current_price:.2f}
    Expiry Date: {expiry_date}
    Days to Expiry: {days_to_expiry}
    
    SPREAD DETAILS:
    --------------
    Strategy Type: {spread_type.title()} Put Spread
    Short Strike: ${spread_analysis.get('short_strike', 'N/A')}
    Long Strike: ${spread_analysis.get('long_strike', 'N/A')}
    Net Premium: ${spread_analysis.get('net_credit', spread_analysis.get('net_debit', 0)):.2f}
    
    PROFIT/LOSS METRICS:
    -------------------
    Max Profit: ${spread_analysis.get('max_profit', 0):.2f}
    Max Loss: ${spread_analysis.get('max_loss', 0):.2f}
    Breakeven Point: ${spread_analysis.get('breakeven', 0):.2f}
    Risk/Reward Ratio: {spread_analysis.get('risk_reward_ratio', 0):.2f}
    
    PROBABILITY ANALYSIS:
    --------------------
    Probability of Max Profit: {spread_analysis.get('prob_max_profit', 0)*100:.1f}%
    Probability of Breakeven: {spread_analysis.get('prob_breakeven', 0)*100:.1f}%
    
    GREEKS:
    -------
    Delta: {spread_analysis.get('greeks', {}).get('delta', 0):.3f}
    Gamma: {spread_analysis.get('greeks', {}).get('gamma', 0):.3f}
    Theta: ${spread_analysis.get('greeks', {}).get('theta', 0):.2f}/day
    Vega: {spread_analysis.get('greeks', {}).get('vega', 0):.2f}
    
    MARKET CONDITIONS:
    -----------------
    Current VIX: {f"{current_vix:.2f}" if current_vix else 'N/A'}
    Implied Volatility: {volatility*100:.1f}%
    """
    
    return formatted_data


def format_iron_condor_data_for_llm(ticker, current_price, expiry_date, days_to_expiry,
                                   strategy_focus, iron_condor, market_conditions, 
                                   historical_vol, current_vix):
    """Format Iron Condor data for LLM analysis"""
    
    formatted_data = f"""
    IRON CONDOR ANALYSIS DATA:
    =========================
    
    Ticker: {ticker}
    Current Price: ${current_price:.2f}
    Expiry Date: {expiry_date}
    Days to Expiry: {days_to_expiry}
    Strategy Focus: {strategy_focus}
    
    IRON CONDOR STRUCTURE:
    ---------------------
    Type: {iron_condor.get('type', 'Standard')}
    Put Long Strike: ${iron_condor.get('put_long_strike', 0):.0f}
    Put Short Strike: ${iron_condor.get('put_short_strike', 0):.0f}
    Call Short Strike: ${iron_condor.get('call_short_strike', 0):.0f}
    Call Long Strike: ${iron_condor.get('call_long_strike', 0):.0f}
    Wing Width: ${iron_condor.get('wing_width', 0):.0f}
    
    PROFIT/LOSS METRICS:
    -------------------
    Net Credit: ${iron_condor.get('net_credit', 0):.2f}
    Max Profit: ${iron_condor.get('max_profit', 0):.2f}
    Max Loss: ${iron_condor.get('max_loss', 0):.2f}
    Risk/Reward Ratio: {iron_condor.get('risk_reward_ratio', 0):.2f}
    
    PROBABILITY ANALYSIS:
    --------------------
    Probability of Profit: {iron_condor.get('pop_percentage', 0):.1f}%
    Breakeven Lower: ${iron_condor.get('breakeven_lower', 0):.2f}
    Breakeven Upper: ${iron_condor.get('breakeven_upper', 0):.2f}
    Profit Zone Width: ${iron_condor.get('breakeven_upper', 0) - iron_condor.get('breakeven_lower', 0):.2f}
    
    MARKET CONDITIONS:
    -----------------
    Current VIX: {f"{current_vix:.2f}" if current_vix else 'N/A'}
    Historical Volatility: {historical_vol*100:.1f}%
    IV Rank: {market_conditions.get('iv_rank', 'N/A')}
    Trade Approved: {'Yes' if market_conditions.get('trade_approved', False) else 'No'}
    VIX Level: {market_conditions.get('vix', 'N/A')}
    
    GREEKS (if available):
    ---------------------
    Delta: {iron_condor.get('delta', 'N/A')}
    Gamma: {iron_condor.get('gamma', 'N/A')}
    Theta: {iron_condor.get('theta', 'N/A')}
    Vega: {iron_condor.get('vega', 'N/A')}
    """
    
    return formatted_data


def format_options_strategy_data_for_llm(ticker, current_price, volatility_data, vix_data, 
                                        confidence_levels, strategy_recommendation):
    """Format general options strategy data for LLM analysis"""
    
    formatted_data = f"""
    OPTIONS STRATEGY ANALYSIS DATA:
    ==============================
    
    Ticker: {ticker}
    Current Price: ${current_price:.2f}
    
    VOLATILITY METRICS:
    ------------------
    ATR (14-day): ${volatility_data.get('atr', 0):.2f}
    Historical Volatility: {volatility_data.get('historical_vol', 0)*100:.1f}%
    Volatility Percentile: {volatility_data.get('vol_percentile', 0):.1f}%
    
    CONFIDENCE LEVELS:
    -----------------
    95% Range: ${confidence_levels.get('lower_95', 0):.2f} - ${confidence_levels.get('upper_95', 0):.2f}
    68% Range: ${confidence_levels.get('lower_68', 0):.2f} - ${confidence_levels.get('upper_68', 0):.2f}
    
    STRATEGY RECOMMENDATION:
    -----------------------
    Recommended Strike: ${strategy_recommendation.get('strike', 0):.2f}
    Strategy Type: {strategy_recommendation.get('strategy', 'N/A')}
    Expected Outcome: {strategy_recommendation.get('expected_outcome', 'N/A')}
    Risk Assessment: {strategy_recommendation.get('risk_assessment', 'N/A')}
    
    MARKET CONDITIONS:
    -----------------
    Current VIX: {vix_data.get('current_vix', 'N/A') if vix_data else 'N/A'}
    VIX Condition: {vix_data.get('condition', 'N/A') if vix_data else 'N/A'}
    Trade Approved: {vix_data.get('trade_approved', 'N/A') if vix_data else 'N/A'}
    """
    
    return formatted_data


def format_master_analysis_data_for_llm(master_data):
    """Format comprehensive master analysis data for LLM recommendations"""
    
    params = master_data['parameters']
    analysis = master_data['analysis_results']
    
    formatted_data = f"""
    MASTER TRADING ANALYSIS REQUEST:
    ===============================
    
    TRADE PARAMETERS:
    ----------------
    Primary Ticker: {params['ticker']}
    Investment Amount: ${params['amount']:,}
    Options Expiry: {params['expiry']}
    Risk Tolerance: {params['risk_tolerance']}
    Time Horizon: {params['time_horizon']}
    Position Preference: {params['position_preference']} {f"(Auto-detected from: {params.get('original_position_preference', 'N/A')})" if 'Auto' in params.get('original_position_preference', '') else ''}
    Market Bias Analysis: {params.get('position_preference', 'Neutral')} bias determined from technical indicators
    
    COMPREHENSIVE ANALYSIS RESULTS:
    ==============================
    
    PRICE PATTERN ANALYSIS:
    ----------------------
    {format_analysis_section('price_charts', analysis.get('price_charts', {}))}
    
    DETAILED STATISTICS:
    -------------------
    {format_analysis_section('detailed_stats', analysis.get('detailed_stats', {}))}
    
    COMPARISON ANALYSIS:
    -------------------
    {format_analysis_section('comparison', analysis.get('comparison', {}))}
    
    VIX MARKET CONDITIONS:
    ---------------------
    {format_analysis_section('vix', analysis.get('vix', {}))}
    
    OPTIONS STRATEGIES:
    ------------------
    {format_analysis_section('options', analysis.get('options', {}))}
    
    PUT SPREAD OPPORTUNITIES:
    ------------------------
    {format_analysis_section('put_spread', analysis.get('put_spread', {}))}
    
    IRON CONDOR STRATEGIES:
    ----------------------
    {format_analysis_section('iron_condor', analysis.get('iron_condor', {}))}
    
    MASTER RECOMMENDATION REQUEST:
    =============================
    Based on this comprehensive analysis, please provide:
    
    1. TOP 3 TRADING RECOMMENDATIONS with specific:
       - Strategy name and type
       - Entry price and timing
       - Risk level (Low/Medium/High)
       - Probability of Profit (POP)
       - Return on Capital (ROC)
       - Maximum profit and risk amounts
       - Specific action steps
    
    2. RISK ASSESSMENT:
       - Overall market risk level
       - Position sizing recommendations
       - Risk management strategies
    
    3. EXECUTION PRIORITIES:
       - Which strategy to execute first
       - Optimal timing considerations
       - Market condition dependencies
    
    Please format response to be concise yet comprehensive, focusing on actionable insights.
    """
    
    return formatted_data


def format_master_analysis_with_actual_data(master_data, actual_strategies):
    """Format master analysis data with ACTUAL strategy numbers for accurate LLM analysis"""
    
    params = master_data['parameters']
    analysis = master_data['analysis_results']
    
    # Format the actual strategy data that matches the displayed table
    strategies_text = ""
    for i, strategy in enumerate(actual_strategies, 1):
        strategies_text += f"""
    STRATEGY {i}: {strategy['name']}
    --------------------------------
    Put Short Strike: {strategy['put_short']}
    Put Long Strike: {strategy['put_long']}
    Call Short Strike: {strategy['call_short']}
    Call Long Strike: {strategy['call_long']}
    Net Credit Received: {strategy['net_credit']}
    Maximum Profit: {strategy['max_profit']}
    Maximum Loss: {strategy['max_loss']}
    Probability of Profit: {strategy['pop']}
    Return on Capital: {strategy['roc']}
    """
    
    formatted_data = f"""
    MASTER TRADING ANALYSIS WITH ACTUAL STRATEGY DATA:
    ================================================
    
    TRADE PARAMETERS:
    ----------------
    Primary Ticker: {params['ticker']}
    Investment Amount: ${params['amount']:,}
    Options Expiry: {params['expiry']}
    Risk Tolerance: {params['risk_tolerance']}
    Time Horizon: {params['time_horizon']}
    Position Preference: {params['position_preference']} {f"(Auto-detected from: {params.get('original_position_preference', 'N/A')})" if 'Auto' in params.get('original_position_preference', '') else ''}
    
    ACTUAL STRATEGY RECOMMENDATIONS (EXACTLY AS DISPLAYED):
    =====================================================
    {strategies_text}
    
    VIX MARKET CONDITIONS:
    ---------------------
    {format_analysis_section('vix', analysis.get('vix', {}))}
    
    ANALYSIS REQUEST:
    ================
    You are analyzing the EXACT strategies and numbers shown above. Please provide:
    
    1. **STRATEGY ANALYSIS**: 
       - Evaluate each of the 3 strategies using the EXACT strikes and amounts provided
       - Comment on the strike selection relative to current market conditions
       - Assess the risk/reward profile of each strategy
    
    2. **RECOMMENDATION RANKING**:
       - Rank the strategies from best to worst for current market conditions
       - Explain WHY each strategy is suitable or unsuitable
       - Consider the user's {params['risk_tolerance']} risk tolerance
    
    3. **EXECUTION GUIDANCE**:
       - Which strategy to execute first and why
       - Entry timing considerations
       - Risk management recommendations
       - Exit strategy planning
    
    4. **MARKET CONTEXT**:
       - How current market conditions affect these specific strategies
       - What market scenarios favor each strategy
       - Key risks to monitor
    
    Please reference the ACTUAL NUMBERS provided above in your analysis. Be specific about the strikes, premiums, and profit/loss figures.
    """
    
    return formatted_data


def format_analysis_section(section_name, data):
    """Helper function to format individual analysis sections"""
    if not data or data.get('status') == 'No data available':
        return f"{section_name.title()}: No data available"
    
    if section_name == 'vix':
        return f"""Current VIX: {data.get('current_vix', 'N/A')}
Market Condition: {data.get('condition', 'N/A')}
Trade Environment: {data.get('trade_environment', 'N/A')}
Recommendation: {data.get('recommendation', 'N/A')}"""
    
    elif section_name == 'options':
        strategies = []
        for strategy, details in data.items():
            if isinstance(details, dict):
                strategies.append(f"{strategy}: POP {details.get('pop', 0)}%, Max Profit ${details.get('max_profit', 0)}")
        return '\n'.join(strategies) if strategies else "Options data available"
    
    elif section_name == 'put_spread':
        spreads = []
        for spread, details in data.items():
            if isinstance(details, dict):
                spreads.append(f"{spread}: POP {details.get('pop', 0)}%, ROC {details.get('roc', 0)}%")
        return '\n'.join(spreads) if spreads else "Put spread data available"
    
    elif section_name == 'iron_condor':
        condors = []
        for condor, details in data.items():
            if isinstance(details, dict):
                condors.append(f"{condor}: POP {details.get('pop', 0)}%, ROC {details.get('roc', 0)}%")
        return '\n'.join(condors) if condors else "Iron condor data available"
    
    else:
        # Generic formatting for other sections
        return f"{section_name.title()}: Analysis completed with {len(data)} data points" 