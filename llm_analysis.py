"""
LLM Analysis Module for Stock Volatility Analyzer

This module provides natural language analysis and recommendations using OpenAI's GPT-4o-mini model.
It analyzes volatility data, options strategy results, and market conditions to generate 
human-readable summaries and actionable trading recommendations.

Features:
- Options strategy analysis and recommendations
- Market condition interpretation
- Risk assessment summaries
- VIX-based trading guidance
- Probability analysis explanations

Author: Mridul Jain
Date: 2025
Version: 1.0 - LLM Integration
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI
import pandas as pd
import numpy as np

class LLMAnalyzer:
    """
    LLM-powered analysis for stock volatility and options strategy data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM analyzer with OpenAI API key
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
    
    def format_options_data(self, 
                          ticker: str,
                          current_price: float,
                          strategy_timeframe: str,
                          recommendations: List[Dict],
                          prob_dist: Dict,
                          vix_data: Optional[Dict] = None,
                          atr: float = 0,
                          confidence_levels: List[Dict] = None) -> Dict[str, Any]:
        """
        Format options strategy data for LLM analysis
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            strategy_timeframe: 'daily' or 'weekly'
            recommendations: List of strike recommendations
            prob_dist: Probability distribution data
            vix_data: VIX market condition data
            atr: Average True Range value
            confidence_levels: 90%, 95%, 99% probability ranges
            
        Returns:
            Formatted data dictionary for LLM analysis
        """
        
        # Best recommendation
        best_rec = recommendations[0] if recommendations else None
        
        # Format recommendations summary
        rec_summary = []
        for i, rec in enumerate(recommendations[:3]):  # Top 3 recommendations
            rec_summary.append({
                'rank': i + 1,
                'strike': rec['strike'],
                'distance_pct': rec['distance_pct'],
                'prob_below': rec['prob_below'],
                'safety_score': rec['safety_score']
            })
        
        # VIX analysis
        vix_summary = None
        if vix_data:
            vix_summary = {
                'current_vix': vix_data.get('current_vix'),
                'condition': vix_data.get('condition'),
                'trade_approved': vix_data.get('trade_approved')
            }
        
        formatted_data = {
            'ticker': ticker,
            'current_price': current_price,
            'strategy_timeframe': strategy_timeframe,
            'atr': atr,
            'best_recommendation': {
                'strike': best_rec['strike'] if best_rec else None,
                'distance_pct': best_rec['distance_pct'] if best_rec else None,
                'prob_below': best_rec['prob_below'] if best_rec else None,
                'safety_score': best_rec['safety_score'] if best_rec else None
            },
            'top_recommendations': rec_summary,
            'probability_stats': {
                'mean_return': prob_dist.get('mean_return'),
                'std_return': prob_dist.get('std_return'),
                'sample_size': prob_dist.get('sample_size')
            },
            'confidence_levels': confidence_levels or [],
            'vix_analysis': vix_summary,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return formatted_data
    
    def create_options_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """
        Create a comprehensive prompt for options strategy analysis
        
        Args:
            data: Formatted options analysis data
            
        Returns:
            Detailed prompt for LLM analysis
        """
        
        prompt = f"""
You are a professional options trading analyst with expertise in volatility analysis and risk management. 
Analyze the following options strategy data and provide a comprehensive summary with actionable recommendations.

ANALYSIS DATA:
=============

Ticker: {data['ticker']}
Current Price: ${data['current_price']:.2f}
Strategy Timeframe: {data['strategy_timeframe']}
ATR (Average True Range): ${data['atr']:.2f}

BEST PUT RECOMMENDATION:
=====================
Strike Price: ${data['best_recommendation']['strike']:.2f}
Distance from Current: {data['best_recommendation']['distance_pct']:.1f}%
Probability of Hit: {data['best_recommendation']['prob_below']:.1%}
Safety Score: {data['best_recommendation']['safety_score']:.1%}

TOP 3 STRIKE RECOMMENDATIONS:
==========================
"""
        
        for rec in data['top_recommendations']:
            prompt += f"{rec['rank']}. ${rec['strike']:.2f} - {rec['distance_pct']:.1f}% away, {rec['prob_below']:.1%} hit prob, {rec['safety_score']:.1%} safety\n"
        
        prompt += f"""
STATISTICAL ANALYSIS:
==================
Mean Return: {data['probability_stats']['mean_return']:.4f}
Standard Deviation: {data['probability_stats']['std_return']:.4f}
Sample Size: {data['probability_stats']['sample_size']} periods

CONFIDENCE LEVELS:
================
"""
        
        for conf in data['confidence_levels']:
            prompt += f"{conf['Confidence Level']}: ${conf['Lower Bound']} - ${conf['Upper Bound']} (Range: ${conf['Range Width']})\n"
        
        if data['vix_analysis']:
            prompt += f"""
VIX MARKET CONDITIONS:
====================
Current VIX: {data['vix_analysis']['current_vix']:.2f}
Market Condition: {data['vix_analysis']['condition']}
Trade Approved: {'Yes' if data['vix_analysis']['trade_approved'] else 'No'}
"""
        
        prompt += """

ANALYSIS REQUIREMENTS:
====================
Please provide a comprehensive analysis including:

1. **EXECUTIVE SUMMARY** (2-3 sentences)
   - Overall market assessment and strategy viability

2. **STRATEGY ANALYSIS** 
   - Quality of the recommended strikes
   - Risk/reward assessment
   - Probability analysis interpretation

3. **MARKET CONDITIONS**
   - VIX impact on strategy (if available)
   - Volatility environment assessment
   - Timing considerations

4. **RISK MANAGEMENT**
   - Key risks to monitor
   - Stop-loss considerations
   - Position sizing guidance

5. **ACTIONABLE RECOMMENDATIONS**
   - Specific trading actions
   - Entry/exit criteria
   - Alternative strategies if current conditions are unfavorable

6. **KEY METRICS TO WATCH**
   - Important indicators to monitor
   - Warning signs to exit

Format your response with clear headers and bullet points. Be specific, actionable, and professional.
Focus on practical trading insights rather than general market commentary.
"""
        
        return prompt
    
    def generate_analysis(self, prompt: str, max_tokens: int = 1500) -> Dict[str, Any]:
        """
        Generate LLM analysis using OpenAI GPT-4o-mini
        
        Args:
            prompt: Analysis prompt
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary with analysis result and metadata
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional options trading analyst with 15+ years of experience in volatility analysis, risk management, and quantitative trading strategies. Provide clear, actionable insights."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more consistent analysis
                top_p=0.9
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                'success': True,
                'analysis': analysis_text,
                'tokens_used': response.usage.total_tokens,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_options_strategy(self,
                                ticker: str,
                                current_price: float,
                                strategy_timeframe: str,
                                recommendations: List[Dict],
                                prob_dist: Dict,
                                vix_data: Optional[Dict] = None,
                                atr: float = 0,
                                confidence_levels: List[Dict] = None) -> Dict[str, Any]:
        """
        Complete options strategy analysis pipeline
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            strategy_timeframe: 'daily' or 'weekly'
            recommendations: List of strike recommendations
            prob_dist: Probability distribution data
            vix_data: VIX market condition data
            atr: Average True Range value
            confidence_levels: 90%, 95%, 99% probability ranges
            
        Returns:
            Complete analysis result with formatted data and LLM insights
        """
        
        # Format data for analysis
        formatted_data = self.format_options_data(
            ticker, current_price, strategy_timeframe,
            recommendations, prob_dist, vix_data, atr, confidence_levels
        )
        
        # Create analysis prompt
        prompt = self.create_options_analysis_prompt(formatted_data)
        
        # Generate LLM analysis
        llm_result = self.generate_analysis(prompt)
        
        return {
            'formatted_data': formatted_data,
            'llm_analysis': llm_result,
            'prompt_used': prompt
        }
    
    def create_market_summary_prompt(self, ticker_results: Dict[str, Any], vix_data: Optional[Dict] = None) -> str:
        """
        Create prompt for overall market condition summary
        
        Args:
            ticker_results: Results from multiple ticker analysis
            vix_data: VIX market condition data
            
        Returns:
            Market summary prompt
        """
        
        prompt = f"""
You are a senior market analyst. Provide a concise market overview based on the following volatility analysis:

ANALYZED TICKERS AND ATR VALUES:
==============================
"""
        
        for ticker, results in ticker_results.items():
            daily_atr = results.get('daily', {}).get('atr', 0) if results.get('daily') else 0
            weekly_atr = results.get('weekly', {}).get('atr', 0) if results.get('weekly') else 0
            prompt += f"{ticker}: Daily ATR ${daily_atr:.2f}, Weekly ATR ${weekly_atr:.2f}\n"
        
        if vix_data:
            prompt += f"""
VIX CONDITIONS:
=============
Current VIX: {vix_data.get('current_vix', 'N/A')}
Market Condition: {vix_data.get('condition', 'Unknown')}
"""
        
        prompt += """
Provide a brief 3-4 sentence market summary focusing on:
1. Overall volatility environment
2. Trading opportunities or risks
3. Key recommendations for options traders
"""
        
        return prompt
    
    def generate_market_summary(self, ticker_results: Dict[str, Any], vix_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate overall market condition summary
        
        Args:
            ticker_results: Results from multiple ticker analysis
            vix_data: VIX market condition data
            
        Returns:
            Market summary analysis
        """
        
        prompt = self.create_market_summary_prompt(ticker_results, vix_data)
        return self.generate_analysis(prompt, max_tokens=300)

    def generate_custom_analysis(self, prompt):
        """
        Generate custom analysis using provided prompt
        
        Args:
            prompt: Custom analysis prompt
            
        Returns:
            Dictionary with analysis results
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional financial analyst specializing in options trading, 
                        volatility analysis, and market conditions. Provide clear, concise, and actionable analysis 
                        based on the data provided. Use specific numbers and be authoritative in your recommendations."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'analysis': analysis_text,
                'tokens_used': response.usage.total_tokens,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis': None,
                'tokens_used': 0,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }

    def format_iron_condor_data(self,
                               ticker: str,
                               current_price: float,
                               strategies: List[Dict],
                               analysis_results: Dict,
                               best_strategy: Dict) -> Dict[str, Any]:
        """
        Format Iron Condor analysis data for LLM analysis
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            strategies: List of all analyzed strategies
            analysis_results: Complete analysis results
            best_strategy: Top recommended strategy
            
        Returns:
            Formatted data dictionary for LLM analysis
        """
        
        # Extract key metrics from best strategy
        credit_check = best_strategy.get('credit_check', {})
        
        # Format top 3 strategies for comparison
        top_strategies = []
        for i, strategy in enumerate(strategies[:3]):
            top_strategies.append({
                'rank': i + 1,
                'strategy_type': strategy.get('strategy_type', 'Unknown'),
                'wing_width': strategy.get('wing_width', 0),
                'total_credit': strategy.get('total_credit', 0),
                'pop_black_scholes': strategy.get('pop_black_scholes', 0),
                'roc_percent': strategy.get('roc_percent', 0),
                'net_theta': strategy.get('net_theta', 0),
                'risk_reward_ratio': strategy.get('risk_reward_ratio', 0),
                'call_structure': f"{strategy.get('call_short', 0):.0f}/{strategy.get('call_long', 0):.0f}",
                'put_structure': f"{strategy.get('put_long', 0):.0f}/{strategy.get('put_short', 0):.0f}",
                'meets_credit_req': strategy.get('credit_check', {}).get('meets_requirement', False)
            })
        
        # Market condition assessment
        market_data = {
            'dte': analysis_results.get('dte', 0),
            'volatility': analysis_results.get('volatility', 0),
            'iv_rank': analysis_results.get('iv_rank'),
            'total_strategies_found': len(strategies),
            'bread_butter_count': analysis_results.get('bread_butter_count', 0),
            'other_strategies_count': analysis_results.get('other_strategies_count', 0)
        }
        
        # Best strategy detailed metrics
        best_metrics = {
            'strategy_type': best_strategy.get('strategy_type', 'Unknown'),
            'wing_width': best_strategy.get('wing_width', 0),
            'total_credit': best_strategy.get('total_credit', 0),
            'max_profit': best_strategy.get('max_profit', 0),
            'max_loss': best_strategy.get('max_loss', 0),
            'pop_black_scholes': best_strategy.get('pop_black_scholes', 0),
            'roc_percent': best_strategy.get('roc_percent', 0),
            'net_theta': best_strategy.get('net_theta', 0),
            'profit_zone_pct': best_strategy.get('profit_zone_pct', 0),
            'risk_reward_ratio': best_strategy.get('risk_reward_ratio', 0),
            'credit_efficiency': credit_check.get('credit_ratio', 0),
            'meets_bread_butter_rule': credit_check.get('meets_requirement', False),
            'breakeven_lower': best_strategy.get('lower_breakeven', 0),
            'breakeven_upper': best_strategy.get('upper_breakeven', 0),
            'call_short': best_strategy.get('call_short', 0),
            'call_long': best_strategy.get('call_long', 0),
            'put_short': best_strategy.get('put_short', 0),
            'put_long': best_strategy.get('put_long', 0)
        }
        
        formatted_data = {
            'ticker': ticker,
            'current_price': current_price,
            'market_environment': market_data,
            'best_strategy_metrics': best_metrics,
            'top_strategies_comparison': top_strategies,
            'credit_analysis': {
                'bread_butter_strategies': analysis_results.get('bread_butter_count', 0),
                'total_strategies': len(strategies),
                'best_credit_efficiency': credit_check.get('credit_ratio', 0),
                'meets_1_3_rule': credit_check.get('meets_requirement', False)
            },
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return formatted_data
    
    def create_iron_condor_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """
        Create comprehensive prompt for Iron Condor strategy analysis
        
        Args:
            data: Formatted Iron Condor analysis data
            
        Returns:
            Detailed prompt for LLM analysis
        """
        
        best = data['best_strategy_metrics']
        market = data['market_environment']
        credit = data['credit_analysis']
        
        prompt = f"""
You are a professional options trader specializing in Iron Condor strategies with 15+ years of experience. 
Analyze the following Iron Condor strategy data and provide concise, actionable trading guidance.

MARKET ENVIRONMENT:
==================
Ticker: {data['ticker']}
Current Price: ${data['current_price']:.2f}
Days to Expiry: {market['dte']} days
Estimated Volatility: {market['volatility']:.1%}
IV Rank: {market['iv_rank']:.0f}% {"(HIGH - Favorable)" if market.get('iv_rank', 0) >= 30 else "(LOW - Unfavorable)" if market.get('iv_rank', 0) < 20 else "(MEDIUM)"}

RECOMMENDED IRON CONDOR:
=======================
Strategy Type: {best['strategy_type']}
Structure: SELL {best['call_short']:.0f}C / BUY {best['call_long']:.0f}C | SELL {best['put_short']:.0f}P / BUY {best['put_long']:.0f}P
Wing Width: ${best['wing_width']:.1f}
Total Credit: ${best['total_credit']:.2f} per share
Credit Efficiency: {best['credit_efficiency']:.1%} ({"✅ Meets 1/3 Rule" if best['meets_bread_butter_rule'] else "❌ Below 1/3 Rule"})

PROBABILITY & RISK METRICS:
=========================
Probability of Profit (Black-Scholes): {best['pop_black_scholes']:.1%}
Return on Capital: {best['roc_percent']:.1f}%
Risk/Reward Ratio: {best['risk_reward_ratio']:.2f}:1
Daily Theta: ${best['net_theta']:.2f}
Profit Zone: {best['profit_zone_pct']:.1f}% of current price
Breakevens: ${best['breakeven_lower']:.2f} - ${best['breakeven_upper']:.2f}

STRATEGY LANDSCAPE:
==================
Total Strategies Analyzed: {credit['total_strategies']}
Bread & Butter Qualified: {credit['bread_butter_strategies']}
Best Credit Efficiency: {credit['best_credit_efficiency']:.1%}

TOP 3 STRATEGY COMPARISON:
========================="""

        for strategy in data['top_strategies_comparison']:
            prompt += f"""
{strategy['rank']}. {strategy['strategy_type']} - ${strategy['wing_width']:.1f} width
   Credit: ${strategy['total_credit']:.2f} | POP: {strategy['pop_black_scholes']:.1%} | ROC: {strategy['roc_percent']:.1f}% | Theta: ${strategy['net_theta']:.2f}
   {"✅ Meets Credit Req" if strategy['meets_credit_req'] else "❌ Below Credit Req"}"""

        prompt += f"""

ANALYSIS REQUIREMENTS:
====================
Provide a focused analysis in 4 concise sections:

1. **STRATEGY ASSESSMENT** (2-3 sentences)
   - Quality of the recommended Iron Condor
   - Key strengths and potential concerns

2. **MARKET TIMING** (2-3 sentences)
   - Is this a good time to trade this strategy?
   - Volatility environment impact

3. **RISK MANAGEMENT** (2-3 bullet points)
   - Key risks to monitor
   - Exit criteria recommendations

4. **TRADE RECOMMENDATION** (2-3 sentences)
   - Clear go/no-go recommendation
   - Specific action items

Be specific, use the exact numbers provided, and focus on actionable insights.
Assume the reader understands Iron Condor basics - focus on this specific trade analysis.
"""
        
        return prompt
    
    def analyze_iron_condor_strategy(self,
                                   ticker: str,
                                   current_price: float,
                                   strategies: List[Dict],
                                   analysis_results: Dict,
                                   best_strategy: Dict) -> Dict[str, Any]:
        """
        Complete Iron Condor strategy analysis pipeline
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            strategies: List of all analyzed strategies
            analysis_results: Complete analysis results
            best_strategy: Top recommended strategy
            
        Returns:
            Complete analysis result with formatted data and LLM insights
        """
        
        # Format data for analysis
        formatted_data = self.format_iron_condor_data(
            ticker, current_price, strategies, analysis_results, best_strategy
        )
        
        # Create analysis prompt
        prompt = self.create_iron_condor_analysis_prompt(formatted_data)
        
        # Generate LLM analysis
        llm_result = self.generate_analysis(prompt, max_tokens=800)
        
        return {
            'formatted_data': formatted_data,
            'llm_analysis': llm_result,
            'prompt_used': prompt
        }

# Utility functions for Streamlit integration
def get_llm_analyzer() -> Optional[LLMAnalyzer]:
    """
    Create LLM analyzer instance with error handling for Streamlit
    
    Returns:
        LLMAnalyzer instance or None if API key not available
    """
    try:
        return LLMAnalyzer()
    except ValueError:
        return None

def format_vix_data_for_llm(vix_value: float, condition: str, trade_approved: bool) -> Dict[str, Any]:
    """
    Format VIX data for LLM analysis
    
    Args:
        vix_value: Current VIX value
        condition: VIX condition description
        trade_approved: Whether trading is approved based on VIX
        
    Returns:
        Formatted VIX data dictionary
    """
    return {
        'current_vix': vix_value,
        'condition': condition,
        'trade_approved': trade_approved
    }

def format_confidence_levels_for_llm(confidence_data: List[Dict]) -> List[Dict]:
    """
    Format confidence level data for LLM analysis
    
    Args:
        confidence_data: List of confidence level dictionaries
        
    Returns:
        Formatted confidence data for LLM
    """
    return confidence_data 