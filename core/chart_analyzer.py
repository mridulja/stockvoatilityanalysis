"""
Advanced Chart Analysis Module with OpenAI o3-mini Integration

This module provides comprehensive chart analysis capabilities including:
- Pattern recognition and technical analysis
- Price action and volume analysis
- Support/resistance level identification
- Gap analysis and trend assessment
- Options strategy recommendations based on chart patterns
- Risk management guidelines

Author: Enhanced by AI Assistant
Date: 2025
Version: 1.0
"""

import streamlit as st
import base64
import io
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class ChartAnalyzer:
    """Advanced chart analyzer using OpenAI o3-mini vision model"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the chart analyzer with OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Please install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        # Use GPT-4V for vision analysis since o3-mini doesn't support vision yet
        self.model = "gpt-4-vision-preview"  # Vision-capable model
        self.fallback_model = "gpt-4o"  # Alternative vision model
        
    def encode_image(self, image_file) -> str:
        """Convert uploaded image to base64 for API"""
        try:
            if hasattr(image_file, 'read'):
                # Streamlit uploaded file
                image_bytes = image_file.read()
                image_file.seek(0)  # Reset file pointer
            else:
                # File path or bytes
                with open(image_file, 'rb') as f:
                    image_bytes = f.read()
            
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Error encoding image: {str(e)}")
    
    def create_analysis_prompt(self, analysis_type: str = "deep") -> str:
        """Create structured prompt for chart analysis"""
        
        base_prompt = """
        You are a professional technical analyst and options strategist with institutional-level expertise. Analyze this stock chart image with the same depth and quality as a professional trading desk analyst.
        
        ### ANALYSIS FRAMEWORK
        Provide comprehensive analysis covering:
        
        1. **Chart Pattern Recognition** - Identify specific patterns (wedges, triangles, channels, flags, head & shoulders, etc.)
        2. **Trend Analysis** - Primary trend direction, strength, and duration assessment  
        3. **Price Action** - Candlestick patterns, momentum shifts, breakout/breakdown signals
        4. **Volume Analysis** - Volume confirmation, accumulation/distribution patterns, volume-price relationships
        5. **Support & Resistance** - Key price levels with strength assessment
        6. **Gap Analysis** - Identify gaps and their trading implications
        7. **Options Strategy Recommendations** - Specific strategies: Put Sell, Put Buy, Call Buy, Call Sell, Put Spreads
        
        ### FORMATTING REQUIREMENTS
        • Use clear section headers (### Pattern Analysis, ### Volume Analysis, etc.)
        • Put all price levels in bold (e.g., **$74.25**)
        • Include specific strike recommendations with rationale
        • Provide probability assessments where relevant
        • Include risk management guidelines (stop losses, position sizing)
        • Add timeframe considerations for each strategy
        
        ### OUTPUT STRUCTURE
        Structure your response with these sections:
        - **Pattern Analysis** (pattern type, confidence level, implications)
        - **Trend & Momentum** (direction, strength, sustainability)  
        - **Volume Insights** (confirmation signals, institutional activity)
        - **Key Levels** (support/resistance with price targets)
        - **Options Strategy Recommendations** (2-3 specific strategies with strikes, timeframes, risk/reward)
        - **Risk Management** (stop levels, position sizing, time considerations)
        
        Be specific with price levels, strike recommendations, and timeframes. Provide actionable insights that a professional trader could implement immediately.
        
        **Disclaimer**: This is educational analysis, not financial advice.
        """
        
        if analysis_type == "quick":
            return base_prompt + "\n\n**QUICK MODE**: Focus on main pattern, trend, key levels, and top 2 options strategies."
        else:
            return base_prompt + "\n\n**DEEP MODE**: Provide exhaustive analysis with multiple scenarios and comprehensive strategy coverage."
    
    def analyze_chart(self, image_file, analysis_type: str = "deep", 
                     additional_context: str = "") -> Dict[str, Any]:
        """Analyze chart image using o3-mini model"""
        try:
            # Encode image
            base64_image = self.encode_image(image_file)
            
            # Create prompt
            prompt = self.create_analysis_prompt(analysis_type)
            
            if additional_context:
                prompt += f"\n\nAdditional Context: {additional_context}"
            
            # Try primary vision model first
            model_to_use = self.model
            try:
                # Make API call with vision model
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
            except Exception as primary_error:
                # Try fallback model if primary fails
                if hasattr(self, 'fallback_model'):
                    model_to_use = self.fallback_model
                    response = self.client.chat.completions.create(
                        model=model_to_use,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=4000,
                        temperature=0.1
                    )
                else:
                    raise primary_error
            
            # Extract response
            content = response.choices[0].message.content
            
            # Create structured response
            analysis_data = {
                'analysis_content': content,
                'metadata': {
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat(),
                    'model': model_to_use,
                    'primary_model': self.model,
                    'fallback_used': model_to_use != self.model
                }
            }
            
            return analysis_data
            
        except Exception as e:
            raise Exception(f"Chart analysis failed: {str(e)}")
    
    def get_strategy_details(self, strategy_name: str, current_price: float, 
                           support_level: float, resistance_level: float) -> Dict[str, Any]:
        """Get detailed options strategy implementation details"""
        
        strategies = {
            'Put Sell': {
                'description': 'Sell cash-secured puts to generate income',
                'strike_selection': f'Strike near support: ${support_level:.2f}',
                'max_profit': 'Premium collected',
                'max_loss': 'Unlimited (if assigned)',
                'break_even': 'Strike - Premium',
                'best_market_condition': 'Bullish to neutral, strong support'
            },
            'Put Buy': {
                'description': 'Buy puts for downside protection or speculation',
                'strike_selection': f'Strike near current price: ${current_price:.2f}',
                'max_profit': 'Unlimited downside',
                'max_loss': 'Premium paid',
                'break_even': 'Strike - Premium',
                'best_market_condition': 'Bearish, breakdown expected'
            },
            'Call Buy': {
                'description': 'Buy calls for upside participation',
                'strike_selection': f'Strike near resistance: ${resistance_level:.2f}',
                'max_profit': 'Unlimited upside',
                'max_loss': 'Premium paid',
                'break_even': 'Strike + Premium',
                'best_market_condition': 'Bullish, breakout expected'
            },
            'Call Sell': {
                'description': 'Sell calls to generate income (covered or naked)',
                'strike_selection': f'Strike near resistance: ${resistance_level:.2f}',
                'max_profit': 'Premium collected',
                'max_loss': 'Unlimited (if naked)',
                'break_even': 'Strike + Premium',
                'best_market_condition': 'Neutral to bearish, strong resistance'
            },
            'Put Spread': {
                'description': 'Bull put spread for defined risk income',
                'strike_selection': f'Short: ${support_level:.2f}, Long: ${support_level-5:.2f}',
                'max_profit': 'Net premium collected',
                'max_loss': 'Spread width - Premium',
                'break_even': 'Short strike - Net premium',
                'best_market_condition': 'Moderately bullish, support holding'
            }
        }
        
        return strategies.get(strategy_name, {
            'description': 'Custom strategy based on chart analysis',
            'implementation': 'See detailed analysis for specifics'
        })

def validate_image_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file type
    if uploaded_file.type not in ['image/png', 'image/jpeg', 'image/jpg']:
        return False, "Please upload a PNG or JPG image file"
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File size too large. Please upload an image smaller than 10MB"
    
    return True, "Valid image file"

def format_analysis_for_display(analysis_data: Dict[str, Any]) -> str:
    """Format analysis data for display in Streamlit"""
    if 'full_analysis' in analysis_data:
        return analysis_data['full_analysis']
    
    formatted = []
    
    # Pattern Analysis
    if 'pattern_analysis' in analysis_data:
        pattern = analysis_data['pattern_analysis']
        formatted.append(f"## 📊 Pattern Analysis")
        formatted.append(f"**Pattern**: {pattern.get('primary_pattern', 'N/A')}")
        formatted.append(f"**Confidence**: {pattern.get('pattern_confidence', 0)*100:.1f}%")
        formatted.append(f"**Description**: {pattern.get('pattern_description', 'N/A')}")
        formatted.append("")
    
    # Trend Analysis
    if 'trend_analysis' in analysis_data:
        trend = analysis_data['trend_analysis']
        formatted.append(f"## 📈 Trend Analysis")
        formatted.append(f"**Primary Trend**: {trend.get('primary_trend', 'N/A')}")
        formatted.append(f"**Strength**: {trend.get('trend_strength', 'N/A')}")
        formatted.append(f"**Analysis**: {trend.get('trend_description', 'N/A')}")
        formatted.append("")
    
    # Options Strategies
    if 'options_strategies' in analysis_data:
        formatted.append(f"## 🎯 Recommended Options Strategies")
        for i, strategy in enumerate(analysis_data['options_strategies'], 1):
            formatted.append(f"### Strategy {i}: {strategy.get('strategy_name', 'N/A')}")
            formatted.append(f"**Type**: {strategy.get('strategy_type', 'N/A')}")
            formatted.append(f"**Rationale**: {strategy.get('rationale', 'N/A')}")
            formatted.append(f"**Risk Level**: {strategy.get('risk_level', 'N/A')}")
            formatted.append("")
    
    return "\n".join(formatted) 