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
        
        # Model configuration - Updated to use GPT-5-mini (now vision-capable)
        self.model = "gpt-5-mini"  # Primary model - GPT-5-mini with vision support
        self.fallback_model = "gpt-4o"  # Fallback - GPT-4o for compatibility
        
        # Debug logging
        print(f"🔧 ChartAnalyzer initialized with:")
        print(f"   Primary Model: {self.model} (Vision: ✅ - GPT-5 models are now multimodal!)")
        print(f"   Fallback Model: {self.fallback_model} (Vision: ✅)")
        print(f"   OpenAI Client: {type(self.client).__name__}")
        
        # Check model availability
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check which models are available and log the status"""
        try:
            # Get available models (this is a simple check)
            print(f"🔍 Checking model availability...")
            print(f"   Primary model set to: {self.model}")
            print(f"   Fallback model set to: {self.fallback_model}")
            
            # Note: We can't easily check model availability without making an API call
            # But we can log what we're configured to use
            print(f"   Ready to use {self.model} for chart analysis")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not verify model availability: {e}")
            print(f"   Will attempt to use configured models during analysis")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured models"""
        return {
            'primary_model': self.model,
            'fallback_model': self.fallback_model,
            'openai_client_type': type(self.client).__name__,
            'vision_support': self._check_vision_support()
        }
    
    def _get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get the correct API parameters for a specific model"""
        if 'gpt-5' in model_name:
            return {
                'max_completion_tokens': 4000,  # GPT-5 models use this
                'temperature': 1,  # GPT-5 models only support default temperature
                'parameter_type': 'max_completion_tokens'
            }
        else:
            return {
                'max_tokens': 4000,  # GPT-4 and other models use this
                'temperature': 0.1,  # GPT-4 models support custom temperature
                'parameter_type': 'max_tokens'
            }
    
    def _make_api_call(self, model_name: str, messages: List[Dict], is_fallback: bool = False) -> Tuple[Any, float]:
        """Make an API call with the correct parameters for the model"""
        model_params = self._get_model_parameters(model_name)
        
        print(f"   Making OpenAI API call{' with fallback' if is_fallback else ''}...")
        print(f"   Model: {model_name}")
        print(f"   {model_params['parameter_type']}: {model_params.get('max_completion_tokens', model_params.get('max_tokens'))}")
        print(f"   Temperature: {model_params['temperature']}")
        
        start_time = datetime.now()
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            **{k: v for k, v in model_params.items() if k != 'parameter_type'}
        )
        end_time = datetime.now()
        
        api_duration = (end_time - start_time).total_seconds()
        print(f"   ✅ API call successful!")
        print(f"   Response time: {api_duration:.2f} seconds")
        print(f"   Model used: {response.model}")
        print(f"   Tokens used: {response.usage.total_tokens if response.usage else 'Unknown'}")
        
        return response, api_duration
    
    def _check_vision_support(self) -> Dict[str, bool]:
        """Check which models support vision capabilities"""
        vision_models = {
            # GPT-4 models
            'gpt-4-vision-preview': True,
            'gpt-4o': True,
            'gpt-4o-mini': True,
            # GPT-5 models - NOW SUPPORT VISION (multimodal)
            'gpt-5-mini': True,  # Updated: GPT-5-mini now supports vision
            'gpt-5': True,       # Updated: GPT-5 now supports vision
            'gpt-5-nano': True,  # Updated: GPT-5-nano now supports vision
            # Legacy models
            'gpt-3.5-turbo': False
        }
        
        return {
            'primary_supports_vision': vision_models.get(self.model, False),
            'fallback_supports_vision': vision_models.get(self.fallback_model, False)
        }
    
    def set_model(self, model_name: str, is_fallback: bool = False):
        """Set the primary or fallback model"""
        if is_fallback:
            self.fallback_model = model_name
            print(f"🔄 Fallback model updated to: {model_name}")
        else:
            self.model = model_name
            print(f"🔄 Primary model updated to: {model_name}")
        
        # Re-check vision support
        vision_info = self._check_vision_support()
        print(f"   Vision support - Primary: {vision_info['primary_supports_vision']}, Fallback: {vision_info['fallback_supports_vision']}")
    
    def list_available_models(self):
        """List commonly available OpenAI models for reference"""
        print(f"\n📋 Commonly Available OpenAI Models:")
        print(f"   Vision Models (for chart analysis):")
        print(f"     • gpt-5-mini - Latest reasoning model with vision support ✅")
        print(f"     • gpt-5 - Latest reasoning model with vision support ✅")
        print(f"     • gpt-5-nano - Latest reasoning model with vision support ✅")
        print(f"     • gpt-4o - High quality vision model ✅")
        print(f"     • gpt-4o-mini - Cost-effective vision model ✅")
        print(f"     • gpt-4-vision-preview - Highest quality vision model ✅")
        print(f"   Text-Only Models (legacy):")
        print(f"     • gpt-3.5-turbo - Legacy model, no vision ❌")
        print(f"   Current Configuration:")
        print(f"     • Primary: {self.model} (Vision: {self._check_vision_support()['primary_supports_vision']})")
        print(f"     • Fallback: {self.fallback_model} (Vision: {self._check_vision_support()['fallback_supports_vision']})")
    
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
    
    def get_default_system_prompt(self) -> str:
        """Get default system prompt for chart analysis"""
        return """You are an expert technical analyst and options strategist with 20+ years of experience on Wall Street. You specialize in:

1. **Chart Pattern Recognition**: Expert identification of technical patterns (wedges, triangles, channels, flags, head & shoulders, etc.)
2. **Price Action Analysis**: Deep understanding of candlestick patterns, momentum shifts, and market microstructure
3. **Volume Analysis**: Institutional activity patterns, accumulation/distribution, and volume-price relationships
4. **Options Strategy**: Professional-grade options recommendations with specific strikes, timeframes, and risk management
5. **Risk Management**: Institutional-level position sizing, stop-loss strategies, and portfolio risk assessment

Your analysis should be:
- **Actionable**: Provide specific price levels, strike recommendations, and timeframes
- **Professional**: Use institutional-grade analysis frameworks and terminology
- **Risk-Aware**: Always include position sizing, stop losses, and risk/reward ratios
- **Evidence-Based**: Support recommendations with technical analysis and pattern recognition

Focus on providing institutional-quality analysis that professional traders can implement immediately.
Your analysis should ALWAYS be based on critical thinking and chart pattern technical analysis, with correct market structure"""

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
    
    def analyze_text_only(self, text_content: str, analysis_type: str = "deep", 
                         additional_context: str = "", system_prompt: str = "") -> Dict[str, Any]:
        """Analyze text-only content using GPT-5-mini (no vision required)"""
        try:
            print(f"\n🚀 Starting text-only analysis...")
            print(f"   Analysis Type: {analysis_type}")
            print(f"   Content Length: {len(text_content)} characters")
            print(f"   Additional Context: {'Yes' if additional_context else 'No'}")
            print(f"   System Prompt: {'Yes' if system_prompt else 'No'}")
            
            # Create prompt
            print(f"📝 Creating analysis prompt...")
            prompt = self.create_analysis_prompt(analysis_type)
            print(f"   Base prompt length: {len(prompt)} characters")
            
            if additional_context:
                prompt += f"\n\nAdditional Context: {additional_context}"
                print(f"   Added context, total prompt length: {len(prompt)} characters")
            
            # Prepare messages array
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
                print(f"   Added system message ({len(system_prompt)} characters)")
            
            # Add user message with text content
            messages.append({
                "role": "user",
                "content": f"{prompt}\n\nContent to Analyze:\n{text_content}"
            })
            print(f"   Added user message with text content")
            print(f"   Total messages: {len(messages)}")
            
            # Use GPT-5-mini for text analysis
            model_to_use = "gpt-5-mini"
            print(f"\n🤖 Using GPT-5-mini for text analysis: {model_to_use}")
            
            try:
                # Make API call with GPT-5-mini
                response, api_duration = self._make_api_call(model_to_use, messages)
                
            except Exception as e:
                print(f"   ❌ GPT-5-mini failed: {str(e)}")
                raise Exception(f"Text analysis failed with GPT-5-mini: {str(e)}")
            
            # Extract response
            print(f"\n📊 Processing API response...")
            content = response.choices[0].message.content
            print(f"   Response content length: {len(content)} characters")
            print(f"   First 100 chars: {content[:100]}...")
            
            # Create structured response
            analysis_data = {
                'analysis_content': content,
                'metadata': {
                    'analysis_type': f"{analysis_type}_text_only",
                    'timestamp': datetime.now().isoformat(),
                    'model': model_to_use,
                    'primary_model': self.model,
                    'fallback_used': False,
                    'api_duration_seconds': api_duration,
                    'tokens_used': response.usage.total_tokens if response.usage else None,
                    'response_model': response.model,
                    'content_type': 'text_only'
                }
            }
            
            print(f"✅ Text analysis completed successfully!")
            print(f"   Model used: {model_to_use}")
            print(f"   Analysis type: {analysis_type}")
            print(f"   Total processing time: {api_duration:.2f} seconds")
            
            return analysis_data
            
        except Exception as e:
            print(f"❌ Text analysis failed: {str(e)}")
            raise Exception(f"Text analysis failed: {str(e)}")
    
    def analyze_chart(self, image_file, analysis_type: str = "deep", 
                     additional_context: str = "", system_prompt: str = "") -> Dict[str, Any]:
        """Analyze chart image using GPT-5-mini model"""
        try:
            print(f"\n🚀 Starting chart analysis...")
            print(f"   Analysis Type: {analysis_type}")
            print(f"   Additional Context: {'Yes' if additional_context else 'No'}")
            print(f"   System Prompt: {'Yes' if system_prompt else 'No'}")
            print(f"   Image File: {getattr(image_file, 'name', 'Unknown')}")
            
            # Encode image
            print(f"📸 Encoding image to base64...")
            base64_image = self.encode_image(image_file)
            print(f"   Image encoded successfully ({len(base64_image)} characters)")
            
            # Create prompt
            print(f"📝 Creating analysis prompt...")
            prompt = self.create_analysis_prompt(analysis_type)
            print(f"   Base prompt length: {len(prompt)} characters")
            
            if additional_context:
                prompt += f"\n\nAdditional Context: {additional_context}"
                print(f"   Added context, total prompt length: {len(prompt)} characters")
            
            # Prepare messages array
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
                print(f"   Added system message ({len(system_prompt)} characters)")
            
            # Add user message with prompt and image
            messages.append({
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
            })
            print(f"   Added user message with prompt and image")
            print(f"   Total messages: {len(messages)}")
            
            # Try primary model first
            model_to_use = self.model
            print(f"\n🤖 Attempting analysis with primary model: {model_to_use}")
            
            try:
                # Make API call with primary model
                response, api_duration = self._make_api_call(model_to_use, messages)
                
            except Exception as primary_error:
                print(f"   ❌ Primary model failed: {str(primary_error)}")
                
                # Try fallback model if primary fails
                if hasattr(self, 'fallback_model'):
                    model_to_use = self.fallback_model
                    print(f"\n🔄 Trying fallback model: {model_to_use}")
                    
                    try:
                        response, api_duration = self._make_api_call(model_to_use, messages, is_fallback=True)
                        
                    except Exception as fallback_error:
                        print(f"   ❌ Fallback model also failed: {str(fallback_error)}")
                        raise Exception(f"Both primary ({self.model}) and fallback ({self.fallback_model}) models failed. Primary error: {str(primary_error)}, Fallback error: {str(fallback_error)}")
                else:
                    raise primary_error
            
            # Extract response
            print(f"\n📊 Processing API response...")
            content = response.choices[0].message.content
            print(f"   Response content length: {len(content)} characters")
            print(f"   First 100 chars: {content[:100]}...")
            
            # Create structured response
            analysis_data = {
                'analysis_content': content,
                'metadata': {
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat(),
                    'model': model_to_use,
                    'primary_model': self.model,
                    'fallback_used': model_to_use != self.model,
                    'api_duration_seconds': api_duration,
                    'tokens_used': response.usage.total_tokens if response.usage else None,
                    'response_model': response.model
                }
            }
            
            print(f"✅ Chart analysis completed successfully!")
            print(f"   Final model used: {model_to_use}")
            print(f"   Analysis type: {analysis_type}")
            print(f"   Total processing time: {api_duration:.2f} seconds")
            
            return analysis_data
            
        except Exception as e:
            print(f"❌ Chart analysis failed: {str(e)}")
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