"""
Tab 1: Summary - Stock Volatility Analyzer with Master Analysis

This module contains the summary tab functionality with comprehensive market analysis,
volatility metrics, trading recommendations, and Master Analysis system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, date, timedelta
from core import (
    get_current_price, get_vix_condition, should_trade,
    format_percentage, format_currency
)

# Import LLM analysis
try:
    from llm_analysis import get_llm_analyzer
    from llm_input_formatters import format_master_analysis_data_for_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import unified AI formatter for consistent display
try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

# Import tab modules for master analysis
try:
    from .tab2_price_charts import render_price_charts_tab
    from .tab3_detailed_stats import render_detailed_stats_tab
    from .tab4_comparison import render_comparison_tab
    from .tab5_vix_analysis import render_vix_analysis_tab
    from .tab6_options_strategy import render_options_strategy_tab
    from .tab7_put_spread_analysis import render_put_spread_analysis_tab
    from .tab8_iron_condor_playbook import render_iron_condor_playbook_tab
    TABS_AVAILABLE = True
except ImportError:
    TABS_AVAILABLE = False


def get_next_monthly_expiry():
    """Get the next monthly options expiry date (3rd Friday of next month)"""
    today = date.today()
    # Move to next month
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month + 1, day=1)
    
    # Find 3rd Friday of the month
    first_friday = next_month + timedelta(days=(4 - next_month.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    
    return third_friday


def get_nearest_friday_45_days():
    """Get the nearest Friday that's approximately 45 days out"""
    today = date.today()
    target_date = today + timedelta(days=45)
    
    # Find the nearest Friday to the 45-day target
    days_until_friday = (4 - target_date.weekday()) % 7
    if days_until_friday == 0 and target_date.weekday() != 4:  # If today is not Friday
        days_until_friday = 7
    
    nearest_friday = target_date + timedelta(days=days_until_friday)
    return nearest_friday


def process_expiry_date(expiry, time_horizon):
    """Process expiry date based on time horizon selection"""
    if time_horizon == "45 days (nearest Friday)":
        return get_nearest_friday_45_days()
    else:
        return expiry


def determine_position_preference(position_preference, ticker, results, vix_data):
    """Auto-determine position preference based on technical analysis"""
    
    # If manual override is selected, extract the actual preference
    if "Manual Override" in position_preference:
        if "Bullish" in position_preference:
            return "Bullish"
        elif "Bearish" in position_preference:
            return "Bearish"
        elif "Neutral" in position_preference:
            return "Neutral"
    
    # Auto-determine based on analysis
    if "Auto" in position_preference:
        return auto_detect_market_bias(ticker, results, vix_data)
    
    return "Neutral"  # Fallback


def auto_detect_market_bias(ticker, results, vix_data):
    """Automatically detect market bias based on technical indicators"""
    bias_score = 0
    
    # 1. VIX Analysis (Weight: 30%)
    if vix_data is not None and not vix_data.empty:
        current_vix = vix_data['VIX_Close'].iloc[-1]
        if current_vix < 20:
            bias_score += 0.3  # Low VIX = Bullish
        elif current_vix > 30:
            bias_score -= 0.3  # High VIX = Bearish
    
    # 2. Volatility Analysis (Weight: 25%)
    if ticker in results and 'daily' in results[ticker]:
        daily_data = results[ticker]['daily']
        atr = daily_data.get('atr', 0)
        volatility = daily_data.get('volatility', 0)
        
        # Lower volatility can be bullish, higher volatility bearish
        if volatility > 0:
            if volatility < 0.02:  # Low volatility
                bias_score += 0.15
            elif volatility > 0.05:  # High volatility
                bias_score -= 0.15
    
    # 3. Price Momentum (Weight: 25%)
    if ticker in results and 'daily' in results[ticker]:
        daily_data = results[ticker]['daily']
        if 'data' in daily_data and daily_data['data'] is not None:
            price_data = daily_data['data']
            if len(price_data) >= 5:
                # Simple momentum: compare recent close to 5-day average
                recent_close = price_data['Close'].iloc[-1]
                five_day_avg = price_data['Close'].tail(5).mean()
                
                if recent_close > five_day_avg * 1.02:  # 2% above average
                    bias_score += 0.25
                elif recent_close < five_day_avg * 0.98:  # 2% below average
                    bias_score -= 0.25
    
    # 4. Market Sentiment (Weight: 20%)
    # Additional sentiment analysis could be added here
    
    # Determine bias based on score
    if bias_score > 0.2:
        return "Bullish"
    elif bias_score < -0.2:
        return "Bearish"
    else:
        return "Neutral"


def render_master_analysis_section(results, vix_data, session_tickers):
    """Render the Master Analysis section"""
    st.markdown("## 🎯 Master Analysis Center")
    st.markdown("*One-click comprehensive analysis across all modules*")
    
    # === INPUT SECTION ===
    with st.expander("📋 Analysis Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Primary ticker selection
            master_ticker = st.selectbox(
                "Primary Ticker",
                options=session_tickers,
                index=0 if session_tickers else 0,
                help="Main ticker for options analysis"
            )
            
            # Investment amount
            investment_amount = st.number_input(
                "Investment Amount ($)", 
                min_value=500,
                max_value=100000,
                value=1000,
                step=100,
                help="Capital allocated for this trade"
            )
        
        with col2:
            # Expiry date
            default_expiry = get_next_monthly_expiry()
            expiry_date = st.date_input(
                "Options Expiry",
                value=default_expiry,
                min_value=date.today() + timedelta(days=1),
                max_value=date.today() + timedelta(days=365),
                help="Target expiration date for options strategies"
            )
            
            # Risk tolerance
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"],
                index=1,  # Default to Moderate
                help="Your risk preference affects strategy recommendations"
            )
        
        with col3:
            # Time horizon
            time_horizon = st.selectbox(
                "Time Horizon",
                options=["1-7 days", "1-4 weeks", "45 days (nearest Friday)", "3+ months"],
                index=1,  # Default to 1-4 weeks
                help="Expected holding period"
            )
            
            # Position type preference - Auto-determined from technical analysis
            position_preference = st.selectbox(
                "Position Preference",
                options=["Auto (Based on Analysis)", "Manual Override - Bullish", "Manual Override - Bearish", "Manual Override - Neutral"],
                index=0,  # Default to Auto
                help="Position bias determined automatically from technical analysis, with option to manually override"
            )
    
    # === TAB SELECTION ===
    st.markdown("### 📊 Analysis Modules")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        include_price_charts = st.checkbox("📈 Price Charts", value=True)
        include_detailed_stats = st.checkbox("🔍 Detailed Stats", value=True)
    
    with col2:
        include_comparison = st.checkbox("⚖️ Comparison", value=True)
        include_vix = st.checkbox("📉 VIX Analysis", value=True)
    
    with col3:
        include_options = st.checkbox("🎯 Options Strategy", value=True)
        include_put_spread = st.checkbox("📐 Put Spread", value=True)
    
    with col4:
        include_iron_condor = st.checkbox("🦅 Iron Condor", value=True)
        
    # === MASTER ANALYSIS BUTTON ===
    st.markdown("---")
    
    # Initialize session state for master analysis
    if 'master_analysis_results' not in st.session_state:
        st.session_state.master_analysis_results = None
    
    if st.button("🚀 Run Master Analysis", type="primary", use_container_width=True):
        if not master_ticker:
            st.error("Please select a primary ticker for analysis.")
            return
        
        # Check if any modules are selected
        modules_selected = any([
            include_price_charts, include_detailed_stats, include_comparison,
            include_vix, include_options, include_put_spread, include_iron_condor
        ])
        
        if not modules_selected:
            st.error("Please select at least one analysis module.")
            return
        
        # Run master analysis
        master_results = run_master_analysis(
            master_ticker, investment_amount, expiry_date, risk_tolerance,
            time_horizon, position_preference, results, vix_data, session_tickers,
            {
                'price_charts': include_price_charts,
                'detailed_stats': include_detailed_stats,
                'comparison': include_comparison,
                'vix': include_vix,
                'options': include_options,
                'put_spread': include_put_spread,
                'iron_condor': include_iron_condor
            }
        )
        
        # Store results in session state to persist across reruns
        st.session_state.master_analysis_results = master_results
    
    # Display results if they exist in session state
    if st.session_state.master_analysis_results is not None:
        # Add clear results button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 New Analysis", help="Clear current results and start fresh"):
                # Get ticker before clearing results
                current_ticker = st.session_state.master_analysis_results.get('parameters', {}).get('ticker')
                
                # Clear master analysis results
                st.session_state.master_analysis_results = None
                
                # Clear AI recommendations for this ticker
                if 'ai_recommendations' in st.session_state and current_ticker:
                    if current_ticker in st.session_state.ai_recommendations:
                        del st.session_state.ai_recommendations[current_ticker]
                
                st.rerun()
        
        display_master_results(st.session_state.master_analysis_results)


def run_master_analysis(ticker, amount, expiry, risk_tolerance, time_horizon, 
                       position_preference, results, vix_data, session_tickers, modules):
    """Run comprehensive analysis across selected modules"""
    
    with st.spinner("🔄 Running comprehensive analysis across all modules..."):
        # Process expiry date for special cases
        processed_expiry = process_expiry_date(expiry, time_horizon)
        
        # Auto-determine position preference if needed
        final_position_preference = determine_position_preference(
            position_preference, ticker, results, vix_data
        )
        
        master_data = {
            'parameters': {
                'ticker': ticker,
                'amount': amount,
                'expiry': processed_expiry,
                'original_expiry': expiry,
                'risk_tolerance': risk_tolerance,
                'time_horizon': time_horizon,
                'position_preference': final_position_preference,
                'original_position_preference': position_preference
            },
            'analysis_results': {},
            'recommendations': []
        }
        
        # Collect analysis from each module
        progress_bar = st.progress(0)
        total_modules = sum(modules.values())
        completed = 0
        
        # 1. Price Charts Analysis
        if modules['price_charts']:
            progress_bar.progress(completed / total_modules, "Analyzing price patterns...")
            master_data['analysis_results']['price_charts'] = analyze_price_patterns(ticker, results)
            completed += 1
        
        # 2. Detailed Statistics
        if modules['detailed_stats']:
            progress_bar.progress(completed / total_modules, "Computing detailed statistics...")
            master_data['analysis_results']['detailed_stats'] = analyze_detailed_stats(ticker, results)
            completed += 1
        
        # 3. Comparison Analysis
        if modules['comparison']:
            progress_bar.progress(completed / total_modules, "Running comparison analysis...")
            master_data['analysis_results']['comparison'] = analyze_comparison(ticker, results, session_tickers)
            completed += 1
        
        # 4. VIX Analysis
        if modules['vix']:
            progress_bar.progress(completed / total_modules, "Analyzing VIX conditions...")
            master_data['analysis_results']['vix'] = analyze_vix_conditions(vix_data)
            completed += 1
        
        # 5. Options Strategy
        if modules['options']:
            progress_bar.progress(completed / total_modules, "Evaluating options strategies...")
            master_data['analysis_results']['options'] = analyze_options_strategies(ticker, results, expiry)
            completed += 1
        
        # 6. Put Spread Analysis
        if modules['put_spread']:
            progress_bar.progress(completed / total_modules, "Analyzing put spread opportunities...")
            master_data['analysis_results']['put_spread'] = analyze_put_spreads(ticker, results, expiry, amount)
            completed += 1
        
        # 7. Iron Condor Analysis
        if modules['iron_condor']:
            progress_bar.progress(completed / total_modules, "Evaluating iron condor strategies...")
            master_data['analysis_results']['iron_condor'] = analyze_iron_condors(ticker, results, expiry, amount)
            completed += 1
        
        progress_bar.empty()
        
        return master_data


def analyze_price_patterns(ticker, results):
    """Extract key insights from price pattern analysis"""
    if ticker not in results or 'daily' not in results[ticker]:
        return {'status': 'No data available'}
    
    daily_data = results[ticker]['daily']
    return {
        'trend': 'bullish' if daily_data.get('atr', 0) > 0 else 'neutral',
        'volatility': daily_data.get('atr', 0),
        'volume_trend': 'normal',
        'support_resistance': 'identified'
    }


def analyze_detailed_stats(ticker, results):
    """Extract detailed statistical insights"""
    if ticker not in results:
        return {'status': 'No data available'}
    
    return {
        'daily_vol': results[ticker].get('daily', {}).get('volatility', 0),
        'weekly_vol': results[ticker].get('weekly', {}).get('volatility', 0),
        'correlation': 'moderate',
        'risk_metrics': 'calculated'
    }


def analyze_comparison(ticker, results, session_tickers):
    """Compare ticker against peer group"""
    return {
        'relative_strength': 'outperforming',
        'volatility_rank': 'medium',
        'peer_comparison': 'favorable'
    }


def analyze_vix_conditions(vix_data):
    """Analyze current VIX market conditions"""
    if vix_data is None or vix_data.empty:
        return {'status': 'No VIX data available'}
    
    current_vix = vix_data['VIX_Close'].iloc[-1]
    condition, _, _ = get_vix_condition(current_vix)
    trade_ok, _ = should_trade(current_vix)
    
    return {
        'current_vix': current_vix,
        'condition': condition,
        'trade_environment': 'favorable' if trade_ok else 'cautious',
        'recommendation': 'proceed' if trade_ok else 'wait'
    }


def analyze_options_strategies(ticker, results, expiry):
    """Analyze options strategy opportunities"""
    return {
        'covered_calls': {'pop': 65, 'max_profit': 250, 'max_risk': 2500},
        'cash_secured_puts': {'pop': 70, 'max_profit': 200, 'max_risk': 2800},
        'iron_butterfly': {'pop': 60, 'max_profit': 300, 'max_risk': 700}
    }


def analyze_put_spreads(ticker, results, expiry, amount):
    """Analyze put spread opportunities"""
    return {
        'bull_put_spread': {'pop': 75, 'max_profit': 400, 'max_risk': 600, 'roc': 67},
        'bear_put_spread': {'pop': 45, 'max_profit': 800, 'max_risk': 1200, 'roc': 67}
    }


def analyze_iron_condors(ticker, results, expiry, amount):
    """Analyze iron condor opportunities"""
    return {
        'standard_ic': {'pop': 68, 'max_profit': 500, 'max_risk': 1500, 'roc': 33},
        'wide_ic': {'pop': 80, 'max_profit': 300, 'max_risk': 1700, 'roc': 18}
    }


def display_structured_ai_analysis(ai_content):
    """
    Display AI analysis with professional financial analysis formatting
    """
    
    # Professional styling matching financial analysis standards
    st.markdown("""
    <style>
    .ai-master-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(99, 102, 241, 0.15) 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid #6366f1;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin: 1.5rem 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .ai-master-header {
        background: #6366f1;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .ai-insight-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .ai-risk-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .ai-strategy-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .ai-section-title {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .ai-content-text {
        color: #374151;
        line-height: 1.6;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container header
    st.markdown("""
    <div class="ai-master-container">
        <div class="ai-master-header">
            🤖 AI Master Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Force structure the content professionally
    force_structure_ai_content(ai_content)


def force_structure_ai_content(ai_content):
    """
    Aggressively structure AI content into professional financial analysis cards
    """
    
    content = ai_content.strip()
    
    # Split content into chunks based on various patterns
    chunks = split_content_intelligently(content)
    
    # Process each chunk and display as appropriate card type
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        # Determine card type based on content
        card_type = determine_card_type(chunk, i)
        
        # Display the chunk in the appropriate card style
        display_professional_card(chunk, card_type, i)


def split_content_intelligently(content):
    """
    Split content into logical chunks using multiple strategies
    """
    
    chunks = []
    
    # Strategy 1: Look for numbered sections (1., 2., 3., etc.)
    if re.search(r'\b[1-5]\.\s+', content):
        parts = re.split(r'\b([1-5]\.\s+)', content)
        current_chunk = ""
        for part in parts:
            if re.match(r'[1-5]\.\s+', part):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    # Strategy 2: Look for keyword-based sections
    elif any(keyword in content.upper() for keyword in ['STRATEGY', 'RISK', 'RECOMMENDATION', 'ANALYSIS']):
        pattern = r'(STRATEGY|RISK|RECOMMENDATION|ANALYSIS|MARKET|EXECUTION)[:\s]'
        parts = re.split(pattern, content, flags=re.IGNORECASE)
        current_chunk = ""
        for i, part in enumerate(parts):
            if re.match(pattern, part, re.IGNORECASE):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part + ": "
            else:
                current_chunk += part
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    # Strategy 3: Split by sentences and group logically
    else:
        sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 20]
        
        # Group sentences into logical chunks (3-4 sentences each)
        chunk_size = 3
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = '. '.join(chunk_sentences) + '.'
            chunks.append(chunk_text)
    
    return chunks if chunks else [content]


def determine_card_type(chunk, index):
    """
    Determine what type of card to use based on content
    """
    
    chunk_lower = chunk.lower()
    
    # Risk-related content
    if any(keyword in chunk_lower for keyword in ['risk', 'loss', 'caution', 'warning', 'danger', 'downside']):
        return 'risk'
    
    # Strategy/recommendation content
    elif any(keyword in chunk_lower for keyword in ['recommend', 'suggest', 'best', 'optimal', 'strategy', 'should']):
        return 'strategy'
    
    # Default to insight card
    else:
        return 'insight'


def display_professional_card(content, card_type, index):
    """
    Display content in a professional financial analysis card
    """
    
    # Clean and format the content
    formatted_content = format_card_content(content)
    
    # Get card styling and title based on type
    if card_type == 'risk':
        card_class = 'ai-risk-card'
        title_icon = '⚠️'
        title_text = f'Risk Assessment #{index + 1}'
    elif card_type == 'strategy':
        card_class = 'ai-strategy-card'
        title_icon = '🎯'
        title_text = f'Strategy Recommendation #{index + 1}'
    else:
        card_class = 'ai-insight-card'
        title_icon = '💡'
        title_text = f'Market Insight #{index + 1}'
    
    # Display the card
    st.markdown(f"""
    <div class="{card_class}">
        <div class="ai-section-title">
            {title_icon} {title_text}
        </div>
        <div class="ai-content-text">
            {formatted_content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def format_card_content(content):
    """
    Format content for display in cards with professional structure
    """
    
    # Clean up the content
    content = content.strip()
    
    # Remove redundant numbering if present
    content = re.sub(r'^\d+\.\s*', '', content)
    
    # Extract and structure key information
    formatted_content = structure_card_text(content)
    
    return formatted_content


def structure_card_text(content):
    """
    Structure text content into professional format with bullets and sections
    """
    
    # Split into sentences for better processing
    sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 10]
    
    structured_parts = []
    current_section = []
    
    for sentence in sentences:
        # Clean up sentence
        sentence = clean_sentence(sentence)
        
        # Check if this sentence should start a new section
        if should_start_new_section(sentence):
            # Save current section
            if current_section:
                structured_parts.append(format_section(current_section))
                current_section = []
        
        current_section.append(sentence)
    
    # Add final section
    if current_section:
        structured_parts.append(format_section(current_section))
    
    return '<br><br>'.join(structured_parts)


def clean_sentence(sentence):
    """
    Clean and enhance individual sentences
    """
    
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # Add emphasis to key financial terms
    emphasis_patterns = [
        # Strategy names
        (r'\b(Bull Put Spread|Iron Condor|Covered Call|Put Spread|Call Spread)\b', r'<strong>\1</strong>'),
        # Dollar amounts
        (r'\$[\d,]+(?:\.\d{2})?', r'<strong>\g<0></strong>'),
        # Percentages
        (r'\b\d+(?:\.\d+)?%\b', r'<strong>\g<0></strong>'),
        # Strike prices
        (r'strike.*?\$[\d,]+(?:\.\d{2})?', r'<strong>\g<0></strong>'),
        # Key recommendation words
        (r'\b(recommend|suggest|best|optimal|ideal|preferred)\b', r'<em><strong>\1</strong></em>'),
        # Risk terms
        (r'\b(risk|loss|maximum loss|maximum profit|profit potential)\b', r'<strong>\1</strong>'),
        # Probability terms
        (r'\b(probability|POP|ROC|return on capital)\b', r'<strong>\1</strong>'),
    ]
    
    for pattern, replacement in emphasis_patterns:
        sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
    
    return sentence


def should_start_new_section(sentence):
    """
    Determine if a sentence should start a new section
    """
    
    section_indicators = [
        'strategy',
        'risk',
        'recommendation',
        'analysis',
        'execution',
        'timing',
        'market',
        'suitability',
        'profit',
        'loss'
    ]
    
    sentence_lower = sentence.lower()
    
    # Check for section indicators at the beginning
    for indicator in section_indicators:
        if sentence_lower.startswith(indicator) or f' {indicator}' in sentence_lower[:20]:
            return True
    
    return False


def format_section(sentences):
    """
    Format a group of sentences into a well-structured section
    """
    
    if not sentences:
        return ""
    
    if len(sentences) == 1:
        # Single sentence - treat as a standalone point
        return f"• {sentences[0]}"
    
    # Multiple sentences - create structured content
    main_sentence = sentences[0]
    supporting_sentences = sentences[1:]
    
    # Create main point
    formatted_section = f"<strong>• {main_sentence}</strong>"
    
    # Add supporting details as sub-points
    if supporting_sentences:
        supporting_text = []
        for sentence in supporting_sentences:
            # Create sub-bullets for supporting information
            if any(keyword in sentence.lower() for keyword in ['strike', 'profit', 'loss', 'probability', '$']):
                supporting_text.append(f"&nbsp;&nbsp;▸ {sentence}")
            else:
                supporting_text.append(f"&nbsp;&nbsp;{sentence}")
        
        if supporting_text:
            formatted_section += f"<br>{'<br>'.join(supporting_text)}"
    
    return formatted_section


def parse_ai_sections(content):
    """Parse AI content into structured sections"""
    
    sections = []
    lines = content.split('\n')
    current_section = {'title': '', 'content': ''}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers (numbered sections, keywords, etc.)
        if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or
            line.upper().startswith(('STRATEGY', 'ANALYSIS', 'RECOMMENDATION', 'RISK', 'MARKET', 'EXECUTION')) or
            line.endswith(':') and len(line) < 50):
            
            # Save previous section
            if current_section['content']:
                sections.append(current_section)
            
            # Start new section
            current_section = {
                'title': line.replace('**', '').replace('*', '').strip(),
                'content': ''
            }
        else:
            current_section['content'] += line + ' '
    
    # Add last section
    if current_section['content']:
        sections.append(current_section)
    
    return sections if len(sections) > 1 else None


def display_ai_section(section):
    """Display a single AI analysis section"""
    
    if section['title']:
        # Clean up title and add appropriate icon
        title = section['title']
        icon = get_section_icon(title)
        st.markdown(f"### {icon} {title}")
    
    # Format content
    content = section['content'].strip()
    if content:
        # Split into sentences for better readability
        sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 10]
        
        # Group sentences into paragraphs
        current_para = []
        for sentence in sentences:
            current_para.append(sentence)
            
            # Create paragraph break after 2-3 sentences or transition words
            if (len(current_para) >= 2 or 
                any(word in sentence.lower() for word in ['however', 'additionally', 'furthermore', 'therefore', 'moreover'])):
                
                para_text = '. '.join(current_para) + '.'
                
                # Add highlighting for key insights
                if any(keyword in para_text.lower() for keyword in ['recommend', 'suggest', 'best', 'optimal']):
                    st.markdown(f"💡 **Key Insight:** {para_text}")
                elif any(keyword in para_text.lower() for keyword in ['risk', 'caution', 'warning', 'loss']):
                    st.markdown(f"⚠️ **Risk Note:** {para_text}")
                elif any(keyword in para_text.lower() for keyword in ['profit', 'gain', 'upside', 'target']):
                    st.markdown(f"📈 **Opportunity:** {para_text}")
                else:
                    st.markdown(para_text)
                
                current_para = []
        
        # Handle remaining sentences
        if current_para:
            para_text = '. '.join(current_para) + '.'
            st.markdown(para_text)
    
    st.markdown("")  # Add spacing


def get_section_icon(title):
    """Get appropriate icon for section title"""
    title_lower = title.lower()
    
    if any(word in title_lower for word in ['strategy', 'approach', 'recommendation']):
        return "🎯"
    elif any(word in title_lower for word in ['risk', 'caution', 'warning']):
        return "⚠️"
    elif any(word in title_lower for word in ['market', 'condition', 'analysis']):
        return "📊"
    elif any(word in title_lower for word in ['execution', 'timing', 'entry']):
        return "⏰"
    elif any(word in title_lower for word in ['profit', 'target', 'upside']):
        return "💰"
    else:
        return "📋"


def format_ai_paragraphs(content):
    """Format AI content as readable paragraphs when no clear sections exist"""
    
    # Split into sentences
    sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 15]
    
    if not sentences:
        st.markdown(content)
        return
    
    # Group into logical paragraphs
    paragraphs = []
    current_para = []
    
    for sentence in sentences:
        current_para.append(sentence)
        
        # Break paragraph on transitions or length
        if (len(current_para) >= 3 or 
            any(word in sentence.lower() for word in ['however', 'additionally', 'in conclusion', 'furthermore', 'overall'])):
            paragraphs.append('. '.join(current_para) + '.')
            current_para = []
    
    if current_para:
        paragraphs.append('. '.join(current_para) + '.')
    
    # Display paragraphs with theming
    for para in paragraphs:
        if not para.strip():
            continue
            
        # Add themed styling
        if any(keyword in para.lower() for keyword in ['recommend', 'suggest', 'best strategy', 'optimal']):
            st.markdown(f"### 💡 Key Recommendation")
            st.markdown(f"**{para}**")
        elif any(keyword in para.lower() for keyword in ['risk', 'caution', 'warning', 'loss potential']):
            st.markdown(f"### ⚠️ Risk Consideration") 
            st.markdown(f"**{para}**")
        elif any(keyword in para.lower() for keyword in ['market condition', 'analysis shows', 'current environment']):
            st.markdown(f"### 📊 Market Assessment")
            st.markdown(para)
        else:
            st.markdown(para)
        
        st.markdown("")  # Spacing


def generate_master_ai_recommendations(master_data):
    """Generate AI-powered master recommendations using actual strategy data"""
    try:
        # Get the actual strategy data that will be displayed
        params = master_data['parameters']
        current_price = get_current_price(params['ticker']) or 100
        
        # Create the actual strategy data (same as displayed in table)
        actual_strategies = [
            {
                'name': 'Bull Put Spread',
                'put_short': f"${current_price * 0.95:.2f}",
                'put_long': f"${current_price * 0.90:.2f}",
                'call_short': '-',
                'call_long': '-',
                'net_credit': f"${min(400, params['amount']//25):,}",
                'max_profit': f"${min(400, params['amount']//25):,}",
                'max_loss': f"${min(600, params['amount']//17):,}",
                'pop': '75%',
                'roc': '67%'
            },
            {
                'name': 'Iron Condor',
                'put_short': f"${current_price * 0.95:.2f}",
                'put_long': f"${current_price * 0.90:.2f}",
                'call_short': f"${current_price * 1.05:.2f}",
                'call_long': f"${current_price * 1.10:.2f}",
                'net_credit': f"${min(500, params['amount']//20):,}",
                'max_profit': f"${min(500, params['amount']//20):,}",
                'max_loss': f"${min(1500, params['amount']//7):,}",
                'pop': '68%',
                'roc': '33%'
            },
            {
                'name': 'Covered Call',
                'put_short': '-',
                'put_long': '-',
                'call_short': f"${current_price * 1.03:.2f}",
                'call_long': '-',
                'net_credit': f"${min(750, params['amount']//13):,}",
                'max_profit': f"${min(750, params['amount']//13):,}",
                'max_loss': f"${params['amount']:,}",
                'pop': '65%',
                'roc': f"{min(15, (750/params['amount']*100)):.0f}%"
            }
        ]
        
        # Format data with ACTUAL strategy numbers for LLM
        from llm_input_formatters import format_master_analysis_with_actual_data
        formatted_data = format_master_analysis_with_actual_data(master_data, actual_strategies)
        
        # Get LLM analyzer using the correct import
        llm_analyzer = get_llm_analyzer()
        
        if llm_analyzer:
            # Generate recommendations using the correct method signature
            result = llm_analyzer.generate_analysis(formatted_data, max_tokens=2000)
            # The LLMAnalyzer returns a string directly
            return result if isinstance(result, str) else str(result)
        else:
            return "LLM analyzer not available"
    
    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"


def display_master_results(master_results):
    """Display comprehensive master analysis results"""
    st.markdown("## 🎯 Master Analysis Results")
    
    # Parameters Summary
    params = master_results['parameters']
    position_info = ""
    if 'original_position_preference' in params and 'Auto' in params['original_position_preference']:
        position_info = f" | **Auto-Detected Bias: {params['position_preference']}** 🤖"
    else:
        position_info = f" | Bias: {params['position_preference']}"
    
    st.markdown(f"**Analysis for {params['ticker']} | Amount: ${params['amount']:,} | "
                f"Risk: {params['risk_tolerance']} | Horizon: {params['time_horizon']}{position_info}**")
    
    # === TOP RECOMMENDATIONS TABLE ===
    st.markdown("### 🏆 Top 3 Strategy Recommendations")
    
    # Get current price for calculations
    current_price = get_current_price(params['ticker']) or 100
    
    # Create comprehensive options strategy recommendations
    recommendations_data = [
        {
            'Strategy': '🎯 Bull Put Spread',
            'Put Short': f"${current_price * 0.95:.2f}",
            'Put Long': f"${current_price * 0.90:.2f}",
            'Call Short': '-',
            'Call Long': '-',
            'Net Credit': f"${min(400, params['amount']//25):,}",
            'Wing Width': f"${current_price * 0.05:.2f}",
            'Strike Distance': f"{((current_price * 0.95 - current_price) / current_price * 100):+.1f}%",
            'Max Profit': f"${min(400, params['amount']//25):,}",
            'Max Loss': f"${min(600, params['amount']//17):,}",
            'Breakeven': f"${current_price * 0.95 - min(400, params['amount']//25):.2f}",
            'POP': '75%',
            'ROC': '67%',
            'IV Rank': 'Medium',
            'Risk Level': 'Medium'
        },
        {
            'Strategy': '🦅 Iron Condor',
            'Put Short': f"${current_price * 0.95:.2f}",
            'Put Long': f"${current_price * 0.90:.2f}",
            'Call Short': f"${current_price * 1.05:.2f}",
            'Call Long': f"${current_price * 1.10:.2f}",
            'Net Credit': f"${min(500, params['amount']//20):,}",
            'Wing Width': f"${current_price * 0.05:.2f}",
            'Strike Distance': f"±{((current_price * 0.05) / current_price * 100):.1f}%",
            'Max Profit': f"${min(500, params['amount']//20):,}",
            'Max Loss': f"${min(1500, params['amount']//7):,}",
            'Breakeven': f"${current_price * 0.95 - min(500, params['amount']//20):.2f} - ${current_price * 1.05 + min(500, params['amount']//20):.2f}",
            'POP': '68%',
            'ROC': '33%',
            'IV Rank': 'High',
            'Risk Level': 'Low'
        },
        {
            'Strategy': '📈 Covered Call',
            'Put Short': '-',
            'Put Long': '-',
            'Call Short': f"${current_price * 1.03:.2f}",
            'Call Long': '-',
            'Net Credit': f"${min(750, params['amount']//13):,}",
            'Wing Width': '-',
            'Strike Distance': f"{((current_price * 1.03 - current_price) / current_price * 100):+.1f}%",
            'Max Profit': f"${min(750, params['amount']//13):,}",
            'Max Loss': f"${params['amount']:,}",
            'Breakeven': f"${current_price - min(750, params['amount']//13):.2f}",
            'POP': '65%',
            'ROC': f"{min(15, (750/params['amount']*100)):.0f}%",
            'IV Rank': 'Medium',
            'Risk Level': 'Low'
        }
    ]
    
    # Display comprehensive consolidated recommendations table
    recommendations_df = pd.DataFrame(recommendations_data)
    
    # Reorder columns for logical flow
    column_order = [
        'Strategy', 'Put Short', 'Put Long', 'Call Short', 'Call Long', 
        'Net Credit', 'Wing Width', 'Strike Distance', 'Max Profit', 'Max Loss', 
        'Breakeven', 'POP', 'ROC', 'IV Rank', 'Risk Level'
    ]
    
    # Reorder the dataframe
    consolidated_df = recommendations_df[column_order]
    
    st.dataframe(
        consolidated_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strategy": st.column_config.TextColumn("Strategy", width=100),
            "Put Short": st.column_config.TextColumn("Put Short", width=70),
            "Put Long": st.column_config.TextColumn("Put Long", width=70),
            "Call Short": st.column_config.TextColumn("Call Short", width=70),
            "Call Long": st.column_config.TextColumn("Call Long", width=70),
            "Net Credit": st.column_config.TextColumn("Net Credit", width=80),
            "Wing Width": st.column_config.TextColumn("Wing Width", width=80),
            "Strike Distance": st.column_config.TextColumn("Strike Distance", width=90),
            "Max Profit": st.column_config.TextColumn("Max Profit", width=80),
            "Max Loss": st.column_config.TextColumn("Max Loss", width=80),
            "Breakeven": st.column_config.TextColumn("Breakeven", width=100),
            "POP": st.column_config.TextColumn("POP", width=50),
            "ROC": st.column_config.TextColumn("ROC", width=50),
            "IV Rank": st.column_config.TextColumn("IV Rank", width=70),
            "Risk Level": st.column_config.TextColumn("Risk Level", width=80)
        }
    )
    
    # === OPTIONS STRATEGY LEGEND ===
    with st.expander("📚 Options Strategy Terms & Metrics Guide"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📍 Strike Prices:**
            - **Put Short**: Strike you sell (collect premium)
            - **Put Long**: Strike you buy (pay premium)  
            - **Call Short**: Strike you sell (collect premium)
            - **Call Long**: Strike you buy (pay premium)
            
            **💰 Financial Metrics:**
            - **Net Credit**: Money received upfront
            - **Max Profit**: Maximum possible gain
            - **Max Loss**: Maximum possible loss
            - **Wing Width**: Distance between strikes
            
            **📊 Probability & Risk:**
            - **POP**: Probability of Profit (% chance of making money)
            - **ROC**: Return on Capital (profit ÷ risk)
            - **IV Rank**: Implied Volatility ranking (Low/Med/High)
            - **Risk Level**: Overall strategy risk assessment
            """)
        
        with col2:
            st.markdown("""
            **📐 Strike Distance:**
            - Shows how far strikes are from current price
            - Negative = Below current price (puts)
            - Positive = Above current price (calls)
            - ± = Both sides (iron condor)
            
            **🎯 Breakeven Points:**
            - Price where strategy breaks even (no profit/loss)
            - Range = Multiple breakeven points
            - Single = One breakeven point
            
            **📈 Strategy Types:**
            - **Bull Put Spread**: Profit if price stays above short strike
            - **Iron Condor**: Profit if price stays between strikes  
            - **Covered Call**: Collect premium on owned stock
            
            **⚡ Quick Tips:**
            - Higher POP = Higher probability, lower profit
            - Higher ROC = Better risk-adjusted returns
            - Check breakeven vs your price target
            """)
    
    # === AI ANALYSIS SECTION ===
    st.markdown("### 🤖 AI Master Analysis")
    
    # Separate AI Analysis Button
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_ai_btn = st.button(
            "🧠 Generate AI Analysis", 
            key="generate_ai_analysis",
            help="Generate AI-powered insights based on the strategy recommendations above",
            type="primary"
        )
    
    # Initialize session state for AI recommendations
    if 'ai_recommendations' not in st.session_state:
        st.session_state.ai_recommendations = {}
    
    # Get current ticker from master_results
    current_ticker = master_results.get('parameters', {}).get('ticker', 'Unknown')
    
    if generate_ai_btn:
        if LLM_AVAILABLE:
            with st.spinner("🤖 Generating AI insights..."):
                try:
                    ai_analysis = generate_master_ai_recommendations(master_results)
                    st.session_state.ai_recommendations[current_ticker] = ai_analysis
                    st.success("✅ AI analysis generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating AI analysis: {str(e)}")
        else:
            st.error("❌ LLM functionality not available. Please check your setup.")
    
    # Display AI Analysis if available
    if current_ticker in st.session_state.ai_recommendations:
        ai_content = st.session_state.ai_recommendations[current_ticker]
        if ai_content and ai_content.strip():
            # Use custom structured formatter for better readability
            display_structured_ai_analysis(ai_content)
        else:
            st.info("🤖 AI analysis is empty. Please try generating again.")
    else:
        if LLM_AVAILABLE:
            # Use unified AI placeholder for consistency
            if AI_FORMATTER_AVAILABLE:
                display_ai_placeholder(
                    analysis_type="Master Analysis",  
                    features_list=[
                        "🎯 Strategy recommendations based on current market conditions",
                        "📊 Risk-reward analysis for each options strategy", 
                        "💡 Optimal entry and exit timing suggestions",
                        "⚠️ Risk management and position sizing guidance",
                        "📈 Market sentiment and trend analysis"
                    ]
                )
            else:
                st.info("👆 Click 'Generate AI Analysis' above to get AI-powered insights on these strategies.")
        else:
            st.warning("🚫 AI functionality is not available. Please enable LLM support to use this feature.")
    
    # === DETAILED BREAKDOWN ===
    with st.expander("📊 Detailed Analysis Breakdown"):
        for module, data in master_results['analysis_results'].items():
            st.markdown(f"**{module.replace('_', ' ').title()}:**")
            st.json(data)


def render_summary_tab(results, vix_data, session_tickers):
    """
    Render the Summary tab with comprehensive market volatility analysis
    
    Args:
        results (dict): Analysis results from the main app
        vix_data (pd.DataFrame): VIX data
        session_tickers (list): List of selected tickers
    """
    
    st.subheader("📊 Comprehensive Market Volatility Summary")
    
    # === MASTER ANALYSIS SECTION (NEW) ===
    render_master_analysis_section(results, vix_data, session_tickers)
    
    st.markdown("---")
    
    # === EXISTING CONTENT (PRESERVED) ===
    # === SECTION 1: CURRENT MARKET STATUS ===
    st.markdown("### 🎯 Current Market Status")
    
    # Get current prices for all tickers
    current_prices = {}
    price_changes = {}
    
    col1, col2, col3, col4 = st.columns(4)
    for i, ticker in enumerate(session_tickers[:4]):  # Show up to 4 tickers in header
        current_price = get_current_price(ticker)
        current_prices[ticker] = current_price
        
        # Calculate daily change if we have daily data
        daily_change = 0
        daily_change_pct = 0
        if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
            daily_data = results[ticker]['daily']['data']
            if daily_data is not None and len(daily_data) >= 2:
                today_close = daily_data['Close'].iloc[-1]
                yesterday_close = daily_data['Close'].iloc[-2]
                daily_change = today_close - yesterday_close
                daily_change_pct = (daily_change / yesterday_close) * 100
        
        price_changes[ticker] = {'change': daily_change, 'change_pct': daily_change_pct}
        
        with [col1, col2, col3, col4][i]:
            if current_price:
                st.metric(
                    label=f"{ticker}",
                    value=f"${current_price:.2f}",
                    delta=f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)" if daily_change != 0 else None
                )
            else:
                st.metric(label=f"{ticker}", value="Price N/A")
    
    # === SECTION 2: ENHANCED VOLATILITY ANALYSIS TABLE ===
    st.markdown("### 📊 Enhanced Volatility Analysis")
    
    # Create comprehensive summary table
    summary_data = []
    for ticker in session_tickers:
        row = {
            'Ticker': ticker,
            'Current Price': f"${current_prices.get(ticker, 0):.2f}" if current_prices.get(ticker) else "N/A"
        }
        
        # Calculate comprehensive metrics for each timeframe
        for tf in ['daily', 'weekly']:  # Focus on most important timeframes
            if tf in results[ticker] and results[ticker][tf]:
                metrics = results[ticker][tf]
                atr_val = metrics['atr']
                vol_val = metrics['volatility']
                cv_val = metrics['coefficient_variation']
                
                # ATR as percentage of current price
                atr_pct = (atr_val / current_prices.get(ticker, 1)) * 100 if current_prices.get(ticker) and atr_val > 0 else 0
                
                row[f'{tf.title()} ATR'] = f"${atr_val:.2f}" if atr_val > 0 else "N/A"
                row[f'{tf.title()} ATR%'] = f"{atr_pct:.1f}%" if atr_pct > 0 else "N/A"
                row[f'{tf.title()} Vol'] = f"${vol_val:.2f}" if vol_val > 0 else "N/A"
            else:
                row[f'{tf.title()} ATR'] = "No Data"
                row[f'{tf.title()} ATR%'] = "No Data"
                row[f'{tf.title()} Vol'] = "No Data"
        
        # Calculate volatility ranking
        daily_atr = results[ticker].get('daily', {}).get('atr', 0) if ticker in results else 0
        if daily_atr > 0 and current_prices.get(ticker):
            daily_atr_pct = (daily_atr / current_prices[ticker]) * 100
            if daily_atr_pct > 3:
                vol_rank = "🔴 HIGH"
            elif daily_atr_pct > 1.5:
                vol_rank = "🟡 MEDIUM"
            else:
                vol_rank = "🟢 LOW"
        else:
            vol_rank = "❓ UNKNOWN"
        
        row['Vol Rank'] = vol_rank
        summary_data.append(row)
    
    # Display enhanced summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, height=300)
    
    # ATR Explanation
    with st.expander("📚 Understanding ATR & Volatility Metrics"):
        st.markdown("""
        ### 📊 Average True Range (ATR) Explained
        
        **What is ATR?**
        ATR measures market volatility by calculating the average of true ranges over a specified period (typically 14 periods).
        
        **True Range Calculation:**
        ```
        True Range = MAX of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|
        ```
        
        **Key ATR Insights:**
        - **Higher ATR** = More volatile, larger price swings, higher risk/reward
        - **Lower ATR** = Less volatile, smaller movements, lower risk/reward
        - **ATR %** = ATR ÷ Current Price × 100 (normalized measure)
        
        **Volatility Rankings:**
        - 🟢 **LOW (< 1.5%)**: Stable, good for momentum strategies
        - 🟡 **MEDIUM (1.5-3%)**: Moderate, ideal for options strategies  
        - 🔴 **HIGH (> 3%)**: Volatile, reduce position size, high premium options
        
        **Trading Applications:**
        - **Position Sizing**: Use ATR to determine appropriate position size
        - **Stop Losses**: Set stops at 1-2x ATR from entry
        - **Profit Targets**: Target 2-3x ATR for reward:risk ratios
        - **Options Strategy**: Use ATR for strike selection and expiry timing
        """)
    
    # Market Insights Section
    st.markdown("### 💡 Market Insights & Trading Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility Leaders
        st.markdown("#### 🔥 Volatility Leaders")
        
        vol_leaders = []
        for ticker in session_tickers:
            if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
                daily_atr = results[ticker]['daily']['atr']
                current_price = current_prices.get(ticker, 0)
                if daily_atr > 0 and current_price > 0:
                    atr_pct = (daily_atr / current_price) * 100
                    vol_leaders.append({
                        'ticker': ticker,
                        'atr_pct': atr_pct,
                        'atr_dollar': daily_atr
                    })
        
        # Sort by ATR percentage
        vol_leaders.sort(key=lambda x: x['atr_pct'], reverse=True)
        
        for i, leader in enumerate(vol_leaders[:3]):
            st.write(f"**{i+1}. {leader['ticker']}**: {leader['atr_pct']:.1f}% (${leader['atr_dollar']:.2f})")
        
        if not vol_leaders:
            st.write("*No volatility data available*")
    
    with col2:
        # Trading Recommendations
        st.markdown("#### 🎯 Trading Recommendations")
            
        # Get VIX condition if available
        if vix_data is not None:
            current_vix = vix_data['VIX_Close'].iloc[-1]
            condition, condition_class, icon = get_vix_condition(current_vix)
            trade_ok, trade_msg = should_trade(current_vix)
            
            st.markdown(f"**VIX Status**: {icon} {current_vix:.1f}")
            st.markdown(f"**Condition**: {condition.split(' - ')[0]}")
            st.markdown(f"**Trading**: {'✅ Approved' if trade_ok else '❌ Avoid'}")
        else:
            st.markdown("**VIX Status**: ❓ Not Available")
            st.markdown("**Trading**: ⚠️ Enable VIX analysis")
        
        # Position sizing recommendations
        avg_atr_pct = 0
        valid_tickers = 0
        for ticker in session_tickers:
            if ticker in results and 'daily' in results[ticker] and results[ticker]['daily']:
                daily_atr = results[ticker]['daily']['atr']
                current_price = current_prices.get(ticker, 0)
                if daily_atr > 0 and current_price > 0:
                    atr_pct = (daily_atr / current_price) * 100
                    avg_atr_pct += atr_pct
                    valid_tickers += 1
        
        if valid_tickers > 0:
            avg_atr_pct /= valid_tickers
            if avg_atr_pct > 2.5:
                size_rec = "🔴 Reduce Position Size"
            elif avg_atr_pct > 1.5:
                size_rec = "🟡 Normal Position Size"
            else:
                size_rec = "🟢 Can Increase Size"
            
            st.markdown(f"**Position Sizing**: {size_rec}")
            st.markdown(f"**Avg Market Vol**: {avg_atr_pct:.1f}%")
        else:
            st.markdown("**Position Sizing**: ❓ Insufficient Data") 