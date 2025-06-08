"""
Unified AI Analysis Formatter for Streamlit Stock Analysis App

This module provides consistent AI analysis display formatting that can be reused
across all tabs for a seamless user experience.

Author: Enhanced for consistency
Date: 2025
"""

import streamlit as st
from datetime import datetime


def format_ai_content(ai_response):
    """
    Extract and format AI content from various response formats
    
    Args:
        ai_response: Can be string, dict, or other format
        
    Returns:
        tuple: (content_text, metadata_dict)
    """
    metadata = {}
    
    if isinstance(ai_response, dict):
        # Extract content from common keys
        content_text = None
        for key in ['content', 'analysis', 'response', 'text', 'message']:
            if key in ai_response:
                content_text = str(ai_response[key])
                break
        
        if content_text is None:
            content_text = str(ai_response)
        
        # Extract metadata
        metadata_keys = ['model', 'tokens_used', 'timestamp', 'success', 'provider', 'cost']
        for key in metadata_keys:
            if key in ai_response:
                metadata[key] = ai_response[key]
                
    elif isinstance(ai_response, str):
        content_text = ai_response
    else:
        content_text = str(ai_response)
    
    return content_text, metadata


def display_ai_analysis(ai_content, analysis_type="Analysis", tab_color="#6366f1", 
                       analysis_key="", session_key="ai_analysis", regenerate_key="regenerate_ai",
                       clear_key="clear_ai", show_debug=True, show_metadata=True):
    """
    Display AI analysis with consistent formatting across all tabs
    
    Args:
        ai_content: The AI response content
        analysis_type: Type of analysis (e.g., "Technical Analysis", "Statistical Analysis")
        tab_color: Hex color for the tab theme
        analysis_key: Key for session state storage
        session_key: Session state key name
        regenerate_key: Unique key for regenerate button
        clear_key: Unique key for clear button
        show_debug: Whether to show debug information
        show_metadata: Whether to show metadata metrics
    """
    
    # Format content and extract metadata
    content_text, metadata = format_ai_content(ai_content)
    
    # Unified CSS for AI analysis display
    st.markdown(f"""
    <style>
    .ai-analysis-unified {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(99, 102, 241, 0.15) 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid {tab_color};
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin: 1.5rem 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .ai-analysis-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    }}
    
    .ai-analysis-title {{
        background: {tab_color};
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .ai-analysis-status {{
        color: {tab_color};
        font-size: 0.9rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .ai-analysis-content {{
        color: #1e293b;
        line-height: 1.7;
        font-size: 1rem;
        margin: 1.5rem 0;
    }}
    
    .ai-analysis-content h1, .ai-analysis-content h2, .ai-analysis-content h3 {{
        color: {tab_color};
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }}
    
    .ai-analysis-content strong {{
        color: #475569;
        font-weight: 600;
    }}
    
    .ai-analysis-content ul, .ai-analysis-content ol {{
        margin: 1rem 0;
        padding-left: 1.5rem;
    }}
    
    .ai-analysis-content li {{
        margin: 0.5rem 0;
    }}
    
    .ai-metadata-container {{
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Main analysis container
    st.markdown(f"""
    <div class="ai-analysis-unified">
        <div class="ai-analysis-header">
            <div class="ai-analysis-title">
                🤖 AI {analysis_type}
            </div>
            <div class="ai-analysis-status">
                <span>✅ Analysis Complete</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Content display with proper markdown rendering
    with st.container():
        st.markdown("### 📋 Analysis Results")
        st.markdown("---")
        
        # Render content as markdown for better formatting
        st.markdown(content_text)
        
        st.markdown("---")
    
    # Metadata display
    if show_metadata and metadata:
        st.markdown("#### 📊 Analysis Metadata")
        
        # Create columns based on available metadata
        metadata_cols = st.columns(min(len(metadata), 4))
        
        for i, (key, value) in enumerate(metadata.items()):
            if i < 4:  # Limit to 4 columns
                with metadata_cols[i]:
                    # Format different metadata types
                    if key == 'model':
                        st.metric("🤖 Model", str(value).upper())
                    elif key == 'tokens_used':
                        st.metric("📝 Tokens", f"{int(value):,}" if isinstance(value, (int, float)) else str(value))
                    elif key == 'timestamp':
                        try:
                            if isinstance(value, str) and 'T' in value:
                                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                formatted_time = dt.strftime('%H:%M:%S')
                            else:
                                formatted_time = str(value)[:10]
                            st.metric("⏰ Generated", formatted_time)
                        except:
                            st.metric("⏰ Generated", str(value)[:10])
                    elif key == 'success':
                        status_icon = "✅" if value else "❌"
                        status_text = "Success" if value else "Failed"
                        st.metric(f"{status_icon} Status", status_text)
                    elif key == 'cost':
                        st.metric("💰 Cost", f"${float(value):.4f}" if isinstance(value, (int, float)) else str(value))
                    elif key == 'provider':
                        st.metric("🏢 Provider", str(value).title())
                    else:
                        st.metric(key.replace('_', ' ').title(), str(value))
    
    # Debug information (collapsible)
    if show_debug and isinstance(ai_content, dict):
        with st.expander("🔧 Technical Details"):
            st.caption("Raw API Response Structure:")
            st.json(ai_content)
    
    # Action buttons with consistent styling
    st.markdown("#### ⚡ Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Regenerate Analysis", 
                    help=f"Generate a new AI {analysis_type.lower()}", 
                    key=regenerate_key,
                    use_container_width=True):
            if session_key in st.session_state and analysis_key in st.session_state[session_key]:
                del st.session_state[session_key][analysis_key]
                st.rerun()
    
    with col2:
        if st.button("📋 Copy to Clipboard", 
                    help="Copy analysis text to clipboard", 
                    key=f"copy_{regenerate_key}",
                    use_container_width=True):
            st.info("💡 Select and copy the analysis text above")
    
    with col3:
        if st.button("🗑️ Clear Analysis", 
                    help=f"Clear the current AI {analysis_type.lower()}", 
                    key=clear_key,
                    use_container_width=True):
            if session_key in st.session_state and analysis_key in st.session_state[session_key]:
                del st.session_state[session_key][analysis_key]
                st.rerun()


def display_ai_placeholder(analysis_type="Analysis", features_list=None):
    """
    Display placeholder content when AI analysis is not available
    
    Args:
        analysis_type: Type of analysis (e.g., "Technical Analysis")
        features_list: List of features the AI analysis provides
    """
    
    if features_list is None:
        features_list = [
            f"Intelligent {analysis_type.lower()} insights",
            "Market condition assessment",
            "Trading recommendations",
            "Risk management guidance",
            "Pattern recognition"
        ]
    
    st.info(f"👆 Click 'Generate AI {analysis_type}' to get intelligent insights")
    
    with st.expander(f"📖 What does AI {analysis_type} provide?"):
        st.markdown(f"### 🎯 {analysis_type} Features:")
        
        for feature in features_list:
            st.markdown(f"- **{feature}**")


def display_ai_setup_instructions(analysis_type="Analysis"):
    """
    Display setup instructions for enabling AI analysis
    
    Args:
        analysis_type: Type of analysis for customized instructions
    """
    
    with st.expander("🔧 How to Enable AI Analysis"):
        st.markdown(f"""
        ### Setup Instructions for AI {analysis_type}:
        
        1. **Install Dependencies**: Ensure `llm_analysis.py` is available
        2. **Configure API Keys**: Set up your preferred LLM provider (OpenAI, Anthropic, etc.)
        3. **Environment Setup**: Add API keys to `.env` file
        4. **Restart Application**: The AI analysis features will become available
        
        **Supported Providers:**
        - OpenAI (GPT-3.5, GPT-4)
        - Anthropic (Claude)
        - Local models (Ollama, etc.)
        
        **Environment Variables:**
        ```
        OPENAI_API_KEY=your_openai_key_here
        ANTHROPIC_API_KEY=your_anthropic_key_here
        ```
        """)


# Color scheme for different tabs
TAB_COLORS = {
    "technical": "#0ea5e9",      # Blue for technical analysis
    "statistical": "#6366f1",    # Indigo for statistical analysis  
    "comparison": "#10b981",     # Green for comparison analysis
    "options": "#8b5cf6",        # Purple for options analysis
    "sentiment": "#059669",      # Emerald for sentiment analysis
    "macro": "#f59e0b",          # Amber for macro analysis
    "risk": "#ef4444",           # Red for risk analysis
    "default": "#6366f1"         # Default indigo
}


def get_tab_color(analysis_type):
    """Get appropriate color for analysis type"""
    analysis_lower = analysis_type.lower()
    
    if "technical" in analysis_lower:
        return TAB_COLORS["technical"]
    elif "statistical" in analysis_lower or "stats" in analysis_lower:
        return TAB_COLORS["statistical"]
    elif "comparison" in analysis_lower or "comparative" in analysis_lower:
        return TAB_COLORS["comparison"]
    elif "option" in analysis_lower:
        return TAB_COLORS["options"]
    elif "sentiment" in analysis_lower:
        return TAB_COLORS["sentiment"]
    elif "macro" in analysis_lower:
        return TAB_COLORS["macro"]
    elif "risk" in analysis_lower:
        return TAB_COLORS["risk"]
    else:
        return TAB_COLORS["default"] 