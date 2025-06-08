"""
Styling and formatting functions for Stock Volatility Analyzer

This module contains CSS styling, formatting utilities, and UI helper functions.
"""


def get_custom_css():
    """Return the custom CSS styling for the application"""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --primary-color: #6366f1;
            --primary-light: #818cf8;
            --primary-dark: #4f46e5;
            --secondary-color: #06b6d4;
            --accent-color: #f59e0b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --background-primary: #ffffff;
            --background-secondary: #f8fafc;
            --background-tertiary: #f1f5f9;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-light: #94a3b8;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --gradient-secondary: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        }
        
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }
        
        .main-header {
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 3rem;
            letter-spacing: -0.025em;
            line-height: 1.1;
        }
        
        /* VIX Condition Styling with Modern Colors */
        .vix-calm { 
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            padding: 1rem; 
            border-radius: 12px; 
            border-left: 4px solid var(--success-color);
            box-shadow: var(--shadow-sm);
        }
        .vix-normal { 
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            padding: 1rem; 
            border-radius: 12px; 
            border-left: 4px solid var(--secondary-color);
            box-shadow: var(--shadow-sm);
        }
        .vix-choppy { 
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            padding: 1rem; 
            border-radius: 12px; 
            border-left: 4px solid var(--warning-color);
            box-shadow: var(--shadow-sm);
        }
        .vix-volatile { 
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
            padding: 1rem; 
            border-radius: 12px; 
            border-left: 4px solid var(--error-color);
            box-shadow: var(--shadow-sm);
        }
        .vix-extreme { 
            background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
            color: white;
            padding: 1rem; 
            border-radius: 12px; 
            border-left: 4px solid #dc2626;
            box-shadow: var(--shadow-lg);
        }
        
        .strike-recommend { 
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            padding: 1.5rem; 
            border-radius: 16px; 
            border: 2px solid var(--success-color);
            box-shadow: var(--shadow-lg);
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .strike-recommend::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--gradient-primary);
        }
        
        /* Button Styling */
        .stButton > button {
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.2s ease-in-out;
            box-shadow: var(--shadow-sm);
        }
        
        .stButton > button:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }
    </style>
    """


def format_percentage(value):
    """Format a decimal value as a percentage"""
    try:
        return f"{value*100:.1f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_currency(value):
    """Format a numeric value as currency"""
    try:
        return f"${value:.2f}"
    except (ValueError, TypeError):
        return "N/A"


def format_large_number(value):
    """Format large numbers with appropriate suffixes"""
    try:
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:.0f}"
    except (ValueError, TypeError):
        return "N/A"


def get_volatility_color(atr_percentage):
    """Get color coding for volatility levels"""
    try:
        if atr_percentage > 3:
            return "🔴", "#ef4444"
        elif atr_percentage > 1.5:
            return "🟡", "#f59e0b"
        else:
            return "🟢", "#10b981"
    except (ValueError, TypeError):
        return "❓", "#64748b"


def create_metric_card(title, value, delta=None, help_text=None):
    """Create a styled metric card HTML"""
    delta_html = ""
    if delta:
        delta_color = "#10b981" if "+" in str(delta) else "#ef4444"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.875rem; font-weight: 500;">{delta}</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">{help_text}</div>'
    
    return f"""
    <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);">
        <div style="color: #64748b; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem;">
            {title}
        </div>
        <div style="color: #1e293b; font-size: 1.875rem; font-weight: 700; line-height: 1;">
            {value}
        </div>
        {delta_html}
        {help_html}
    </div>
    """ 