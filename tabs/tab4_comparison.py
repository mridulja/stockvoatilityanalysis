"""
Tab 4: Multi-Ticker Comparison Analysis - Stock Volatility Analyzer

This module provides comprehensive comparison analysis across multiple tickers including:
- Performance comparison charts and rankings
- Risk-return analysis and correlation matrices
- Volatility regime comparison
- Sector/style analysis
- AI-powered comparative insights

Author: Enhanced Analysis Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Import unified AI formatter
try:
    from shared.ai_formatter import display_ai_analysis, display_ai_placeholder, display_ai_setup_instructions, get_tab_color
    AI_FORMATTER_AVAILABLE = True
except ImportError:
    AI_FORMATTER_AVAILABLE = False

# Import LLM analysis functionality
try:
    from llm_analysis import get_llm_analyzer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import utility functions
try:
    from core import get_current_price, get_vix_condition, format_percentage, format_currency
except ImportError:
    # Fallback functions
    def get_current_price(ticker):
        return None
    def get_vix_condition(vix_value):
        return "Unknown", "vix-normal", "🤷"
    def format_percentage(value):
        return f"{value*100:.1f}%"
    def format_currency(value):
        return f"${value:.2f}"


def create_performance_comparison_chart(results, session_tickers, timeframes=['daily']):
    """Create comprehensive performance comparison chart"""
    
    # Prepare data for comparison
    comparison_data = []
    
    for ticker in session_tickers:
        if ticker in results:
            ticker_data = {'ticker': ticker}
            
            for timeframe in timeframes:
                if timeframe in results[ticker] and results[ticker][timeframe]:
                    stats = results[ticker][timeframe]['stats']
                    metrics = results[ticker][timeframe]
                    
                    # Add key metrics
                    ticker_data.update({
                        f'{timeframe}_atr': metrics.get('atr', 0),
                        f'{timeframe}_volatility': metrics.get('volatility', 0),
                        f'{timeframe}_mean_range': stats.get('mean', 0) if stats is not None else 0,
                        f'{timeframe}_std_range': stats.get('std', 0) if stats is not None else 0,
                        f'{timeframe}_cv': metrics.get('coefficient_variation', 0),
                        f'{timeframe}_max_range': stats.get('max', 0) if stats is not None else 0,
                        f'{timeframe}_q75_range': stats.get('75%', 0) if stats is not None else 0
                    })
            
            comparison_data.append(ticker_data)
    
    if not comparison_data:
        return None
    
    df = pd.DataFrame(comparison_data)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Average True Range (ATR) Comparison',
            'Volatility vs Risk-Adjusted Return',
            'Range Distribution Comparison',
            'Coefficient of Variation Analysis'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Chart 1: ATR Comparison
    fig.add_trace(
        go.Bar(
            x=df['ticker'],
            y=df['daily_atr'],
            name='ATR',
            marker_color='rgba(99, 102, 241, 0.8)',
            text=[f'${val:.2f}' for val in df['daily_atr']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Chart 2: Risk-Return Scatter  
    fig.add_trace(
        go.Scatter(
            x=df['daily_volatility'],
            y=df['daily_mean_range'],
            mode='markers+text',
            text=df['ticker'],
            textposition='top center',
            marker=dict(
                size=df['daily_atr'] * 10,  # Size by ATR
                color=df['daily_cv'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="CV", x=1.02)
            ),
            name='Risk-Return'
        ),
        row=1, col=2
    )
    
    # Chart 3: Range Distribution
    for i, ticker in enumerate(df['ticker']):
        fig.add_trace(
            go.Bar(
                x=[f"Mean", f"Std", f"75%", f"Max"],
                y=[df.loc[i, 'daily_mean_range'], df.loc[i, 'daily_std_range'], 
                   df.loc[i, 'daily_q75_range'], df.loc[i, 'daily_max_range']],
                name=f'{ticker} Range',
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Chart 4: Coefficient of Variation
    fig.add_trace(
        go.Bar(
            x=df['ticker'],
            y=df['daily_cv'],
            name='Coefficient of Variation',
            marker_color='rgba(255, 159, 64, 0.8)',
            text=[f'{val:.3f}' for val in df['daily_cv']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="📊 Comprehensive Multi-Ticker Performance Comparison",
        title_x=0.5,
        height=800,
        showlegend=True,
        font=dict(size=10)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Ticker", row=1, col=1)
    fig.update_yaxes(title_text="ATR ($)", row=1, col=1)
    
    fig.update_xaxes(title_text="Volatility", row=1, col=2)
    fig.update_yaxes(title_text="Mean Range", row=1, col=2)
    
    fig.update_xaxes(title_text="Range Metrics", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    
    fig.update_xaxes(title_text="Ticker", row=2, col=2)
    fig.update_yaxes(title_text="Coefficient of Variation", row=2, col=2)
    
    return fig


def create_correlation_matrix(results, session_tickers, timeframe='daily'):
    """Create correlation matrix for price movements"""
    
    # Collect price data for all tickers
    price_data = {}
    
    for ticker in session_tickers:
        if ticker in results and timeframe in results[ticker] and results[ticker][timeframe]:
            data = results[ticker][timeframe]['data']
            if data is not None and not data.empty and 'Close' in data.columns:
                price_data[ticker] = data['Close']
    
    if len(price_data) < 2:
        return None
    
    # Create DataFrame with aligned indices
    df = pd.DataFrame(price_data)
    
    # Calculate returns for correlation
    returns_df = df.pct_change().dropna()
    
    if returns_df.empty:
        return None
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=f'📈 Price Movement Correlation Matrix ({timeframe.title()})',
        title_x=0.5,
        width=600,
        height=500,
        xaxis_title="Ticker",
        yaxis_title="Ticker"
    )
    
    return fig


def create_ranking_table(results, session_tickers, timeframe='daily'):
    """Create comprehensive ranking table"""
    
    ranking_data = []
    
    for ticker in session_tickers:
        if ticker in results and timeframe in results[ticker] and results[ticker][timeframe]:
            stats = results[ticker][timeframe]['stats']
            metrics = results[ticker][timeframe]
            
            current_price = get_current_price(ticker)
            
            ranking_data.append({
                'Ticker': ticker,
                'Current Price': f"${current_price:.2f}" if current_price else "N/A",
                'ATR': f"${metrics.get('atr', 0):.2f}",
                'ATR %': f"{(metrics.get('atr', 0) / current_price * 100):.1f}%" if current_price else "N/A",
                'Volatility': f"${metrics.get('volatility', 0):.2f}",
                'Mean Range': f"${stats.get('mean', 0):.2f}" if stats is not None else "N/A",
                'Max Range': f"${stats.get('max', 0):.2f}" if stats is not None else "N/A",
                'CV': f"{metrics.get('coefficient_variation', 0):.3f}",
                'ATR Raw': metrics.get('atr', 0),
                'Volatility Raw': metrics.get('volatility', 0),
                'CV Raw': metrics.get('coefficient_variation', 0)
            })
    
    if not ranking_data:
        return None
    
    df = pd.DataFrame(ranking_data)
    
    # Create rankings
    df['ATR Rank'] = df['ATR Raw'].rank(ascending=False).astype(int)
    df['Volatility Rank'] = df['Volatility Raw'].rank(ascending=False).astype(int)
    df['CV Rank'] = df['CV Raw'].rank(ascending=False).astype(int)
    df['Overall Score'] = (df['ATR Rank'] + df['Volatility Rank'] + df['CV Rank']) / 3
    df['Overall Rank'] = df['Overall Score'].rank().astype(int)
    
    # Select display columns
    display_df = df[['Ticker', 'Current Price', 'ATR', 'ATR %', 'Volatility', 
                     'Mean Range', 'CV', 'ATR Rank', 'Volatility Rank', 'Overall Rank']]
    
    return display_df.sort_values('Overall Rank')


def create_risk_return_scatter(results, session_tickers, timeframe='daily'):
    """Create advanced risk-return scatter plot"""
    
    scatter_data = []
    
    for ticker in session_tickers:
        if ticker in results and timeframe in results[ticker] and results[ticker][timeframe]:
            stats = results[ticker][timeframe]['stats']
            metrics = results[ticker][timeframe]
            data = results[ticker][timeframe]['data']
            
            if data is not None and not data.empty:
                # Calculate simple return (price change over period)
                returns = data['Close'].pct_change().dropna()
                mean_return = returns.mean() * 252  # Annualized
                volatility = metrics.get('volatility', 0)
                atr = metrics.get('atr', 0)
                
                current_price = get_current_price(ticker)
                
                scatter_data.append({
                    'ticker': ticker,
                    'return': mean_return * 100,  # Percentage
                    'risk': volatility,
                    'atr': atr,
                    'sharpe': mean_return / (volatility + 1e-6),  # Avoid division by zero
                    'current_price': current_price or 0
                })
    
    if not scatter_data:
        return None
    
    df = pd.DataFrame(scatter_data)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['risk'],
        y=df['return'],
        mode='markers+text',
        text=df['ticker'],
        textposition='top center',
        marker=dict(
            size=df['atr'] * 10,  # Size by ATR
            color=df['sharpe'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Risk (Volatility): %{x:.2f}<br>' +
                      'Return: %{y:.2f}%<br>' +
                      'ATR: $%{marker.size:.2f}<br>' +
                      '<extra></extra>',
        name='Tickers'
    ))
    
    # Add quadrant lines at mean values
    if len(df) > 1:
        mean_risk = df['risk'].mean()
        mean_return = df['return'].mean()
        
        fig.add_hline(y=mean_return, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=mean_risk, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        max_risk = df['risk'].max() * 1.1
        max_return = df['return'].max() * 1.1
        min_return = df['return'].min() * 1.1
        
        fig.add_annotation(x=max_risk, y=max_return, text="High Risk<br>High Return", 
                          showarrow=False, bgcolor="rgba(0,255,0,0.1)")
        fig.add_annotation(x=0, y=max_return, text="Low Risk<br>High Return", 
                          showarrow=False, bgcolor="rgba(0,255,0,0.3)")
        fig.add_annotation(x=max_risk, y=min_return, text="High Risk<br>Low Return", 
                          showarrow=False, bgcolor="rgba(255,0,0,0.1)")
        fig.add_annotation(x=0, y=min_return, text="Low Risk<br>Low Return", 
                          showarrow=False, bgcolor="rgba(255,255,0,0.1)")
    
    fig.update_layout(
        title='📊 Risk-Return Analysis (Size = ATR, Color = Sharpe Ratio)',
        title_x=0.5,
        xaxis_title='Risk (Volatility)',
        yaxis_title='Annualized Return (%)',
        height=600,
        font=dict(size=12)
    )
    
    return fig


def render_comparison_tab(results, vix_data, session_tickers):
    """
    Render the comprehensive comparison analysis tab
    
    Args:
        results (dict): Analysis results from the main app
        vix_data (pd.DataFrame): VIX data
        session_tickers (list): List of selected tickers
    """
    
    st.subheader("⚖️ Multi-Ticker Comparative Analysis")
    
    if len(session_tickers) < 2:
        st.warning("⚠️ **Multiple tickers required for comparison analysis**")
        st.info("Please select at least 2 tickers in the sidebar and run the analysis to enable comprehensive comparison features.")
        
        # Show what's available with multiple tickers
        st.markdown("""
        ### 🔍 Available with Multiple Tickers:
        
        **📊 Performance Comparison**
        - ATR and volatility comparison charts
        - Risk-return scatter plots with Sharpe ratios
        - Range distribution analysis
        
        **🔗 Correlation Analysis**  
        - Price movement correlation matrices
        - Sector/style correlation insights
        - Diversification benefits analysis
        
        **🏆 Ranking & Scoring**
        - Comprehensive multi-metric rankings
        - Risk-adjusted performance scores
        - Trading opportunity identification
        
        **🤖 AI Comparative Insights**
        - Intelligent portfolio construction suggestions
        - Relative strength analysis
        - Market regime adaptability comparison
        """)
        return
    
    # === ANALYSIS CONFIGURATION ===
    st.markdown("### ⚙️ Comparison Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        analysis_timeframe = st.selectbox(
            "📅 Analysis Timeframe:",
            ['daily', 'hourly', 'weekly'],
            index=0,
            help="Select timeframe for comparison analysis",
            key="comparison_timeframe"
        )
    
    with config_col2:
        comparison_focus = st.selectbox(
            "🎯 Focus Area:",
            ['Overall Comparison', 'Risk Analysis', 'Performance Analysis', 'Correlation Study'],
            help="Choose the primary focus of comparison",
            key="comparison_focus"
        )
    
    with config_col3:
        include_ai_analysis = st.checkbox(
            "🤖 Include AI Analysis",
            value=True,
            help="Generate AI-powered comparative insights"
        )
    
    # Verify all tickers have data for the selected timeframe
    available_tickers = [
        ticker for ticker in session_tickers 
        if ticker in results and analysis_timeframe in results[ticker] and results[ticker][analysis_timeframe]
    ]
    
    if len(available_tickers) < 2:
        st.error(f"⚠️ Not enough tickers have {analysis_timeframe} data. Available: {available_tickers}")
        st.info("Please ensure at least 2 tickers have data for the selected timeframe.")
        return
    
    st.success(f"✅ Comparing {len(available_tickers)} tickers: {', '.join(available_tickers)}")
    
    # === PERFORMANCE COMPARISON SECTION ===
    st.markdown("### 📊 Performance Comparison Dashboard")
    
    # Create comprehensive comparison chart
    comparison_chart = create_performance_comparison_chart(results, available_tickers, [analysis_timeframe])
    if comparison_chart:
        st.plotly_chart(comparison_chart, use_container_width=True)
    else:
        st.error("Unable to create performance comparison chart")
    
    # === RANKING TABLE ===
    st.markdown("### 🏆 Comprehensive Rankings")
    
    ranking_table = create_ranking_table(results, available_tickers, analysis_timeframe)
    if ranking_table is not None:
        # Display with color coding
        st.dataframe(
            ranking_table,
            use_container_width=True,
            height=300
        )
        
        # Top performers summary
        top_performer = ranking_table.iloc[0]
        st.info(f"🥇 **Top Overall Performer**: {top_performer['Ticker']} "
                f"(ATR: {top_performer['ATR']}, Volatility Rank: {top_performer['Volatility Rank']})")
    
    # === CORRELATION & RISK ANALYSIS ===
    correlation_col, risk_return_col = st.columns(2)
    
    with correlation_col:
        st.markdown("#### 🔗 Correlation Matrix")
        corr_chart = create_correlation_matrix(results, available_tickers, analysis_timeframe)
        if corr_chart:
            st.plotly_chart(corr_chart, use_container_width=True)
        else:
            st.warning("Unable to create correlation matrix")
    
    with risk_return_col:
        st.markdown("#### 📈 Risk-Return Analysis")
        risk_return_chart = create_risk_return_scatter(results, available_tickers, analysis_timeframe)
        if risk_return_chart:
            st.plotly_chart(risk_return_chart, use_container_width=True)
        else:
            st.warning("Unable to create risk-return chart")
    
    # === AI COMPARATIVE ANALYSIS ===
    if include_ai_analysis:
        st.markdown("### 🤖 AI-Powered Comparative Analysis")
        
        if LLM_AVAILABLE and AI_FORMATTER_AVAILABLE:
            ai_col1, ai_col2 = st.columns([1, 3])
            
            with ai_col1:
                if st.button("🧠 Generate Comparative Analysis", type="primary", key="generate_comparison_ai"):
                    _generate_comparison_ai_analysis(results, available_tickers, analysis_timeframe, vix_data)
            
            with ai_col2:
                _display_comparison_ai_results(available_tickers, analysis_timeframe)
        
        elif AI_FORMATTER_AVAILABLE:
            display_ai_setup_instructions("Comparative Analysis")
        else:
            st.info("🤖 AI analysis requires the unified AI formatter. Please ensure shared/ai_formatter.py is available.")
    
    # === DETAILED METRICS SECTION ===
    with st.expander("📋 Detailed Metrics Breakdown"):
        st.markdown("#### 🔍 Detailed Performance Metrics")
        
        detailed_metrics = []
        for ticker in available_tickers:
            if ticker in results and analysis_timeframe in results[ticker]:
                stats = results[ticker][analysis_timeframe]['stats']
                metrics = results[ticker][analysis_timeframe]
                
                current_price = get_current_price(ticker)
                
                detailed_metrics.append({
                    'Ticker': ticker,
                    'Current Price': current_price,
                    'ATR': metrics.get('atr', 0),
                    'Volatility': metrics.get('volatility', 0),
                                    'Mean Range': stats.get('mean', 0) if stats is not None else 0,
                'Std Range': stats.get('std', 0) if stats is not None else 0,
                'Min Range': stats.get('min', 0) if stats is not None else 0,
                'Max Range': stats.get('max', 0) if stats is not None else 0,
                'Q25 Range': stats.get('25%', 0) if stats is not None else 0,
                'Q75 Range': stats.get('75%', 0) if stats is not None else 0,
                    'CV': metrics.get('coefficient_variation', 0)
                })
        
        if detailed_metrics:
            detailed_df = pd.DataFrame(detailed_metrics)
            st.dataframe(detailed_df, use_container_width=True)


def _generate_comparison_ai_analysis(results, tickers, timeframe, vix_data):
    """Generate AI analysis for ticker comparison"""
    
    try:
        llm_analyzer = get_llm_analyzer()
        
        # Prepare comparison data for AI
        comparison_summary = []
        for ticker in tickers:
            if ticker in results and timeframe in results[ticker]:
                stats = results[ticker][timeframe]['stats']
                metrics = results[ticker][timeframe]
                current_price = get_current_price(ticker)
                
                comparison_summary.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'atr': metrics.get('atr', 0),
                    'volatility': metrics.get('volatility', 0),
                    'cv': metrics.get('coefficient_variation', 0),
                                    'mean_range': stats.get('mean', 0) if stats is not None else 0,
                'max_range': stats.get('max', 0) if stats is not None else 0
                })
        
        # Create analysis prompt
        vix_current = vix_data['VIX_Close'].iloc[-1] if vix_data is not None else None
        vix_condition = get_vix_condition(vix_current)[0] if vix_current else "Unknown"
        
        tickers_str = ", ".join([f"{s['ticker']} (ATR: ${s['atr']:.2f}, Vol: ${s['volatility']:.2f})" for s in comparison_summary])
        
        analysis_prompt = f"""
        Analyze this multi-ticker comparison and provide professional trading insights:

        **Comparison Set**: {len(tickers)} tickers ({timeframe} timeframe)
        **Tickers**: {tickers_str}
        **Market Context**: VIX {vix_current:.1f} - {vix_condition}
        
        **Detailed Metrics**:
        {chr(10).join([f"• {s['ticker']}: Price ${s['current_price']:.2f}, ATR ${s['atr']:.2f} ({s['atr']/s['current_price']*100:.1f}%), Volatility ${s['volatility']:.2f}, CV {s['cv']:.3f}" for s in comparison_summary])}

        Please provide:
        1. **Relative Strength Ranking** (Which tickers show best risk-adjusted profiles?)
        2. **Portfolio Construction** (How to combine these tickers for diversification?)
        3. **Trading Opportunities** (Which tickers offer best trading setups now?)
        4. **Risk Assessment** (Concentration risk and correlation concerns)
        5. **Market Regime Suitability** (Which tickers perform best in current VIX environment?)

        Keep analysis practical and actionable for portfolio managers and active traders.
        """
        
        # Generate AI response
        ai_response = None
        for method_name in ['analyze', 'get_analysis', 'generate_analysis', 'chat', 'query']:
            if hasattr(llm_analyzer, method_name):
                ai_response = getattr(llm_analyzer, method_name)(analysis_prompt)
                break
        
        if not ai_response and callable(llm_analyzer):
            ai_response = llm_analyzer(analysis_prompt)
        
        if ai_response:
            # Store in session state
            if 'ai_comparison_analysis' not in st.session_state:
                st.session_state.ai_comparison_analysis = {}
            
            analysis_key = f"comparison_{'-'.join(tickers)}_{timeframe}"
            st.session_state.ai_comparison_analysis[analysis_key] = ai_response
            st.success("✅ AI Comparative Analysis completed!")
        else:
            st.warning("⚠️ AI Analysis could not be generated. Check LLM configuration.")
            
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        st.info("💡 AI analysis requires proper LLM configuration.")


def _display_comparison_ai_results(tickers, timeframe):
    """Display AI comparison analysis results"""
    
    analysis_key = f"comparison_{'-'.join(tickers)}_{timeframe}"
    
    if 'ai_comparison_analysis' in st.session_state and analysis_key in st.session_state.ai_comparison_analysis:
        ai_content = st.session_state.ai_comparison_analysis[analysis_key]
        
        if AI_FORMATTER_AVAILABLE:
            display_ai_analysis(
                ai_content=ai_content,
                analysis_type="Comparative Analysis",
                tab_color=get_tab_color("comparison"),
                analysis_key=analysis_key,
                session_key="ai_comparison_analysis",
                regenerate_key="regenerate_comparison_ai",
                clear_key="clear_comparison_ai",
                show_debug=True,
                show_metadata=True
            )
        else:
            st.markdown("#### 🧠 AI Comparative Analysis Results")
            content_text = str(ai_content.get('content', ai_content)) if isinstance(ai_content, dict) else str(ai_content)
            st.markdown(content_text)
    
    else:
        if AI_FORMATTER_AVAILABLE:
            display_ai_placeholder(
                analysis_type="Comparative Analysis",
                features_list=[
                    "Relative strength ranking and performance comparison",
                    "Diversification benefits and correlation analysis",
                    "Risk-adjusted portfolio construction recommendations",
                    "Market regime suitability assessment",
                    "Trading opportunity identification across tickers",
                    "Concentration risk and allocation guidance"
                ]
            )
        else:
            st.info("👆 Click 'Generate Comparative Analysis' to get AI insights on ticker relationships and portfolio optimization")

 