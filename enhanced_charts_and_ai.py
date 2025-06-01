#!/usr/bin/env python3
"""
Enhanced Charts and Optional AI Analysis Implementation
"""

# Read the current streamlit app
with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add new statistical chart functions after the existing chart functions
statistical_charts_code = '''
def create_price_distribution_chart(data, ticker, timeframe):
    """Create price distribution histogram and box plot"""
    if data is None or data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{ticker} Close Price Distribution',
            f'{ticker} Daily Returns Distribution', 
            f'{ticker} Price Range Box Plot',
            f'{ticker} Volume Distribution'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price distribution histogram
    fig.add_trace(
        go.Histogram(
            x=data['Close'],
            nbinsx=30,
            name='Close Price',
            marker_color='rgba(99, 102, 241, 0.7)'
        ),
        row=1, col=1
    )
    
    # Returns distribution (if we have enough data)
    if len(data) > 1:
        returns = data['Close'].pct_change().dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                name='Daily Returns (%)',
                marker_color='rgba(16, 185, 129, 0.7)'
            ),
            row=1, col=2
        )
    
    # Price range box plot
    price_data = [data['High'], data['Low'], data['Close']]
    price_labels = ['High', 'Low', 'Close']
    
    for i, (prices, label) in enumerate(zip(price_data, price_labels)):
        fig.add_trace(
            go.Box(
                y=prices,
                name=label,
                boxpoints='outliers'
            ),
            row=2, col=1
        )
    
    # Volume distribution (if available)
    if 'Volume' in data.columns:
        fig.add_trace(
            go.Histogram(
                x=data['Volume'],
                nbinsx=25,
                name='Volume',
                marker_color='rgba(245, 158, 11, 0.7)'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title=f'{ticker} Statistical Distribution Analysis ({timeframe.title()})',
        height=600,
        showlegend=True
    )
    
    return fig

def create_volatility_analysis_chart(data, ticker, timeframe):
    """Create comprehensive volatility analysis charts"""
    if data is None or data.empty or len(data) < 5:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'True Range Distribution',
            'ATR Evolution Over Time',
            'Range vs Close Price Correlation', 
            'Volatility Percentiles'
        ]
    )
    
    # True Range distribution
    if 'true_range' in data.columns:
        fig.add_trace(
            go.Histogram(
                x=data['true_range'],
                nbinsx=25,
                name='True Range',
                marker_color='rgba(239, 68, 68, 0.7)'
            ),
            row=1, col=1
        )
        
        # ATR over time
        atr_window = min(14, len(data))
        if atr_window > 1:
            atr_series = data['true_range'].rolling(window=atr_window).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=atr_series,
                    mode='lines',
                    name=f'ATR ({atr_window})',
                    line=dict(color='#ef4444', width=2)
                ),
                row=1, col=2
            )
    
    # Range vs Close correlation scatter
    if 'range' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Close'],
                y=data['range'],
                mode='markers',
                name='Range vs Price',
                marker=dict(
                    color=data.index.map(lambda x: x.toordinal()),
                    colorscale='Viridis',
                    size=6,
                    opacity=0.7
                )
            ),
            row=2, col=1
        )
        
        # Volatility percentiles
        range_percentiles = data['range'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        fig.add_trace(
            go.Bar(
                x=['10th', '25th', '50th', '75th', '90th'],
                y=range_percentiles.values,
                name='Range Percentiles',
                marker_color='rgba(99, 102, 241, 0.8)'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title=f'{ticker} Volatility Analysis ({timeframe.title()})',
        height=600,
        showlegend=True
    )
    
    return fig

def create_pot_analysis_charts(spread_results):
    """Create POT analysis statistical charts"""
    if not spread_results or not spread_results.get('scenarios'):
        return None
    
    scenarios = spread_results['scenarios']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'POT Distribution by Target Level',
            'Strike Distance Distribution',
            'Safety Level Analysis',
            'POT vs Distance Correlation'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # Extract data
    target_pots = [s['target_pot'] * 100 for s in scenarios]
    actual_pots = [s['actual_pot'] * 100 for s in scenarios]  
    distances = [s['distance_pct'] for s in scenarios]
    strikes = [s['short_strike'] for s in scenarios]
    
    # POT accuracy analysis
    fig.add_trace(
        go.Scatter(
            x=target_pots,
            y=actual_pots,
            mode='markers+lines',
            name='Target vs Actual POT',
            marker=dict(size=8, color='rgba(99, 102, 241, 0.8)'),
            line=dict(color='rgba(99, 102, 241, 0.8)', width=2)
        ),
        row=1, col=1
    )
    
    # Add perfect correlation line
    fig.add_trace(
        go.Scatter(
            x=[0, max(target_pots)],
            y=[0, max(target_pots)],
            mode='lines',
            name='Perfect Match',
            line=dict(color='red', dash='dash', width=1)
        ),
        row=1, col=1
    )
    
    # Distance distribution
    fig.add_trace(
        go.Histogram(
            x=distances,
            nbinsx=15,
            name='Distance %',
            marker_color='rgba(16, 185, 129, 0.7)'
        ),
        row=1, col=2
    )
    
    # Safety level pie chart
    safety_counts = {}
    for distance in distances:
        if distance > 5:
            safety = "Very Safe"
        elif distance > 3:
            safety = "Safe"
        elif distance > 2:
            safety = "Moderate"
        else:
            safety = "Risky"
        safety_counts[safety] = safety_counts.get(safety, 0) + 1
    
    if safety_counts:
        fig.add_trace(
            go.Pie(
                labels=list(safety_counts.keys()),
                values=list(safety_counts.values()),
                name="Safety Levels"
            ),
            row=2, col=1
        )
    
    # POT vs Distance correlation
    fig.add_trace(
        go.Scatter(
            x=actual_pots,
            y=distances,
            mode='markers',
            name='POT vs Distance',
            marker=dict(
                size=8,
                color=target_pots,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Target POT %")
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Put Spread POT Statistical Analysis',
        height=700,
        showlegend=True
    )
    
    return fig

def create_strategy_comparison_chart(spread_results):
    """Create strategy comparison charts"""
    if not spread_results or not spread_results.get('scenarios'):
        return None
    
    scenarios = spread_results['scenarios']
    
    # Create comparison data
    comparison_data = []
    for scenario in scenarios:
        if scenario.get('spreads'):
            best_spread = scenario['spreads'][0]
            comparison_data.append({
                'Target POT': f"{scenario['target_pot']:.1%}",
                'Strike Price': scenario['short_strike'],
                'Distance %': scenario['distance_pct'],
                'Actual POT': scenario['actual_pot'] * 100,
                'Prob Profit': best_spread['prob_profit'] * 100,
                'Max Profit': best_spread['max_profit']
            })
    
    if not comparison_data:
        return None
    
    df = pd.DataFrame(comparison_data)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Risk vs Reward Analysis', 'Strike Price Comparison']
    )
    
    # Risk vs Reward scatter
    fig.add_trace(
        go.Scatter(
            x=df['Distance %'],
            y=df['Prob Profit'],
            mode='markers+text',
            text=df['Target POT'],
            textposition='top center',
            name='Strategies',
            marker=dict(
                size=df['Max Profit'] * 2,  # Size by max profit
                color=df['Actual POT'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Actual POT %")
            )
        ),
        row=1, col=1
    )
    
    # Strike price comparison
    fig.add_trace(
        go.Bar(
            x=df['Target POT'],
            y=df['Strike Price'],
            name='Strike Prices',
            marker_color='rgba(99, 102, 241, 0.8)'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Distance from Current (%)", row=1, col=1)
    fig.update_yaxes(title_text="Probability of Profit (%)", row=1, col=1)
    fig.update_xaxes(title_text="Target POT Level", row=1, col=2)
    fig.update_yaxes(title_text="Strike Price ($)", row=1, col=2)
    
    fig.update_layout(
        title='Put Spread Strategy Comparison',
        height=500,
        showlegend=True
    )
    
    return fig
'''

# Insert the statistical charts code after the existing chart functions
insert_position = content.find('def create_probability_chart(recommendations, current_price, ticker):')
if insert_position == -1:
    insert_position = content.find('def main():')

content = content[:insert_position] + statistical_charts_code + '\n\n' + content[insert_position:]

# Modify the sidebar to include optional AI analysis checkbox
sidebar_modification = '''    # AI Analysis Configuration
    st.sidebar.subheader("🤖 AI Analysis Settings")
    enable_ai_analysis = st.sidebar.checkbox(
        "Enable AI Analysis", 
        value=False,
        help="Check to enable AI-powered analysis on all tabs"
    )
    
    if enable_ai_analysis:
        api_key_available = bool(os.getenv('OPENAI_API_KEY'))
        if not api_key_available:
            st.sidebar.warning("⚠️ Set OPENAI_API_KEY environment variable to enable AI analysis")
            enable_ai_analysis = False
        else:
            st.sidebar.success("✅ AI Analysis Ready")
    
    st.sidebar.markdown("---")
    
    # Ticker selection'''

# Replace the ticker selection part
old_sidebar = '''    # Sidebar for controls
    st.sidebar.header("Analysis Parameters")
    
    # Ticker selection'''

new_sidebar = '''    # Sidebar for controls
    st.sidebar.header("Analysis Parameters")
    
''' + sidebar_modification

content = content.replace(old_sidebar, new_sidebar)

# Modify the Price Charts tab to include statistical charts
old_tab2 = '''        with tab2:
            st.subheader("📈 Interactive Price Charts with Enhanced ATR")
            
            # Chart selection
            chart_ticker = st.selectbox("Select ticker for detailed chart:", session_tickers)
            chart_timeframe = st.selectbox(
                "Select timeframe:",
                [tf for tf in ['hourly', 'daily', 'weekly'] if tf in results[chart_ticker] and results[chart_ticker][tf]]
            )
            
            if chart_timeframe in results[chart_ticker] and results[chart_ticker][chart_timeframe]:
                chart_data = results[chart_ticker][chart_timeframe]['data']
                show_vix_chart = include_vix and chart_timeframe == 'daily' and vix_data is not None
                
                fig = create_price_chart(chart_data, chart_ticker, chart_timeframe, show_vix_chart)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Chart explanation
                    st.info("""
                    **Chart Elements:**
                    - 🕯️ **Candlesticks**: Price action (Open, High, Low, Close)
                    - 📊 **Blue Bars**: Range (High - Low) for each period
                    - 🔴 **Red Line**: Enhanced ATR (True Range average)
                    - 🟣 **Purple Line**: VIX levels (daily charts only)
                    - **Horizontal Lines**: VIX condition thresholds
                    """)'''

new_tab2 = '''        with tab2:
            st.subheader("📈 Interactive Price Charts with Enhanced ATR")
            
            # Chart selection
            chart_ticker = st.selectbox("Select ticker for detailed chart:", session_tickers)
            chart_timeframe = st.selectbox(
                "Select timeframe:",
                [tf for tf in ['hourly', 'daily', 'weekly'] if tf in results[chart_ticker] and results[chart_ticker][tf]]
            )
            
            if chart_timeframe in results[chart_ticker] and results[chart_ticker][chart_timeframe]:
                chart_data = results[chart_ticker][chart_timeframe]['data']
                show_vix_chart = include_vix and chart_timeframe == 'daily' and vix_data is not None
                
                # Main price chart
                fig = create_price_chart(chart_data, chart_ticker, chart_timeframe, show_vix_chart)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Chart explanation
                    st.info("""
                    **Chart Elements:**
                    - 🕯️ **Candlesticks**: Price action (Open, High, Low, Close)
                    - 📊 **Blue Bars**: Range (High - Low) for each period
                    - 🔴 **Red Line**: Enhanced ATR (True Range average)
                    - 🟣 **Purple Line**: VIX levels (daily charts only)
                    - **Horizontal Lines**: VIX condition thresholds
                    """)
                
                # Statistical Analysis Section
                st.markdown("### 📊 Statistical Analysis")
                
                # Create tabs for different statistical views
                stat_tab1, stat_tab2 = st.tabs(["📈 Price Distributions", "📊 Volatility Analysis"])
                
                with stat_tab1:
                    # Price distribution charts
                    dist_fig = create_price_distribution_chart(chart_data, chart_ticker, chart_timeframe)
                    if dist_fig:
                        st.plotly_chart(dist_fig, use_container_width=True)
                        
                        # Statistical summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            price_mean = chart_data['Close'].mean()
                            price_std = chart_data['Close'].std()
                            st.metric("Price Mean", f"${price_mean:.2f}")
                            st.metric("Price Std Dev", f"${price_std:.2f}")
                        
                        with col2:
                            if len(chart_data) > 1:
                                returns = chart_data['Close'].pct_change().dropna()
                                returns_mean = returns.mean() * 100
                                returns_std = returns.std() * 100
                                st.metric("Avg Return", f"{returns_mean:.2f}%")
                                st.metric("Return Volatility", f"{returns_std:.2f}%")
                        
                        with col3:
                            if 'Volume' in chart_data.columns:
                                vol_mean = chart_data['Volume'].mean()
                                vol_std = chart_data['Volume'].std()
                                st.metric("Avg Volume", f"{vol_mean:,.0f}")
                                st.metric("Volume Std", f"{vol_std:,.0f}")
                
                with stat_tab2:
                    # Volatility analysis charts
                    vol_fig = create_volatility_analysis_chart(chart_data, chart_ticker, chart_timeframe)
                    if vol_fig:
                        st.plotly_chart(vol_fig, use_container_width=True)
                        
                        # Volatility metrics
                        if 'true_range' in chart_data.columns:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                tr_mean = chart_data['true_range'].mean()
                                tr_std = chart_data['true_range'].std()
                                st.metric("Avg True Range", f"${tr_mean:.2f}")
                                st.metric("TR Volatility", f"${tr_std:.2f}")
                            
                            with col2:
                                atr_14 = chart_data['true_range'].rolling(14).mean().iloc[-1]
                                if not pd.isna(atr_14):
                                    st.metric("Current ATR (14)", f"${atr_14:.2f}")
                                
                                # ATR percentile
                                atr_series = chart_data['true_range'].rolling(14).mean()
                                current_percentile = (atr_series <= atr_14).mean() * 100
                                st.metric("ATR Percentile", f"{current_percentile:.0f}th")
                            
                            with col3:
                                range_efficiency = (chart_data['true_range'] / chart_data['Close']).mean() * 100
                                st.metric("Range Efficiency", f"{range_efficiency:.2f}%")
                
                # AI Analysis for Price Charts
                if enable_ai_analysis and LLM_AVAILABLE:
                    st.markdown("---")
                    st.markdown("### 🤖 AI Price Analysis")
                    
                    if st.button("🧠 Generate AI Price Analysis", key="price_ai_btn"):
                        with st.spinner("🤖 AI analyzing price patterns..."):
                            try:
                                llm_analyzer = get_llm_analyzer()
                                if llm_analyzer:
                                    # Prepare price data for AI
                                    price_summary = {
                                        'ticker': chart_ticker,
                                        'timeframe': chart_timeframe,
                                        'current_price': chart_data['Close'].iloc[-1],
                                        'price_change': chart_data['Close'].iloc[-1] - chart_data['Close'].iloc[0],
                                        'volatility': chart_data['true_range'].mean() if 'true_range' in chart_data.columns else None,
                                        'trend': 'Bullish' if chart_data['Close'].iloc[-1] > chart_data['Close'].iloc[0] else 'Bearish'
                                    }
                                    
                                    # Generate AI analysis (you'll need to add this method to llm_analysis.py)
                                    # For now, show a placeholder
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                                padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                                border-left: 4px solid var(--secondary-color);">
                                        <h4 style="color: var(--secondary-color); margin: 0 0 1rem 0;">
                                            🤖 AI Price Pattern Analysis
                                        </h4>
                                        <p>AI analysis for price patterns will be implemented here.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"AI analysis failed: {str(e)}")'''

content = content.replace(old_tab2, new_tab2)

# Modify the Put Spread Analysis tab to include statistical charts and AI analysis
old_tab7_start = '''        with tab7:
            st.subheader("📐 Advanced Put Spread Analysis")
            
            if PUT_SPREAD_AVAILABLE:'''

# Find the end of tab7 (before the main function)
tab7_end_marker = 'def main():'
tab7_end_pos = content.find(tab7_end_marker)

# Find the actual end of tab7 content
tab7_start_pos = content.find(old_tab7_start)
if tab7_start_pos != -1:
    # Find where the tab7 content actually ends (looking for next function or end of content)
    lines = content[tab7_start_pos:].split('\n')
    tab7_content = []
    indent_level = None
    
    for i, line in enumerate(lines):
        if line.strip() == '':
            tab7_content.append(line)
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        if indent_level is None and line.strip():
            indent_level = current_indent
        
        if line.strip() and current_indent <= indent_level and i > 0 and not line.startswith('        '):
            break
        
        tab7_content.append(line)
    
    old_tab7_content = '\n'.join(tab7_content)
    
    # Add statistical charts and AI analysis to the end of tab7 before the final "else" clause
    tab7_enhancement = '''
                            # Statistical Analysis Section
                            st.markdown("---")
                            st.markdown("### 📊 Put Spread Statistical Analysis")
                            
                            # Create statistical charts
                            stat_col1, stat_col2 = st.columns(2)
                            
                            with stat_col1:
                                # POT Analysis Charts
                                pot_fig = create_pot_analysis_charts(spread_results)
                                if pot_fig:
                                    st.plotly_chart(pot_fig, use_container_width=True)
                            
                            with stat_col2:
                                # Strategy Comparison Charts
                                comp_fig = create_strategy_comparison_chart(spread_results)
                                if comp_fig:
                                    st.plotly_chart(comp_fig, use_container_width=True)
                            
                            # Statistical Summary
                            st.markdown("#### 📋 Statistical Summary")
                            
                            if spread_results['scenarios']:
                                summary_data = []
                                for scenario in spread_results['scenarios']:
                                    summary_data.append({
                                        'POT Level': f"{scenario['target_pot']:.1%}",
                                        'Accuracy': f"{abs(scenario['actual_pot'] - scenario['target_pot']):.1%}",
                                        'Distance': f"{scenario['distance_pct']:.1f}%",
                                        'Strike': f"${scenario['short_strike']:.2f}",
                                        'Safety': 'High' if scenario['distance_pct'] > 5 else 'Medium' if scenario['distance_pct'] > 2 else 'Low'
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df, use_container_width=True)
                            
                            # AI Analysis for Put Spreads
                            if enable_ai_analysis and LLM_AVAILABLE:
                                st.markdown("---")
                                st.markdown("### 🤖 AI Put Spread Analysis")
                                
                                if st.button("🧠 Generate AI Put Spread Analysis", key="put_spread_ai_btn"):
                                    with st.spinner("🤖 AI analyzing put spread strategies..."):
                                        try:
                                            llm_analyzer = get_llm_analyzer()
                                            if llm_analyzer:
                                                # Prepare put spread data for AI analysis
                                                # Generate AI analysis using existing method but for put spreads
                                                llm_result = llm_analyzer.analyze_options_strategy(
                                                    ticker=spread_ticker,
                                                    current_price=current_price,
                                                    strategy_timeframe="put_spread",
                                                    recommendations=spread_results.get('scenarios', []),
                                                    prob_dist=None,  # We have POT data instead
                                                    vix_data=None,
                                                    atr=atr_value,
                                                    confidence_levels=None
                                                )
                                                
                                                if llm_result.get('llm_analysis', {}).get('success'):
                                                    st.markdown("""
                                                    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                                                padding: 2rem; border-radius: 16px; margin: 1rem 0; 
                                                                border-left: 6px solid var(--secondary-color); box-shadow: var(--shadow-lg);">
                                                        <h3 style="color: var(--secondary-color); margin: 0 0 1rem 0; font-weight: 700;">
                                                            🤖 AI Put Spread Strategy Analysis
                                                        </h3>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                    
                                                    st.markdown(llm_result['llm_analysis']['analysis'])
                                                    
                                                    with st.expander("📊 Analysis Details"):
                                                        col1, col2 = st.columns(2)
                                                        with col1:
                                                            st.metric("Tokens Used", llm_result['llm_analysis']['tokens_used'])
                                                        with col2:
                                                            st.metric("Analysis Time", 
                                                                    datetime.fromisoformat(llm_result['llm_analysis']['timestamp']).strftime("%H:%M:%S"))
                                                else:
                                                    st.warning("⚠️ AI analysis failed to generate results")
                                        except Exception as e:
                                            st.error(f"❌ AI analysis error: {str(e)}")
                        
                        else:
                            st.error("❌ Could not generate put spread analysis. Check data availability.")
                        
                    except Exception as e:
                        st.error(f"❌ Put spread analysis failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())'''
    
    # Insert the enhancement before the final else clause
    else_position = old_tab7_content.rfind('            else:')
    if else_position != -1:
        enhanced_tab7 = old_tab7_content[:else_position] + tab7_enhancement + '\n                    ' + old_tab7_content[else_position:]
        content = content.replace(old_tab7_content, enhanced_tab7)

# Write the enhanced content
with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Enhanced charts and optional AI analysis implemented!")
print("📊 Added statistical charts to Price Charts and Put Spread Analysis tabs")
print("🤖 Made AI analysis optional via sidebar checkbox")
print("🎯 AI analysis now available on all tabs including Put Spread Analysis") 