# Fix tab7 implementation
with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the expandable section with proper tab7
old_section = '''        # Add the new Put Spread Analysis tab (tab7)
        if PUT_SPREAD_AVAILABLE:
            with st.expander("📐 Advanced Put Spread Analysis (Click to expand)", expanded=False):'''

new_section = '''        
        with tab7:
            st.subheader("📐 Advanced Put Spread Analysis")
            
            if PUT_SPREAD_AVAILABLE:'''

content = content.replace(old_section, new_section)

# Also fix the else clause at the end
content = content.replace(
    '''        else:
            st.warning("⚠️ Put Spread Analysis module not available. Please ensure put_spread_analysis.py is properly installed.")''',
    '''            else:
                st.warning("⚠️ Put Spread Analysis module not available. Please ensure put_spread_analysis.py is properly installed.")'''
)

# Write back
with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully converted expandable section to proper tab7') 