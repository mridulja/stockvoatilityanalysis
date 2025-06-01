#!/usr/bin/env python3
"""
Fix the indentation error around line 2357 in streamlit_stock_app_complete.py
"""

# Read the current streamlit app
with open('streamlit_stock_app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("🔧 Fixing indentation error around line 2357...")

# Find and fix the problematic section
problematic_section = '''                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        
                        with tab7:'''

fixed_section = '''                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with tab7:'''

if problematic_section in content:
    content = content.replace(problematic_section, fixed_section)
    print("✅ Fixed indentation issue with tab7")
else:
    print("⚠️ Problematic section not found, checking for similar patterns...")
    
    # Alternative pattern
    alt_pattern1 = '''        
        
                        with tab7:'''
    alt_fix1 = '''        
        with tab7:'''
    
    if alt_pattern1 in content:
        content = content.replace(alt_pattern1, alt_fix1)
        print("✅ Fixed alternative indentation pattern")

# Write the fixed content
with open('streamlit_stock_app_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Indentation error fixed!") 