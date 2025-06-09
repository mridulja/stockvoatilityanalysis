#!/usr/bin/env python3
"""
Simple Project Cleanup

Identifies and removes non-essential project files.
"""

import os
from pathlib import Path

def main():
    print("🧹 Simple Project Cleanup")
    print("="*50)
    
    # Files to remove (non-essential development files)
    files_to_remove = [
        # Backup files
        "streamlit_stock_app_complete_backup.py",
        
        # Temporary files
        "temp_file.py",
        
        # Fix scripts (development only)
        "fix_indentation_error.py",
        "fix_put_spread_and_remove_individual_ai.py", 
        "fix_options_strategy_ai.py",
        "fix_pop_and_add_charts.py",
        "fix_pot_formula.py",
        "fix_tab7.py",
        
        # Enhancement scripts (development only)
        "enhance_ai_and_distributions.py",
        "enhanced_charts_and_ai.py",
        "improve_strategy.py",
        "add_session_state_storage.py",
        
        # Test files (development only)
        "test_pop_formula.py",
        "test_pot_formula.py", 
        "test_strategy_logic.py",
        "test_improved_strategy.py",
        "test_realistic_dates.py",
        
        # Old version (superseded)
        "streamlit_stock_app.py",
        
        # Individual analysis (superseded by tabs)
        "put_spread_analysis.py",
        "stock_analyst.py",
        
        # Analysis files (no longer needed)
        "analyze_unused_files.py",
        "cleanup_project_files.py",
    ]
    
    # Check which files exist and calculate total size
    existing_files = []
    total_size = 0
    
    for filename in files_to_remove:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            size_mb = size / (1024 * 1024)
            existing_files.append((filepath, size_mb))
            total_size += size
            print(f"📂 {filename} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {filename} - Not found")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n📊 Summary:")
    print(f"Files to remove: {len(existing_files)}")
    print(f"Space to free: {total_mb:.2f} MB")
    
    if existing_files:
        print(f"\n⚠️ This will remove {len(existing_files)} files ({total_mb:.2f} MB)")
        response = input("Proceed? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            removed = 0
            for filepath, size_mb in existing_files:
                try:
                    filepath.unlink()
                    print(f"✅ Removed: {filepath.name}")
                    removed += 1
                except Exception as e:
                    print(f"❌ Failed to remove {filepath.name}: {e}")
            
            print(f"\n🎉 Cleanup complete! Removed {removed} files.")
        else:
            print("Cleanup cancelled.")
    else:
        print("No files to remove.")

if __name__ == "__main__":
    main() 