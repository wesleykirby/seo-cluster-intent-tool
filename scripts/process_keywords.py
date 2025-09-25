#!/usr/bin/env python3
"""
Direct Keyword Analyzer - No Web Interface Required
Upload your CSV and get instant 4-column analysis with URL enhancement
"""

import pandas as pd
import sys
from cluster.pipeline import run_pipeline

def process_csv(input_file, output_file=None, min_sim=0.8):
    """Process a CSV file with keyword analysis"""
    
    if not output_file:
        output_file = input_file.replace('.csv', '_analyzed.csv')
    
    print(f"🎯 Processing: {input_file}")
    print(f"📊 Output will be saved to: {output_file}")
    
    try:
        # Run the analysis pipeline
        run_pipeline(input_file, output_file, min_sim=min_sim)
        
        # Show results
        df = pd.read_csv(output_file)
        print(f"\n✅ Analysis complete! Processed {len(df)} keywords")
        print("\n📋 Sample results:")
        print(df.head(10).to_string(index=False))
        
        if len(df) > 10:
            print(f"\n... and {len(df) - 10} more rows")
        
        print(f"\n💾 Full results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_keywords.py input.csv [output.csv] [similarity_threshold]")
        print("Example: python process_keywords.py my_keywords.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    min_sim = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    
    process_csv(input_file, output_file, min_sim)