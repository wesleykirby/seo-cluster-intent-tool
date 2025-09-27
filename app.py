import pandas as pd
import streamlit as st
import tempfile

from cluster.active_learning import (
    DEFAULT_QUEUE_PATH,
    filter_low_confidence,
    load_label_queue,
    save_label_queue,
    update_label_queue,
)
from cluster.pipeline import run_pipeline
from cluster.text_utils import detect_industry_vertical

def process_file(uploaded_file, min_sim, config_path):
    # Handle both UploadedFile and file path strings
    if isinstance(uploaded_file, str):
        # File path from override
        input_path = uploaded_file
    else:
        # UploadedFile from Streamlit
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_in:
            tmp_in.write(uploaded_file.getbuffer())
            tmp_in.flush()
            input_path = tmp_in.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_out:
        output_path = tmp_out.name
    run_pipeline(input_path, output_path, min_sim=min_sim, config_path=config_path or None)
    df = pd.read_csv(output_path)
    return df, output_path

st.title("🎯 Smart SEO Keyword Analyzer")

st.info("🚀 **Enhanced Semantic Analysis!** Upload a CSV with a 'keyword' column. If you have URL data (e.g., 'Current URL'), the system will use ranking page intelligence for superior accuracy!")

# File uploader outside form for immediate preview
uploaded = st.file_uploader("Upload keyword CSV", type=["csv"], help="Required: CSV with 'keyword' or 'Keyword' column. Optional: Include 'Current URL' for enhanced analysis with ranking page intelligence.")

# Form for processing parameters
with st.form("cluster_form"):
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.8, step=0.05)
    config_path = st.text_input("Config path (optional)")
    active_threshold = st.slider(
        "Label queue confidence threshold",
        0.0,
        1.0,
        0.6,
        step=0.05,
        help="Predictions below this confidence will be sent to the human label queue.",
    )
    submitted = st.form_submit_button("Process")

# Preview uploaded file - Now shows immediately when file is selected!
if uploaded is not None:
    st.info("🔍 **Instant Preview** - File loaded, analyzing structure...")
    try:
        temp_df = pd.read_csv(uploaded)
        st.write(f"📊 **File loaded successfully!** Found {len(temp_df)} rows and {len(temp_df.columns)} columns")
        
        # Show all columns for debugging
        st.write(f"**Columns in your file:** {', '.join(temp_df.columns)}")
        
        # Check for keyword column (case-insensitive)
        keyword_col = None
        if "keyword" in temp_df.columns:
            keyword_col = "keyword"
        elif "Keyword" in temp_df.columns:
            keyword_col = "Keyword"
        
        if keyword_col:
            # Check for URL columns
            url_columns = [col for col in temp_df.columns if 'url' in col.lower() and 'current' in col.lower()]
            serp_columns = [col for col in temp_df.columns if col in ['Volume', 'Current position', 'KD', 'CPC', 'Organic traffic']]
            
            # Enhanced preview message
            enhancement_msg = f"✅ Ready to analyze **{len(temp_df)} keywords** with semantic ML!"
            if url_columns:
                enhancement_msg += f"\n🔗 **URL-Enhanced Analysis Available!** Found ranking URL data in '{url_columns[0]}' column."
            if serp_columns:
                enhancement_msg += f"\n📈 **SERP Data Detected!** Will preserve {len(serp_columns)} metrics: {', '.join(serp_columns)}"
            
            st.success(enhancement_msg)
            
            # Show sample data
            st.write("**Sample data from your file:**")
            preview_cols = [keyword_col]
            if url_columns:
                preview_cols.append(url_columns[0])
            if serp_columns:
                preview_cols.extend(serp_columns[:3])  # Show first 3 SERP metrics
                
            sample_data = temp_df[preview_cols].head(5)
            st.dataframe(sample_data, use_container_width=True)
            
            if len(temp_df) > 5:
                st.write(f"... and {len(temp_df) - 5} more rows")
                
            st.info("👆 **Preview ready!** Click 'Process' button below to start analysis.")
                
        else:
            st.error("❌ CSV must contain a 'keyword' or 'Keyword' column")
            
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.write("Please make sure your file is a valid CSV format.")

if submitted:
    if not uploaded:
        st.warning("Please upload a CSV file before processing.")
    else:
        with st.spinner("🧠 Running semantic analysis with ML pattern discovery..."):
            df, out_path = process_file(uploaded, min_sim, config_path)
        
        # Check if URL enhancement was used
        url_enhanced = False
        if len(df) > 0 and 'URL_Enhanced' in df.columns:
            url_enhanced = df['URL_Enhanced'].iloc[0]
        
        if url_enhanced:
            st.success("🎉 **URL-Enhanced Analysis Complete!** Used ranking page intelligence to improve classification accuracy.")
        else:
            st.success("🎉 **Analysis Complete!** Your keywords have been organized into clean semantic structure.")
        
        # Show results summary
        col1, col2, col3, col4 = st.columns(4)
        
        if 'Main' in df.columns:
            with col1:
                main_topics = df['Main'].value_counts()
                st.metric("Main Topics", len(main_topics))
                st.write("**Top topics:**")
                for topic, count in main_topics.head(3).items():
                    st.write(f"• {topic}: {count}")
        
        if 'Sub' in df.columns:
            with col2:
                sub_topics = df['Sub'].value_counts()
                st.metric("Sub Topics", len(sub_topics))
                st.write("**Top brands/categories:**")
                for topic, count in sub_topics.head(3).items():
                    st.write(f"• {topic}: {count}")
        
        if 'Mod' in df.columns:
            with col3:
                modifiers = df['Mod'].value_counts()
                st.metric("Intent Modifiers", len(modifiers))
                st.write("**Top intents:**")
                for mod, count in modifiers.head(3).items():
                    st.write(f"• {mod}: {count}")
        
        with col4:
            st.metric("Total Keywords", len(df))
            st.write("**Output structure:**")
            st.write("✅ Main Topic")
            st.write("✅ Sub Topic") 
            st.write("✅ Modifier")
            st.write("✅ Keyword")
            
            # Show enhanced features
            serp_cols = [col for col in df.columns if col in ['Volume', 'Current position', 'KD', 'CPC', 'Organic traffic']]
            if serp_cols:
                st.write(f"📈 + {len(serp_cols)} SERP metrics")
        
        st.divider()
        
        # Show preview of results
        st.subheader("📋 Results Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        if len(df) > 20:
            st.info(f"Showing first 20 rows of {len(df)} total keywords. Download the full CSV below.")
        
        # Download button
        with open(out_path, "rb") as f:
            st.download_button(
                "📥 Download Complete Analysis (CSV)",
                data=f.read(),
                file_name="semantic_keyword_analysis.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

st.divider()

# Information section
st.subheader("🧠 How This Analysis Works")

col1, col2 = st.columns(2)

with col1:
    st.write("**🔍 Semantic Pattern Discovery:**")
    st.write("• Automatically discovers brands from your data")
    st.write("• Identifies intent modifiers (Login, App, Registration, etc.)")
    st.write("• Groups keywords by semantic meaning, not just similarity")
    st.write("• Adapts to different markets and regions")

with col2:
    st.write("**📊 Output Structure:**")
    st.write("• **Main**: Core topic (Branded, Betting, Sports, Casino)")
    st.write("• **Sub**: Specific brand or category")
    st.write("• **Mod**: Intent modifier (the key to user intent!)")
    st.write("• **Keyword**: Your original search term")

st.info("💡 **Pro Tip**: The 'Modifier' column reveals the true user intent - Login (access), App (mobile), Ghana (geo-targeting), etc. This is perfect for SEO content strategy!")

st.divider()

# Training Data Management Section
st.subheader("📚 Training Data Management")
st.write("Upload labeled training data to continuously improve the model's accuracy. This helps the system learn new keywords, brands, and patterns.")

# Create tabs for different training functions
train_tab1, train_tab2 = st.tabs(["📤 Upload Training Data", "📊 Training Status"])

with train_tab1:
    st.write("**Upload labeled keywords to expand the model's knowledge base:**")
    
    # Training data uploader
    training_file = st.file_uploader(
        "Upload training CSV", 
        type=["csv"], 
        key="training_upload",
        help="CSV should contain columns: 'keyword' and 'intent'. Supported intents: transactional, informational, commercial, navigational"
    )
    
    # Show expected format
    st.write("**Expected format:**")
    sample_training_data = pd.DataFrame({
        'keyword': ['betway register bonus', 'football betting tips', 'odds comparison'],
        'intent': ['transactional', 'informational', 'commercial']
    })
    st.dataframe(sample_training_data, use_container_width=True)
    
    if training_file is not None:
        try:
            # Preview training data
            train_df = pd.read_csv(training_file)
            st.success(f"✅ **Training file loaded!** Found {len(train_df)} labeled keywords")
            
            # Validate required columns
            required_cols = ['keyword', 'intent']
            missing_cols = [col for col in required_cols if col not in train_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                # Show preview
                st.write("**Preview of uploaded training data:**")
                st.dataframe(train_df.head(10), use_container_width=True)
                
                # Show intent distribution
                intent_counts = train_df['intent'].value_counts()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Intent Distribution:**")
                    for intent, count in intent_counts.items():
                        st.write(f"• {intent}: {count} keywords")
                
                with col2:
                    st.write("**Keywords by Intent:**")
                    total_keywords = len(train_df)
                    for intent, count in intent_counts.items():
                        percentage = (count / total_keywords) * 100
                        st.progress(percentage / 100, text=f"{intent}: {percentage:.1f}%")
                
                # Add to training data button
                if st.button("📚 Add to Training Data", type="primary", key="add_training"):
                    try:
                        # Import the retraining script function
                        import sys
                        from pathlib import Path
                        
                        # Add the scripts directory to path
                        scripts_path = Path("scripts")
                        if str(scripts_path) not in sys.path:
                            sys.path.append(str(scripts_path))
                        
                        from scripts.weekly_retrain import DEFAULT_TRAINING_PATH, load_training_data
                        
                        # Load existing training data
                        existing_data = load_training_data(DEFAULT_TRAINING_PATH)
                        
                        # Combine with new data
                        combined_data = pd.concat([existing_data, train_df], ignore_index=True)
                        
                        # Remove duplicates (keep latest)
                        combined_data = combined_data.drop_duplicates(subset=['keyword'], keep='last')
                        
                        # Save updated training data
                        combined_data.to_csv(DEFAULT_TRAINING_PATH, index=False)
                        
                        st.success(f"🎉 **Training data updated!** Added {len(train_df)} new keywords. Total training data now contains {len(combined_data)} keywords.")
                        
                        # Show updated stats
                        new_intent_counts = combined_data['intent'].value_counts()
                        st.write("**Updated training data distribution:**")
                        for intent, count in new_intent_counts.items():
                            st.write(f"• {intent}: {count} keywords")
                            
                    except Exception as e:
                        st.error(f"❌ Error adding training data: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error reading training file: {str(e)}")

with train_tab2:
    st.write("**Current training data statistics:**")
    
    try:
        from scripts.weekly_retrain import DEFAULT_TRAINING_PATH, load_training_data
        
        # Load current training data
        current_training = load_training_data(DEFAULT_TRAINING_PATH)
        
        if current_training.empty:
            st.info("No training data found. Upload some labeled keywords to get started!")
        else:
            # Show overall stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Keywords", len(current_training))
            
            with col2:
                unique_intents = current_training['intent'].nunique()
                st.metric("Intent Categories", unique_intents)
            
            with col3:
                # Calculate average keywords per intent
                avg_per_intent = len(current_training) / unique_intents if unique_intents > 0 else 0
                st.metric("Avg Keywords/Intent", f"{avg_per_intent:.1f}")
            
            # Show intent breakdown
            st.write("**Training data by intent:**")
            intent_breakdown = current_training['intent'].value_counts()
            
            # Create a nice chart
            chart_data = pd.DataFrame({
                'Intent': intent_breakdown.index,
                'Count': intent_breakdown.values
            })
            st.bar_chart(chart_data.set_index('Intent'))
            
            # Show recent additions (if we can determine them)
            st.write("**Sample training keywords:**")
            sample_data = current_training.sample(min(10, len(current_training)))
            st.dataframe(sample_data, use_container_width=True)
            
            # Download current training data
            csv_data = current_training.to_csv(index=False)
            st.download_button(
                "📥 Download Current Training Data",
                data=csv_data,
                file_name="current_training_data.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error loading training data: {str(e)}")

# Option to trigger retraining
st.write("**Model Retraining:**")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Trigger Immediate Retraining", help="Retrain the model with current training data"):
        try:
            from scripts.weekly_retrain import run_retraining
            
            with st.spinner("🧠 Retraining model with updated data..."):
                results = run_retraining()
            
            st.success("✅ **Model retrained successfully!**")
            
            # Show retraining results
            if results['before_f1'] is not None and results['after_f1'] is not None:
                improvement = results['after_f1'] - results['before_f1']
                st.write(f"**Performance change:** {improvement:+.3f} F1 score")
                
                col_before, col_after = st.columns(2)
                with col_before:
                    st.metric("Before F1", f"{results['before_f1']:.3f}")
                with col_after:
                    st.metric("After F1", f"{results['after_f1']:.3f}")
            
            st.write(f"**Evaluated on:** {results['evaluated_rows']} keywords")
            st.write(f"**Added to training:** {results['added_rows']} new labels")
            
        except Exception as e:
            st.error(f"❌ Error during retraining: {str(e)}")

with col2:
    st.info("🗓️ **Automatic retraining** runs weekly on Sundays at 2:00 AM, incorporating new labels from the review queue.")
