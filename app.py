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
    st.write("**Upload enhanced training data to improve the vector-based semantic model:**")
    
    # Show format options
    format_option = st.selectbox(
        "Training data format:",
        ["Enhanced (Main/Sub/Modifier)", "Legacy (Intent only)"],
        help="Enhanced format trains the sophisticated semantic analyzer. Legacy format only trains basic intent classification."
    )
    
    # Training data uploader
    training_file = st.file_uploader(
        "Upload training CSV", 
        type=["csv"], 
        key="training_upload",
        help=("Enhanced format: 'keyword', 'main_topic', 'sub_topic', 'modifier' columns. " +
              "Legacy format: 'keyword', 'intent' columns.")
    )
    
    # Show expected format based on selection
    if format_option == "Enhanced (Main/Sub/Modifier)":
        st.write("**🚀 Enhanced Format (Recommended):**")
        st.info("💡 This format trains the full semantic analyzer with Main Topic, Sub Topic, and Modifier prediction!")
        
        sample_training_data = pd.DataFrame({
            'keyword': ['betway register bonus', 'sportybet app download', 'football betting tips', 'aviator game crash'],
            'main_topic': ['Branded', 'Branded', 'Sports', 'Casino'],
            'sub_topic': ['Betway', 'SportyBet', 'Football', 'Aviator'],
            'modifier': ['Registration', 'App', 'Tips', 'Game']
        })
        st.dataframe(sample_training_data, use_container_width=True)
        
        # Show supported categories
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Main Topics:**")
            st.write("• Branded")
            st.write("• Sports") 
            st.write("• Casino")
            st.write("• Lottery")
            st.write("• Horse Racing")
            st.write("• Betting")
        
        with col2:
            st.write("**Sub Topics:**")
            st.write("• Brand names (Betway, SportyBet)")
            st.write("• Sports (Football, Basketball)")
            st.write("• Games (Aviator, PowerBall)")
            st.write("• Categories (Live Casino)")
            st.write("• General")
        
        with col3:
            st.write("**Modifiers (Intent):**")
            st.write("• Registration/Login")
            st.write("• App/Download")
            st.write("• Tips/Predictions")
            st.write("• Results/Live")
            st.write("• Ghana/Regional")
            st.write("• General")
    else:
        st.write("**Legacy Format:**")
        st.warning("⚠️ Legacy format only trains basic intent classification, not the full semantic analyzer.")
        
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
            
            # Determine format and validate columns
            if format_option == "Enhanced (Main/Sub/Modifier)":
                required_cols = ['keyword', 'main_topic', 'sub_topic', 'modifier']
                format_type = "enhanced"
            else:
                required_cols = ['keyword', 'intent']
                format_type = "legacy"
            
            missing_cols = [col for col in required_cols if col not in train_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.write(f"**Found columns:** {', '.join(train_df.columns)}")
            else:
                # Show preview
                st.write("**Preview of uploaded training data:**")
                st.dataframe(train_df.head(10), use_container_width=True)
                
                # Show distribution based on format
                if format_type == "enhanced":
                    # Show enhanced statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        main_counts = train_df['main_topic'].value_counts()
                        st.write("**Main Topics:**")
                        for topic, count in main_counts.items():
                            st.write(f"• {topic}: {count}")
                    
                    with col2:
                        sub_counts = train_df['sub_topic'].value_counts()
                        st.write("**Sub Topics:**")
                        for topic, count in sub_counts.head(5).items():
                            st.write(f"• {topic}: {count}")
                        if len(sub_counts) > 5:
                            st.write(f"... and {len(sub_counts) - 5} more")
                    
                    with col3:
                        mod_counts = train_df['modifier'].value_counts()
                        st.write("**Modifiers:**")
                        for mod, count in mod_counts.head(5).items():
                            st.write(f"• {mod}: {count}")
                        if len(mod_counts) > 5:
                            st.write(f"... and {len(mod_counts) - 5} more")
                    
                    # Show semantic analysis capabilities
                    st.info(f"🧠 **This training data will teach the semantic analyzer:**\n"
                           f"• {len(main_counts)} main topic categories\n"
                           f"• {len(sub_counts)} sub-topic patterns (brands, games, etc.)\n"
                           f"• {len(mod_counts)} intent modifiers\n"
                           f"• Advanced pattern recognition from {len(train_df)} keyword examples")
                    
                else:
                    # Show legacy statistics
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
                button_text = "🚀 Add Enhanced Training Data" if format_type == "enhanced" else "📚 Add Legacy Training Data"
                if st.button(button_text, type="primary", key="add_training"):
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
                        
                        # Show success message based on format
                        if format_type == "enhanced":
                            st.success(f"🎉 **Enhanced training data added!** The vector-based semantic model will learn from {len(train_df)} new examples. Total training data: {len(combined_data)} keywords.")
                            
                            # Trigger immediate retraining for enhanced data
                            with st.spinner("🧠 Training vector-based semantic model..."):
                                from scripts.weekly_retrain import train_vector_semantic_model
                                vector_results = train_vector_semantic_model(combined_data)
                                
                                if vector_results.get('vector_training_status') == 'success':
                                    st.success("✅ **Vector semantic model trained successfully!**")
                                    
                                    col_acc1, col_acc2, col_acc3 = st.columns(3)
                                    with col_acc1:
                                        st.metric("Main Topic Accuracy", f"{vector_results.get('main_accuracy', 0):.1%}")
                                    with col_acc2:
                                        st.metric("Sub Topic Accuracy", f"{vector_results.get('sub_accuracy', 0):.1%}")
                                    with col_acc3:
                                        st.metric("Modifier Accuracy", f"{vector_results.get('modifier_accuracy', 0):.1%}")
                                    
                                    st.write(f"**Learned:** {vector_results.get('learned_brands', 0)} brands, {vector_results.get('learned_patterns', 0)} patterns")
                                
                        else:
                            st.success(f"🎉 **Legacy training data updated!** Added {len(train_df)} new keywords. Total training data: {len(combined_data)} keywords.")
                        
                        # Show updated stats
                        if format_type == "enhanced":
                            st.write("**Updated semantic training capabilities:**")
                            main_counts = combined_data['main_topic'].value_counts()
                            st.write("**Main Topics:**")
                            for topic, count in main_counts.items():
                                st.write(f"• {topic}: {count} examples")
                        else:
                            new_intent_counts = combined_data['intent'].value_counts() if 'intent' in combined_data.columns else {}
                            st.write("**Updated training data distribution:**")
                            for intent, count in new_intent_counts.items():
                                st.write(f"• {intent}: {count}")
                                
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
            from scripts.weekly_retrain import run_retraining, load_training_data, DEFAULT_TRAINING_PATH
            
            with st.spinner("🧠 Retraining model with current training data..."):
                results = run_retraining()
            
            # Check if retraining actually occurred or was skipped
            if results['added_rows'] == 0 and results['evaluated_rows'] == 0:
                # No new labels from queue, but we might have training data
                training_data = load_training_data(DEFAULT_TRAINING_PATH)
                if not training_data.empty:
                    st.warning("⚠️ **Retraining was skipped** - No new labels in the review queue. The weekly retraining process only runs when there are new human-labeled keywords from the active learning queue.")
                    st.info("💡 **Tip:** The model will automatically retrain weekly when new labels are added through the active learning process, or upload keywords that need human review to trigger retraining.")
                else:
                    st.warning("⚠️ **No training data available** - Upload some labeled training data first, then try retraining.")
            else:
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
