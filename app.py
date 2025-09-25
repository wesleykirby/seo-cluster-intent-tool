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

with st.form("cluster_form"):
    uploaded = st.file_uploader("Upload keyword CSV", type=["csv"], help="Required: CSV with 'keyword' or 'Keyword' column. Optional: Include 'Current URL' for enhanced analysis with ranking page intelligence.")
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

# Preview uploaded file
if uploaded is not None:
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
