from pathlib import Path

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
from cluster.text_utils import normalize_kw
from scripts.weekly_retrain import DEFAULT_TRAINING_PATH, run_retraining

TRAINING_DATA_PATH = Path(DEFAULT_TRAINING_PATH)


def _load_training_data() -> pd.DataFrame:
    if not TRAINING_DATA_PATH.exists():
        return pd.DataFrame(columns=["keyword", "intent", "keyword_norm"])
    df = pd.read_csv(TRAINING_DATA_PATH)
    if "keyword" not in df.columns or "intent" not in df.columns:
        st.warning(
            "Existing training data is missing required columns. A fresh upload will overwrite it."
        )
        return pd.DataFrame(columns=["keyword", "intent", "keyword_norm"])
    if "keyword_norm" not in df.columns:
        df["keyword_norm"] = df["keyword"].astype(str).apply(normalize_kw)
    return df


def _persist_training_data(df: pd.DataFrame) -> None:
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATA_PATH, index=False)


def _prepare_training_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    rename_map = {}
    if "human_intent" in df.columns and "intent" not in df.columns:
        rename_map["human_intent"] = "intent"
    if rename_map:
        df = df.rename(columns=rename_map)
    missing = {"keyword", "intent"} - set(df.columns)
    if missing:
        raise ValueError(
            "Uploaded training data must include the following columns: "
            + ", ".join(sorted(missing))
        )
    result = df.copy()
    result["keyword"] = result["keyword"].astype(str).str.strip()
    result["intent"] = result["intent"].astype(str).str.strip()
    result = result[result["keyword"] != ""]
    result = result[result["intent"] != ""]
    if result.empty:
        raise ValueError("No valid (keyword, intent) pairs found in uploaded file.")
    result["keyword_norm"] = result["keyword"].apply(normalize_kw)
    keep_cols = [col for col in ["keyword", "intent", "keyword_norm"] if col in result.columns]
    return result[keep_cols]

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

        # Active learning queue section
        if 'intent_conf' in df.columns:
            low_conf_mask = pd.to_numeric(df['intent_conf'], errors='coerce').fillna(0.0) < active_threshold
            newly_flagged = int(low_conf_mask.sum())
            queue_df = update_label_queue(
                df,
                threshold=active_threshold,
                queue_path=DEFAULT_QUEUE_PATH,
            )

            st.divider()
            st.subheader("📝 Review low-confidence predictions")
            st.caption(
                "Edit the `human_intent` column to override model predictions. "
                "Changes persist to `data/label_queue.csv`."
            )

            status_cols = st.columns(3)
            with status_cols[0]:
                st.metric("New rows queued", newly_flagged)
            with status_cols[1]:
                st.metric("Total queue size", len(queue_df))
            with status_cols[2]:
                st.metric("Confidence threshold", active_threshold)

            if st.button("🔄 Reload saved queue", key="reload_queue"):
                queue_df = load_label_queue(DEFAULT_QUEUE_PATH)

            edited_queue = st.data_editor(
                queue_df,
                key="label_queue_editor",
                num_rows="dynamic",
                use_container_width=True,
            )

            if st.button("💾 Save corrections", key="save_queue", type="primary"):
                save_label_queue(edited_queue, queue_path=DEFAULT_QUEUE_PATH)
                st.success("Label queue saved. Retraining scripts can now ingest the corrections.")
        else:
            st.warning(
                "The current analysis did not include intent confidence scores, so the label queue could not be updated."
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

st.subheader("🤖 Train or Update the Intent Model")
st.write(
    "Upload freshly labeled intent data and trigger retraining without leaving the app."
)

training_stats = _load_training_data()
if not training_stats.empty:
    intents = training_stats["intent"].nunique()
    st.success(
        f"Current training set: {len(training_stats)} examples across {intents} unique intents."
    )
else:
    st.info("No training data found yet. Upload a labeled CSV to get started.")

with st.expander("📤 Upload labeled training data", expanded=False):
    st.write(
        "Provide a CSV with at least `keyword` and `intent` columns (or `human_intent`)."
    )
    training_upload = st.file_uploader(
        "Upload labeled CSV", type=["csv"], key="training_upload"
    )
    overwrite = st.checkbox(
        "Overwrite existing training data instead of merging", value=False
    )
    if st.button("Add to training set", type="primary"):
        if not training_upload:
            st.warning("Please upload a labeled CSV first.")
        else:
            try:
                new_examples = _prepare_training_upload(training_upload)
                if overwrite:
                    combined = new_examples
                else:
                    existing = _load_training_data()
                    combined = pd.concat([existing, new_examples], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["keyword_norm", "intent"], keep="last"
                ).sort_values("keyword_norm")
                _persist_training_data(combined)
                st.success(
                    f"Saved {len(combined)} total training examples. Last upload contributed {len(new_examples)} rows."
                )
            except Exception as exc:
                st.error(f"❌ Could not process uploaded training data: {exc}")

st.markdown("---")

st.subheader("🪄 Retrain intent classifier from the label queue")
st.write(
    "This will pull newly labeled rows from the active-learning queue, evaluate performance, "
    "and update the training CSV."
)

holdout_ratio = st.slider(
    "Holdout ratio for evaluation",
    0.1,
    0.9,
    0.4,
    step=0.05,
    help="Portion of new labels reserved for before/after evaluation during retraining.",
)

if st.button("Run retraining", use_container_width=True):
    with st.spinner("Retraining model with latest human labels..."):
        metrics = run_retraining(
            training_path=TRAINING_DATA_PATH,
            queue_path=DEFAULT_QUEUE_PATH,
            holdout_ratio=holdout_ratio,
        )
    before = metrics.get("before_f1")
    after = metrics.get("after_f1")
    evaluated = metrics.get("evaluated_rows", 0)
    added = metrics.get("added_rows", 0)
    if added == 0:
        st.warning("No newly labeled rows found in the queue. Nothing to retrain.")
    else:
        cols = st.columns(2)
        cols[0].metric(
            "Before retraining F1",
            "n/a" if before is None else f"{before:.3f}",
        )
        cols[1].metric(
            "After retraining F1",
            "n/a" if after is None else f"{after:.3f}",
        )
        st.success(
            f"Evaluated on {evaluated} labeled rows and appended {added} examples to the training set."
        )
