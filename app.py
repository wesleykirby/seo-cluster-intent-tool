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

st.title("SEO Cluster & Intent Tool")

st.info("💡 Smart CSV Processing: Just upload a CSV with a 'keyword' column - the tool will automatically detect the industry vertical and process everything!")

with st.form("cluster_form"):
    uploaded = st.file_uploader("Upload keyword CSV", type=["csv"], help="Required: CSV with 'keyword' column. Industry vertical will be auto-detected.")
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

# Industry vertical detection and override
if uploaded is not None:
    # Preview the uploaded file to show detection
    temp_df = pd.read_csv(uploaded)
    if "keyword" in temp_df.columns:
        detected_vertical = detect_industry_vertical(temp_df["keyword"].tolist())
        st.success(f"🎯 Auto-detected industry: **{detected_vertical}**")
        
        # Allow manual override
        override_vertical = st.selectbox(
            "Override industry vertical (optional):",
            options=["Use auto-detected"] + ["betting", "ecommerce", "finance", "health", "education", "technology", "travel", "real_estate", "automotive", "food", "general"],
            key="vertical_override"
        )
    else:
        st.error("CSV must contain a 'keyword' column")
        override_vertical = "general"

if submitted:
    if not uploaded:
        st.warning("Please upload a CSV file before processing.")
    else:
        # Apply vertical override if selected
        if override_vertical != "Use auto-detected":
            # Temporarily modify the uploaded file to include the vertical column
            temp_df = pd.read_csv(uploaded)
            temp_df["vertical"] = override_vertical
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_df.to_csv(temp_file.name, index=False)
                uploaded = temp_file.name
        
        with st.spinner("Running pipeline..."):
            df, out_path = process_file(uploaded, min_sim, config_path)
        st.success("Processing complete. Preview below:")
        
        # Show detected/overridden vertical in results
        if "vertical" in df.columns:
            vertical_used = df["vertical"].iloc[0]
            st.info(f"📊 Industry vertical used: **{vertical_used}**")
        
        st.dataframe(df.head())
        low_conf = filter_low_confidence(df, threshold=active_threshold)
        update_label_queue(candidates=low_conf)
        if low_conf.empty:
            st.info("No low-confidence keywords were added to the label queue.")
        else:
            st.info(
                f"Queued {len(low_conf)} keywords with intent confidence below "
                f"{active_threshold:.2f}."
            )
            st.dataframe(
                low_conf[[c for c in ["keyword", "intent", "intent_conf"] if c in low_conf.columns]],
                use_container_width=True,
            )
        with open(out_path, "rb") as f:
            st.download_button(
                "Download full results",
                data=f.read(),
                file_name="keywords_tagged.csv",
                mime="text/csv",
            )

st.divider()
st.header("Active Learning Label Queue")
st.caption(f"Queue stored at `{DEFAULT_QUEUE_PATH}`")
queue_df = load_label_queue()
if queue_df.empty:
    st.info("No low-confidence keywords are awaiting review.")
else:
    unlabeled_mask = queue_df["human_intent"].astype(str).str.strip() == ""
    st.write(
        f"{int(unlabeled_mask.sum())} of {len(queue_df)} keywords still need a human label."
    )
    edited_queue = st.data_editor(
        queue_df,
        num_rows="dynamic",
        key="label_queue_editor",
        use_container_width=True,
    )
    col_save, col_download = st.columns(2)
    with col_save:
        if st.button("Save queue updates", key="save_queue_button"):
            save_label_queue(edited_queue)
            st.success("Label queue saved.")
            queue_df = edited_queue
    labeled_rows = edited_queue[
        edited_queue["human_intent"].astype(str).str.strip() != ""
    ]
    csv_bytes = labeled_rows.to_csv(index=False).encode("utf-8") if not labeled_rows.empty else b""
    with col_download:
        st.download_button(
            "Download labeled rows",
            data=csv_bytes,
            file_name="new_labels.csv",
            mime="text/csv",
            disabled=labeled_rows.empty,
            help="Exports rows that already have a human-provided intent label.",
        )
