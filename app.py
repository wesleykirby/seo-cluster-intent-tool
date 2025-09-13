import pandas as pd
import streamlit as st
import tempfile
from cluster.pipeline import run_pipeline

def process_file(uploaded_file, min_sim, config_path):
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

with st.form("cluster_form"):
    uploaded = st.file_uploader("Upload keyword CSV", type=["csv"])
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.8, step=0.05)
    config_path = st.text_input("Config path (optional)")
    submitted = st.form_submit_button("Process")

if submitted:
    if not uploaded:
        st.warning("Please upload a CSV file before processing.")
    else:
        with st.spinner("Running pipeline..."):
            df, out_path = process_file(uploaded, min_sim, config_path)
        st.success("Processing complete. Preview below:")
        st.dataframe(df.head())
        with open(out_path, "rb") as f:
            st.download_button(
                "Download full results",
                data=f.read(),
                file_name="keywords_tagged.csv",
                mime="text/csv",
            )
