from pathlib import Path
from typing import List, Tuple

import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app_tlpa import create_distances, create_freq, create_mdvr, tw_freq
from corpora import Corpus
from helpers import write

st.set_page_config(layout="centered", page_title="Upload Dataset", page_icon="💾")

data_cols = {"element", "frequency_in_category", "category"}
data_cols_msg = f"All data files must contain 3 columns: {', '.join(['`' + e  +'`' for e in data_cols])}. "
metadata_cols_msg = "Make sure metadata file is named `metadata.csv` and has two columns: `category` and `date`."


def validate(file_list: List[UploadedFile]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    metadata = None
    metadata_candidates = [f for f in file_list if f.name == "metadata.csv"]
    if metadata_candidates:
        metadata = pd.read_csv(metadata_candidates[0], parse_dates=["date"])
        if not set(metadata.columns) - {"date", "category"}:
            st.error(metadata_cols_msg)
            st.stop()
    data = [pd.read_csv(f) for f in file_list if f.name != "metadata.csv"]
    if any([set(d.columns) - data_cols for d in data]):
        st.error(data_cols_msg)
        st.stop()
    return metadata, data


##### config
config = {}
st.info(data_cols_msg + metadata_cols_msg)
col1, col2 = st.columns(2)
with col1:
    st.text("Data example")
    st.dataframe(
        pd.DataFrame(
            [
                ["apple", 5, "category_a"],
                ["pear", 2, "category_a"],
                ["orange", 6, "category_b"],
            ],
            columns=["element", "frequency_in_category", "category"],
        ),
    )
with col2:
    st.text("Metadata example")
    st.dataframe(
        pd.DataFrame(
            [
                ["category_a", "2022-07-01"],
                ["category_b", "2022-06-31"],
            ],
            columns=["category", "date"],
        )
    )
with st.form("my_form"):
    file_list = st.file_uploader("Upload your data", accept_multiple_files=True)
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
    with col1:
        config["corpus"] = st.text_input("Corpus name")
        config["freq"] = st.selectbox(
            "Initial temporal bin size",
            options=["D", "W", "MS"],
            index=2,
            format_func=lambda x: {"D": "Day", "W": "Week", "MS": "Month"}[x],
        )
        save = st.checkbox("Save dataset")
    with col2:
        start_date, end_date = None, None
        config["start_date"] = st.date_input(
            "Starting date", value=datetime.today() - timedelta(1)
        )
        config["end_date"] = st.date_input("End date", value=datetime.today())
        if config["end_date"] <= config["start_date"]:
            st.error("End date must be later than start date")
    with col3:
        config["epsilon"] = st.number_input("Epsilon if known", value=0)
        config["lambda"] = st.number_input(
            "Lambda", min_value=0.01, step=0.01, value=1.0
        )  # > 0
    with col4:
        config["metric"] = st.selectbox(
            "Timeline segmentation algorithm",
            ["KLD_divergence_consecutive", "JSD", "JSD_max", "sqrt_JSD"],
        )
        config["threshold"] = st.number_input(
            "Bin content threshold",
            min_value=0,
            value=0,
            step=1,
        )
    submitted = st.form_submit_button("Run")
    if submitted:
        metadata, data = validate(file_list)

if submitted:
    st.info("this might take a few minutes")
    progress_bar = st.progress(0)
    cname = config["corpus"].lower().replace(" ", "_")
    RESULTS = Path("results") / cname
    DATA = Path("data") / cname
    for p in (RESULTS, DATA):
        if not p.exists():
            RESULTS.mkdir()
    if save:
        write(DATA, (metadata, "metadata"))
        write(DATA, (metadata, *[(df, str(i)) for i, df in enumerate(data)]))
    base_freq = create_freq(metadata, data)
    progress_bar.progress(25)
    # # write(PATH, (base_freq, "base_freq"))
    # # base_freq = read("base_freq.csv")
    tw_freq_df = tw_freq(base_freq, config["freq"])
    progress_bar.progress(35)
    write(RESULTS, (tw_freq_df, "tw_freq"))
    # tw_freq_df = read(PATH, f"tw_freq.csv")
    corpus = Corpus(tw_freq_df["date"], tw_freq_df["element"])
    matrix = corpus.pivot(tw_freq_df)
    matrix = matrix.epsilon_modification(
        epsilon=config["epsilon"], lambda_=config["lambda"]
    )
    progress_bar.progress(50)
    # matrix, corpus = dBTC(base_freq, matrix, vectorizor, delta="median")
    dvr = create_mdvr(matrix, corpus)
    progress_bar.progress(60)
    write(RESULTS, (dvr, "dvr"))
    write(RESULTS, (matrix, "matrix"))
    # matrix = Matrix(read(PATH, "matrix.npy"))
    sigs, max_distances = create_distances(matrix, dvr, corpus)
    progress_bar.progress(70)
    for sig in sigs:
        name = sig.name.strftime("%Y-%m-%d")
        write(RESULTS, (sig.rename("KL").reset_index(), f"sigs/sigs_{name}"))
        progress_bar.progress(progress_bar.value + 1)
    progress_bar.progress(100)
    st.dataframe(max_distances)
    st.dataframe()
