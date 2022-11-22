import numpy as np
import os
import pandas as pd

import streamlit as st
from visualize import timeline

st.set_page_config(layout="wide", page_title="Visualize", page_icon="📈")

dataset = st.selectbox("Select a dataset", sorted(os.listdir("results/")))
dvr = (
    pd.read_csv(f"results/{dataset}/dvr.csv")
    .sort_values("global_weight", ascending=False)
    .head(15)
)
with open(f"results/{dataset}/matrix.npy", "rb") as f:
    matrix = np.load(f)
pdf = (
    pd.DataFrame(matrix[:, dvr["element_code"]], columns=dvr["element"])
    .reset_index()
    .melt(id_vars="index")
    .rename(columns={"index": "xdate", "value": "global_weight"})
)
# pdf["xdate"] = pdf["date"].astype(str).str[:7]
st.altair_chart(
    timeline(
        pdf,
        x="xdate",
        y="global_weight",
        subcorpus="enron",
        stack="center",
        order=dvr["element"].to_list(),
        name=f"vis_1_enron_timeline",
    ),
    use_container_width=True,
)
