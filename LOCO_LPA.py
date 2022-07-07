import os
from collections import Counter
from pathlib import Path
from typing import Literal

import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

from LPA import LPA

Subcorpus = Literal["full", "conspiracy", "mainstream"]

if "WSL_DISTRO_NAME" not in os.environ.keys():
    nltk.download("stopwords")
    nltk.download("punkt")
    stemmer = SnowballStemmer("english")
    ignored_words = nltk.corpus.stopwords.words("english")
DATA_FOLDER = Path("./data")


def create_metadata() -> pd.DataFrame:
    metadata = pd.read_csv("data/loco/metadata.csv", parse_dates=["date"])
    metadata = metadata[metadata["date"].dt.year >= 1990]
    return metadata


def create_freq() -> pd.DataFrame:
    metadata = create_metadata()
    base_freq = []
    for i in range(0, 96000, 1000):
        base_freq.append(pd.read_csv(f"data/loco/np_freq/frequency_{i}.csv"))
    base_freq = pd.merge(
        pd.concat(base_freq),
        metadata[["category", "date", "subcorpus"]],
        on="category",
        how="inner",
    ).rename(columns={"count": "frequency_in_category"})
    return base_freq


def freq_window(
    base_freq: pd.DataFrame,
    quantity: int | str | tuple,
    cumulative: bool,
    direction: Literal["from", "to", "range"] = "to",
    subcorpus: Subcorpus = "full",
) -> pd.DataFrame:
    if isinstance(quantity, str) or len(str(quantity)) == 4:
        quantity = pd.to_datetime(quantity, format=("%Y"))
    if subcorpus != "full":
        base_freq = base_freq[base_freq["subcorpus"] == subcorpus].reset_index(
            drop=True
        )
    # quantity = pd.Period(**{p: quantity, "freq": freq})
    if cumulative:
        if direction == "to":
            return base_freq[base_freq["date"] <= quantity].reset_index(drop=True)
        elif direction == "from":
            return base_freq[base_freq["date"] >= quantity].reset_index(drop=True)
    elif direction == "range":
        return base_freq[
            (base_freq["date"] >= quantity[0]) & (base_freq["date"] <= quantity[1])
        ].reset_index(drop=True)
    else:
        return base_freq[base_freq["date"] == quantity].reset_index(drop=True)


if __name__ == "__main__":
    dvr = pd.read_csv("data/dvr.csv")
    ilpa = IterLPA(blocks=97, size=1000, dvr=dvr)
    ilpa.iter_sigs()
