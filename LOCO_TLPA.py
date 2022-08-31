from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import tomli
from more_itertools import consecutive_groups

from corpora import Corpus, Matrix
from helpers import read, write
from LPA import LPA
from visualize import metric_bar_chart, moving_avg

with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)
PATH = Path("results") / config["corpus"]


def create_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(
        f"data/{config['corpus']}/metadata.csv", parse_dates=["date"]
    )
    metadata = metadata[
        (metadata["date"] >= pd.Timestamp(config["start_date"]))
        & (metadata["date"] <= pd.Timestamp(config["end_date"]))
    ]
    if config["threshold"] > 0:
        resampled = metadata.resample(config["freq"], on="date").count()
        filter_ = resampled[resampled > config["threshold"]].dropna()
        filter_ = filter_.index.astype(str).str[:7]
        metadata = metadata[metadata["date"].astype(str).str[:7].isin(filter_)]
    return metadata


# def create_freq(threshold=20) -> pd.DataFrame:
#     metadata, filter_ = create_metadata(threshold)
#     base_freq = []
#     for i in range(0, 96000, 1000):
#         base_freq.append(pd.read_csv(f"data/loco/np_freq/frequency_{i}.csv"))
#     base_freq = pd.merge(
#         pd.concat(base_freq),
#         metadata[["category", "date", "subcorpus"]],
#         on="category",
#         how="inner",
#     ).rename(columns={"count": "frequency_in_category"})
#     base_freq["dt"] = base_freq["date"].astype("str").str[:7]

#     base_freq = base_freq.set_index(["subcorpus", "dt"])
#     base_freq = (
#         base_freq.loc[filter_.index]
#         .sort_values("date")
#         .reset_index(level=0)
#         .reset_index(drop=True)
#     )
#     return base_freq


def create_freq() -> pd.DataFrame:
    metadata = create_metadata()
    base_freq = []
    for p in glob(f"data/{config['corpus']}/np_freq/*.csv"):
        base_freq.append(pd.read_csv(p))
    base_freq = pd.merge(
        pd.concat(base_freq),
        metadata[["category", "date"]],
        on="category",
        how="inner",
    )
    base_freq["category"] = pd.Categorical(base_freq["category"])
    # .rename(columns={"count": "frequency_in_category"}) #FIXME: frequency_in_category in the firstplace
    return base_freq


def tw_freq(
    base_freq: pd.DataFrame, freq: Literal["MS", "D", "W"] = "MS"
) -> pd.DataFrame:
    df = base_freq.groupby([pd.Grouper(freq=freq, key="date"), "element"]).sum()
    res = (
        (df / base_freq.resample(freq, on="date").sum())
        .reset_index()
        .rename(columns={"frequency_in_category": "global_weight"})
    )
    return res


def check_metric(
    matrix: Matrix, delta: str | float, iter_: int
) -> List[Tuple[int, int]]:
    metric = config["metric"]
    df = matrix.apply(metric)
    if delta == "median":
        delta = df[metric].median()
    metric_bar_chart(df, rule_value=delta, metric=metric).save(
        f"results/{config['corpus']}/bar_charts/{metric}_delta_{delta}_iter_{iter_}.html"
    )
    low = df[df[metric] < delta]
    groups = [
        (min(i), max(i))
        for i in [list(x) for x in consecutive_groups(low.index)]
        if len(i) > 1
    ]
    return groups


def dBTC(
    base_freq: pd.DataFrame,
    matrix: Matrix,
    corpus: Corpus,
    delta: float | str,
) -> Tuple[Matrix, Corpus]:
    """
    δ-bounded timeline compression using Kullback-Leibler divergence under a delta threshold - in this case the median.
    """
    iter_ = 0
    groups = check_metric(matrix, delta, iter_)
    print(f"should be around {sum(b-a for a, b in groups)} iterations")
    while len(groups) > 0:
        iter_ += 1
        date_code = groups[0][0]
        date = corpus.code_to_cat(date_code)
        next_date = corpus.code_to_cat(date_code + 1)
        matrix.delete(date_code + 1, 0)
        corpus.update_dates(corpus.code_to_cat(date_code + 1))
        squeezed_matrix = squeeze_freq(base_freq, date, next_date, corpus)
        matrix.matrix[date_code] = squeezed_matrix.epsilon_modification(
            epsilon=config["epsilon"], lambda_=config["lambda"]
        )
        groups = check_metric(matrix, delta, iter_)
        print(f"finished iteration {iter_}")
        if len(groups) == 0:
            return matrix, corpus


def create_mdvr(matrix: Matrix, corpus: Corpus) -> pd.DataFrame:
    dvr = pd.DataFrame(
        {
            "element": corpus.element_cat.categories,
            "global_weight": matrix.normalized_average_weight(),
        }
    )
    dvr = (
        dvr.reset_index()
        .rename(columns={"index": "element_code"})
        .sort_values("global_weight", ascending=False)
        .reset_index(drop=True)
    )
    return dvr


def squeeze_freq(
    base_freq: pd.DataFrame,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    corpus: Corpus,
) -> Matrix:
    squeezed = (
        base_freq[(base_freq["date"] >= min_date) & (base_freq["date"] <= max_date)]
        .groupby("element", as_index=False)
        .agg({"frequency_in_category": "sum"})
    )
    squeezed["date"] = min_date
    squeezed["global_weight"] = squeezed["frequency_in_category"] / sum(
        squeezed["frequency_in_category"]
    )
    matrix = corpus.pivot(squeezed)
    return matrix


def create_and_cut(vectors):
    # FIXME for matrix
    vectors = vectors.rename(
        columns={"date": "category", "global_weight": "local_weight"}
    ).drop(columns=["frequency_in_category"])
    dvr = create_mdvr(vectors)
    lpa = LPA(dvr, epsilon_frac=2)
    vectors = vectors.sort_values(["category", "element"]).reset_index(drop=True)
    lpa.create_arrays(vectors)
    sigs, max_distances = lpa.create_distances(vectors)
    return dvr, sigs, max_distances


if __name__ == "__main__":
    """
    The main process of this code reads
    """
    base_freq = create_freq()
    # write(PATH, (base_freq, "base_freq"))
    # base_freq = read("base_freq")
    tw_freq_df = tw_freq(base_freq, config["freq"])
    write((tw_freq_df, "tw_freq"))
    # tw_freq_df = read(f"tw_freq")
    corpus = Corpus(tw_freq_df)
    matrix = corpus.pivot(tw_freq_df)
    # moving_average(matrix, window=3)
    matrix = matrix.epsilon_modification(
        epsilon=config["epsilon"], lambda_=config["lambda"]
    )
    check_metric(matrix, "median", 1)
    # matrix, vectorizor = dBTC(base_freq, matrix, vectorizor, delta="median")
    ## write(PATH, (squeezed_freq, "squeezed_freq"), (kldf, "final_kldf"))
    ## squeezed_freq = read("squeezed_freq")
    dvr = create_mdvr(matrix, corpus)
    write(PATH, (dvr, "dvr"))
    write(PATH, (matrix, "matrix"))

    # dvr, sigs, max_distances = create_and_cut(squeezed_freq)
    # write(PATH, (max_distances, "max_distances"))
    # for sig in sigs:
    #     name = sig.name.strftime("%Y-%m-%d")
    #     write(PATH, (sig.rename("KL").reset_index(), f"sigs/sigs_{name}"))
