from pathlib import Path
from typing import Literal, Tuple

from more_itertools import consecutive_groups
import numpy as np
import pandas as pd

from LPA import LPA
from algo import KLD_divergence_consecutive
from visualize import show_kldf

Subcorpus = Literal["full", "conspiracy", "mainstream"]


def create_metadata(threshold=20) -> pd.DataFrame:
    metadata = pd.read_csv("data/loco/metadata.csv", parse_dates=["date"])
    metadata = metadata[metadata["date"].dt.year >= 1990]
    metadata["dt"] = metadata["date"].astype("str").str[:7]
    grouped = metadata.groupby(["subcorpus", "dt"])["category"].count()
    grouped = grouped[grouped > threshold]
    return metadata, grouped


def create_freq(threshold=20) -> pd.DataFrame:
    metadata, filter_ = create_metadata(threshold)
    base_freq = []
    for i in range(0, 96000, 1000):
        base_freq.append(pd.read_csv(f"data/loco/np_freq/frequency_{i}.csv"))
    base_freq = pd.merge(
        pd.concat(base_freq),
        metadata[["category", "date", "subcorpus"]],
        on="category",
        how="inner",
    ).rename(columns={"count": "frequency_in_category"})
    base_freq["dt"] = base_freq["date"].astype("str").str[:7]

    base_freq = base_freq.set_index(["subcorpus", "dt"])
    base_freq = (
        base_freq.loc[filter_.index]
        .sort_values("date")
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    return base_freq


def freq_window(
    base_freq: pd.DataFrame,
    quantity: int | str | tuple,
    cumulative: bool,
    direction: Literal["from", "to", "range"] = "range",
    subcorpus: Subcorpus = "full",
) -> pd.DataFrame:
    if subcorpus != "full":
        base_freq = base_freq[base_freq["subcorpus"] == subcorpus].reset_index(
            drop=True
        )
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


def tw_freq(
    base_freq: pd.DataFrame,
    subcorpus: Subcorpus = "full",
    freq: Literal["MS", "D", "W"] = "MS",
    start_date: str = "1990-01-01",
    end_date: str = "2020-07-01",
    filter_: pd.Series | None = None,
) -> pd.DataFrame:
    l = []
    s = pd.to_datetime(start_date, format="%Y-%m-%d")
    e = pd.to_datetime(end_date, format="%Y-%m-%d")
    for dr in zip(pd.date_range(s, e, freq=freq), pd.date_range(s, e, freq=freq[:1])):
        conditional_dir = {"direction": "range"} if freq in ("W", "MS") else {}
        if freq == "W":
            dr = dr[0] - pd.Timedelta("6D"), dr[0]
        freq_df = freq_window(
            base_freq, dr, False, subcorpus=subcorpus, **conditional_dir
        )
        dvr = LPA.create_dvr(freq_df).assign(**{"date": dr[0]})
        if filter_:
            dvr = dvr[dvr["element"] == filter_]
        l.append(dvr)
    tw_freq = pd.concat(l).reset_index(drop=True)
    return tw_freq


def kld_single(tw_freq):
    kld_res = []
    date_col = tw_freq["date"].drop_duplicates().to_list()
    for i in range(len(date_col) - 1):
        Q = tw_freq[tw_freq["date"] == date_col[i]]
        P = tw_freq[tw_freq["date"] == date_col[i + 1]]
        merged = pd.merge(Q, P, on="element", how="outer")[
            ["global_weight_x", "global_weight_y"]
        ].fillna(1e-7)
        merged /= merged.sum()
        kld_res.append(np.sum(KLD_divergence_consecutive(merged.to_numpy())))
    kldf = pd.DataFrame({"date": date_col[1:], "KLD": kld_res})
    return kldf


def dBTC(base_freq, tw_freq, subcorpus):
    """
    δ-bounded timeline compression using Kullback-Leibler divergence under a delta threshold - in this case the median.
    """
    iter_ = 0
    kldf = kld_single(tw_freq)
    cutoff = kldf["KLD"].median()
    show_kldf(kldf, rule_value=cutoff).save(
        f"results/{subcorpus}/bar_charts/bar_iter_{iter_}.html"
    )
    low_xentropy = kldf[kldf["KLD"] < cutoff]
    groups = [
        (min(i), max(i))
        for i in [list(x) for x in consecutive_groups(low_xentropy.index)]
        if len(i) > 1
    ]
    print(f"should be around {sum(b-a for a, b in groups)} iterations")
    while len(groups) > 0:
        iter_ += 1
        date = low_xentropy.loc[groups[0][0], "date"]
        next_date = low_xentropy.loc[groups[0][0] + 1, "date"]
        sq = squeeze_freq(base_freq, date, next_date, subcorpus)
        split_freq = tw_freq[~tw_freq["date"].isin((date, next_date))]
        tw_freq = (
            pd.concat([split_freq, sq])
            .sort_values(["date", "global_weight"], ascending=[True, False])
            .reset_index(drop=True)
        )
        kldf = kld_single(tw_freq)
        kldf_barchart = show_kldf(kldf, rule_value=cutoff)
        kldf_barchart.save(f"results/{subcorpus}/bar_charts/bar_iter_{iter_}.html")
        low_xentropy = kldf[kldf["KLD"] < cutoff]
        groups = [
            (min(i), max(i))
            for i in [list(x) for x in consecutive_groups(low_xentropy.index)]
            if len(i) > 1
        ]
        print(f"finished iteration {iter_}")
        if len(groups) == 0:
            return tw_freq, kldf


def squeeze_freq(base_freq, min_date, max_date, subcorpus):
    base_freq = base_freq[base_freq["subcorpus"] == subcorpus]
    squeezed = (
        base_freq[(base_freq["date"] >= min_date) & (base_freq["date"] <= max_date)]
        .groupby("element", as_index=False)
        .agg({"frequency_in_category": "sum"})
    )
    squeezed["date"] = min_date
    squeezed["global_weight"] = squeezed["frequency_in_category"] / sum(
        squeezed["frequency_in_category"]
    )
    return squeezed.sort_values(by=["date", "frequency_in_category"], ascending=False)


def create_mdvr(vectors):
    average_weight = vectors.groupby("element")["local_weight"].sum() / len(
        vectors["category"].drop_duplicates()
    )
    normalized_average_weight = average_weight / average_weight.sum()
    dvr = (
        normalized_average_weight.rename("global_weight")
        .sort_values(ascending=False)
        .reset_index()
    )
    return dvr


def create_and_cut(vectors):
    vectors = vectors.rename(
        columns={"date": "category", "global_weight": "local_weight"}
    ).drop(columns=["frequency_in_category"])
    dvr = create_mdvr(vectors)
    lpa = LPA(dvr, epsilon_frac=2)
    vectors = vectors.sort_values(["category", "element"]).reset_index(drop=True)
    lpa.create_arrays(vectors)
    sigs, max_distances = lpa.create_distances(vectors)
    return dvr, sigs, max_distances


def write(*args: Tuple[pd.DataFrame, str]):
    for df, name in args:
        df.to_csv(PATH / f"{name}.csv", index=False)
        print(f"wrote {name}")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(PATH / f"{name}.csv", parse_dates=["date"])


if __name__ == "__main__":
    """
    The main process of this code reads
    """
    PATH = Path("results")
    base_freq = create_freq(threshold=20)
    write((base_freq, "base_freq"))
    # base_freq = read("base_freq")
    for subcorpus in ("mainstream", "conspiracy"):
        PATH = Path("results") / subcorpus
        tw_freq_df = tw_freq(base_freq, subcorpus, "MS", "2000-01-01", "2020-07-01")
        write((tw_freq_df, "tw_freq"))
        # tw_freq_df = read("tw_freq")
        squeezed_freq, kldf = dBTC(base_freq, tw_freq_df, subcorpus)
        write((squeezed_freq, "squeezed_freq"), (kldf, "final_kldf"))
        # squeezed_freq = read("squeezed_freq")
        dvr, sigs, max_distances = create_and_cut(squeezed_freq)
        write((dvr, "dvr"), (max_distances, "max_distances"))
        for sig in sigs:
            name = sig.name.strftime("%Y-%m-%d")
            write((sig.rename("KL").reset_index(), f"sigs/sigs_{name}"))
