from typing import Optional
import pandas as pd
import numpy as np
from scipy.spatial import distance


def create_dvr(x):
    """Creates the DVR table of the domain"""
    dvr = (
        x.groupby("element", as_index=False)
        .sum()
        .sort_values(by="frequency_in_category", ascending=False)
    )
    dvr["global_weight"] = dvr["frequency_in_category"] / sum(
        dvr["frequency_in_category"]
    )
    return dvr


def create_vector(df):
    """Creates a vector for every category in the domain"""
    df["local_weight"] = df["frequency_in_category"] / df.groupby("category")[
        "frequency_in_category"
    ].transform("sum")
    return df


def calc_distance_summary(dvr_vec, sample_vec):
    """Auxillary func for distance summary"""
    return sum(((sample_vec - dvr_vec) * np.log10(sample_vec / dvr_vec)))


def distance_summary(dvr, vecs):
    # TODO: fix
    """Calculates the distance of every category from the domain"""
    dvr_vec = np.array(dvr.sort_values("element")["global_weight"])
    dist_sum = pd.DataFrame(index=vecs.index, columns=["distance_summary"], data=0)

    for i in range(len(vecs)):
        sample_vec = np.array(vecs.iloc[i, :])
        dist_sum.iloc[i, :] = calc_distance_summary(dvr_vec, sample_vec)
    f_max = round(max(dist_sum["distance_summary"]) + 0.005, 3)
    if f_max > 1:
        dist_sum["distance_summary"] = dist_sum["distance_summary"] / f_max
    return dist_sum


def KLDS(signatures):
    """Auxillary func for Sock Puppet Distance Calculation"""
    signatures["existing_element_flag"].replace(0, -1, inplace=True)
    signatures["KLR"] = signatures["KL"] * signatures["existing_element_flag"]
    signatures["KLR"] = signatures["KLR"] + signatures["existing_element_flag"]
    return signatures


def cross_categories(categories):
    """Auxillary func for Sock Puppet Distance Calculation"""
    index = pd.MultiIndex.from_product(
        [categories, categories], names=["user1", "user2"]
    )
    return pd.DataFrame(index=index).reset_index()


def SPD(klds, cc):
    """Calculates the L1 distance between every pair of categories in the domain (Sock Puppet Distance)"""
    spd = cc.copy()
    spd["distance_between_users"] = 0

    ## This code can be made more efficient by not calculating twice when user1 and user2 are interchanged, using the other masks
    for i in range(len(cc)):
        user1 = cc.iloc[i, 0]
        user2 = cc.iloc[i, 1]

        mask1 = klds["category"] == user1
        mask2 = klds["category"] == user2
        x = klds[mask1 | mask2]

        x = x.pivot(index="category", columns="element", values="KLR")
        csize = len(x.columns)

        # using just the data in klds and the rest is 0
        vec_a = np.array(x.fillna(0).iloc[0, :])
        vec_b = np.array(x.fillna(0).iloc[-1, :])

        spd_mask1 = spd["user1"] == user1
        spd_mask2 = spd["user1"] == user2
        spd_mask3 = spd["user2"] == user1
        spd_mask4 = spd["user2"] == user2

        spd.loc[spd[(spd_mask1 & spd_mask4)].index[0], "distance_between_users"] = (
            distance.cityblock(vec_a, vec_b) / csize
        )

    f_max = round(max(spd["distance_between_users"]) + 0.005, 3)
    if f_max > 1:
        spd["distance_between_users"] = spd["distance_between_users"] / f_max
    return spd


def SockPuppetDistance(signatures, df):
    """Wrapper func for L1 Distance Calculation"""
    return SPD(KLDS(signatures), cross_categories(df))


def create_signatures(df, epsilon_frac=2, dvr=None):
    """Prepares the raw data and creates signatures for every category in the domain.
    `epsilon_frac` defines the size of epsilon, default is 1/(corpus size * 2)
    `sig_length` defines the length of the signature, default is 500"""
    if not isinstance(dvr, pd.DataFrame):
        dvr = create_dvr(df)
    epsilon = 1 / (len(dvr) * epsilon_frac)
    vecs = create_vector(df).pivot_table(
        values="local_weight", index="category", columns="element"
    )
    mask = vecs.isna().stack(dropna=False).reset_index(drop=True)
    vecs = vecs.fillna(epsilon)
    dvr_array = (
        dvr[dvr["element"].isin(vecs.columns)]
        .sort_values("element")["global_weight"]
        .to_numpy()
    )
    vecs_array = vecs.to_numpy()
    subtracted = (
        pd.DataFrame(
            (vecs_array - dvr_array) * np.log10(vecs_array / dvr_array),
            index=vecs.index,
            columns=vecs.columns,
        )
        .stack(dropna=False)
        .reset_index()
        .rename(columns={0: "KL"})
    )
    subtracted["missing"] = mask
    subtracted = subtracted.sort_values(["category", "KL"], ascending=[True, False])
    return subtracted


def diminishing_return(
    sigs: pd.DataFrame, sig_length: int = 500, categories: int = 1000
):
    length = len(sigs) // categories
    return sigs[([True] * sig_length + [False] * (length - sig_length)) * categories]


def create_and_cut(
    df,
    dvr: Optional[pd.DataFrame] = None,
    epsilon_frac: int = 2,
    sig_length: int = 500,
    categories: int = 1000,
):
    sigs = create_signatures(df, epsilon_frac, dvr)
    return diminishing_return(sigs, sig_length, categories)


def distance_from_domain(df: pd.DataFrame, dvr: Optional[pd.DataFrame] = None):
    """Prepares the data and returns the distance of every category in the domain"""
    if not isinstance(dvr, pd.DataFrame):
        dvr = create_dvr(df)
    epsilon = 1 / (len(dvr) * 2)
    vector = create_vector(df)
    num_of_elements = vector.groupby("category").max()["lrnk"]
    betas = 1 - ((len(dvr) - num_of_elements) * epsilon)
    beta_vecs = (
        vector.pivot_table(
            values="local_weight", index="category", columns="element"
        ).fillna(0)
        * betas
    ).replace(0, epsilon, inplace=True)
    return distance_summary(dvr, beta_vecs)
