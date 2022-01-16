from itertools import combinations_with_replacement as cwr, product
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import numpy.ma as ma
from helpers import timing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class LPA:
    def __init__(self, dvr, epsilon_frac=2, categories=1000):
        self.dvr = dvr
        self.epsilon = 1 / (len(dvr) * epsilon_frac)
        self.categories = categories

    def create_pvr(self, df):
        """Creates a vector for every category in the domain"""
        df["local_weight"] = df["frequency_in_category"] / df.groupby("category")[
            "frequency_in_category"
        ].transform("sum")
        return df

    def heuristic(self, signatures):
        """Auxillary func for Sock Puppet Distance Calculation"""
        signatures["missing"].replace(0, -1, inplace=True)
        signatures["KLR"] = signatures["KL"] * signatures["existing_element_flag"]
        signatures["KLR"] = signatures["KLR"] + signatures["existing_element_flag"]
        return signatures

    def cross_categories(self, categories):
        """Auxillary func for Sock Puppet Distance Calculation"""
        index = pd.MultiIndex.from_product(
            [categories, categories], names=["user1", "user2"]
        )
        return pd.DataFrame(index=index).reset_index()

    @staticmethod
    def KLD(P, Q):
        # P represents the data, the observations, or a measured probability distribution.
        # Q represents instead a theory, a model, a description or an approximation of P.
        return (P - Q) * (ma.log(P) - ma.log(Q))

    def extended_freq_vec(self, vec, vec_lengths, missing):
        # TODO: rename
        betas = [
            item
            for sublist in [
                times * [(1 - missing * self.epsilon)[i]]
                for i, times in enumerate(vec_lengths)
            ]
            for item in sublist
        ]
        vec["local_weight"] = vec["local_weight"] * pd.Series(betas)
        return vec

    def betas(self, pvr):
        pvr_lengths = (
            pvr["category"].drop_duplicates(keep="last").index
            - pvr["category"].drop_duplicates(keep="first").index
            + 1
        ).to_numpy()
        missing = len(self.dvr) - pvr_lengths
        return self.extended_freq_vec(pvr, pvr_lengths, missing)

    def get_missing(self, length) -> pd.DataFrame:
        missing = self.dvr.loc[: length - 1, "element"].copy().to_frame()
        missing["KL"] = self.KLD(
            self.dvr.loc[: length - 1, "global_weight"].copy().to_numpy(), self.epsilon
        )
        missing["missing"] = True
        return missing

    def create_signatures(self, df):
        """Prepares the raw data and creates signatures for every category in the domain.
        `epsilon_frac` defines the size of epsilon, default is 1/(corpus size * 2)
        `sig_length` defines the length of the signature, default is 500"""
        vecs = self.betas(self.create_pvr(df)).pivot_table(
            values="local_weight", index="element", columns="category"
        )
        dvr_array = (
            self.dvr[self.dvr["element"].isin(vecs.index)]
            .sort_values("element")["global_weight"]
            .to_numpy()
        )
        vecs_array = vecs.fillna(0).to_numpy().T
        distances = (
            pd.DataFrame(
                self.KLD(dvr_array, vecs_array),
                index=vecs.columns,
                columns=vecs.index,
            )
            .stack()
            .reset_index()
            .rename(columns={0: "KL"})
        )
        return distances

    def diminishing_return(self, sigs: pd.DataFrame, sig_length: int = 500):
        sigs["missing"] = False
        missing = self.get_missing(sig_length)
        categories = sigs["category"].drop_duplicates().reset_index(drop=True)
        merged = pd.merge(categories, missing, how="cross")
        sigs = (
            sigs.append(merged)
            .sort_values(["category", "KL"], ascending=[True, False])
            .groupby("category")
            .head(sig_length)
            .reset_index(drop=True)
        )
        return sigs

    def create_and_cut(self, df, sig_length=500):
        sigs = self.create_signatures(df)
        return self.diminishing_return(sigs, sig_length)

    def distance_summary(self, df):
        sigs = self.create_signatures(df)
        return sigs.groupby("category").sum()

    @timing
    def sockpuppet_distance(
        self, signatures1: pd.DataFrame, signatures2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns size*size df
        """
        categories1 = signatures1["category"].drop_duplicates()
        categories2 = signatures2["category"].drop_duplicates()
        pivot = signatures1.append(signatures2).pivot_table(
            values="KL", index="category", columns="element", fill_value=0
        )
        XA = pivot.filter(categories1, axis="index")  # .to_numpy()
        XB = pivot.filter(categories2, axis="index")  # .to_numpy()
        df = pd.DataFrame(
            cdist(XA, XB, metric="cityblock"), index=categories1, columns=categories2
        )
        return df


class IterLPA(LPA):
    def __init__(self, blocks, size, dvr):
        super().__init__(dvr=dvr)
        self.blocks = blocks
        self.size = size
        self._range = range(0, self.size * self.blocks, self.size)

    @staticmethod
    def create_dvr(df):
        """Creates the DVR table of the domain"""
        dvr = (
            df.groupby("element", as_index=False)
            .sum()
            .sort_values(by="frequency_in_category", ascending=False)
        )
        dvr["global_weight"] = dvr["frequency_in_category"] / sum(
            dvr["frequency_in_category"]
        )
        return dvr

    def _shorthand(self, stage):
        shorthand = {"frequency": "freq", "signatures": "sigs"}
        return shorthand[stage]

    def _grid(self, symmetric=True):
        if symmetric:
            return list(cwr(self._range, r=2))
        else:
            return list(product(self._range, r=2))

    def iter_dvr(self):
        l = []
        for i in self._range:
            l.append(pd.read_csv(f"data/freq/frequency_{i}.csv"))
        self.create_dvr(pd.concat(l)).to_csv("dvr1.csv", index=False)

    def run_sockpuppets(self):
        for i, j in self._grid():
            sigs1 = self.create_and_cut(pd.read_csv(f"data/freq/frequency_{i}.csv"))
            sigs2 = self.create_and_cut(pd.read_csv(f"data/freq/frequency_{j}.csv"))
            self.sockpuppet_distance(sigs1, sigs2).to_csv(
                f"data/sockpuppets/sp_{i}_{j}.csv"
            )
            print(f"finished spd for blocks {i}, {j}")
            break

    def PCA(self):
        df = pd.read_csv(f"data/sockpuppets/sp_{i}_{j}.csv").set_index("category")
        df = StandardScaler().fit_transform(df)
        pca = PCA(n_components=2)
        pcdf = pca.fit_transform(df)
