from itertools import combinations_with_replacement as cwr
from itertools import product

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helpers import timing
from algo import KLD_distance


class LPA:
    def __init__(self, dvr: pd.DataFrame, epsilon_frac: int = 2):
        self.dvr = dvr.sort_values("global_weight", ascending=False)
        self.epsilon = 1 / (len(dvr) * epsilon_frac)

    @staticmethod
    def create_dvr(frequency: pd.DataFrame) -> pd.DataFrame:
        """Creates the DVR table of the domain"""
        dvr = frequency.groupby("element", as_index=False).sum()
        dvr["global_weight"] = dvr["frequency_in_category"] / sum(
            dvr["frequency_in_category"]
        )
        dvr = (
            dvr.drop(columns=["frequency_in_category"])
            .sort_values("global_weight", ascending=False)
            .reset_index(drop=True)
        )
        return dvr

    def create_pvr(self, frequency: pd.DataFrame) -> pd.DataFrame:
        """Creates a vector for every category in the domain"""
        frequency["local_weight"] = frequency[
            "frequency_in_category"
        ] / frequency.groupby("category")["frequency_in_category"].transform("sum")
        return frequency

    def normalize_pvr(
        self, pvr: pd.DataFrame, pvr_lengths: np.array, missing: np.array
    ) -> pd.DataFrame:
        """
        The extended pvr (with ɛ) is no longer a probability vector - the sum of all
        coordinates is now larger than 1. We correct this by multiplying all non-ɛ
        frequencies by a normalization coefficient β. This normalization coefficient is
        given by the formula β=1-N*ɛ, where N is the number of words missing in one vector compared to the other (variable named `missing`).
        """
        betas = [
            item
            for sublist in [
                times * [(1 - missing * self.epsilon)[i]]
                for i, times in enumerate(pvr_lengths)
            ]
            for item in sublist
        ]
        pvr["normalized_weight"] = pvr["local_weight"] * pd.Series(betas)
        return pvr

    def betas(self, pvr: pd.DataFrame) -> pd.DataFrame:
        pvr_lengths = (
            pvr["category"].drop_duplicates(keep="last").index
            - pvr["category"].drop_duplicates(keep="first").index
            + 1
        ).to_numpy()
        missing = len(self.dvr) - pvr_lengths
        return self.normalize_pvr(pvr, pvr_lengths, missing)

    def create_arrays(self, pvr: pd.DataFrame) -> pd.DataFrame:
        """Prepares the raw data and creates signatures for every category in the domain.
        `epsilon_frac` defines the size of epsilon, default is 1/(corpus size * 2)
        `sig_length` defines the length of the signature, default is 500"""
        vecs = self.betas(pvr)
        vecs = vecs.pivot_table(
            values="normalized_weight", index="element", columns="category"
        )
        self.dvr_array = (
            self.dvr[self.dvr["element"].isin(vecs.index)]
            .sort_values("element")["global_weight"]
            .to_numpy()
        )
        self.vecs_array = (
            vecs.fillna(self.epsilon).replace(0, self.epsilon).to_numpy().T
        )

    def create_distances(self, frequency: pd.DataFrame) -> pd.DataFrame:
        try:
            dvr_array, vecs_array = getattr(self, "dvr_array"), getattr(
                self, "vecs_array"
            )
        except AttributeError:
            frequency = frequency.sort_values("category").reset_index(drop=True)
            pvr = self.create_pvr(frequency)
            self.create_arrays(pvr)
        categories = frequency["category"].drop_duplicates()
        elements = frequency["element"].drop_duplicates().dropna().sort_values()
        distances = (
            pd.DataFrame(
                KLD_distance(dvr_array, vecs_array), index=categories, columns=elements
            )
            .stack()
            .reset_index()
            .rename(columns={0: "KL"})
        )
        return distances

    def add_overused(self, distances: pd.DataFrame) -> pd.DataFrame:
        overused = np.less(self.dvr_array, self.vecs_array)
        overused = overused.reshape(np.multiply(*overused.shape))
        distances["overused"] = overused
        return distances

    def cut(self, sigs: pd.DataFrame, sig_length: int = 500) -> pd.DataFrame:
        # TODO: diminishing return
        return (
            sigs.sort_values(["category", "KL"], ascending=[True, False])
            .groupby("category")
            .head(sig_length)
            .reset_index(drop=True)
        )

    def create_and_cut(
        self, frequency: pd.DataFrame, sig_length: int = 500
    ) -> pd.DataFrame:
        distances = self.create_distances(frequency)
        sigs = self.add_overused(distances)
        cut = self.cut(sigs, sig_length)
        return cut

    def distance_summary(self, frequency: pd.DataFrame) -> pd.DataFrame:
        sigs = self.create_distances(frequency)
        return sigs.groupby("category").sum()

    @timing
    def sockpuppet_distance(
        self, signatures1: pd.DataFrame, signatures2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns size*size df
        """
        # TODO: triu
        categories1 = signatures1["category"].drop_duplicates()
        categories2 = signatures2["category"].drop_duplicates()
        pivot = pd.concat([signatures1, signatures2]).pivot_table(
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
    def create_dvr(freq):
        """Creates the DVR table of the domain"""
        dvr = freq.groupby("element", as_index=False).sum()
        dvr["global_weight"] = dvr["frequency_in_category"] / sum(
            dvr["frequency_in_category"]
        )
        return dvr.sort_values(by="global_weight", ascending=False)

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

    def create_and_cut(
        self, frequency: pd.DataFrame, sig_length: int = 500
    ) -> pd.DataFrame:
        frequency = frequency.sort_values("category").reset_index(drop=True)
        pvr = self.create_pvr(frequency)
        self.create_arrays(pvr)
        distances = self.create_distances(frequency)
        sigs = self.add_overused(distances)
        cut = self.cut(sigs, sig_length)
        return cut

    def run_sockpuppets(self):
        for i, j in self._grid():
            sigs1 = self.create_and_cut(pd.read_csv(f"data/freq/frequency_{i}.csv"))
            sigs2 = self.create_and_cut(pd.read_csv(f"data/freq/frequency_{j}.csv"))
            self.sockpuppet_distance(sigs1, sigs2).to_csv(
                f"data/sockpuppets/sp_{i}_{j}.csv"
            )
            print(f"finished spd for blocks {i}, {j}")

    def PCA(self):
        df = pd.read_csv(f"data/sockpuppets/sp_{i}_{j}.csv").set_index("category")
        df = StandardScaler().fit_transform(df)
        pca = PCA(n_components=2)
        pcdf = pca.fit_transform(df)
