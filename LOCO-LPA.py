import os
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

from LPA import LPA, IterLPA

if "WSL_DISTRO_NAME" not in os.environ.keys():
    nltk.download("stopwords")
    nltk.download("punkt")
    stemmer = SnowballStemmer("english")
    ignored_words = nltk.corpus.stopwords.words("english")
DATA_FOLDER = Path("./data")


if __name__ == "__main__":
    dvr = pd.read_csv("data/dvr.csv")
    ilpa = IterLPA(blocks=97, size=1000, dvr=dvr)
    ilpa.run_sockpuppets()
