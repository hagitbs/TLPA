from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import nltk
from collections import Counter
from pathlib import Path
import LPA

nltk.download("stopwords")
nltk.download("punkt")
stemmer = SnowballStemmer("english")
ignored_words = nltk.corpus.stopwords.words("english")
DATA_FOLDER = Path("./data")


def count_em(article):
    c = Counter()
    for sentence in article.split("\r\n\r\n"):
        word_list = nltk.word_tokenize(sentence)
        word_list = [
            word.lower()
            for word in word_list
            if word.isalpha() and word.lower() not in ignored_words
        ]
        stem_words = [stemmer.stem(word) for word in word_list]
        c += Counter(stem_words)
    return c


def df_it(d, i, j):
    df = (
        pd.DataFrame.from_dict(d, orient="index")
        .rename_axis(index="the")
        .reset_index()
        .melt(id_vars="the")
        .dropna(subset=["value"])
        .rename(
            columns={
                "variable": "element",
                "the": "category",
                "value": "frequency_in_category",
            }
        )
        .sort_values("category")
    )
    df.to_csv(f"./data/frequency_{i + j + 1}.csv", index=False)


if __name__ == "__main__":
    dvr = pd.read_csv("data/dvr.csv")
    for i in range(1000, 97000, 1000):
        df = pd.read_csv(f"data/freq/frequency_{i}.csv")
        sig = LPA.create_and_cut(df, dvr)
        sig.to_csv(f"data/sigs/sigs500_{i}.csv", index=False)
        print(f"did {i}")
    # spd = LPA.SockPuppetDistance(sig, fdf)
    # spd.to_csv("spd.csv", index=False)
    # print("finished spd")
