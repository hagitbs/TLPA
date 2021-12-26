import simplejson as json
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import nltk
from collections import Counter
from google_cloud import GoogleCloud
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
    gcloud = GoogleCloud()
    # for i in range(0, 100_000, 5000):
    #     gcloud.download(
    #         f"LOCO_{i}.json", destination=DATA_FOLDER, bucket_name="loco_data"
    #     )
    #     with open(DATA_FOLDER/f"LOCO_{i}.json", "r") as f:
    #         data = json.load(f)
    #     d = {}
    #     for j, article in enumerate(data):
    #         d[article["doc_id"]] = dict(count_em(article["txt"]))
    #         if len(d) % 1000 == 0:
    #             df_it(d, i, j)
    #             d = {}
    l = []
    for i in range(1000, 97000, 1000):
        df = pd.read_csv(f"./data/freq/frequency_{i}.csv")
        l.append(df)
    fdf = pd.concat(l)
    #  LPA.create_dvr(fdf)
    dvr = pd.read_csv("data/dvr.csv")
    # dvr.to_csv("data/dvr.csv", index=False)
    # for i in range(1000, 97000, 1000):
    #     df = pd.read_csv(f"data/freq/frequency_{i}.csv")
    #     sig = LPA.create_signatures(df, epsilon_frac=2, sig_length=200, dvr=dvr)
    #     sig.to_csv(f"data/sigs/sigs200_{i}.csv", index=False)
    #     print(f"did {i}")
    sig = LPA.create_signatures(fdf, epsilon_frac=5, sig_length=200, dvr=dvr)
    sig.to_csv("sigs.csv", index=False)
    print("finished sigs")
    spd = LPA.SockPuppetDistance(sig, fdf)
    spd.to_csv("spd.csv", index=False)
    # print("finished spd")
