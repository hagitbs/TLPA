import os
from collections import Counter
from pathlib import Path

import ijson
import nltk
import pandas as pd
import simplejson as json
from nltk.stem.snowball import SnowballStemmer

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
    df.to_csv(f"./data/frequency_{(i + j + 1)-1000}.csv", index=False)


def break_up_json():
    with open("data/LOCO.json") as f:
        data = ijson.items(f, "item")
        items = [v for v in data]
        for i in range(0, len(items), 5000):
            with open(f"data/LOCO_{i}.json", "w") as fp:
                json.dump(items[i : i + 5000], fp)


def create_frequencies():
    # FIXME: misses last 743
    for i in range(95000, 100_000, 5000):
        with open(f"data/LOCO_{i}.json", "r") as f:
            data = json.load(f)
        d = {}
        for j, article in enumerate(data):
            d[article["doc_id"]] = dict(count_em(article["txt"]))
            if len(d) % 1000 == 0:
                df_it(d, i, j)
                d = {}
        break


create_frequencies()
