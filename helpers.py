from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        print(f"func: {f.__name__} took: {te-ts}")
        return result

    return wrap


def write(path: Path, *args: Tuple[pd.DataFrame, str]):
    for df, name in args:
        if isinstance(df, np.array):
            with open(path / f"{name}.npy", "wb") as f:
                np.save(f, df)
        else:
            df.to_csv(path / f"{name}.csv", index=False)
        print(f"wrote {name}")


def read(path: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path / f"{name}.csv", parse_dates=["date"])
