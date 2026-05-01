import numpy as np
import pandas as pd


def get_avg_uniqueness(
    n: int,
    max_hold: int,
    t1_indices: np.ndarray | None = None,
) -> np.ndarray:
    i_arr = np.arange(n)
    if t1_indices is not None:
        t1_idx = np.clip(t1_indices, i_arr, n - 1)
    else:
        t1_idx = np.minimum(i_arr + max_hold, n - 1)

    diff = np.zeros(n + 1, dtype=np.float64)
    np.add.at(diff, i_arr,       1.0)
    np.add.at(diff, t1_idx + 1, -1.0)
    c = np.cumsum(diff[:n])

    inv_c     = 1.0 / np.maximum(c, 1.0)
    cum_inv_c = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(inv_c, out=cum_inv_c[1:])

    lengths = (t1_idx - i_arr + 1).astype(np.float64)
    return (cum_inv_c[t1_idx + 1] - cum_inv_c[i_arr]) / lengths


class PurgedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        t1_col: str = "close_time",
    ):
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct
        self.t1_col      = t1_col

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None):
        n = len(X)
        embargo_size = int(n * self.embargo_pct)
        indices = np.arange(n)

        open_ns = X["open_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        t1_ns = X[self.t1_col].to_numpy(dtype="datetime64[ns]").astype(np.int64)

        test_folds = np.array_split(indices, self.n_splits)

        for fold in test_folds:
            test_t_start = open_ns[fold[0]]
            test_t_end   = t1_ns[fold[-1]]

            train_mask = np.ones(n, dtype=bool)
            train_mask[fold] = False

            purge_mask = (open_ns <= test_t_end) & (t1_ns >= test_t_start)
            train_mask[purge_mask] = False

            emb_start = fold[-1] + 1
            emb_end   = min(emb_start + embargo_size, n)
            train_mask[emb_start:emb_end] = False

            yield indices[train_mask], fold

    @staticmethod
    def approximate_t1(df: pd.DataFrame, max_hold: int) -> pd.Series:
        ct  = df["close_time"].to_numpy(dtype="datetime64[ns]")
        idx = np.arange(len(df))
        t1  = ct[np.minimum(idx + max_hold, len(df) - 1)]
        return pd.Series(t1, index=df.index, name="t1")