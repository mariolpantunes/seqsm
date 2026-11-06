"""State and K-window bi-directional encoders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import polars as pl
from numpy.typing import NDArray


class StateEncoder:
    """Bijective encoder between aligned Polars feature columns and int32 atomic state IDs."""

    def __init__(self, feature_cols: Sequence[str] | None = None) -> None:
        self._feature_cols: tuple[str, ...] | None = tuple(feature_cols) if feature_cols is not None else None
        self._tuple_to_id: dict[tuple[Any, ...], int] = {}
        self._id_to_tuple: dict[int, tuple[Any, ...]] = {}
        self._vocabulary: pl.DataFrame | None = None

    @property
    def n_states(self) -> int:
        """Returns the number of unique atomic states."""
        if self._vocabulary is None:
            raise ValueError("StateEncoder has not been fitted.")
        return len(self._id_to_tuple)

    @property
    def feature_cols(self) -> tuple[str, ...]:
        """Returns the tuple of feature column names."""
        if self._feature_cols is None:
            raise ValueError("StateEncoder has not been fitted.")
        return self._feature_cols

    @property
    def vocabulary(self) -> pl.DataFrame:
        """Returns the mapping DataFrame of state_id to feature values."""
        if self._vocabulary is None:
            raise ValueError("StateEncoder has not been fitted.")
        return self._vocabulary

    def fit(self, df: pl.DataFrame | pl.LazyFrame) -> Self:
        """Learns the unique atomic state combinations from the discrete feature columns."""
        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        if self._feature_cols is None:
            self._feature_cols = tuple(frame.columns)

        missing = [c for c in self._feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"Feature columns {missing} not found in DataFrame.")

        unique_df = frame.select(list(self._feature_cols)).unique().sort(list(self._feature_cols))
        rows = unique_df.iter_rows()

        self._tuple_to_id.clear()
        self._id_to_tuple.clear()

        for idx, row in enumerate(rows):
            self._tuple_to_id[row] = idx
            self._id_to_tuple[idx] = row

        # Build vocabulary table
        id_series = pl.Series("state_id", np.arange(len(self._id_to_tuple), dtype=np.int32))
        self._vocabulary = unique_df.with_columns(id_series).select(["state_id", *self._feature_cols])
        return self

    def partial_fit(self, df: pl.DataFrame | pl.LazyFrame) -> Self:
        """Incrementally adds unseen state tuples to the vocabulary."""
        if self._vocabulary is None or self._feature_cols is None:
            return self.fit(df)

        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        missing = [c for c in self._feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"Feature columns {missing} not found in DataFrame.")

        selected = frame.select(list(self._feature_cols)).unique()
        cur_id = len(self._id_to_tuple)
        new_rows: list[tuple[Any, ...]] = []
        for row in selected.iter_rows():
            if row not in self._tuple_to_id:
                self._tuple_to_id[row] = cur_id
                self._id_to_tuple[cur_id] = row
                new_rows.append(row)
                cur_id += 1

        if new_rows:
            new_df = pl.DataFrame({col: [r[i] for r in new_rows] for i, col in enumerate(self._feature_cols)})
            start_idx = len(self._id_to_tuple) - len(new_rows)
            new_ids = pl.Series("state_id", np.arange(start_idx, len(self._id_to_tuple), dtype=np.int32))
            appended = new_df.with_columns(new_ids).select(["state_id", *self._feature_cols])
            self._vocabulary = pl.concat([self._vocabulary, appended])

        return self

    def transform(self, df: pl.DataFrame | pl.LazyFrame) -> NDArray[np.int32]:
        """Transforms DataFrame rows into int32 atomic state IDs. Unseen rows map to -1."""
        if self._vocabulary is None or self._feature_cols is None:
            raise ValueError("StateEncoder has not been fitted.")

        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        missing = [c for c in self._feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"Feature columns {missing} not found in DataFrame.")

        selected = frame.select(list(self._feature_cols))
        n_rows = len(selected)
        out = np.empty(n_rows, dtype=np.int32)

        for i, row in enumerate(selected.iter_rows()):
            out[i] = self._tuple_to_id.get(row, -1)

        return out

    def fit_transform(self, df: pl.DataFrame | pl.LazyFrame) -> NDArray[np.int32]:
        """Fits the encoder and transforms the DataFrame in one step."""
        return self.fit(df).transform(df)

    def inverse_transform(self, state_ids: Sequence[int] | NDArray[np.int32]) -> pl.DataFrame:
        """Maps an array/sequence of state IDs back to a Polars DataFrame."""
        if self._feature_cols is None:
            raise ValueError("StateEncoder has not been fitted.")

        arr = np.asarray(state_ids, dtype=np.int32)
        col_data: dict[str, list[Any]] = {col: [] for col in self._feature_cols}

        for sid in arr:
            if sid in self._id_to_tuple:
                row_tuple = self._id_to_tuple[int(sid)]
                for col_idx, col_name in enumerate(self._feature_cols):
                    col_data[col_name].append(row_tuple[col_idx])
            else:
                for col_name in self._feature_cols:
                    col_data[col_name].append(None)

        return pl.DataFrame(col_data)

    def state_to_dict(self, state_id: int) -> dict[str, Any]:
        """Returns the dictionary of feature name to value for a given state ID."""
        if self._feature_cols is None or state_id not in self._id_to_tuple:
            raise KeyError(f"State ID {state_id} not found in fitted vocabulary.")
        row_tuple = self._id_to_tuple[state_id]
        return dict(zip(self._feature_cols, row_tuple, strict=True))

    def dict_to_state(self, state_dict: dict[str, Any]) -> int:
        """Maps a feature dictionary back to its integer state ID, or -1 if unseen."""
        if self._feature_cols is None:
            raise ValueError("StateEncoder has not been fitted.")
        row_tuple = tuple(state_dict.get(c) for c in self._feature_cols)
        return self._tuple_to_id.get(row_tuple, -1)

    def state_diff(self, from_state: int, to_state: int) -> dict[str, tuple[Any, Any]]:
        """Returns the mapping of features that changed between from_state and to_state."""
        d_from = self.state_to_dict(from_state)
        d_to = self.state_to_dict(to_state)
        return {k: (d_from[k], d_to[k]) for k in d_from if d_from[k] != d_to[k]}


class KWindowEncoder:
    """Compact bijection between observed K-length state tuples and contiguous int32 indices."""

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"Markov history order k must be >= 1, got {k}.")
        self._k: int = k
        self._window_to_id: dict[tuple[int, ...], int] = {}
        self._id_to_window: dict[int, tuple[int, ...]] = {}

    @property
    def k(self) -> int:
        """Returns the Markov history order k."""
        return self._k

    @property
    def n_windows(self) -> int:
        """Returns the number of unique observed K-windows (N_K)."""
        return len(self._id_to_window)

    @property
    def k_tuples(self) -> NDArray[np.int32]:
        """Returns array of shape (N_K, K) containing all observed K-tuples."""
        n = len(self._id_to_window)
        out = np.empty((n, self._k), dtype=np.int32)
        for idx in range(n):
            out[idx, :] = self._id_to_window[idx]
        return out

    def fit_transform(
        self,
        states: NDArray[np.int32],
        episode_ids: NDArray[np.int32] | None = None,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
        """
        Discovers all valid K-history windows and returns (source_u, target_v, time_indices).

        Windows crossing episode breaks or containing unknown (-1) states are excluded.
        """
        states_arr = np.asarray(states, dtype=np.int32)
        n = len(states_arr)
        if episode_ids is not None:
            ep_arr = np.asarray(episode_ids, dtype=np.int32)
            if len(ep_arr) != n:
                raise ValueError("states and episode_ids must have the same length.")
        else:
            ep_arr = None

        valid_windows: list[tuple[int, ...]] = []
        source_tuples: list[tuple[int, ...]] = []
        target_states: list[int] = []
        time_idxs: list[int] = []

        # Find transitions w_t -> s_{t+1}
        for t in range(self._k - 1, n - 1):
            target = int(states_arr[t + 1])
            if target < 0:
                continue

            if ep_arr is not None and ep_arr[t + 1] != ep_arr[t]:
                continue

            # Check window validity
            start = t - self._k + 1
            win_states = tuple(int(states_arr[idx]) for idx in range(start, t + 1))
            if any(s < 0 for s in win_states):
                continue

            if ep_arr is not None:
                first_ep = ep_arr[start]
                if any(ep_arr[idx] != first_ep for idx in range(start + 1, t + 1)):
                    continue

            valid_windows.append(win_states)
            source_tuples.append(win_states)
            target_states.append(target)
            time_idxs.append(t)

        # Unique windows sorted for determinism
        unique_windows = sorted(set(valid_windows))
        self._window_to_id = {win: idx for idx, win in enumerate(unique_windows)}
        self._id_to_window = {idx: win for idx, win in enumerate(unique_windows)}

        m = len(source_tuples)
        sources_u = np.empty(m, dtype=np.int32)
        targets_v = np.empty(m, dtype=np.int32)
        time_arr = np.empty(m, dtype=np.int32)

        for i in range(m):
            sources_u[i] = self._window_to_id[source_tuples[i]]
            targets_v[i] = target_states[i]
            time_arr[i] = time_idxs[i]

        return sources_u, targets_v, time_arr

    def partial_fit_transform(
        self,
        states: NDArray[np.int32],
        episode_ids: NDArray[np.int32] | None = None,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
        """Discovers new K-history windows and transforms (source_u, target_v, time_indices)."""
        states_arr = np.asarray(states, dtype=np.int32)
        n = len(states_arr)
        if episode_ids is not None:
            ep_arr = np.asarray(episode_ids, dtype=np.int32)
            if len(ep_arr) != n:
                raise ValueError("states and episode_ids must have the same length.")
        else:
            ep_arr = None

        source_tuples: list[tuple[int, ...]] = []
        target_states: list[int] = []
        time_idxs: list[int] = []

        for t in range(self._k - 1, n - 1):
            target = int(states_arr[t + 1])
            if target < 0:
                continue

            if ep_arr is not None and ep_arr[t + 1] != ep_arr[t]:
                continue

            start = t - self._k + 1
            win_states = tuple(int(states_arr[idx]) for idx in range(start, t + 1))
            if any(s < 0 for s in win_states):
                continue

            if ep_arr is not None:
                first_ep = ep_arr[start]
                if any(ep_arr[idx] != first_ep for idx in range(start + 1, t + 1)):
                    continue

            if win_states not in self._window_to_id:
                new_id = len(self._id_to_window)
                self._window_to_id[win_states] = new_id
                self._id_to_window[new_id] = win_states

            source_tuples.append(win_states)
            target_states.append(target)
            time_idxs.append(t)

        m = len(source_tuples)
        sources_u = np.empty(m, dtype=np.int32)
        targets_v = np.empty(m, dtype=np.int32)
        time_arr = np.empty(m, dtype=np.int32)

        for i in range(m):
            sources_u[i] = self._window_to_id[source_tuples[i]]
            targets_v[i] = target_states[i]
            time_arr[i] = time_idxs[i]

        return sources_u, targets_v, time_arr

    def encode_window(self, window: Sequence[int] | NDArray[np.int32]) -> int:
        """Maps a K-length window to its integer index u, or -1 if unseen."""
        win_tuple = tuple(int(x) for x in window)
        if len(win_tuple) != self._k:
            raise ValueError(f"Window length must be {self._k}, got {len(win_tuple)}.")
        return self._window_to_id.get(win_tuple, -1)

    def decode_window(self, source_id: int) -> tuple[int, ...]:
        """Maps source window index u back to its K-tuple of atomic states."""
        if source_id not in self._id_to_window:
            raise KeyError(f"Window index {source_id} not found in encoder.")
        return self._id_to_window[source_id]


__all__ = ["KWindowEncoder", "StateEncoder"]
