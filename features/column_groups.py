"""
Group identical columns of a dense k-mer feature matrix.

Many k-mers have exactly the same values in every genome (e.g. k-mers tiling one gene), and
tree models only ever split on the first copy of such a group. Grouping keeps one column per
distinct column pattern and drops columns that are constant across all genomes.

- The representative of a group is its lowest original column index. Columns are ordered by
  total k-mer count, so this is the group's first k-mer in that order.
- Kept columns are ordered by their representative, so using only the first N original
  columns (max_features) is the same as taking a prefix of the grouped columns.
- In cross-validation, regroup_fold groups each fold again on its training rows only and builds the
  test fold like inference does (the median over each group's k-mers), so CV evaluates the deployed pipeline.
"""
import logging
import zipfile

import numpy as np
from numpy.lib import format as npy_format

# Multipliers for the two independent rolling column hashes
_P1 = np.uint64(1000003)
_P2 = np.uint64(0x9E3779B97F4A7C15)

# Rows are read in blocks of about this many bytes
BLOCK_BYTES = 256 * 2**20


def _rows_per_block(n_cols, itemsize):
    return max(1, BLOCK_BYTES // max(1, n_cols * itemsize))


def iter_dense_rows(X):
    """Yield blocks of rows from an in-memory 2-D array."""
    block = _rows_per_block(X.shape[1], X.dtype.itemsize)
    for start in range(0, X.shape[0], block):
        yield X[start:start + block]


def npz_array_header(path, key="matrix"):
    """Return (shape, fortran_order, dtype) of an array stored in an .npz, without reading its data."""
    with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
        return _read_header(f)


def _read_header(f):
    version = npy_format.read_magic(f)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(f)
    return npy_format.read_array_header_2_0(f)


def iter_npz_rows(path, key="matrix"):
    """Yield blocks of rows of a C-ordered 2-D array stored in an .npz (compressed or not),
    so that a large cached matrix can be processed without loading it into memory."""
    with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
        (n_rows, n_cols), fortran_order, dtype = _read_header(f)
        if fortran_order:
            raise ValueError(f"{path}: expected a C-ordered matrix")
        block = _rows_per_block(n_cols, dtype.itemsize)
        row_bytes = n_cols * dtype.itemsize
        for start in range(0, n_rows, block):
            n = min(block, n_rows - start)
            buf = f.read(n * row_bytes)
            if len(buf) != n * row_bytes:
                raise ValueError(f"{path}: matrix data ended early (file truncated?)")
            yield np.frombuffer(buf, dtype=dtype).reshape(n, n_cols)


def _hashable(values):
    """View of the values that converts exactly to uint64; floats are hashed by their bit pattern."""
    if np.issubdtype(values.dtype, np.floating):
        return values.view(np.dtype(f"u{values.dtype.itemsize}"))
    return values


def group_columns(row_blocks, n_rows, n_cols, dtype):
    """
    Find groups of identical columns and build the grouped matrix.

    Parameters:
    - row_blocks: callable returning a fresh iterator over 2-D blocks of rows (it is iterated twice)
    - n_rows, n_cols, dtype: shape and dtype of the full matrix

    Returns:
    - rep_cols: int64 array (G,), original column index of each kept column's representative, ascending
    - group_of: int32 array (n_cols,), kept column of each original column, or -1 if the column is constant
    - X_grouped: array (n_rows, G), the representative columns
    - const_values: array (n_cols,), the value of each constant column (0 for the other columns)
    """
    # Pass 1: per-column min/max (to find constant columns) and two independent hashes
    col_min = col_max = None
    h1 = np.zeros(n_cols, np.uint64)
    h2 = np.zeros(n_cols, np.uint64)
    v = np.empty(n_cols, np.uint64)     # reused buffers: per-row allocations of this size are slow
    tmp = np.empty(n_cols, np.uint64)
    row_idx = 0
    with np.errstate(over="ignore"):
        for block in row_blocks():
            if col_min is None:
                col_min, col_max = block.min(axis=0), block.max(axis=0)
            else:
                np.minimum(col_min, block.min(axis=0), out=col_min)
                np.maximum(col_max, block.max(axis=0), out=col_max)
            for row in _hashable(block):
                np.copyto(v, row, casting="unsafe")
                h1 *= _P1
                np.add(v, np.uint64(1), out=tmp)
                h1 += tmp
                np.add(v, np.uint64(row_idx), out=tmp)
                h2 ^= tmp
                h2 *= _P2
                row_idx += 1
    del v, tmp
    if row_idx != n_rows:
        raise ValueError(f"Expected {n_rows} rows, got {row_idx}")

    # Group non-constant columns with equal hashes; sort by (h1, h2, column index)
    nonconst = np.flatnonzero(col_min != col_max)
    const_values = np.where(col_min == col_max, col_min, 0).astype(dtype)
    del col_min, col_max
    group_of = np.full(n_cols, -1, np.int32)
    if len(nonconst) == 0:
        logging.warning("All columns are constant; the grouped matrix has no columns.")
        return np.zeros(0, np.int64), group_of, np.zeros((n_rows, 0), dtype), const_values

    order = nonconst[np.lexsort((nonconst, h2[nonconst], h1[nonconst]))]
    k1, k2 = h1[order], h2[order]
    del h1, h2
    new_group = np.ones(len(order), bool)
    new_group[1:] = (k1[1:] != k1[:-1]) | (k2[1:] != k2[:-1])
    del k1, k2
    group_idx = np.cumsum(new_group) - 1
    group_rep = order[new_group]                  # first of each run is its lowest column index
    rank = np.argsort(group_rep)
    rep_cols = group_rep[rank].astype(np.int64)
    kept_of_group = np.empty_like(rank)
    kept_of_group[rank] = np.arange(len(rank))
    group_of[order] = kept_of_group[group_idx]
    del order, new_group, group_idx, group_rep, rank, kept_of_group

    # Pass 2: verify every member equals its representative exactly, and copy the representatives
    members = np.flatnonzero(group_of >= 0)
    members = members[rep_cols[group_of[members]] != members]
    rep_of_members = rep_cols[group_of[members]]
    X_grouped = np.empty((n_rows, len(rep_cols)), dtype)
    start = 0
    for block in row_blocks():
        if not np.array_equal(block[:, members], block[:, rep_of_members]):
            raise RuntimeError("Column hash collision: two different columns hashed equal. Please report this.")
        X_grouped[start:start + len(block)] = block[:, rep_cols]
        start += len(block)

    n_const = n_cols - len(nonconst)
    logging.info(f"Grouped {n_cols:,} columns into {len(rep_cols):,} "
                 f"({n_const:,} constant columns dropped, {len(members):,} duplicates merged)")
    return rep_cols, group_of, X_grouped, const_values


def prefix_groups(rep_cols, group_of, n_features):
    """
    Number of kept columns and their group sizes when only the first n_features original columns are used.

    Exact: if any member of a group is among the first n_features columns, so is its representative.
    """
    n_kept = int(np.searchsorted(rep_cols, n_features))
    g = group_of[:n_features]
    group_sizes = np.bincount(g[g >= 0], minlength=n_kept)
    return n_kept, group_sizes


def constant_sq_norm(const_values, n_features):
    """Sum of squares of the dropped constant columns among the first n_features.

    A constant nonzero column has idf 1 and adds the same amount to every row's squared TF-IDF norm,
    so adding this back keeps grouped TF-IDF values equal to the ungrouped ones."""
    v = const_values[:n_features].astype(np.float64)
    return float(np.dot(v, v))


def members_of_groups(group_of, n_features, n_kept):
    """Original column indices of each kept column's members among the first n_features, representative first."""
    cols = np.flatnonzero(group_of[:n_features] >= 0)
    groups = group_of[cols]
    order = np.lexsort((cols, groups))
    cols, groups = cols[order], groups[order]
    bounds = np.searchsorted(groups, np.arange(n_kept + 1))
    return [cols[bounds[j]:bounds[j + 1]] for j in range(n_kept)]


def weighted_group_medians(X, group_of, weights, n_groups):
    """
    For each row, the median of each group's values, where column j stands for weights[j] identical
    k-mers (so it counts weights[j] times). Columns with group_of == -1 are ignored.

    This is the inference-time median over a group's k-mers, computed from a grouped matrix.
    Returns a float array (n_rows, n_groups).
    """
    out = np.empty((X.shape[0], n_groups), np.float64 if X.dtype == np.float64 else np.float32)
    cols = np.flatnonzero(group_of >= 0)
    g = group_of[cols]
    n_members = np.bincount(g, minlength=n_groups)

    # Groups made of a single column: the median is that column's value
    single = n_members[g] == 1
    out[:, g[single]] = X[:, cols[single]]
    if single.all():
        return out

    # Other groups: sort each row's values by (group, value) and find the median positions
    # using the cumulative weights. Every row has the same group boundaries.
    order = np.argsort(g[~single], kind="stable")
    multi_cols, multi_g = cols[~single][order], g[~single][order]
    w = np.asarray(weights, dtype=np.int64)[multi_cols]
    groups, start = np.unique(multi_g, return_index=True)
    w_group = np.add.reduceat(w, start)
    w_before = np.cumsum(w_group) - w_group
    w_total = int(w_group.sum())
    pos_lo = w_before + (w_group - 1) // 2
    pos_hi = w_before + w_group // 2

    chunk = max(1, (32 * 2**20) // (8 * len(multi_cols)))
    for s in range(0, X.shape[0], chunk):
        V = X[s:s + chunk][:, multi_cols]
        r = V.shape[0]
        idx = np.lexsort((V, np.broadcast_to(multi_g, V.shape)), axis=1)
        values = np.take_along_axis(V, idx, axis=1).ravel()
        # Offset each row so that one searchsorted call handles all rows
        offsets = (np.arange(r, dtype=np.int64) * (w_total + 1))[:, None]
        cum_w = (np.cumsum(w[idx], axis=1) + offsets).ravel()
        lo = np.searchsorted(cum_w, (pos_lo + offsets).ravel(), side="right")
        hi = np.searchsorted(cum_w, (pos_hi + offsets).ravel(), side="right")
        out[s:s + r, groups] = ((values[lo].astype(np.float64) + values[hi]) / 2).reshape(r, -1)
    return out


def regroup_fold(X_train, X_test, group_sizes, constant_sq_norm=0.0):
    """
    Group a grouped matrix's columns again using only a fold's training rows, as the final model's
    groups use only its training data, and build the test fold's columns as inference does: the
    median over each group's k-mers. Columns constant on the training rows are dropped.

    Parameters:
    - X_train, X_test: the fold's rows of the grouped matrix
    - group_sizes: number of k-mers in each column of the grouped matrix
    - constant_sq_norm: TF-IDF norm contribution of the constant columns already dropped

    Returns X_train_fold, X_test_fold, fold_group_sizes, fold_constant_sq_norm
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    _, group_of, X_train_fold, const_values = group_columns(
        lambda: iter_dense_rows(X_train), *X_train.shape, X_train.dtype)
    n_groups = X_train_fold.shape[1]
    kept = group_of >= 0
    fold_sizes = np.bincount(group_of[kept], weights=sizes[kept], minlength=n_groups).astype(np.int64)
    dropped_values = const_values[~kept].astype(np.float64)
    fold_constant_sq_norm = constant_sq_norm + float(np.dot(sizes[~kept] * dropped_values, dropped_values))
    X_test_fold = weighted_group_medians(X_test, group_of, sizes, n_groups)
    return X_train_fold, X_test_fold, fold_sizes, fold_constant_sq_norm
