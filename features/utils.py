import logging

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def compute_idf(X):
    """Smoothed inverse document frequency, as in sklearn's TfidfTransformer(smooth_idf=True)."""
    n_samples = X.shape[0]
    df = np.asarray((X != 0).sum(axis=0)).ravel()
    return np.log((1 + n_samples) / (1 + df)) + 1.0


def apply_tfidf(X, idf, group_sizes=None, constant_sq_norm=0.0, scale_groups=False):
    """
    TF-IDF with l2 row normalization, matching sklearn's TfidfTransformer(norm='l2'), on dense float64 output.

    With group_sizes (a grouped matrix), column j counts group_sizes[j] times in each row's norm,
    and constant_sq_norm adds back the dropped constant columns, which gives the same values as the
    ungrouped matrix at the representative columns. With scale_groups, column j is then multiplied by
    sqrt(group_sizes[j]), so dot products and euclidean distances between rows are the ungrouped ones
    (without the constant columns), e.g. for kNN. Works in row chunks, so the only large allocation is the output.
    """
    n_rows, n_cols = X.shape
    weights = None if group_sizes is None else np.asarray(group_sizes, dtype=np.float64)
    col_scale = np.sqrt(weights) if scale_groups and weights is not None else None
    chunk = max(1, (64 * 2**20) // max(1, n_cols * 8))
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    for start in range(0, n_rows, chunk):
        rows = X[start:start + chunk]
        rows = rows.toarray() if sparse.issparse(rows) else rows
        block = out[start:start + chunk]
        np.multiply(rows, idf, out=block)
        if weights is None:
            sq_norms = np.einsum("ij,ij->i", block, block)
        else:
            sq_norms = np.einsum("ij,ij,j->i", block, block, weights)
        norms = np.sqrt(sq_norms + constant_sq_norm)
        norms[norms == 0] = 1.0
        block /= norms[:, None]
        if col_scale is not None:
            block *= col_scale
    return out

def train_val_svd(X_train, X_val, k=150):
    """
    Fits TruncatedSVD on training data and transforms both training and validation data.
    Both projections are shifted by the projected mean of the training rows, as if the data had been centered (as in PCA).
    """
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    # Verify that k is less than min(n_samples, n_features)
    if k >= min(n_samples, n_features):
        logging.warning(f"k={k} is too large for data with shape {X_train.shape}. Setting k to min(n_samples-1, n_features-1).")
        k = min(n_samples - 1, n_features - 1)

    svd = TruncatedSVD(n_components=k, random_state=1993)
    X_train_proj = svd.fit_transform(X_train)
    X_val_proj = svd.transform(X_val)
    components = svd.components_

    # Determine correction factor for train matrix
    mu = np.asarray(X_train.mean(axis=0)).ravel()
    mean_proj = mu @ components.T
    
    X_train_proj -= mean_proj
    X_val_proj -= mean_proj

    return X_train_proj, X_val_proj
