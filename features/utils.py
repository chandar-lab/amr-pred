import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import umap


# TODO: change the name of the function
def efficient_svd(X, k=150):
    """
    Efficient PCA-like approximation using TruncatedSVD on high-dimensional sparse TF-IDF data.
    Simulates centering to estimate explained variance ratio as in true PCA.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    # Verify that k is less than min(n_samples, n_features)
    if k >= min(n_samples, n_features):
        print(f"Warning: k={k} is too large for data with shape {X.shape}. Setting k to min(n_samples-1, n_features-1).")
        k = min(n_samples - 1, n_features - 1)

    # 1. Fit TruncatedSVD (does not center X)
    svd = TruncatedSVD(n_components=k, random_state=1993)
    X_proj = svd.fit_transform(X)  # shape: (n_samples, k)
    components = svd.components_  # shape: (k, n_features)

    # 2. Estimate mean of features
    mu = np.asarray(X.mean(axis=0)).ravel()

    # 3. Estimate projections of mean onto components
    mean_proj = mu @ components.T  # shape: (k,)
    X_proj -= mean_proj  # simulate centering in-place to save memory
    X_proj_centered = X_proj

    # 4. Compute explained variance (simulate covariance matrix on centered X)
    explained_variance = np.sum(X_proj_centered**2, axis=0) / (n_samples - 1)

    # 5. Total variance of centered data
    norm_squared = np.sum(X.power(2)) if sparse.issparse(X) else np.sum(X**2)
    mean_norm = np.dot(mu, mu)
    total_variance = (norm_squared - n_samples * mean_norm) / (n_samples - 1)

    # 6. Explained variance ratio
    explained_variance_ratio = explained_variance / total_variance

    return explained_variance, explained_variance_ratio, X_proj_centered

def train_val_svd(X_train, X_val, k=150):
    """
    Fits TruncatedSVD on training data and transforms both training and validation data.
    """
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    # Verify that k is less than min(n_samples, n_features)
    if k >= min(n_samples, n_features):
        print(f"Warning: k={k} is too large for data with shape {X_train.shape}. Setting k to min(n_samples-1, n_features-1).")
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

# import umap
# def plot_umap(X, y, metric='cosine', title=None, save_path=None):
#     """
#     Plots UMAP projection of the data.
#     """
#     # NOTE: do not use random_state because it hard-codes n_jobs=1 to be deterministic
#     reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric)
#     X_umap = reducer.fit_transform(X)
#     unique_labels = np.unique(y)

def plot_umap(X, y, metric='cosine', title=None, save_path=None):
    """
    Plots UMAP projection of the data.
    """
    # NOTE: do not use random_state because it hard-codes n_jobs=1 to be deterministic
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric)
    X_umap = reducer.fit_transform(X)
    unique_labels = np.unique(y)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=7)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(boundaries=np.arange(len(unique_labels)+1)-0.5).set_ticks(np.arange(len(unique_labels)))
    
#     if title:
#         plt.title(title)
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     #test
#     plt.show()