import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import umap


# TODO: change the name of the function
def efficient_svd(X_sparse, k):
    """
    Efficient PCA-like approximation using TruncatedSVD on high-dimensional sparse TF-IDF data.
    Simulates centering to estimate explained variance ratio as in true PCA.
    """
    assert sparse.issparse(X_sparse), "Input must be a sparse matrix."

    n_samples = X_sparse.shape[0]

    # 1. Fit TruncatedSVD (does not center X)
    svd = TruncatedSVD(n_components=k, random_state=0)
    X_proj = svd.fit_transform(X_sparse)  # shape: (n_samples, k)
    components = svd.components_  # shape: (k, n_features)

    # 2. Estimate mean of features
    mu = np.asarray(X_sparse.mean(axis=0)).ravel()

    # 3. Estimate projections of mean onto components
    mean_proj = mu @ components.T  # shape: (k,)
    X_proj_centered = X_proj - mean_proj  # simulate centering

    # 4. Compute explained variance (simulate covariance matrix on centered X)
    explained_variance = np.sum(X_proj_centered**2, axis=0) / (n_samples - 1)

    # 5. Total variance of centered data
    norm_squared = X_sparse.multiply(X_sparse).sum()
    mean_norm = np.dot(mu, mu)
    total_variance = (norm_squared - n_samples * mean_norm) / (n_samples - 1)

    # 6. Explained variance ratio
    explained_variance_ratio = explained_variance / total_variance

    return explained_variance, explained_variance_ratio, X_proj_centered

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
    
#     plt.show()