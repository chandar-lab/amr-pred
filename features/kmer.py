import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
from Bio import SeqIO
from multiprocessing import Pool
import psutil
from tqdm import tqdm
import logging
import gc
import pickle
import os
import re

try:
    NUM_PROC = len(psutil.Process().cpu_affinity())
except AttributeError:
    NUM_PROC = os.cpu_count()

from features.utils import efficient_svd
from features.column_groups import (
    group_columns, iter_dense_rows, iter_npz_rows, npz_array_header, prefix_groups, members_of_groups,
    constant_sq_norm
)

# Any run of characters other than A/C/G/T (N, or another IUPAC ambiguity code such as R, Y, K, M)
NON_ACGT = re.compile(r"[^ACGT]+")

class KmerFeatures:
    def __init__(self, k=31, mode="binary", path_precomputed=None, reduce_dimensionality=False,
                 min_kmer_freq=2, use_canonical=True, batch_size=500, max_features=None, use_sparse=False,
                 path_kmers=None, path_full_matrix=None, group_columns=False, path_kmer_groups=None):
        """
        Memory-efficient k-mer feature extraction with advanced feature selection.
        
        Parameters:
        - k: int, length of k-mers
        - mode: str, 'binary' or 'count' or 'entropy'
        - path_precomputed: str, path to precomputed k-mer matrix (if available)
        - reduce_dimensionality: bool, whether to apply dimensionality reduction
        - min_kmer_freq: int, minimum frequency to keep a k-mer (frequency filtering)
        - use_canonical: bool, use canonical k-mers (reduces features by ~50%)
        - batch_size: int, process sequences in batches to manage memory
        - max_features: int, maximum number of features to keep (most frequent)
        - use_sparse: bool, use sparse CSR matrix (True) or dense numpy arrays (False)
        - path_kmers: str, path to precomputed frequent k-mers dict (pkl file)
        - path_full_matrix: str, path to a large, precomputed matrix for sub-sampling
        - group_columns: bool, merge columns that are identical across all genomes and drop constant
          columns (training only; see features/column_groups.py). Dense matrices only.
        - path_kmer_groups: str, path to the k-mer groups saved with a grouped model (inference only).
          Each column's value is the median over its member k-mers, counting absent members as 0.
        """
        assert mode in ("binary", "count", "entropy", "svd"), "Mode must be one of: 'binary', 'count', 'entropy', 'svd'"

        logging.info(f"Initializing KmerFeatures with k={k}, mode={mode}, min_freq={min_kmer_freq}, use_sparse={use_sparse}")

        self.k = k
        self.mode = mode
        self.kmer_to_col = {}
        # self.path_precomputed = path_precomputed
        self.reduce_dim = reduce_dimensionality
        self.min_kmer_freq = min_kmer_freq
        self.use_canonical = use_canonical
        self.batch_size = batch_size
        self.max_features = max_features
        self.use_sparse = use_sparse
        self.path_kmers = path_kmers
        self.path_full_matrix = path_full_matrix
        self.group_columns = group_columns
        self.path_kmer_groups = path_kmer_groups
        self.group_sizes = None         # number of k-mers merged into each column (grouped runs only)
        self.constant_sq_norm = 0.0     # TF-IDF norm contribution of the dropped constant columns (grouped runs only)
        self.inference_groups = False   # True when columns are medians over k-mer groups

        if group_columns and use_sparse:
            raise ValueError("Column grouping supports dense matrices only; set use_sparse=False.")
        if group_columns and path_kmer_groups:
            raise ValueError("group_columns is for training and path_kmer_groups is for inference; use one.")
        # Grouped matrices are cached next to the ungrouped matrix they are derived from
        self.path_grouped_matrix = (path_full_matrix.replace(".npz", "__grouped.npz")
                                    if group_columns and path_full_matrix else None)

        # Check if the parent directory of the kmers exists, and create it if not
        if self.path_kmers:
            if not os.path.exists(os.path.dirname(self.path_kmers)):
                os.makedirs(os.path.dirname(self.path_kmers))
                logging.info(f"Created directory {os.path.dirname(self.path_kmers)} for storing kmers.")

        # For two-pass approach
        self.kmer_global_counts = Counter()
        self.selected_kmers = set()

        if path_kmer_groups:
            self._load_kmer_groups(path_kmer_groups)

    def _reverse_complement(self, seq):
        """Get reverse complement of DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(complement.get(base, base) for base in reversed(seq))

    def _canonical_kmer(self, kmer):
        """Return canonical k-mer (lexicographically smaller of kmer and its reverse complement)."""
        if not self.use_canonical:
            return kmer
        rev_comp = self._reverse_complement(kmer)
        return min(kmer, rev_comp)

    def _extract_kmers_from_file(self, fna_path):
        """Extract k-mers from FASTA file with memory-efficient streaming.

        Only k-mers made entirely of A/C/G/T are kept: any other character splits the
        sequence into segments, and k-mers are taken within each segment."""
        kmers = []
        for record in SeqIO.parse(fna_path, "fasta"):
            seq = str(record.seq).upper()
            for segment in NON_ACGT.split(seq):
                for i in range(len(segment) - self.k + 1):
                    kmers.append(self._canonical_kmer(segment[i:i + self.k]))
        return kmers

    def _count_global_kmers(self, dataset):
        """First pass: count all k-mers globally for initial filtering.
        
        Only called when a pre-computed matrix is not available."""

        # Check if precomputed frequent k-mers are available
        if self.path_kmers:
            logging.info(f"Loading precomputed frequent k-mers from {self.path_kmers}...")
            try:
                with open(self.path_kmers, 'rb') as f:
                    frequent_kmers = pickle.load(f)
                logging.info(f"Loaded {len(frequent_kmers)} frequent k-mers from file")
            except (FileNotFoundError, pickle.PickleError) as e:
                logging.warning(f"Failed to load precomputed kmers: {e}. Computing from scratch...")
                frequent_kmers = self._compute_kmers(dataset)
        else:
            # Compute frequent k-mers from scratch
            frequent_kmers = self._compute_kmers(dataset)
        
        # "frequent-kmers" contains kmer: total-count after min frequency filtering
        # Initial selection based on frequency (if max_features specified)
        if self.max_features and len(frequent_kmers) > self.max_features:
            # Sort so frequent k-mers are first
            sorted_kmers = sorted(frequent_kmers.items(), key=lambda x: x[1], reverse=True)
            # Select top N
            selected_kmers_with_counts = sorted_kmers[:self.max_features]
            # Drop the counts, keep only k-mers
            self.selected_kmers = {kmer for kmer, _ in selected_kmers_with_counts}
            logging.info(f"Pre-selected top {self.max_features} most frequent k-mers")
        else:
            # This will get sorted in the next step; order does not matter here
            self.selected_kmers = set(frequent_kmers.keys())
        
        # Create initial k-mer to column mapping
        # The sort ensures that column 0 is the most frequent k-mer, 1 is the second, and so on.
        sorted_frequent_kmers = sorted(self.selected_kmers, key=lambda kmer: frequent_kmers.get(kmer, 0), reverse=True)
        self.kmer_to_col = {kmer: idx for idx, kmer in enumerate(sorted_frequent_kmers)}
        
        logging.info(f"Initial feature set: {len(self.kmer_to_col)} k-mers")
        
        # Clear memory
        del frequent_kmers
        gc.collect()

    def _compute_kmers(self, dataset):
        """Compute frequent k-mers from dataset and optionally save to file.
        
        If saved, it will be to a pickle of k-mer: total-count dictionary. This is not in sorted order"""
        logging.info("Computing k-mer frequencies from dataset...")
        
        for i, entry in enumerate(tqdm(dataset, desc="Counting k-mers")):
            kmers = self._extract_kmers_from_file(entry['seq']) # List of all kmers in order of appearance
            self.kmer_global_counts.update(kmers) # Counter object; counts all kmers across the dataset
            
            if i % 100 == 0:
                gc.collect()    # Periodically clear memory
        
        logging.info(f"Found {len(self.kmer_global_counts)} unique k-mers")
        
        # Filter by minimum frequency
        frequent_kmers = {kmer: count for kmer, count in self.kmer_global_counts.items() if count >= self.min_kmer_freq}
        
        logging.info(f"After frequency filtering (min_freq={self.min_kmer_freq}): {len(frequent_kmers)} k-mers")
        
        # Save frequent k-mers to file if path is provided
        if self.path_kmers:
            self._save_frequent_kmers(frequent_kmers, self.path_kmers)
        
        # Clear global counts to free memory
        self.kmer_global_counts.clear()
        
        return frequent_kmers

    def _save_frequent_kmers(self, frequent_kmers, filepath):
        """Save frequent k-mers dictionary to pickle file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(frequent_kmers, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Saved frequent k-mers dictionary to {filepath}")
        except Exception as e:
            logging.warning(f"Failed to save frequent k-mers: {e}")

    def _process_sequence_batch_sparse(self, batch, start_row_idx):
        """
        Parallel sparse processing using the same _process_single_sequence
        worker function as the dense version.
        Returns: (data, rows, cols)
        """

        data, rows, cols = [], [], []

        # Prepare args for workers — note: row_idx is relative,
        # so we still add start_row_idx after
        worker_args = [
            (
                i,                      # relative row index
                entry,
                self.mode,
                self.selected_kmers,
                self.kmer_to_col,
                self._extract_kmers_from_file,
                self.group_sizes if self.inference_groups else None
            )
            for i, entry in enumerate(batch)
        ]

        # Process in parallel
        with Pool(processes=NUM_PROC) as pool:
            for rel_row_idx, col_dict in tqdm(
                pool.imap_unordered(self._process_single_sequence, worker_args),
                total=len(worker_args),
                desc=f"Processing sparse batch ({NUM_PROC} processes)",
            ):
                if not col_dict:
                    continue  # skip empty rows

                # Convert relative index → absolute index
                abs_row_idx = start_row_idx + rel_row_idx

                # Append all sparse entries
                for col_idx, val in col_dict.items():
                    rows.append(abs_row_idx)
                    cols.append(col_idx)
                    data.append(val)

        return data, rows, cols


    def _process_single_sequence(self, args):
        """
        Standalone worker function so it can be pickled by multiprocessing.
        Returns: (row_idx, {col_idx: value})
        """
        (row_idx, entry, mode, selected_kmers, kmer_to_col, extract_func, group_sizes) = args

        kmers = extract_func(entry['seq'])
        filtered = [k for k in kmers if k in selected_kmers]
        if not filtered:
            return row_idx, {}  # empty row

        # build statistic depending on mode
        if mode == "binary":
            kmer_dict = {k: 1 for k in set(filtered)}
        elif mode == "count":
            kmer_dict = Counter(filtered)
        elif mode == "entropy":
            kmer_dict = Counter(filtered)
            total = sum(kmer_dict.values())
            if total > 0:
                kmer_dict = {k: v / total for k, v in kmer_dict.items()}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # convert to column-indexed dictionary
        if group_sizes is not None:
            return row_idx, self._group_medians(kmer_dict, kmer_to_col, group_sizes)
        col_dict = {kmer_to_col[k]: val for k, val in kmer_dict.items()}
        return row_idx, col_dict

    @staticmethod
    def _group_medians(kmer_dict, kmer_to_col, group_sizes):
        """Median of each column's member k-mer values, counting members absent from the genome as 0.

        Returns {col_idx: median} for columns with a nonzero median."""
        n = len(kmer_dict)
        cols = np.fromiter((kmer_to_col[k] for k in kmer_dict), dtype=np.int64, count=n)
        vals = np.fromiter(kmer_dict.values(), dtype=np.float64, count=n)
        order = np.lexsort((vals, cols))
        cols, vals = cols[order], vals[order]
        present_cols, start, n_present = np.unique(cols, return_index=True, return_counts=True)
        n_absent = group_sizes[present_cols] - n_present

        def kth_value(pos):
            # pos-th smallest of the column's values, where the absent members are the smallest (zeros)
            idx = pos - n_absent
            out = np.zeros(len(present_cols))
            ok = idx >= 0
            out[ok] = vals[start[ok] + idx[ok]]
            return out

        m = group_sizes[present_cols]
        medians = (kth_value((m - 1) // 2) + kth_value(m // 2)) / 2
        keep = medians != 0
        return dict(zip(present_cols[keep].tolist(), medians[keep].tolist()))



    def _process_sequence_batch_dense(self, batch, start_row_idx, X):
        """Parallel version using multiprocessing. Fills the batch's rows of X in place.

        Returns X, which is a widened copy if a count did not fit in X's dtype."""
        value_dtype = np.float64 if self.mode == "entropy" or self.inference_groups else np.int64

        # Prepare args for workers
        worker_args = [
            (
                i,
                entry,
                self.mode,
                self.selected_kmers,
                self.kmer_to_col,
                self._extract_kmers_from_file,
                self.group_sizes if self.inference_groups else None
            )
            for i, entry in enumerate(batch)
        ]

        # Process using a pool + tqdm
        with Pool(processes=NUM_PROC) as pool:
            for row_idx, col_dict in tqdm(
                pool.imap_unordered(self._process_single_sequence, worker_args),
                total=len(worker_args),
                desc=f"Processing dense batch ({NUM_PROC} processes)",
            ):
                if not col_dict:
                    continue  # skip empty rows

                cols = np.fromiter(col_dict.keys(), dtype=np.int64, count=len(col_dict))
                vals = np.fromiter(col_dict.values(), dtype=value_dtype, count=len(col_dict))

                # Numpy wraps out-of-range integers silently, so check before writing the row
                new_dtype = self._fit_dtype(X.dtype, vals.max())
                if new_dtype != X.dtype:
                    logging.warning(f"Count {int(vals.max())} does not fit in {X.dtype}; widening the feature matrix "
                                    f"to {new_dtype} (copies {X.nbytes / 1024**3:.1f} GB)")
                    X = X.astype(new_dtype)

                X[start_row_idx + row_idx, cols] = vals

        return X

    def _get_dtype(self):
        """Get the initial dtype based on mode.

        Integer matrices start as uint8 and are widened with _fit_dtype when a count does not
        fit, so the final dtype is the smallest one that holds every value. Group medians can be
        fractional, so they are stored as floats."""
        if self.mode in ("binary", "count") and not self.inference_groups:
            return np.uint8
        else:
            return np.float32

    @staticmethod
    def _fit_dtype(dtype, max_val):
        """Return dtype if max_val fits in it, else the smallest unsigned integer dtype that holds max_val."""
        if np.issubdtype(dtype, np.integer) and max_val > np.iinfo(dtype).max:
            return np.min_scalar_type(int(max_val))
        return np.dtype(dtype)

    def build_feature_matrix(self, dataset): 
        """
        Build feature matrix with a priority system:
        1. Load and sub-sample from a full matrix if available.
        2. Load from a standard precomputed matrix.
        3. Build from scratch and save the full matrix if a path is given.

        With group_columns, the same steps run on the grouped matrix (see _build_grouped_matrix).
        """
        if self.group_columns:
            return self._build_grouped_matrix(dataset)
        
        # --- NEW: Priority 1 - Load and sub-sample from a full matrix ---
        if self.path_full_matrix and os.path.exists(self.path_full_matrix):
            logging.info(f"Loading full feature matrix from {self.path_full_matrix} to sub-sample...")
            X_full, y = self.load_matrix(self.path_full_matrix)  # This also loads the full kmer_to_col map

            n_features_full = X_full.shape[1]
            
            if self.max_features and self.max_features < n_features_full:
                logging.info(f"Sub-sampling to top {self.max_features} features from {n_features_full} total features.")
                
                # Columns are already ordered by k-mer frequency, so we can slice directly
                X_sub = X_full[:, :self.max_features].copy()
                del X_full
                gc.collect()
                
                # Update the k-mer map to reflect the sub-sampling
                col_to_kmer_full = {v: k for k, v in self.kmer_to_col.items()}
                self.kmer_to_col = {col_to_kmer_full[i]: i for i in range(self.max_features)}
                
                logging.info(f"Sub-sampled matrix shape: {X_sub.shape}")
                if self.reduce_dim:
                    logging.info("Applying dimensionality reduction using SVD...")
                    _, _, X_reduced = efficient_svd(X_sub)
                    return X_reduced, y
                return X_sub, y
            else:
                logging.info("Using the full pre-loaded matrix as max_features is not smaller.")
                if self.reduce_dim:
                    logging.info("Applying dimensionality reduction using SVD...")
                    _, _, X_reduced = efficient_svd(X_full)
                    return X_reduced, y
                return X_full, y

        # # --- EXISTING: Priority 2 - Load from standard precomputed file ---
        # if self.path_precomputed:
        #     logging.info(f"Loading precomputed k-mer matrix from {self.path_precomputed}...")
        #     X, y = self.load_matrix(self.path_precomputed)
        #     return X, y

        # --- Fallback: Build matrix from scratch ---
        X, y = self._build_from_scratch(dataset)

        if self.reduce_dim:
            logging.info("Applying dimensionality reduction using SVD...")
            _, _, X_reduced = efficient_svd(X)
            return X_reduced, y
            
        return X, y

    def _build_from_scratch(self, dataset):
        """Count k-mers, build the feature matrix and save it to path_full_matrix (if given)."""
        logging.info("No precomputed matrix found, building from scratch...")
        
        # Step 1: Count and filter k-mers (inference with k-mer groups already has its columns)
        if not self.inference_groups:
            self._count_global_kmers(dataset)
        
        if self._n_columns() == 0:
            raise ValueError("No k-mers passed the filtering criteria. Consider lowering min_kmer_freq.")

        # Step 2: Build feature matrix
        if self.use_sparse:
            X, y = self._build_sparse_matrix(dataset)
        else:
            X, y = self._build_dense_matrix(dataset)
        
        # Save the newly built matrix as a 'full' matrix for future runs
        if self.path_full_matrix:
            logging.info(f"Saving the newly built matrix to {self.path_full_matrix} for future use.")
            self.save_matrix(X, y, self.path_full_matrix)

        return X, y

    def _n_columns(self):
        """Number of feature columns (with k-mer groups, several k-mers map to one column)."""
        return len(self.group_sizes) if self.inference_groups else len(self.kmer_to_col)

    # ------------------------------------------------------------------
    # Column grouping (training)
    # ------------------------------------------------------------------
    def _build_grouped_matrix(self, dataset):
        """Grouped version of build_feature_matrix: one column per distinct column pattern.

        Uses the grouped cache if present, else groups the ungrouped cache (streamed, without
        loading it into memory), else builds from scratch and groups the result."""
        if self.path_grouped_matrix and os.path.exists(self.path_grouped_matrix):
            logging.info(f"Loading grouped feature matrix from {self.path_grouped_matrix}...")
            X, y = self._load_grouped_matrix(self.path_grouped_matrix)
        else:
            if self.path_full_matrix and os.path.exists(self.path_full_matrix):
                logging.info(f"Grouping the columns of {self.path_full_matrix} (one-time conversion)...")
                X, y = self._group_cached_matrix(self.path_full_matrix)
            else:
                X_full, y = self._build_from_scratch(dataset)
                kmers = self._kmers_in_column_order(self.kmer_to_col, X_full.shape[1])
                self.rep_cols, self.group_of, X, self.const_values = group_columns(
                    lambda: iter_dense_rows(X_full), *X_full.shape, X_full.dtype)
                del X_full
                gc.collect()
                self.kmers = kmers
            if self.path_grouped_matrix:
                self._save_grouped_matrix(X, y, self.path_grouped_matrix)

        X = self._take_grouped_prefix(X)
        if self.reduce_dim:
            logging.info("Applying dimensionality reduction using SVD...")
            _, _, X_reduced = efficient_svd(X)
            return X_reduced, y
        return X, y

    def _kmers_in_column_order(self, kmer_to_col, n_cols):
        """K-mers as a fixed-width bytes array, ordered by column."""
        kmers = np.empty(n_cols, dtype=f"S{self.k}")
        for kmer, col in kmer_to_col.items():
            kmers[col] = kmer
        return kmers

    def _group_cached_matrix(self, filepath):
        """Group the columns of a cached ungrouped .npz matrix, reading it in row blocks."""
        with np.load(filepath, allow_pickle=True) as data:
            matrix_type = str(data['matrix_type']) if 'matrix_type' in data else ('dense' if 'matrix' in data else 'sparse')
            if matrix_type != 'dense':
                raise ValueError(f"Column grouping supports dense matrices only, but {filepath} is {matrix_type}.")
            y = data['labels']
            kmer_to_col = data['kmer_to_col'].item()
        (n_rows, n_cols), _, dtype = npz_array_header(filepath)
        self.kmers = self._kmers_in_column_order(kmer_to_col, n_cols)
        del kmer_to_col
        gc.collect()
        self.rep_cols, self.group_of, X, self.const_values = group_columns(
            lambda: iter_npz_rows(filepath), n_rows, n_cols, dtype)
        return X, y

    def _take_grouped_prefix(self, X):
        """Keep the grouped columns of the top max_features k-mers, and set kmer_to_col and group_sizes."""
        n_total = len(self.group_of)
        self.n_prefix = min(self.max_features, n_total) if self.max_features else n_total
        n_kept, self.group_sizes = prefix_groups(self.rep_cols, self.group_of, self.n_prefix)
        self.constant_sq_norm = constant_sq_norm(self.const_values, self.n_prefix)
        if n_kept < X.shape[1]:
            X = X[:, :n_kept].copy()
        rep_kmers = self.kmers[self.rep_cols[:n_kept]].astype(str).tolist()
        self.kmer_to_col = {kmer: j for j, kmer in enumerate(rep_kmers)}
        logging.info(f"Using {n_kept:,} grouped columns for the top {self.n_prefix:,} k-mers. Matrix shape: {X.shape}")
        return X

    def get_column_groups(self):
        """Member k-mers of each grouped column (within the top max_features), representative first."""
        members = members_of_groups(self.group_of, self.n_prefix, len(self.group_sizes))
        return [self.kmers[cols].astype(str).tolist() for cols in members]

    def _save_grouped_matrix(self, X, y, filepath):
        logging.info(f"Saving grouped matrix with shape {X.shape} to {filepath}")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        np.savez_compressed(filepath, matrix_type='grouped', matrix=X, labels=y,
                            rep_cols=self.rep_cols, group_of=self.group_of, kmers=self.kmers,
                            const_values=self.const_values)

    def _load_grouped_matrix(self, filepath):
        with np.load(filepath) as data:
            X, y = data['matrix'], data['labels']
            self.rep_cols, self.group_of, self.kmers = data['rep_cols'], data['group_of'], data['kmers']
            self.const_values = data['const_values']
        logging.info(f"Loaded grouped matrix with shape {X.shape}")
        return X, y

    # ------------------------------------------------------------------
    # K-mer groups (inference)
    # ------------------------------------------------------------------
    def _load_kmer_groups(self, filepath):
        """Load the k-mer groups saved with a grouped model: a list of member k-mers per column."""
        logging.info(f"Loading k-mer groups from {filepath}...")
        with open(filepath, 'rb') as f:
            groups = pickle.load(f)
        self.kmer_to_col = {kmer: col for col, members in enumerate(groups) for kmer in members}
        self.selected_kmers = set(self.kmer_to_col)
        self.group_sizes = np.array([len(members) for members in groups], dtype=np.int64)
        self.representatives = [members[0] for members in groups]
        self.inference_groups = True
        logging.info(f"Loaded {len(groups):,} columns made of {len(self.kmer_to_col):,} k-mers")

    def _build_sparse_matrix(self, dataset):
        """Build sparse CSR matrix."""
        all_data, all_rows, all_cols = [], [], []
        labels = []
        
        logging.info("Building sparse feature matrix...")
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
            batch = dataset[i:i + self.batch_size]
            batch_data, batch_rows, batch_cols = self._process_sequence_batch_sparse(batch, i)
            
            all_data.extend(batch_data)
            all_rows.extend(batch_rows)
            all_cols.extend(batch_cols)
            
            labels.extend([entry['label'] for entry in batch])
            
            if i % (self.batch_size * 10) == 0:
                gc.collect()    # Periodically clear memory

        # Build sparse matrix
        n_rows = len(dataset)
        n_cols = self._n_columns()
        
        logging.info(f"Building sparse matrix: {n_rows} samples x {n_cols} features")
        
        dtype = self._fit_dtype(self._get_dtype(), max(all_data, default=0))
        logging.info(f"Using dtype {dtype} for the sparse matrix")
        X = csr_matrix((all_data, (all_rows, all_cols)), shape=(n_rows, n_cols), dtype=dtype)
        y = np.array(labels)
        
        # Clean up memory
        del all_data, all_rows, all_cols
        gc.collect()
        
        if hasattr(X, 'nnz'):
            logging.info(f"Final sparse matrix density: {X.nnz / (X.shape[0] * X.shape[1]):.6f}")
            logging.info(f"Memory usage: ~{X.data.nbytes / 1024**2:.1f} MB")

        if self.reduce_dim:
            logging.info("Applying dimensionality reduction using SVD...")
            from sklearn.decomposition import TruncatedSVD
            n_components = 100 #min(min(X.shape) - 1, 1000)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            # Fit to get the uncorrected verion, then apply correction for column means
            X_uncorrected = svd.fit_transform(X)
            V = svd.components_
            # correct using column means; project each component onto the mean vector, then subtract
            mu = np.asarray(X.mean(axis=0)).ravel()
            mean_proj = mu @ V.T  # shape (n_components,)
            X_corrected = X_uncorrected - mean_proj[None, :]
            logging.info(f"Reduced dimensions from {X.shape[1]} to {X_corrected.shape[1]}")
            return X_corrected, y

        return X, y

    def _build_dense_matrix(self, dataset):
        """Build dense numpy matrix."""
        n_rows = len(dataset)
        n_cols = self._n_columns() # kmer_to_col only includes as many columns as was specified
        
        logging.info(f"Building dense matrix: {n_rows} samples x {n_cols} features")
        
        X = np.zeros((n_rows, n_cols), dtype=self._get_dtype())
        labels = []
        
        logging.info("Building dense feature matrix...")
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
            batch = dataset[i:i + self.batch_size]
            X = self._process_sequence_batch_dense(batch, i, X)
            
            labels.extend([entry['label'] for entry in batch])
            
            if i % (self.batch_size * 10) == 0:
                gc.collect()    # Periodically clear memory

        y = np.array(labels)
        
        logging.info(f"Final dense matrix shape: {X.shape}, dtype: {X.dtype}")
        logging.info(f"Memory usage: ~{X.nbytes / 1024**2:.1f} MB")

        return X, y

    def get_feature_names(self):
        """Return list of k-mer features in column order."""
        if self.inference_groups:
            return list(self.representatives)
        if not self.kmer_to_col:
            return []
        return [kmer for kmer, _ in sorted(self.kmer_to_col.items(), key=lambda x: x[1])]

    def save_matrix(self, X, y, filepath):
        """Save matrix (sparse or dense) and labels to file."""
        from scipy.sparse import issparse

        # Check that the parent folder exists, and if not create it
        parent_folder = os.path.dirname(filepath)
        if parent_folder and not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
            logging.info(f"Created directory {parent_folder} for saving the matrix.")
        
        if issparse(X):  # Sparse matrix
            logging.info(f"Saving sparse matrix ({X.format}) with shape {X.shape}")
            np.savez_compressed(filepath, 
                                matrix_type='sparse',
                                matrix_format=X.format,
                                data=X.data, 
                                indices=X.indices, 
                                indptr=X.indptr, 
                                shape=X.shape,
                                dtype=str(X.dtype),
                                labels=y,
                                kmer_to_col=self.kmer_to_col,
                                use_sparse=self.use_sparse)
        else:  # Dense matrix
            logging.info(f"Saving dense matrix with shape {X.shape}")
            np.savez_compressed(filepath, 
                                matrix_type='dense',
                                matrix=X, 
                                labels=y,
                                kmer_to_col=self.kmer_to_col,
                                use_sparse=self.use_sparse)

        logging.info(f"Saved matrix and k-mer mapping to {filepath}")           

    def load_matrix(self, filepath):
        """Load matrix (sparse or dense), labels, and k-mer mapping from file."""
        data = np.load(filepath, allow_pickle=True)
        
        # Determine matrix type
        if 'matrix_type' in data:
            matrix_type = str(data['matrix_type'])
        else:
            # Backward compatibility: infer from presence of 'matrix' key
            matrix_type = 'dense' if 'matrix' in data else 'sparse'
        
        if matrix_type == 'dense':
            logging.info("Loading dense matrix")
            X = data['matrix']
        else:  # sparse
            logging.info(f"Loading sparse matrix")
            # Get format if available, default to 'csr'
            matrix_format = str(data.get('matrix_format', 'csr'))
            
            if matrix_format == 'csr':
                X = csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
            else:
                # For other sparse formats, reconstruct as CSR (most common)
                X = csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
                logging.info(f"Loaded as CSR format (original was {matrix_format})")
        
        y = data['labels']
        
        # Load k-mer mapping if available
        if 'kmer_to_col' in data:
            self.kmer_to_col = data['kmer_to_col'].item()
        
        # Update use_sparse setting if available
        if 'use_sparse' in data:
            saved_use_sparse = bool(data['use_sparse'])
            if saved_use_sparse != self.use_sparse:
                logging.warning(f"Matrix was saved with use_sparse={saved_use_sparse}, "
                                f"but current setting is use_sparse={self.use_sparse}")
        
        logging.info(f"Loaded matrix with shape {X.shape}")
        
        return X, y