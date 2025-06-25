import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
from Bio import SeqIO
from tqdm import tqdm
import logging
import gc
import pickle


class KmerFeatures:
    def __init__(self, k=31, mode="binary", path_precomputed=None, reduce_dimensionality=False, 
                 min_kmer_freq=2, use_canonical=True, batch_size=500, max_features=None, use_sparse=False, 
                 path_kmers=None):
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
        """
        assert mode in ("binary", "count", "entropy", "svd"), "Mode must be one of: 'binary', 'count', 'entropy', 'svd'"

        logging.info(f"Initializing KmerFeatures with k={k}, mode={mode}, min_freq={min_kmer_freq}, use_sparse={use_sparse}")

        self.k = k
        self.mode = mode
        self.kmer_to_col = {}
        self.path_precomputed = path_precomputed
        self.reduce_dim = reduce_dimensionality
        self.min_kmer_freq = min_kmer_freq
        self.use_canonical = use_canonical
        self.batch_size = batch_size
        self.max_features = max_features
        self.use_sparse = use_sparse
        self.path_kmers = path_kmers

        # For two-pass approach
        self.kmer_global_counts = Counter()
        self.selected_kmers = set()
        self.feature_selector = None
        self.selected_feature_indices = None

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
        """Extract k-mers from FASTA file with memory-efficient streaming."""
        kmers = []
        for record in SeqIO.parse(fna_path, "fasta"):
            seq = str(record.seq).upper()
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k]
                if "N" not in kmer:
                    canonical_kmer = self._canonical_kmer(kmer)
                    kmers.append(canonical_kmer)
        return kmers

    def _count_global_kmers(self, dataset):
        """First pass: count all k-mers globally for initial filtering."""

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
        
        # Initial selection based on frequency (if max_features specified)
        if self.max_features and len(frequent_kmers) > self.max_features:
            sorted_kmers = sorted(frequent_kmers.items(), key=lambda x: x[1], reverse=True)
            selected_kmers_with_counts = sorted_kmers[:self.max_features]
            self.selected_kmers = {kmer for kmer, _ in selected_kmers_with_counts}
            logging.info(f"Pre-selected top {self.max_features} most frequent k-mers")
        else:
            self.selected_kmers = set(frequent_kmers.keys())
        
        # Create initial k-mer to column mapping
        self.kmer_to_col = {kmer: idx for idx, kmer in enumerate(sorted(self.selected_kmers))}
        
        logging.info(f"Initial feature set: {len(self.kmer_to_col)} k-mers")
        
        # Clear memory
        del frequent_kmers
        gc.collect()

    def _compute_kmers(self, dataset):
        """Compute frequent k-mers from dataset and optionally save to file."""
        logging.info("Computing k-mer frequencies from dataset...")
        
        for i, entry in enumerate(tqdm(dataset, desc="Counting k-mers")):
            kmers = self._extract_kmers_from_file(entry['seq'])
            self.kmer_global_counts.update(kmers)
            
            if i % 100 == 0:
                gc.collect()    # Periodically clear memory
        
        logging.info(f"Found {len(self.kmer_global_counts)} unique k-mers")
        
        # Filter by minimum frequency
        frequent_kmers = {kmer: count for kmer, count in self.kmer_global_counts.items() 
                         if count > self.min_kmer_freq}
        
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

    def _process_sequence_batch(self, batch, start_row_idx):
        """Process a batch of sequences efficiently."""
        if self.use_sparse:
            return self._process_sequence_batch_sparse(batch, start_row_idx)
        else:
            return self._process_sequence_batch_dense(batch, start_row_idx)

    def _process_sequence_batch_sparse(self, batch, start_row_idx):
        """Process a batch of sequences for sparse matrix representation."""
        data, rows, cols = [], [], []
        
        for batch_idx, entry in enumerate(batch):
            row_idx = start_row_idx + batch_idx
            kmers = self._extract_kmers_from_file(entry['seq'])
            
            # Filter k-mers to only include selected ones
            filtered_kmers = [kmer for kmer in kmers if kmer in self.selected_kmers]
            
            if not filtered_kmers:
                continue
                
            if self.mode == "binary":
                kmer_dict = {k: 1 for k in set(filtered_kmers)}
            elif self.mode == "count":
                kmer_dict = Counter(filtered_kmers)
            elif self.mode == "entropy":
                kmer_dict = Counter(filtered_kmers)
                total_count = sum(kmer_dict.values())
                if total_count > 0:
                    kmer_dict = {k: v / total_count for k, v in kmer_dict.items()}
            
            for kmer, val in kmer_dict.items():
                col_idx = self.kmer_to_col[kmer]
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(val)
        
        return data, rows, cols

    def _process_sequence_batch_dense(self, batch, start_row_idx):
        """Process a batch of sequences for dense matrix representation."""
        n_cols = len(self.kmer_to_col)
        batch_matrix = np.zeros((len(batch), n_cols), dtype=self._get_dtype())
        
        for batch_idx, entry in enumerate(batch):
            kmers = self._extract_kmers_from_file(entry['seq'])
            
            # Filter k-mers to only include selected ones
            filtered_kmers = [kmer for kmer in kmers if kmer in self.selected_kmers]
            
            if not filtered_kmers:
                continue
                
            if self.mode == "binary":
                kmer_dict = {k: 1 for k in set(filtered_kmers)}
            elif self.mode == "count":
                kmer_dict = Counter(filtered_kmers)
            elif self.mode == "entropy":
                kmer_dict = Counter(filtered_kmers)
                total_count = sum(kmer_dict.values())
                if total_count > 0:
                    kmer_dict = {k: v / total_count for k, v in kmer_dict.items()}
            
            for kmer, val in kmer_dict.items():
                col_idx = self.kmer_to_col[kmer]
                batch_matrix[batch_idx, col_idx] = val
        
        return batch_matrix

    def _get_dtype(self):
        """Get appropriate dtype based on mode."""
        if self.mode == "binary":
            return np.uint8
        elif self.mode == "count":
            return np.uint16
        else:
            return np.float32

    def build_feature_matrix(self, dataset):
        """Build feature matrix with advanced feature selection."""
        
        if self.path_precomputed:
            logging.info(f"Loading precomputed k-mer matrix from {self.path_precomputed}...")
            X, y = self.load_matrix(self.path_precomputed)
            return X, y

        # Step 1: Count and filter k-mers
        self._count_global_kmers(dataset)
        
        if len(self.kmer_to_col) == 0:
            raise ValueError("No k-mers passed the filtering criteria. Consider lowering min_kmer_freq.")

        # Step 2: Build feature matrix
        labels = []
        
        if self.use_sparse:
            return self._build_sparse_matrix(dataset)
        else:
            return self._build_dense_matrix(dataset)

    def _build_sparse_matrix(self, dataset):
        """Build sparse CSR matrix."""
        all_data, all_rows, all_cols = [], [], []
        labels = []
        
        logging.info("Building sparse feature matrix...")
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
            batch = dataset[i:i + self.batch_size]
            batch_data, batch_rows, batch_cols = self._process_sequence_batch(batch, i)
            
            all_data.extend(batch_data)
            all_rows.extend(batch_rows)
            all_cols.extend(batch_cols)
            
            labels.extend([entry['label'] for entry in batch])
            
            if i % (self.batch_size * 10) == 0:
                gc.collect()    # Periodically clear memory

        # Build sparse matrix
        n_rows = len(dataset)
        n_cols = len(self.kmer_to_col)
        
        logging.info(f"Building sparse matrix: {n_rows} samples x {n_cols} features")
        
        X = csr_matrix((all_data, (all_rows, all_cols)), shape=(n_rows, n_cols), dtype=self._get_dtype())
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
            n_components = min(min(X.shape) - 1, 1000)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = svd.fit_transform(X)
            logging.info(f"Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
            return X_reduced, y

        return X, y

    def _build_dense_matrix(self, dataset):
        """Build dense numpy matrix."""
        n_rows = len(dataset)
        n_cols = len(self.kmer_to_col)
        
        logging.info(f"Building dense matrix: {n_rows} samples x {n_cols} features")
        
        X = np.zeros((n_rows, n_cols), dtype=self._get_dtype())
        labels = []
        
        logging.info("Building dense feature matrix...")
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
            batch = dataset[i:i + self.batch_size]
            batch_matrix = self._process_sequence_batch(batch, i)
            
            end_idx = min(i + self.batch_size, n_rows)
            X[i:end_idx] = batch_matrix
            
            labels.extend([entry['label'] for entry in batch])
            
            if i % (self.batch_size * 10) == 0:
                gc.collect()    # Periodically clear memory

        y = np.array(labels)
        
        logging.info(f"Final dense matrix shape: {X.shape}")
        logging.info(f"Memory usage: ~{X.nbytes / 1024**2:.1f} MB")

        if self.reduce_dim:
            logging.info("Applying dimensionality reduction using SVD...")
            from sklearn.decomposition import TruncatedSVD
            n_components = min(min(X.shape) - 1, 1000)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = svd.fit_transform(X)
            logging.info(f"Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
            return X_reduced, y

        return X, y

    def get_feature_names(self):
        """Return list of k-mer features in column order."""
        if not self.kmer_to_col:
            return []
        return [kmer for kmer, _ in sorted(self.kmer_to_col.items(), key=lambda x: x[1])]

    def save_matrix(self, X, y, filepath):
        """Save matrix (sparse or dense) and labels to file."""
        from scipy.sparse import issparse
        
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
