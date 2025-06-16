from Bio import SeqIO
from collections import Counter, defaultdict
import numpy as np
import logging
from scipy.sparse import csr_matrix
from tqdm import tqdm
from preprocessing.utils import efficient_svd, plot_umap


class KmerFeatures:
    def __init__(self, k=31, mode="binary", path_precomputed=None, reduce_dimensionality=False):
        """
        Parameters:
        - k: int, length of k-mers
        - mode: str, 'binary' or 'count' or 'entropy'
        - path_precomputed: str, path to precomputed k-mer matrix (if available)
        - reduce_dimensionality: bool, whether to apply dimensionality reduction (e.g., SVD)
        """
        assert mode in ("binary", "count", "entropy", "svd"), "Mode must be one of: 'binary', 'count', 'entropy', 'svd'"
        logging.info(f"Initializing KmerFeatures with k={k}, mode={mode}, precomputed={path_precomputed is not None}")
        
        self.k = k
        self.mode = mode
        self.kmer_to_col = {}  # Maps k-mer to column index
        self.path_precomputed = path_precomputed
        self.reduce_dim = reduce_dimensionality

    def _extract_kmers(self, fna_path):
        kmers = []
        for record in SeqIO.parse(fna_path, "fasta"):
            seq = str(record.seq).upper()
            # print(record.id, len(s eq))
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k]
                if "N" not in kmer:
                    kmers.append(kmer)
        return kmers
    

    def build_feature_matrix(self, dataset):
        """Builds the feature matrix using scipy's CSR matrix with dynamic k-mer indexing."""
        data, rows, cols = [], [], []
        labels = []
        next_col = 0

        if self.path_precomputed:
            logging.info(f"Loading precomputed k-mer matrix from {self.path_precomputed}...")
            X = np.load(self.path_precomputed, allow_pickle=True)

            # get the labels from the precomputed matrix
            for row_idx, entry in enumerate(tqdm(dataset, desc="Extracting k-mers")):
                labels.append(entry['label'])

            return csr_matrix((X['data'], X['indices'], X['indptr']), shape=X['shape']), np.array(labels)


        for row_idx, entry in enumerate(tqdm(dataset, desc="Extracting k-mers")):
            kmers = self._extract_kmers(entry['seq'])
            
            if self.mode == "binary":
                kmer_dict = {k: 1 for k in set(kmers)}
            
            elif self.mode == "count":
                kmer_dict = Counter(kmers)
            
            elif self.mode == "entropy":
                kmer_dict = Counter(kmers)
                total_count = sum(kmer_dict.values())
                kmer_dict = {k: v / total_count for k, v in kmer_dict.items()}

            for kmer, val in kmer_dict.items():
                if kmer not in self.kmer_to_col:
                    self.kmer_to_col[kmer] = next_col
                    next_col += 1
                col_idx = self.kmer_to_col[kmer]
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(val)

            labels.append(entry['label'])

        n_rows = len(dataset)
        n_cols = next_col
        
        logging.info(f"Building feature matrix with {n_rows} samples (rows) and {n_cols} features (cols)...")
        X = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.uint16)
        y = np.array(labels)

        if self.reduce_dim:
            logging.info("Applying dimensionality reduction using SVD...")
            explained_variance, explained_variance_ratio, X_proj_centered = efficient_svd(X, k=self.k)
            # logging.info(f"Explained variance ratio: {explained_variance_ratio}")

            return X_proj_centered, y

        else:
            logging.info("No dimensionality reduction applied.")
            return X, y


    def analyze_kmer_stats(self, dataset, top_n=10):
        """Analyze global and per-sequence k-mer stats."""
        print("Analyzing k-mer statistics...")

        global_counter = Counter()
        shared_kmer_tracker = defaultdict(int)
        unique_counts = []

        for entry in tqdm(dataset, desc="Collecting k-mer stats"):
            kmers = []
            for record in SeqIO.parse(entry['seq'], "fasta"):
                seq = str(record.seq).upper()
                for i in range(len(seq) - self.k + 1):
                    kmer = seq[i:i + self.k]
                    if "N" not in kmer:
                        kmers.append(kmer)

            unique_in_seq = set(kmers)
            unique_counts.append(len(unique_in_seq))
            global_counter.update(kmers)
            for kmer in unique_in_seq:
                shared_kmer_tracker[kmer] += 1

        total_unique = len(global_counter)
        shared_across_seqs = sum(1 for v in shared_kmer_tracker.values() if v > 1)
        avg_unique_per_seq = np.mean(unique_counts)
        most_common = global_counter.most_common(top_n)

        print("\n K-mer Statistics:")
        print(f"• Total unique k-mers: {total_unique:,}")
        print(f"• K-mers shared across ≥2 sequences: {shared_across_seqs:,}")
        print(f"• Avg unique k-mers per sequence: {avg_unique_per_seq:,.2f}")
        print(f"• Top {top_n} most common k-mers:")
        for kmer, count in most_common:
            print(f"    {kmer} : {count:,}")