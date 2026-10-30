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
import shutil
import tempfile

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

# The KmerFeatures object used by a pool worker. It is set once when the worker starts (inherited, not
# pickled, under fork), so the large k-mer -> column map is not sent to the workers with every genome.
_worker_features = None


def _init_worker(kmer_features):
    global _worker_features
    _worker_features = kmer_features


def _process_entry(args):
    """Pool task: one genome's row as (row_idx, cols, vals)."""
    row_idx, fna_path = args
    return _worker_features._process_single_sequence(row_idx, fna_path)


# ----------------------------------------------------------------------
# K-mers as integers (counting pass)
# ----------------------------------------------------------------------
# A k-mer is stored in 2 bits per base (A=0, C=1, G=2, T=3), 32 bases per uint64 word, its first bases in
# the first word. For k <= 32 the keys are a uint64 array; for longer k they are a structured array with
# one uint64 field per word. Both sort in the same order as the k-mer strings.
_BASES_PER_WORD = 32
_MAX_WORDS = 4
MAX_K = _MAX_WORDS * _BASES_PER_WORD
_INVALID = 4
_BASE_CODE = np.full(256, _INVALID, dtype=np.uint8)
for _code, _base in enumerate(b"ACGT"):
    _BASE_CODE[_base] = _code
    _BASE_CODE[ord(chr(_base).lower())] = _code
_CODE_BASE = np.frombuffer(b"ACGT", dtype=np.uint8)
# Genomes merged into a shard's running counts at a time
_MERGE_EVERY = 32


def _word_spans(k):
    """(first base, number of bases) of each word of a k-mer."""
    return [(off, min(_BASES_PER_WORD, k - off)) for off in range(0, k, _BASES_PER_WORD)]


def _key_dtype(k):
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}, got {k}.")
    n_words = len(_word_spans(k))
    if n_words == 1:
        return np.dtype(np.uint64)
    return np.dtype([(f"w{i}", np.uint64) for i in range(n_words)])


def _key_columns(keys):
    """The uint64 words of the keys, most significant first."""
    if keys.dtype.names is None:
        return [keys]
    return [keys[name] for name in keys.dtype.names]


def _pack_words(base_at, k, n):
    """Words of n k-mers, where base_at(j) is the array of the n k-mers' j-th base codes."""
    words = []
    for off, n_bases in _word_spans(k):
        word = np.zeros(n, dtype=np.uint64)
        for j in range(off, off + n_bases):
            word <<= np.uint64(2)
            word |= base_at(j)
        words.append(word)
    return words


def _keys_from_words(words, k):
    if len(words) == 1:
        return words[0]
    keys = np.empty(len(words[0]), dtype=_key_dtype(k))
    for name, word in zip(keys.dtype.names, words):
        keys[name] = word
    return keys


def _encode_sequence(seq, k, canonical):
    """Keys of the k-mers of one sequence (bytes) that are made entirely of A/C/G/T, as in
    KmerFeatures._extract_kmers_from_file."""
    b = _BASE_CODE[np.frombuffer(seq, dtype=np.uint8)]
    n = len(b) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=_key_dtype(k))
    invalid = b == _INVALID
    n_invalid = np.concatenate(([0], np.cumsum(invalid)))
    valid = n_invalid[k:] - n_invalid[:-k] == 0
    b[invalid] = 0
    words = _pack_words(lambda j: b[j:j + n], k, n)
    if canonical:
        # Base j of the reverse complement of the k-mer at i is the complement of base k-1-j
        comp = 3 - b
        rc_words = _pack_words(lambda j: comp[k - 1 - j:k - 1 - j + n], k, n)
        # The reverse complement is smaller where it is smaller in the first word that differs
        rc_smaller = np.zeros(n, dtype=bool)
        decided = np.zeros(n, dtype=bool)
        for fw, rw in zip(words, rc_words):
            rc_smaller |= ~decided & (rw < fw)
            decided |= rw != fw
        words = [np.where(rc_smaller, rw, fw) for fw, rw in zip(words, rc_words)]
    return _keys_from_words([w[valid] for w in words], k)


def _encode_genome(fna_path, k, canonical):
    """Keys of all k-mers of a FASTA file, in order of appearance."""
    keys = [_encode_sequence(bytes(record.seq), k, canonical) for record in SeqIO.parse(fna_path, "fasta")]
    return np.concatenate(keys) if keys else np.zeros(0, dtype=_key_dtype(k))


def _encode_strings(kmers, k):
    """Keys of k-mer strings (already canonical), e.g. the keys of a legacy {kmer: count} pickle."""
    mat = _BASE_CODE[np.array(kmers, dtype=f"S{k}").view(np.uint8).reshape(-1, k)]
    if (mat == _INVALID).any():
        raise ValueError("K-mers must be made of A/C/G/T only.")
    return _keys_from_words(_pack_words(lambda j: mat[:, j], k, len(mat)), k)


def _decode_keys(keys, k, chunk_size=1_000_000):
    """K-mer strings of the keys."""
    kmers = []
    for start in range(0, len(keys), chunk_size):
        words = _key_columns(keys[start:start + chunk_size])
        letters = np.empty((len(words[0]), k), dtype=np.uint8)
        for word, (off, n_bases) in zip(words, _word_spans(k)):
            for j in range(n_bases):
                shift = np.uint64(2 * (n_bases - 1 - j))
                letters[:, off + j] = _CODE_BASE[((word >> shift) & np.uint64(3)).astype(np.intp)]
        # Slicing one string is much faster than converting a fixed-width array to Python strings
        text = letters.tobytes().decode("ascii")
        kmers.extend(text[i:i + k] for i in range(0, len(text), k))
    return kmers


def _sort_order(keys, counts=None):
    """Order sorting keys ascending; with counts, by count descending first."""
    cols = _key_columns(keys)
    if counts is None and len(cols) == 1:
        return np.argsort(keys, kind="stable")
    sort_keys = cols[::-1]   # np.lexsort sorts by its last key first
    if counts is not None:
        sort_keys.append(-counts.astype(np.int64))
    return np.lexsort(sort_keys)


def _sum_by_key(keys, counts=None):
    """Unique keys (sorted) and the summed counts of each (counts of 1 if counts is None)."""
    if len(keys) == 0:
        return keys, np.zeros(0, dtype=np.int64)
    order = _sort_order(keys)
    keys = keys[order]
    is_new = np.zeros(len(keys), dtype=bool)
    is_new[0] = True
    for col in _key_columns(keys):
        is_new[1:] |= col[1:] != col[:-1]
    starts = np.flatnonzero(is_new)
    if counts is None:
        summed = np.diff(np.append(starts, len(keys))).astype(np.int64)
    else:
        summed = np.add.reduceat(counts[order].astype(np.int64), starts)
    return keys[starts], summed


def _partition_of(keys, n_parts):
    """Partition (0 .. n_parts-1) of each key, from a hash of its words."""
    h = np.zeros(len(keys), dtype=np.uint64)
    for col in _key_columns(keys):
        h ^= col
    h *= np.uint64(0x9E3779B97F4A7C15)
    return ((h >> np.uint64(32)) % np.uint64(n_parts)).astype(np.int64)


def _part_paths(tmpdir, shard, part):
    stem = os.path.join(tmpdir, f"shard{shard}_part{part}")
    return f"{stem}_keys.npy", f"{stem}_counts.npy"


def _count_shard(args):
    """Pool task: count the k-mers of a shard of genomes, and save the counts split into partitions.

    Returns the number of distinct k-mers in the shard."""
    shard, fna_paths, k, canonical, n_parts, tmpdir = args
    keys, counts = np.zeros(0, dtype=_key_dtype(k)), np.zeros(0, dtype=np.int64)
    pending = []
    for i, fna_path in enumerate(fna_paths):
        pending.append(_sum_by_key(_encode_genome(fna_path, k, canonical)))
        if len(pending) == _MERGE_EVERY or i == len(fna_paths) - 1:
            keys, counts = _sum_by_key(np.concatenate([keys] + [kk for kk, _ in pending]),
                                       np.concatenate([counts] + [cc for _, cc in pending]))
            pending = []
    part = _partition_of(keys, n_parts)
    order = np.argsort(part, kind="stable")
    bounds = np.searchsorted(part[order], np.arange(n_parts + 1))
    for p in range(n_parts):
        idx = order[bounds[p]:bounds[p + 1]]
        path_keys, path_counts = _part_paths(tmpdir, shard, p)
        np.save(path_keys, keys[idx])
        np.save(path_counts, counts[idx])
    return len(keys)


def _reduce_partition(args):
    """Pool task: sum one partition's counts over all shards, keeping k-mers seen at least min_freq times.

    Returns (keys, counts, number of distinct k-mers before filtering)."""
    part, n_shards, tmpdir, min_freq = args
    keys, counts = [], []
    for shard in range(n_shards):
        path_keys, path_counts = _part_paths(tmpdir, shard, part)
        keys.append(np.load(path_keys))
        counts.append(np.load(path_counts))
        os.remove(path_keys)
        os.remove(path_counts)
    keys, counts = _sum_by_key(np.concatenate(keys), np.concatenate(counts))
    n_distinct = len(keys)
    keep = counts >= min_freq
    keys, counts = keys[keep], counts[keep]
    if counts.max(initial=0) <= np.iinfo(np.uint32).max:
        counts = counts.astype(np.uint32)
    return keys, counts, n_distinct

class KmerFeatures:
    def __init__(self, k=31, mode="binary", path_precomputed=None, reduce_dimensionality=False,
                 min_kmer_freq=2, use_canonical=True, batch_size=500, max_features=None, use_sparse=False,
                 path_kmers=None, path_full_matrix=None, cache_features=None, group_columns=False,
                 path_kmer_groups=None):
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
        - path_kmers: str, path to the frequent k-mer counts: an .npz cache of the counting pass (written
          if missing), or a {kmer: count} pickle (a model's k-mers at inference).

        The caches (path_kmers .npz, path_full_matrix and its grouped version) record the assembly IDs of
        the genomes they were built from, in row order, and are only reused for exactly the same genomes.
        - path_full_matrix: str, path to a large, precomputed matrix for sub-sampling
        - cache_features: int, number of top k-mers to build and cache in path_full_matrix, so that runs with
          different max_features can share it (default: max_features). Must be at least max_features.
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
        if cache_features is not None and (max_features is None or cache_features < max_features):
            raise ValueError(f"cache_features ({cache_features}) must be at least max_features ({max_features}).")
        # Number of top k-mers to build (and cache); None keeps all frequent k-mers
        self.cache_features = (cache_features or max_features) if path_full_matrix else max_features
        self.group_columns = group_columns
        self.path_kmer_groups = path_kmer_groups
        self.group_sizes = None         # number of k-mers merged into each column (grouped runs only)
        self.constant_sq_norm = 0.0     # TF-IDF norm contribution of the dropped constant columns (grouped runs only)
        self.inference_groups = False   # True when columns are medians over k-mer groups
        self.assemblies = None          # assembly IDs of the dataset's genomes, in row order (set per build)

        if group_columns and use_sparse:
            raise ValueError("Column grouping supports dense matrices only; set use_sparse=False.")
        if group_columns and path_kmer_groups:
            raise ValueError("group_columns is for training and path_kmer_groups is for inference; use one.")
        # Grouped matrices are cached next to the ungrouped matrix they are derived from: the metadata in an
        # .npz, and the matrix in an uncompressed .npy that is memory-mapped, so only the columns used are read
        self.path_grouped_matrix = (path_full_matrix.replace(".npz", "__grouped.npz")
                                    if group_columns and path_full_matrix else None)

        # Check if the parent directory of the kmers exists, and create it if not
        if self.path_kmers:
            if not os.path.exists(os.path.dirname(self.path_kmers)):
                os.makedirs(os.path.dirname(self.path_kmers))
                logging.info(f"Created directory {os.path.dirname(self.path_kmers)} for storing kmers.")

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
        """First pass: count all k-mers globally for initial filtering, and map the top k-mers to columns.

        Only called when a pre-computed matrix is not available."""
        keys, counts = self._load_kmer_counts()
        if keys is None:
            keys, counts = self._compute_kmers(dataset)

        # The sort ensures that column 0 is the most frequent k-mer, 1 is the second, and so on (ties in
        # k-mer order). Initial selection of the top cache_features (runs keep their top max_features later)
        order = _sort_order(keys, counts)
        if self.cache_features and len(order) > self.cache_features:
            order = order[:self.cache_features]
            logging.info(f"Pre-selected top {self.cache_features} most frequent k-mers")
        self.kmer_to_col = {kmer: idx for idx, kmer in enumerate(_decode_keys(keys[order], self.k))}

        logging.info(f"Initial feature set: {len(self.kmer_to_col)} k-mers")

        # Clear memory
        del keys, counts, order
        gc.collect()

    def _compute_kmers(self, dataset):
        """Count the k-mers of the dataset in parallel, keeping those seen at least min_kmer_freq times.

        Each worker counts a shard of the genomes and saves its counts split into partitions (by k-mer);
        then each worker sums one partition over all shards, so no process holds every distinct k-mer.
        Returns (keys, counts), unsorted, and saves them to path_kmers if it is an .npz path."""
        logging.info("Computing k-mer frequencies from dataset...")
        fna_paths = [entry['seq'] for entry in dataset]
        n_shards = max(1, min(NUM_PROC, len(fna_paths)))
        # Shard and partition files go to the node's local disk when running under SLURM
        tmp_parent = os.environ.get("SLURM_TMPDIR") or (os.path.dirname(self.path_kmers) if self.path_kmers else None)
        tmpdir = tempfile.mkdtemp(prefix="kmer_counts_", dir=tmp_parent)
        try:
            with Pool(processes=n_shards) as pool:
                shard_args = [(shard, fna_paths[shard::n_shards], self.k, self.use_canonical, n_shards, tmpdir)
                              for shard in range(n_shards)]
                for _ in tqdm(pool.imap_unordered(_count_shard, shard_args), total=n_shards,
                              desc=f"Counting k-mers ({n_shards} shards of ~{len(fna_paths) // n_shards} genomes)"):
                    pass
                part_args = [(part, n_shards, tmpdir, self.min_kmer_freq) for part in range(n_shards)]
                parts = list(tqdm(pool.imap_unordered(_reduce_partition, part_args), total=n_shards,
                                  desc="Merging k-mer counts"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        logging.info(f"Found {sum(n for _, _, n in parts)} unique k-mers")
        keys = np.concatenate([part_keys for part_keys, _, _ in parts])
        counts = np.concatenate([part_counts for _, part_counts, _ in parts])
        del parts
        logging.info(f"After frequency filtering (min_freq={self.min_kmer_freq}): {len(keys)} k-mers")

        if self.path_kmers and self.path_kmers.endswith(".npz"):
            self._save_kmer_counts(keys, counts, self.path_kmers)
        return keys, counts

    def _load_kmer_counts(self):
        """Frequent k-mer counts from path_kmers as (keys, counts), or (None, None) if there are none."""
        path = self.path_kmers
        if not path:
            return None, None
        try:
            if not path.endswith(".npz"):
                return self._load_kmer_pickle(path)
            if os.path.exists(path):
                return self._load_kmer_npz(path)
            path_legacy = path[:-len(".npz")] + ".pkl"
            if os.path.exists(path_legacy):
                # Pickled counts do not record which genomes were counted, so they cannot be trusted
                logging.warning(f"Ignoring the legacy k-mer counts in {path_legacy} (no record of the genomes "
                                f"counted); it is no longer used and can be deleted.")
        except (OSError, pickle.PickleError, ValueError, KeyError) as e:
            logging.warning(f"Failed to load precomputed kmers: {e}. Computing from scratch...")
        return None, None

    def _load_kmer_npz(self, path):
        """Counts saved by _save_kmer_counts; (None, None) if they were counted with other settings."""
        logging.info(f"Loading precomputed frequent k-mers from {path}...")
        with np.load(path) as data:
            k, canonical, min_freq = int(data['k']), bool(data['use_canonical']), int(data['min_kmer_freq'])
            if (k, canonical) != (self.k, self.use_canonical) or min_freq > self.min_kmer_freq:
                logging.warning(f"{path} was counted with k={k}, use_canonical={canonical}, min_kmer_freq="
                                f"{min_freq}, which does not match this run. Computing from scratch...")
                return None, None
            if not self._same_assemblies(data, path):
                return None, None
            keys, counts = data['keys'], data['counts']
        if min_freq < self.min_kmer_freq:
            keep = counts >= self.min_kmer_freq
            keys, counts = keys[keep], counts[keep]
        logging.info(f"Loaded {len(keys)} frequent k-mers from file")
        return keys, counts

    def _load_kmer_pickle(self, path, chunk_size=1_000_000):
        """A {kmer: count} pickle as (keys, counts). Its counts are used as they are (not filtered)."""
        logging.info(f"Loading precomputed frequent k-mers from {path}...")
        with open(path, 'rb') as f:
            frequent_kmers = pickle.load(f)
        logging.info(f"Loaded {len(frequent_kmers)} frequent k-mers from file")
        kmers = list(frequent_kmers)
        counts = np.fromiter(frequent_kmers.values(), dtype=np.int64, count=len(kmers))
        del frequent_kmers
        if kmers and len(kmers[0]) != self.k:
            raise ValueError(f"{path} holds k-mers of length {len(kmers[0])}, not k={self.k}.")
        keys = [_encode_strings(kmers[i:i + chunk_size], self.k) for i in range(0, len(kmers), chunk_size)]
        keys = np.concatenate(keys) if keys else np.zeros(0, dtype=_key_dtype(self.k))
        return keys, counts

    def _save_kmer_counts(self, keys, counts, filepath):
        """Save the frequent k-mer counts as arrays (written to a temporary file first, so an interrupted
        save leaves no cache)."""
        try:
            path_tmp = filepath + ".tmp"
            with open(path_tmp, 'wb') as f:
                np.savez(f, keys=keys, counts=counts, k=self.k, use_canonical=self.use_canonical,
                         min_kmer_freq=self.min_kmer_freq, **self._assembly_fields())
            os.replace(path_tmp, filepath)
            logging.info(f"Saved frequent k-mer counts to {filepath}")
        except Exception as e:
            logging.warning(f"Failed to save frequent k-mers: {e}")

    def _worker_pool(self):
        """Pool whose workers receive this object (with its k-mer -> column map) once, at startup."""
        return Pool(processes=NUM_PROC, initializer=_init_worker, initargs=(self,))

    def _process_sequence_batch_sparse(self, batch, start_row_idx, pool):
        """
        Parallel sparse processing using the same _process_single_sequence
        worker function as the dense version.
        Returns: (data, rows, cols), lists of per-genome arrays
        """

        data, rows, cols = [], [], []

        # Workers only need the row index (relative to the batch) and the genome path
        worker_args = [(i, entry['seq']) for i, entry in enumerate(batch)]

        for rel_row_idx, row_cols, row_vals in tqdm(
            pool.imap_unordered(_process_entry, worker_args),
            total=len(worker_args),
            desc=f"Processing sparse batch ({NUM_PROC} processes)",
        ):
            if len(row_cols) == 0:
                continue  # skip empty rows

            # Convert relative index → absolute index
            rows.append(np.full(len(row_cols), start_row_idx + rel_row_idx, dtype=np.int64))
            cols.append(row_cols)
            data.append(row_vals)

        return data, rows, cols


    def _value_dtype(self):
        """Dtype of the row values computed by the workers (before fitting them into the matrix dtype)."""
        return np.float64 if self.mode == "entropy" or self.inference_groups else np.int64

    def _process_single_sequence(self, row_idx, fna_path):
        """
        Worker function (run through _process_entry): one genome's row.
        Returns: (row_idx, cols, vals), with cols an int64 array and vals a _value_dtype() array
        """
        mode, kmer_to_col = self.mode, self.kmer_to_col

        kmers = self._extract_kmers_from_file(fna_path)
        filtered = [k for k in kmers if k in kmer_to_col]
        if not filtered:
            return row_idx, np.zeros(0, np.int64), np.zeros(0, self._value_dtype())  # empty row

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

        # convert to column indices and values
        if self.inference_groups:
            return (row_idx, *self._group_medians(kmer_dict, kmer_to_col, self.group_sizes))
        n = len(kmer_dict)
        cols = np.fromiter((kmer_to_col[k] for k in kmer_dict), dtype=np.int64, count=n)
        vals = np.fromiter(kmer_dict.values(), dtype=self._value_dtype(), count=n)
        return row_idx, cols, vals

    @staticmethod
    def _group_medians(kmer_dict, kmer_to_col, group_sizes):
        """Median of each column's member k-mer values, counting members absent from the genome as 0.

        Returns (cols, medians) for the columns with a nonzero median."""
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
        return present_cols[keep], medians[keep]



    def _process_sequence_batch_dense(self, batch, start_row_idx, X, pool):
        """Parallel version using multiprocessing. Fills the batch's rows of X in place.

        Returns X, which is a widened copy if a count did not fit in X's dtype."""
        # Workers only need the row index (relative to the batch) and the genome path
        worker_args = [(i, entry['seq']) for i, entry in enumerate(batch)]

        for row_idx, cols, vals in tqdm(
            pool.imap_unordered(_process_entry, worker_args),
            total=len(worker_args),
            desc=f"Processing dense batch ({NUM_PROC} processes)",
        ):
            if len(cols) == 0:
                continue  # skip empty rows

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
        self.assemblies = self._assembly_ids(dataset)
        if self.group_columns:
            return self._build_grouped_matrix(dataset)
        
        # --- NEW: Priority 1 - Load and sub-sample from a full matrix ---
        if self._usable_cache(self.path_full_matrix):
            logging.info(f"Loading full feature matrix from {self.path_full_matrix} to sub-sample...")
            X, y = self.load_matrix(self.path_full_matrix)  # This also loads the full kmer_to_col map
            y = self._dataset_labels(dataset, y)
        else:
            # --- Fallback: Build matrix from scratch (and cache it) ---
            X, y = self._build_from_scratch(dataset)

        X = self._take_top_columns(X)
        if self.reduce_dim:
            logging.info("Applying dimensionality reduction using SVD...")
            _, _, X_reduced = efficient_svd(X)
            return X_reduced, y
        return X, y

        # # --- EXISTING: Priority 2 - Load from standard precomputed file ---
        # if self.path_precomputed:
        #     logging.info(f"Loading precomputed k-mer matrix from {self.path_precomputed}...")
        #     X, y = self.load_matrix(self.path_precomputed)
        #     return X, y

    def _take_top_columns(self, X):
        """Keep the columns of the top max_features k-mers (columns are ordered by k-mer frequency)."""
        n_features_full = X.shape[1]
        if not self.max_features or self.max_features >= n_features_full:
            logging.info(f"Using all {n_features_full:,} columns of the matrix (max_features={self.max_features}).")
            return X
        logging.info(f"Sub-sampling to top {self.max_features} features from {n_features_full} total features.")
        X_sub = X[:, :self.max_features].copy()
        # Update the k-mer map to reflect the sub-sampling
        col_to_kmer_full = {v: k for k, v in self.kmer_to_col.items()}
        self.kmer_to_col = {col_to_kmer_full[i]: i for i in range(self.max_features)}
        logging.info(f"Sub-sampled matrix shape: {X_sub.shape}")
        return X_sub

    @staticmethod
    def _assembly_ids(dataset):
        """Assembly IDs of the dataset's genomes in row order, or None if an entry has no 'assembly'."""
        if not all('assembly' in entry for entry in dataset):
            return None
        return np.array([entry['assembly'] for entry in dataset], dtype=str)

    def _assembly_fields(self):
        """Fields recording the dataset's assembly IDs in a saved cache (none if they are unknown)."""
        return {} if self.assemblies is None else {'assemblies': self.assemblies}

    def _same_assemblies(self, data, filepath):
        """Whether a cache (an open .npz) was built from the dataset's genomes, in the same order.

        Caches saved without assembly IDs are not trusted. Without assembly IDs for the dataset, there is
        nothing to check against."""
        if self.assemblies is None:
            return True
        if 'assemblies' not in data:
            logging.warning(f"{filepath} does not record the genomes it was built from; it will be rebuilt.")
            return False
        cached = data['assemblies']
        if np.array_equal(cached, self.assemblies):
            return True
        n_shared = len(np.intersect1d(cached, self.assemblies))
        logging.warning(f"{filepath} was built from {len(cached):,} genomes, which do not match (in order) the "
                        f"dataset's {len(self.assemblies):,} genomes ({n_shared:,} in common); it will be rebuilt.")
        return False

    @staticmethod
    def _dataset_labels(dataset, y_cached):
        """Labels of the dataset, which may have been revised since a cached matrix saved its labels."""
        y = np.array([entry['label'] for entry in dataset])
        if len(y) == len(y_cached) and not np.array_equal(y, y_cached):
            logging.warning(f"{int((y != y_cached).sum()):,} labels differ from the cached matrix's; using the "
                            f"dataset's labels.")
        return y

    @staticmethod
    def _cache_columns(filepath):
        """Number of k-mer columns of a cached matrix and the k-mer cap it was built with, read without
        loading the matrix. The cap is 0 for no cap, and -1 for caches saved before the cap was recorded."""
        with np.load(filepath, allow_pickle=True) as data:
            n_kmers_cap = int(data['n_kmers_cap']) if 'n_kmers_cap' in data else -1
            matrix_type = str(data['matrix_type']) if 'matrix_type' in data else ('dense' if 'matrix' in data else 'sparse')
            if matrix_type == 'grouped':
                return len(data['group_of']), n_kmers_cap   # group_of has one entry per ungrouped column
            if matrix_type == 'sparse':
                return int(data['shape'][1]), n_kmers_cap
        return npz_array_header(filepath)[0][1], n_kmers_cap

    def _usable_cache(self, filepath):
        """Whether the cached matrix at filepath exists and holds the top max_features k-mers.

        A cache with fewer columns than its cap holds every frequent k-mer (e.g. small k), so it is enough
        for any max_features."""
        if not filepath or not os.path.exists(filepath):
            return False
        with np.load(filepath, allow_pickle=True) as data:
            if not self._same_assemblies(data, filepath):
                return False
        n_cols, n_kmers_cap = self._cache_columns(filepath)
        if self.max_features is not None and n_cols >= self.max_features:
            return True
        if n_kmers_cap == -1:
            # Caches without a recorded cap were built with max_features=None when no limit was set
            enough = self.max_features is None
        else:
            enough = n_kmers_cap == 0 or n_cols < n_kmers_cap
        if not enough:
            cap = "unknown" if n_kmers_cap == -1 else f"{n_kmers_cap:,}"
            logging.warning(f"Cached matrix {filepath} has {n_cols:,} columns (k-mer cap: {cap}), fewer than "
                            f"max_features={self.max_features}; it will be rebuilt with cache_features="
                            f"{self.cache_features}.")
        return enough

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
        if self._usable_cache(self.path_grouped_matrix) and self._has_grouped_array(self.path_grouped_matrix):
            logging.info(f"Loading grouped feature matrix from {self.path_grouped_matrix}...")
            X, y = self._load_grouped_matrix(self.path_grouped_matrix)
        else:
            if self._usable_cache(self.path_full_matrix):
                logging.info(f"Grouping the columns of {self.path_full_matrix} (one-time conversion)...")
                X, y = self._group_cached_matrix(self.path_full_matrix)
                n_kmers_cap = self._cache_columns(self.path_full_matrix)[1]
            else:
                X_full, y = self._build_from_scratch(dataset)
                kmers = self._kmers_in_column_order(self.kmer_to_col, X_full.shape[1])
                # The k-mers are kept in column order in self.kmers; free the large lookup structures
                self.kmer_to_col, self.selected_kmers = {}, set()
                self.rep_cols, self.group_of, X, self.const_values = group_columns(
                    lambda: iter_dense_rows(X_full), *X_full.shape, X_full.dtype)
                del X_full
                gc.collect()
                self.kmers = kmers
                n_kmers_cap = self.cache_features or 0
            if self.path_grouped_matrix:
                self._save_grouped_matrix(X, y, self.path_grouped_matrix, n_kmers_cap)
                # Re-open it memory-mapped, so the first run uses as little memory as later runs
                del X
                gc.collect()
                X, y = self._load_grouped_matrix(self.path_grouped_matrix)

        y = self._dataset_labels(dataset, y)
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
        """Keep the grouped columns of the top max_features k-mers, and set group_sizes."""
        n_total = len(self.group_of)
        self.n_prefix = min(self.max_features, n_total) if self.max_features else n_total
        n_kept, self.group_sizes = prefix_groups(self.rep_cols, self.group_of, self.n_prefix)
        self.constant_sq_norm = constant_sq_norm(self.const_values, self.n_prefix)
        if n_kept < X.shape[1]:
            X = X[:, :n_kept]   # a view: on a memory-mapped matrix, only these columns are ever read
        logging.info(f"Using {n_kept:,} grouped columns for the top {self.n_prefix:,} k-mers. Matrix shape: {X.shape}")
        return X

    def get_column_groups(self):
        """Member k-mers of each grouped column (within the top max_features), representative first."""
        members = members_of_groups(self.group_of, self.n_prefix, len(self.group_sizes))
        return [self.kmers[cols].astype(str).tolist() for cols in members]

    @staticmethod
    def _grouped_array_path(filepath):
        return filepath.replace(".npz", ".npy")

    def _has_grouped_array(self, filepath):
        """Whether the grouped cache's matrix exists: its .npy, or in the .npz for caches saved before memory-mapping."""
        if os.path.exists(self._grouped_array_path(filepath)):
            return True
        with np.load(filepath) as data:
            return 'matrix' in data

    def _save_grouped_matrix(self, X, y, filepath, n_kmers_cap):
        """n_kmers_cap is the k-mer cap of the ungrouped matrix (see _cache_columns).

        The matrix goes to an uncompressed .npy and the metadata to the .npz. The .npz is written last,
        so a cache interrupted while saving has no .npz and counts as missing."""
        path_array = self._grouped_array_path(filepath)
        logging.info(f"Saving grouped matrix with shape {X.shape} to {path_array}")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        np.save(path_array, np.ascontiguousarray(X))
        np.savez_compressed(filepath, matrix_type='grouped', labels=y,
                            rep_cols=self.rep_cols, group_of=self.group_of, kmers=self.kmers,
                            const_values=self.const_values, n_kmers_cap=n_kmers_cap, **self._assembly_fields())

    def _load_grouped_matrix(self, filepath):
        """Load the grouped metadata, and memory-map the matrix (read-only)."""
        path_array = self._grouped_array_path(filepath)
        with np.load(filepath) as data:
            y = data['labels']
            self.rep_cols, self.group_of, self.kmers = data['rep_cols'], data['group_of'], data['kmers']
            self.const_values = data['const_values']
            n_kmers_cap = int(data['n_kmers_cap'])
            # Caches saved before memory-mapping keep the matrix in the compressed .npz: move it out once
            # (always rewritten from the .npz, in case an earlier conversion was interrupted)
            X = data['matrix'] if 'matrix' in data else None
        if X is not None:
            logging.info(f"Moving the grouped matrix of {filepath} to {path_array} (one-time conversion)...")
            self._save_grouped_matrix(X, y, filepath, n_kmers_cap)
            del X
            gc.collect()
        X = np.load(path_array, mmap_mode='r')
        logging.info(f"Memory-mapped grouped matrix with shape {X.shape} from {path_array}")
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
        with self._worker_pool() as pool:
            for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
                batch = dataset[i:i + self.batch_size]
                batch_data, batch_rows, batch_cols = self._process_sequence_batch_sparse(batch, i, pool)

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
        
        all_data = np.concatenate(all_data) if all_data else np.zeros(0, self._value_dtype())
        all_rows = np.concatenate(all_rows) if all_rows else np.zeros(0, np.int64)
        all_cols = np.concatenate(all_cols) if all_cols else np.zeros(0, np.int64)
        dtype = self._fit_dtype(self._get_dtype(), all_data.max(initial=0))
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
        with self._worker_pool() as pool:
            for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
                batch = dataset[i:i + self.batch_size]
                X = self._process_sequence_batch_dense(batch, i, X, pool)

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
        if self.group_sizes is not None:
            # Grouped training: each column's representative k-mer
            return self.kmers[self.rep_cols[:len(self.group_sizes)]].astype(str).tolist()
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
                                use_sparse=self.use_sparse,
                                n_kmers_cap=self.cache_features or 0,
                                **self._assembly_fields())
        else:  # Dense matrix
            logging.info(f"Saving dense matrix with shape {X.shape}")
            np.savez_compressed(filepath, 
                                matrix_type='dense',
                                matrix=X, 
                                labels=y,
                                kmer_to_col=self.kmer_to_col,
                                use_sparse=self.use_sparse,
                                n_kmers_cap=self.cache_features or 0,
                                **self._assembly_fields())

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