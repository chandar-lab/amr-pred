"""
Utility functions for the testing scripts, both for inference and explainability.
"""

import os
import re
import time
import logging
import joblib
import argparse
import shap
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, classification_report

from data.data_utils import load_data
from features.kmer import KmerFeatures
from features.utils import apply_tfidf

PATHOGEN_NAME_MAPPER = {
    "a_baumannii": "acinetobacter_baumannii",
    "e_coli": "escherichia_coli",
    "s_aureus": "staphylococcus_aureus",
    "k_pneumoniae": "klebsiella_pneumoniae",
    "p_aeruginosa": "pseudomonas_aeruginosa",
}


def get_parser():
    
    parser = argparse.ArgumentParser(description="Run inference.")

    parser.add_argument("--path-fna", type=str, 
                        help="Root path of the dataset genomic sequences for all species. Assumes the genomic files are organized in subdirectories named after the species. "
                        "E.g., <path_fna>/escherichia_coli/, <path_fna>/staphylococcus_aureus/, etc.")
    parser.add_argument("--path-csv", type=str, required=True,
                        help="Path to the csv file containing specie x drug assembly id and mic values")
    parser.add_argument("--path-model", type=str, required=True,
                        help="Path to the trained model (must be a pkl file).")
    parser.add_argument("--path-out", type=str, default=".",
                        help="Directory to save the results and the log file.")
    
    return parser

# Helper function to convert 'K' notation (e.g., '5000K') to numeric values
def _convert_k_to_numeric(val):
    val = str(val).upper().replace('K', '')
    return int(val) * 1000


def _parse_model_filename(filename):
    """
    Parse model filename and extract structured information.

    Expected format: model_<model_name>__<pathogen>__<antibiotic>__k_<k>__mode_<mode>[__split_<split>]__feat_<size>[__no_sweep].pkl
    Fields after the antibiotic are matched by prefix, so files saved without a split segment also parse.
    
    Args:
        filename (str): The input filename to parse
        
    Returns:
        dict: Dictionary with parsed components
    """
    stem = os.path.basename(filename)
    stem = stem[:-len(".pkl")] if stem.endswith(".pkl") else stem
    parts = stem.split("__")

    result = {
        'model_name': parts[0][len("model_"):] if parts[0].startswith("model_") else parts[0],
        'pathogen': parts[1] if len(parts) > 1 else None,
        'antibiotic': parts[2] if len(parts) > 2 else None,
        'kmer_size': None,
        'kmer_mode': None,
        'split_method': None,
        'feature_size': None,
    }
    prefixes = {'k_': 'kmer_size', 'mode_': 'kmer_mode', 'split_': 'split_method', 'feat_': 'feature_size'}
    for part in parts[3:]:
        for prefix, key in prefixes.items():
            if part.startswith(prefix):
                result[key] = part[len(prefix):]
                break
    
    return result


def load_model_data(path_root_fna, path_data_csv, path_model, path_out):
    ''' Given a path to a trained model, load the model and the test data, by extracting k-mer features.'''

    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"Dataset CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")
    
    # Ensure output directory exists
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(os.path.join(path_out, "feature_matrices"), exist_ok=True)
    os.makedirs(os.path.join(path_out, "unique_kmers"), exist_ok=True)

    # ---------------------
    # Load model
    # ---------------------
    logging.info("Loading model from %s", path_model)
    if not os.path.isfile(path_model):
        raise FileNotFoundError(f"Model file not found: {path_model}")
    model = joblib.load(path_model)

    # extract kmer size and feature matrix size from the model_path
    model_fname = path_model.split('/')[-1]
    model_info = _parse_model_filename(model_fname)
    k = int(model_info['kmer_size'])
    kmer_mode = model_info['kmer_mode']
    fmat_size = model_info['feature_size']
    max_features = None if fmat_size == 'full' else _convert_k_to_numeric(fmat_size)
    pathogen, antibiotic = model_info['pathogen'], model_info['antibiotic']

    # ---------------------
    # Load data
    # ---------------------
    path_root_fna = os.path.join(path_root_fna, PATHOGEN_NAME_MAPPER[pathogen])
    logging.info("Loading test data from %s", path_data_csv)
    test_datalist = load_data(path_data_csv, path_root_fna)
    logging.info("Number of test samples: %d", len(test_datalist))

    # ---------------------
    # Load kmers
    # ---------------------
    model_parent_path = os.path.dirname(path_model)

    # Logistic regression needs the TF-IDF parameters from training; check before the slow feature extraction
    path_tfidf = None
    if model_info['model_name'].startswith('LGR'):
        model_stem = os.path.basename(path_model)[len("model_"):-len(".pkl")]
        path_tfidf = os.path.join(model_parent_path, f"tfidf_{model_stem}.npz")
        if not os.path.isfile(path_tfidf):
            raise FileNotFoundError(
                f"TF-IDF parameters not found: {path_tfidf}. This LGR model was trained before the TF-IDF "
                "parameters were saved, so its inputs cannot be reproduced; please retrain it.")

    # Grouped models (model name ends in "_grp") use k-mer groups: each column is the median over its k-mers
    grouped = model_info['model_name'].endswith('_grp')
    kmers_fname = "kmer_groups" if grouped else "kmers"
    path_kmers = os.path.join(model_parent_path, f"{kmers_fname}_{pathogen}__{antibiotic}__k_{k}__feat_{fmat_size}.pkl")
    if not os.path.isfile(path_kmers):
        # Throw an error if kmer file not found; it's necessary
        raise FileNotFoundError(f"K-mer file not found: {path_kmers}. Necessary for feature extraction.")
    
    # ---------------------
    # Feature extraction
    # ---------------------
    logging.info("Initializing K-mer class to build feature matrix ...")
    if grouped:
        # Columns of the 10M test matrix are single k-mers, so grouped models use their own cache
        full_matrix_path = os.path.join(
            path_out, "feature_matrices", f"{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_{fmat_size}__grouped.npz")
    else:
        # Check if the full 10M feature matrix exists from training
        full_matrix_path = os.path.join(
            path_out, "feature_matrices", f"{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_10M.npz")
        if not os.path.isfile(full_matrix_path):
            full_matrix_path = os.path.join(
                path_out, "feature_matrices", f"{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_{fmat_size}.npz")
    kmer_features = KmerFeatures(
        k=k,
        mode=kmer_mode,
        min_kmer_freq=2,        # Remove extremely rare k-mers
        max_features=None if grouped else max_features,  # the k-mer groups already define the columns
        batch_size=256,
        use_canonical=True,
        use_sparse=False,
        path_full_matrix=full_matrix_path,
        path_kmers=None if grouped else path_kmers,
        path_kmer_groups=path_kmers if grouped else None,
    )

    logging.info("Extracting k-mer features")
    X_test, y_test = kmer_features.build_feature_matrix(dataset=test_datalist)

    logging.info(f"Feature matrix shape: {X_test.shape}")
    logging.info(f"Number of samples: {X_test.shape[0]}")

    # ---------------------
    # TF-IDF (logistic regression is trained on TF-IDF features)
    # ---------------------
    if path_tfidf:
        logging.info("Applying the training TF-IDF transformation from %s", path_tfidf)
        with np.load(path_tfidf) as tfidf_params:
            idf = tfidf_params['idf']
            group_sizes = tfidf_params['group_sizes'] if 'group_sizes' in tfidf_params else None
            const_sq = float(tfidf_params['constant_sq_norm']) if 'constant_sq_norm' in tfidf_params else 0.0
        X_test = apply_tfidf(X_test, idf, group_sizes, const_sq)

    return model, X_test, y_test, model_info, kmer_features