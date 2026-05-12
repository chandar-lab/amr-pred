"""
Script to run inference on a trained model and evaluate its performance on a test dataset.
Saves predictions and evaluation metrics to specified output directory.

Usage:
    PYTHONPATH=<path_to_repository> python testing/explain_model.py \
        --path-fna <path_to_fna_files> \
        --path-csv <path_to_test_csv> \
        --path-model <path_to_trained_model> \
        --path-out <output_directory>
"""

import os
import re
import time
import logging
import joblib
import argparse
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, classification_report

from data.data_utils import load_data
from features.kmer import KmerFeatures
from testing.utils import get_parser, load_model_data

PATHOGEN_NAME_MAPPER = {
    "a_baumannii": "acinetobacter_baumannii",
    "e_coli": "escherichia_coli",
    "s_aureus": "staphylococcus_aureus",
    "k_pneumoniae": "klebsiella_pneumoniae",
    "p_aeruginosa": "pseudomonas_aeruginosa",
}


def main():
    
    parser = get_parser()
    args = parser.parse_args()
    path_root_fna = args.path_fna
    path_data_csv = args.path_csv
    path_model = args.path_model
    path_out = args.path_out

    # Start logging
    file_handler = logging.FileHandler(os.path.join(args.path_out, f"important_features.log"))
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )

    # Load model and data
    model, X_test, y_test, model_info, kmer_features = load_model_data(
        path_root_fna, path_data_csv, path_model, path_out
    )
    pathogen = model_info['pathogen']
    antibiotic = model_info['antibiotic']
    k = model_info['kmer_size']
    kmer_mode = model_info['kmer_mode']
    fmat_size = model_info['feature_size']

    # get the column names from kmer_features
    kmer_column_names = kmer_features.get_feature_names()

    # Compare the list of column names to the list of feature importances from the model
    if len(kmer_column_names) != len(model.feature_importances_):
        logging.error("Number of k-mer features does not match number of feature importances from the model.")
        return
    # Get the total and mean counts for each k-mer that is important
    logging.info("Calculating k-mer counts for important features...")
    kmer_stats = []
    feature_importances = model.feature_importances_
    for idx in range(len(feature_importances)):
        importance = feature_importances[idx]
        if importance == 0:
            continue  # Skip k-mers with zero importance
        kmer = kmer_column_names[idx]
        mean_count = np.mean(X_test[:, idx])
        total_count = np.sum(X_test[:, idx])
        kmer_stats.append({
            "kmer": kmer,
            "importance": importance,
            "mean_count": mean_count,
            "total_count": total_count
        })

    kmer_stats = pd.DataFrame(kmer_stats)
    kmer_stats = kmer_stats.sort_values(by="importance", ascending=False)
    kmer_stats.to_csv(os.path.join(path_out, f"important_kmer_counts_{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_{fmat_size}.csv"), index=False)
    logging.info(f"Important k-mer counts saved to {os.path.join(path_out, f'important_kmer_counts_{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_{fmat_size}.csv')}")

if __name__ == "__main__":
    main()