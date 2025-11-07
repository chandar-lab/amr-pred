"""
Script to run inference on a trained model and evaluate its performance on a test dataset.
Saves predictions and evaluation metrics to specified output directory.

Usage:
    PYTHONPATH=<path_to_repository> python testing/test_model.py \
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

from sklearn.metrics import balanced_accuracy_score, classification_report

from data.data_utils import load_data
from features.kmer import KmerFeatures

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
def convert_k_to_numeric(val):
    val = str(val).upper().replace('K', '')
    return int(val) * 1000


def parse_model_filename(filename):
    """
    Parse model filename and extract structured information.
    
    Args:
        filename (str): The input filename to parse
        
    Returns:
        dict: Dictionary with parsed components
    """
    # Pattern to match '__' followed by any characters until the next '__' or end
    pattern = r'__([^_]+(?:_[^_]+)*?)(?=__|\.)'
    
    matches = re.findall(pattern, filename)
    
    # Map to structured dictionary
    result = {
        'pathogen': matches[0] if len(matches) > 0 else None,
        'antibiotic': matches[1] if len(matches) > 1 else None,
        'kmer_size': matches[2] if len(matches) > 2 else None,
        'kmer_mode': matches[3] if len(matches) > 3 else None,
        'split_method': matches[4] if len(matches) > 4 else None,
        'feature_size': matches[5] if len(matches) > 5 else None,
    }
    
    # Extract k-mer value (e.g., "k_17" -> 17)
    if result['kmer_size'] and result['kmer_size'].startswith('k_'):
        result['kmer_size'] = result['kmer_size'].split('_')[1]

    # Extract mode type (e.g., "mode_count" -> "count")
    if result['kmer_mode'] and result['kmer_mode'].startswith('mode_'):
        result['kmer_mode'] = result['kmer_mode'].split('_', 1)[1]

    # Extract split type (e.g., "split_random" -> "random")
    if result['split_method'] and result['split_method'].startswith('split_'):
        result['split_method'] = result['split_method'].split('_', 1)[1]

    # Extract feature count (e.g., "feat_1000K" -> "1000K")
    if result['feature_size'] and result['feature_size'].startswith('feat_'):
        result['feature_size'] = result['feature_size'].split('_', 1)[1]
    
    return result


def main():
    
    parser = get_parser()
    args = parser.parse_args()
    path_root_fna = args.path_fna
    path_data_csv = args.path_csv
    path_model = args.path_model
    path_out = args.path_out

    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"Dataset CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")
    
    # Ensure output directory exists
    os.makedirs(args.path_out, exist_ok=True)
    os.makedirs(os.path.join(args.path_out, "feature_matrices"), exist_ok=True)
    os.makedirs(os.path.join(args.path_out, "unique_kmers"), exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(args.path_out, f"inference.log"))
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )

    # ---------------------
    # Load model
    # ---------------------
    logging.info("Loading model from %s", path_model)
    if not os.path.isfile(path_model):
        raise FileNotFoundError(f"Model file not found: {path_model}")
    model = joblib.load(path_model)

    # extract kmer size and feature matrix size from the model_path
    model_fname = path_model.split('/')[-1]
    model_info = parse_model_filename(model_fname)
    k = int(model_info['kmer_size'])
    kmer_mode = model_info['kmer_mode']
    fmat_size = model_info['feature_size']
    max_features = convert_k_to_numeric(fmat_size)
    pathogen, antibiotic = model_info['pathogen'], model_info['antibiotic']

    # ---------------------
    # Load data
    # ---------------------
    path_root_fna = os.path.join(path_root_fna, PATHOGEN_NAME_MAPPER[pathogen])
    logging.info("Loading test data from %s", path_data_csv)
    test_datalist = load_data(path_data_csv, path_root_fna)
    logging.info("Number of test samples: %d", len(test_datalist))
    
    # ---------------------
    # Feature extraction
    # ---------------------
    logging.info("Initializing K-mer class to build feature matrix ...")
    kmer_features = KmerFeatures(
        k=k,
        mode=kmer_mode,
        min_kmer_freq=2,        # Remove extremely rare k-mers
        max_features=max_features,
        batch_size=256,
        use_canonical=True,
        use_sparse=False,
        path_full_matrix=os.path.join(
            path_out, "feature_matrices", f"{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_{fmat_size}.npz"),
        path_kmers=os.path.join(
            path_out, "unique_kmers", f"{pathogen}__{antibiotic}__k_{k}.pkl"),
    )

    logging.info("Extracting k-mer features")
    X_test, y_test = kmer_features.build_feature_matrix(dataset=test_datalist)

    logging.info(f"Feature matrix shape: {X_test.shape}")
    logging.info(f"Number of samples: {X_test.shape[0]}")

    # ---------------------
    # Run inference
    # ---------------------
    logging.info("Running inference ...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    accuracy = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logging.info("Balanced Accuracy: %.4f", accuracy)
    logging.info("Classification Report:\n%s", report)
    logging.info("Inference Time: %.2f seconds", end_time - start_time)

    # ---------------------
    # Save 
    # ---------------------
    logging.info("Saving predictions and metrics ...")
    results_df = pd.DataFrame({
        "True_Label": y_test,
        "Predicted_Label": y_pred
    })
    results_df.to_csv(os.path.join(path_out, "predictions.csv"), index=False)
    with open(os.path.join(path_out, "metrics.txt"), 'w') as f:
        f.write(f"Balanced Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"Inference Time (seconds): {end_time - start_time:.2f}\n")

    logging.info("Done.")

if __name__ == "__main__":
    main()