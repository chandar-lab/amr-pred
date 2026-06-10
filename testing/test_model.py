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
from testing.utils import get_parser,  load_model_data

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

    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"Dataset CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")
    

    file_handler = logging.FileHandler(os.path.join(args.path_out, f"inference.log"))
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )

    model, X_test, y_test, model_info, kmer_features = load_model_data(
        path_root_fna, path_data_csv, path_model, path_out
    )
    

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
        f.write(report) # type: ignore
        f.write(f"Inference Time (seconds): {end_time - start_time:.2f}\n")

    logging.info("Done.")

if __name__ == "__main__":
    main()