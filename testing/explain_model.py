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
import shap
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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
    file_handler = logging.FileHandler(os.path.join(args.path_out, f"inference.log"))
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

    # ---------------------
    # Run SHAP explainability
    # ---------------------
    logging.info("Running SHAP explainability ...")
    explainer = shap.Explainer(model)
    col_to_kmer = {v: k for k, v in kmer_features.kmer_to_col.items()}
    feature_names = [col_to_kmer[i] for i in range(len(col_to_kmer))]
    explanation = explainer(X_test)
    explanation.feature_names = feature_names
    
    # Plot results
    logging.info("Generating SHAP summary plot ...")
    plot_path = os.path.join(path_out, "shap",
                             f"shap_summary_{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_{fmat_size}.png")

    # Create the figures directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    # Change heatmap to read "Count of K-mer"
    plt.xlabel("SHAP value (Negative: Susceptible, Positive: Resistant)")
    plt.title(f"SHAP Summary Plot for {pathogen.replace('_', ' ').title()} - {antibiotic.replace('_', ' ').title()} with k={k}, feat={fmat_size}")

    # Find colorbar axis created by SHAP
    fig = plt.gcf()
    cbar_ax = fig.axes[-1]  # SHAP's colorbar is last
    cbar_ax.set_ylabel("Count of K-mer")

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')    
    plt.close()        
    logging.info(f"Saved SHAP summary plot to {plot_path}")

    # Print the top 10 most important k-mers and associated SHAP values
    shap_values_mean = np.abs(explanation.values).mean(axis=0)
    top_kmer_indices = np.argsort(shap_values_mean)[-10:][::-1]  # Top 10 indices in descending order
    logging.info("Top 10 most important k-mers and their mean absolute SHAP values:")
    for idx in top_kmer_indices:
        kmer = explanation.feature_names[idx]
        shap_value = shap_values_mean[idx]
        logging.info(f"K-mer: {kmer}, Mean |SHAP value|: {shap_value:.6f}")
    
    logging.info("Saving k-mers and their mean |SHAP| values to CSV ...")
    shap_df = pd.DataFrame({
        "kmer": explanation.feature_names,
        "mean_abs_shap_value": shap_values_mean
    })
    # Sort by mean_abs_shap_value descending
    shap_df = shap_df.sort_values(by="mean_abs_shap_value", ascending=False)
    shap_df = shap_df[shap_df["mean_abs_shap_value"] > 0]  # Keep only k-mers with non-zero SHAP values
    
    # Make output directory if it doesn't exist
    csv_path = os.path.join(path_out, "shap", f"shap_values_{pathogen}__{antibiotic}__k_{k}__mode_{kmer_mode}__feat_{fmat_size}.csv")
    shap_df.to_csv(csv_path, index=False)
    logging.info(f"Saved SHAP values to {csv_path}")
    
    logging.info("Done.")

if __name__ == "__main__":
    main()