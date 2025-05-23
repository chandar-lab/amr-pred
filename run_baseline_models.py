import os
import glob
import pandas as pd
import numpy as np
import argparse
import time
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import xgboost as xgb
from datetime import datetime

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix
)
from kmer import KmerFeatures


# TODO: 
# 1. add other options to create the feature matrix
# 2. add options for other ML models; use Hydra for setting up models

SEED = 42
LABEL_MAPPING = {
    # NOTE: convention is to use 0 for susceptible and 1 for resistant
    "resistant": 1,
    "susceptible": 0
}

def get_parser():
    
    parser = argparse.ArgumentParser(description="Train XGBoost model on k-mer features")
    parser.add_argument("--path-dataroot", type=str, help="Root path of the dataset genomic sequences for all species")
    parser.add_argument("--path-csv", type=str, required=True, help="Path to the folder containing CSV files for each specie-drug combination")
    parser.add_argument("--pathogen", type=str, help="Pathogen name (lowercase, separated by underscores) e.g. neisseria_gonorrhoeae")
    parser.add_argument("--antibiotic", type=str, help="Antibiotic name (lowercase) e.g. azithromycin")
    parser.add_argument("--k", type=int, default=8, help="Length of k-mers to extract (default: 8)")
    parser.add_argument("--path-precomputed", type=str, default=None, help="Path to precomputed k-mer matrix (optional)")
    
    return parser


def load_data(path_data_csv, path_root_fna):

    df_data = pd.read_csv(path_data_csv)
    specie_drug_assemblies = df_data['Assembly'].tolist()
    # get the corresponding phenotypic labels for each assembly
    phenotypes = df_data['Resistance phenotype'].tolist()
    # print(phenotypes)

    dataset_list = []
    for assembly, phenotype in zip(specie_drug_assemblies, phenotypes):
        temp_dict = {}
        # if assembly not in dataset_dict:
        fname = glob.glob(os.path.join(path_root_fna, assembly, "ncbi_dataset", "data", assembly, "*.fna"))
        if len(fname) == 0:
            logging.info(f"WARNING: No .fna file found for assembly {assembly}. Skipping this entry.")
            continue

        temp_dict["seq"] = fname[0]
        temp_dict["label"] = LABEL_MAPPING[phenotype]
        dataset_list.append(temp_dict)
    
    print("Number of samples in the dataset: ", len(dataset_list))
    
    return dataset_list


def fitmodel_nestedcv(X, pheno, estimator, parameters, modelname, path_out=None):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    all_scores = []
    all_reports = []
    all_conf_matrices = []
    fold_times = []
    best_fold_index = -1

    logging.info(f" Starting CV for {modelname}...")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, pheno)):
        start = time.process_time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pheno[train_idx], pheno[test_idx]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        # grid = GridSearchCV(estimator=estimator,
        #                     param_grid=parameters,
        #                     scoring='balanced_accuracy',
        #                     cv=5, # uses StratifiedKFold
        #                     n_jobs=(mp.cpu_count()//4),
        #                     )
        # grid.fit(X_train, y_train)

        # best_model = grid.best_estimator_
        # y_pred = best_model.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        all_scores.append(score)
        all_reports.append(report)
        all_conf_matrices.append(cm)
        fold_times.append(time.process_time() - start)

        logging.info(f"\n Fold {fold_idx+1} - Balanced Accuracy: {score:.3f}")
        # TODO: what are 0 and 1 ?
        logging.info(f"\n {report}")
        # logging.info("Confusion Matrix:\n", cm)

        if score == max(all_scores):
            best_fold_index = fold_idx

    logging.info(f"\n {modelname} Nested CV Summary:")
    logging.info(f"Mean Balanced Accuracy: {np.mean(all_scores):.3f} ± {np.std(all_scores):.3f}")
    logging.info(f"Avg Fold Time: {np.mean(fold_times):.2f}s")

    # Plot best confusion matrix
    best_cm = all_conf_matrices[best_fold_index]
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Susceptible", "Resistant"], yticklabels=["Susceptible", "Resistant"])
    plt.title(f"{modelname} - (Fold {best_fold_index + 1})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, f"{modelname}_best_cm.png"), dpi=300)

    return all_scores, all_reports, all_conf_matrices


def main():

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    path_root_fna = args.path_dataroot
    path_data_csv = args.path_csv

    now = datetime.now().strftime(f"%Y%m%d-%H%M%S")
    path_out = os.path.join("/home/mila/n/naga-karthik.enamundram/genomicFMs/results", now)
    os.makedirs(path_out, exist_ok=True)

    path_root_fna = "/home/mila/n/naga-karthik.enamundram/scratch/dataset_ncbi_asts"
    # dataset = "neisseria_gonorrhoeae__azithromycin.csv"
    dataset = f"{args.pathogen}__{args.antibiotic}.csv"
    path_data_csv = os.path.join(args.path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")

    file_handler = logging.FileHandler(os.path.join(path_out, f"log__{args.pathogen}__{args.antibiotic}__k{args.k}.txt"))
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )

    logging.info(f"Training on {args.pathogen} for {args.antibiotic} using k-mers of length {args.k} ...")

    # NOTE: this is a list of dicts pointing for the paths to the genomic sequences (.fna files) and the corresponding S/R labels
    dataset_list = load_data(path_data_csv, path_root_fna)

    # Parameters
    k = args.k

    logging.info("Initializing K-mer class to build feature matrix ...")
    builder = KmerFeatures(k=k, mode="count", path_precomputed=args.path_precomputed)

    # # Analyze k-mer repetition
    # repetition_stats = builder.analyze_kmer_repetition(dataset_list, top_n=20, output_file="None")
    
    # Build feature matrix
    X, y = builder.build_feature_matrix(dataset=dataset_list)
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Number of samples: {X.shape[0]}")

    # save the matrix to a file
    if args.path_precomputed is None:
        save_fname = os.path.join(args.path_csv, f"{args.pathogen}__{args.antibiotic}__k{k}.npz")
        np.savez_compressed(save_fname, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)

        logging.info(f"Feature matrix saved to {save_fname}")

    # # Print information about the sparse matrix
    # sparse_bytes = X.data.nbytes + X.indptr.nbytes + X.indices.nbytes
    # print(f"Feature matrix shape: {X.shape}")
    # print(f"Number of non-zero elements: {X.nnz}")
    # print(f"Sparsity: {X.nnz / (X.shape[0] * X.shape[1]):.6f}")
    # print(f"Memory usage: {sparse_bytes} bytes or {sparse_bytes / 1e9:.2f} GB")

    # # Compare to what a dense matrix would use
    # dense_bytes = X.shape[0] * X.shape[1] * 8  # 8 bytes per float64
    # print(f"Equivalent dense matrix would use: {dense_bytes} bytes or {dense_bytes / 1e9:.2f} GB")
    # print(f"Memory savings: {dense_bytes / (sparse_bytes):.1f}x")

    # # ==========
    # # logistic regression
    # # ==========
    # elastic_net = SGDClassifier(
    #     loss="log",
    #     penalty="elasticnet",
    #     random_state=SEED,
    # )
    # elastic_net_params = {
    #     "alpha": [0.001, 0.01],
    #     "l1_ratio": [0.1, 0.5, 0.9],
    # }

    # enet_scores, enet_reports, enet_conf_matrices = fitmodel_nestedcv(
    #     X, y, elastic_net, elastic_net_params, "ElasticNet"
    # )

    # # ==========
    # # Random Forest
    # # ==========
    # rf = RandomForestClassifier(
    #     n_jobs=-1,
    #     random_state=SEED,
    #     class_weight="balanced",
    # )
    # rf_params = {
    #     "n_estimators": [100, 200],
    # }
    # rf_scores, rf_reports, rf_conf_matrices = fitmodel_nestedcv(
    #     X, y, rf, rf_params, "RandomForest"
    # )

    # Train XGBoost
    # # print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        # verbosity=1
    )
    # xgb_params = {
    # 'alpha': [1e-5, 1e-4], 
    # 'colsample_bytree': [0.6],
    # 'gamma': [0.05, 0.1], 
    # 'learning_rate': [0.01, 0.1], 
    # 'max_depth': [2],
    # 'subsample': [0.2, 0.4, 0.6]
    # }

    xgb_scores, xgb_reports, xgb_conf_matrices = fitmodel_nestedcv(
        X, y, xgb_model, None, "XGBoost", path_out
    )

    # # Evaluation
    # print("Evaluating model...")
    # y_pred = model.predict(X_test)
    # print("\nClassification Report:")
    # target_names = ["Susceptible (0)", "Resistant (1)"]
    # print(classification_report(y_test, y_pred, target_names=target_names))



if __name__ == "__main__":
    main()
