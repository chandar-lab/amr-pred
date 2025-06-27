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
import time
import pickle

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix
)
from features.kmer import KmerFeatures
from features.utils import plot_umap


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
    parser.add_argument("--path-precomputed", type=str, default=None, help="Path to precomputed k-mer matrix (optional)")
    parser.add_argument("--debug",  action='store_true', help="Enable debug mode for verbose logging")
    parser.add_argument("--k", type=int, default=8, help="Length of k-mers to extract (default: 8)")
    parser.add_argument("--feat-mode", type=str, default="count", choices=["binary", "count", "entropy"], help="Mode for k-mer extraction (default: count)")
    parser.add_argument("--max-features", type=int, default=1000000, help="Maximum number of top most frequent k-mers to extract (default: 1M)")
    parser.add_argument("--use-sparse", action='store_true', help="Use sparse matrix for k-mer features (default: False)")

    return parser


def load_data(path_data_csv, path_root_fna):

    df_data = pd.read_csv(path_data_csv)
    specie_drug_assemblies = df_data['Assembly'].tolist()
    # get the corresponding phenotypic labels for each assembly
    phenotypes = df_data['Phenotype_revised'].tolist()
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

    bal_acc_scores, f1_macro_scores = [], []
    very_major_errors, major_errors = [], []
    fold_times = []

    # Track best model information
    best_score = -1
    best_model = None
    best_y_test = None
    best_y_pred = None
    best_fold_idx = -1

    logging.info(f"Starting CV for {modelname}...")
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, pheno)):
        start = time.process_time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pheno[train_idx], pheno[test_idx]

        if parameters is None:
            # If no parameters are provided, use the estimator as is
            logging.info(f"Using default parameters for {modelname} in fold {fold_idx+1}")
            
            current_model = clone(estimator)
            current_model.fit(X_train, y_train)
            y_pred = current_model.predict(X_test)

        else:
            logging.info(f"Performing GridSearchCV for {modelname} in fold {fold_idx+1}")
            
            # Perform grid search to find the best hyperparameters
            grid = GridSearchCV(
                estimator=estimator, 
                param_grid=parameters, 
                scoring='balanced_accuracy',
                cv=5, # uses StratifiedKFold
            )
            grid.fit(X_train, y_train)

            current_model = grid.best_estimator_
            y_pred = current_model.predict(X_test)

            logging.info(f"Best parameters in fold {fold_idx+1}: {grid.best_params_}")

        score = balanced_accuracy_score(y_test, y_pred)
        f1_macro = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
        cm = confusion_matrix(y_test, y_pred)

        # Compute Very Major Error and Major Error
        # NOTE: # VME is the false negative rate; proportion of resistant samples misclassified as susceptible; aka Type II error
        # ME is the false positive rate; proportion of susceptible samples misclassified as resistant; aka Type I error
        # To be cleared for marketing in the USA, the FDA requires that VMEs attributable to a test device should be <1.5% for individual species/drug comparisons, 
        # MEs should not exceed 3%, and an overall essential MIC agreement of >90% of device MICs within one doubling dilution of a CLSI reference MIC.
        # More info: https://academic.oup.com/cid/article-abstract/49/11/1749/344384

        tn, fp, fn, tp = cm.ravel()
        vme = fn / (fn + tp) if (fn + tp) > 0 else 0
        me = fp / (fp + tn) if (fp + tn) > 0 else 0

        bal_acc_scores.append(score)
        f1_macro_scores.append(f1_macro)
        very_major_errors.append(vme)
        major_errors.append(me)
        fold_times.append(time.process_time() - start)

        # Track best performing fold
        if score > best_score:
            best_score = score
            best_model = current_model
            best_y_test = y_test.copy()
            best_y_pred = y_pred.copy()
            best_fold_idx = fold_idx

        logging.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f}")
        logging.info(f"Very Major Error (VME): {(vme)*100:.2f}% | Major Error (ME): {(me)*100:.2f}%")

    # Generate detailed report for best model
    best_report = classification_report(best_y_test, best_y_pred, digits=4)
    
    # Convert classification report to dictionary for easy access/plotting
    best_report_dict = classification_report(best_y_test, best_y_pred, digits=4, output_dict=True)    

    logging.info(f"{modelname} Nested CV Summary:")
    logging.info(f"\tMean Balanced Accuracy: {np.mean(bal_acc_scores):.3f} ± {np.std(bal_acc_scores):.3f}")
    logging.info(f"\tMean F1 Macro: {np.mean(f1_macro_scores):.3f} ± {np.std(f1_macro_scores):.3f}")
    logging.info(f"\tMean Very Major Error (VME): {np.mean(very_major_errors)*100:.2f}% ± {np.std(very_major_errors)*100:.2f}%")
    logging.info(f"\tMean Major Error (ME): {np.mean(major_errors)*100:.2f}% ± {np.std(major_errors)*100:.2f}%")
    logging.info(f"\tAvg Fold Time: {np.mean(fold_times):.2f}s")
    logging.info(f"Best Model (Fold {best_fold_idx+1}) - Balanced Accuracy: {best_score:.3f}")
    logging.info(f"Best Model Classification Report:\n{best_report}")    

    # Save results if path provided
    if path_out:
        results = {
            'model_name': modelname,
            'best_fold_idx': best_fold_idx,
            'best_score': best_score,
            'best_classification_report': best_report_dict,
            # 'best_model': best_model,
            'balanced_accuracy_scores': bal_acc_scores,
            'f1_macro_scores': f1_macro_scores,
        }
        
        # Save as pickle for easy loading
        with open(f"{path_out}.pkl", 'wb') as f:
            pickle.dump(results, f)    

    return very_major_errors, major_errors


def main():

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    path_root_fna = args.path_dataroot
    path_data_csv = args.path_csv

    # dataset = "neisseria_gonorrhoeae__azithromycin.csv"
    dataset = f"{args.pathogen}/filtered/{args.pathogen}__{args.antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(args.path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")

    now = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    path_out = os.path.join("/home/mila/n/naga-karthik.enamundram/genomicFMs/results", now.split('_')[0], now.split('_')[1])
    os.makedirs(path_out, exist_ok=True)

    # Parameters
    min_kmer_freq = 2   # filters out k-mers that appear less than at least this many times in the dataset
    if args.path_precomputed is not None:
        if not os.path.isfile(args.path_precomputed):
            raise FileNotFoundError(f"Precomputed file not found: {args.path_precomputed}")
        
        # Extract k, feature mode, and max features from the filename
        k = int(args.path_precomputed.split('__')[2].split('_')[1])
        feature_mode = args.path_precomputed.split('__')[3].split('_')[1]
        max_features = int(args.path_precomputed.split('__')[4].split('_')[1].split('.')[0][:-1]) * 1000
    else:        
        k = args.k
        feature_mode = args.feat_mode
        max_features = args.max_features

    patho_short = f"{args.pathogen.split('_')[0][0]}_{args.pathogen.split('_')[1]}"
    fname = f"{patho_short}__{args.antibiotic}__k_{k}__mode_{feature_mode}__feat_{(max_features//1000)}K"
    file_handler = logging.FileHandler(os.path.join(path_out, f"log__{fname}.txt"))
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler]
    )
    if args.path_precomputed is not None:
        logging.info(f"Using precomputed k-mer matrix from {args.path_precomputed}")

    logging.info(f"Training on {args.pathogen} for {args.antibiotic} using k-mers of length {args.k} ...")

    # NOTE: this is a list of dicts pointing for the paths to the genomic sequences (.fna files) and the corresponding S/R labels
    if args.debug:
        dataset_list = load_data(path_data_csv, path_root_fna)[:75]  # Load only first 75 samples for debugging
    else:
        dataset_list = load_data(path_data_csv, path_root_fna)

    logging.info("Initializing K-mer class to build feature matrix ...")
    kmer_features = KmerFeatures(
        k=k,
        mode=feature_mode,                                      # Essential for resistance gene dosage effects
        min_kmer_freq=min_kmer_freq,                            # Remove extremely rare k-mers
        max_features=max_features,                                   # Very wide initial net for diverse mechanisms
        batch_size=256,                                         # Smaller batches for memory efficiency
        use_canonical=True,
        use_sparse=args.use_sparse,                            # Use sparse matrix for k-mer features
        path_precomputed=args.path_precomputed,                 # Path to precomputed k-mer matrix if available
        path_kmers=os.path.join(args.path_csv, f"{patho_short}__{args.antibiotic}__k_{k}.pkl"),
    )
    
    # Build feature matrix
    X, y = kmer_features.build_feature_matrix(dataset=dataset_list)
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Number of samples: {X.shape[0]}")

    # save the matrix to a file
    if args.path_precomputed is None:
        save_fname = os.path.join(args.path_csv, f"{fname}.npz")
        kmer_features.save_matrix(X, y, save_fname)
        logging.info(f"Feature matrix saved to {save_fname}")

    if args.debug:
        # plot UMAP of the feature matrix with different metrics
        # metrics = ["jaccard", "rogerstanimoto", "euclidean", "canberra", "cosine"]
        metrics = ["jaccard", "cosine"]
        for metric in metrics:
            logging.info(f"Plotting UMAP with metric: {metric}")
            start_time = time.time()
            plot_umap(X, y, metric=metric,
                    title=f"UMAP of feature matrix; K: {k}; Mode: {feature_mode}; Metric: {metric}", 
                    save_path=os.path.join(path_out, f"umap__{fname}__{metric}.png"))
            logging.info(f"UMAP plot for {metric} completed in {time.time() - start_time:.2f} seconds")


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
    xgb_params = None
    # {
    #     'learning_rate': [0.1],              # Fix at good default
    #     'max_depth': [3],                 # Most important for genomic data
    #     'subsample': [0.8],            # Key for preventing overfitting
    #     'colsample_bytree': [0.6, 0.8],     # Important for k-mer features
    #     'gamma': [0.1],                  # Simple regularization choice
    # }

    save_fname = f"results_xgb__{fname}"
    path_out = os.path.join(path_out, save_fname)
    vmes, mes = fitmodel_nestedcv(X, y, xgb_model, xgb_params, "xgb", path_out=path_out)

    # # Evaluation
    # print("Evaluating model...")
    # y_pred = model.predict(X_test)
    # print("\nClassification Report:")
    # target_names = ["Susceptible (0)", "Resistant (1)"]
    # print(classification_report(y_test, y_pred, target_names=target_names))



if __name__ == "__main__":
    main()
