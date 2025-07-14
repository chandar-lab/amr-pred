import os
import numpy as np
import argparse
import time
import logging
import xgboost as xgb
from datetime import datetime
import time
import pickle
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix
)
from features.kmer import KmerFeatures
from features.utils import plot_umap
from data.data_utils import load_data, LABEL_MAPPING
from training.train_utils import compute_major_errors
from training.wandb_utils import WandbFoldTrainer, init_summary_wandb


SEED = 42
RUNS_PER_FOLD = 50 # Number of runs per fold for hyperparameter tuning, must be fixed
NUM_FOLDS = 5  # Number of folds for cross-validation

# Global variable to store the fixed parameters for the sweep and summary runs
fixed_config = {}



def get_parser():
    
    parser = argparse.ArgumentParser(description="Train XGBoost model on k-mer features")
    parser.add_argument("--path-dataroot", type=str, help="Root path of the dataset genomic sequences for all species")
    parser.add_argument("--path-csv", type=str, help="Path to the folder containing CSV files for each specie-drug combination")
    parser.add_argument("--pathogen", type=str, help="Pathogen name (lowercase, separated by underscores) e.g. neisseria_gonorrhoeae")
    parser.add_argument("--antibiotic", type=str, help="Antibiotic name (lowercase) e.g. azithromycin")
    parser.add_argument("--path-precomputed", type=str, default=None, help="Path to precomputed k-mer matrix (optional)")
    parser.add_argument("--debug",  action='store_true', help="Enable debug mode for verbose logging")
    parser.add_argument("--k", type=int, default=8, help="Length of k-mers to extract (default: 8)")
    parser.add_argument("--feat-mode", type=str, default="count", choices=["binary", "count", "entropy"], help="Mode for k-mer extraction (default: count)")
    parser.add_argument("--max-features", type=int, default=1000000, help="Maximum number of top most frequent k-mers to extract (default: 1M)")
    parser.add_argument("--use-sparse", action='store_true', help="Use sparse matrix for k-mer features (default: False)")
    # Args for WANDB array jobs, these are unused for the time being
    # parser.add_argument("--fold-idx", type=int, default=None, help="Fold index for cross-validation. This will run only the specified fold, for parallel execution. Default is None, which runs all folds.")
    # parser.add_argument("--summary_wandb_id", type=str, default=None, help="ID of the summary wandb run to log results to. If not provided, a new summary run will be created. If specifying fold_idx, this value is required.")

    return parser


def fitmodel_nestedcv(X, pheno, estimator, wandb_config, modelname, path_out=None, prescribed_fold=None):
    api = wandb.Api()
    # Open the summary run to log results
    summary_run = api.run(f"amr-prediction/{fixed_config['summary_wandb_id']}")

    outer_cv = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Set the best score and fold idx in the wandb summary run
    summary_run.summary["best_score"] = -1
    summary_run.summary["best_fold_idx"] = -1
    summary_run.summary["best_y_pred"] = None
    summary_run.summary["best_y_test"] = None
    summary_run.update()

    logging.info(f"Starting CV for {modelname}...")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, pheno)):
        if prescribed_fold is not None and fold_idx != prescribed_fold:
            continue # Skip to the next fold if not the prescribed one
        start = time.process_time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pheno[train_idx], pheno[test_idx]

        if wandb_config is None:
            # If no parameters are provided, use the estimator as is
            logging.info(f"Using default parameters for {modelname} in fold {fold_idx+1}")
            
            current_model = clone(estimator)
            # Create a wandb run for this fold
            fold_wandb_run = wandb.init(
                project="amr-prediction",
                name=f"{fixed_config['pathogen_short']}__{fixed_config['antibiotic']}__id_{fixed_config['id']}__k_{fixed_config['k']}__mode_{fixed_config['feat_mode']}__feat_{fixed_config['max_features']}__fold_{fold_idx+1}",
                config=fixed_config,
                tags=["fold", "fixed", "baseline"]
            )
            current_model.fit(X_train, y_train)
            y_pred = current_model.predict(X_test)
            fold_wandb_run.finish()

            # Update the summary run with the wandb run ID
            summary_run.summary[f"fold_runs"].append(fold_wandb_run.id)
            summary_run.update()

            # Re-open the wandb run for this fold to log metrics below
            fold_wandb_run = api.run(f"amr-prediction/{fold_wandb_run.id}")
        else:
            # Use weights and biases for hyperparameter tuning
            logging.info(f"Performing grid search for {modelname} in fold {fold_idx+1} using weights and biases")
            wandb_config['name'] = f"{fixed_config['pathogen']}-{fixed_config['antibiotic']}__id_{fixed_config['id']}__fold_{fold_idx+1}__k{fixed_config['k']}__mode_{fixed_config['feat_mode']}__feat_{(fixed_config['max_features']//1000)}K"
            sweep_id = wandb.sweep(wandb_config, project="amr-prediction")

            # Create the trainer obj and run the sweep. 
            trainer = WandbFoldTrainer(estimator, fixed_config, X_train, y_train, fold_idx)
            wandb.agent(sweep_id, function=trainer, count=RUNS_PER_FOLD)

            # Output notification of best model found
            logging.info(f"Best parameters in fold {fold_idx+1}: {trainer.best_params}")
            logging.info(f"Training on full training set with best parameters...")

            # Use the best setting to predict on the test set
            current_model = trainer.best_model
            current_model.fit(X_train, y_train) # Fit the best model on the full training set
            y_pred = current_model.predict(X_test)
            
            # Add the sweep ID to the summary run
            wandb.teardown() # This will close the sweep run
            # Update the summary run with the ID of the best performing hyperparam combo
            summary_run.summary[f"fold_runs"].append(trainer.best_id)
            summary_run.update()

            # Log metrics to the best run from the sweep
            fold_wandb_run = api.run(f"amr-prediction/{trainer.best_id}")

        # Evaluate the model
        score = balanced_accuracy_score(y_test, y_pred)
        f1_macro = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
        vme, me = compute_major_errors(y_test, y_pred)

        # Log to the wandb run
        wandb_run_metrics = {
            "fold_idx": int(fold_idx + 1),
            "balanced_accuracy": float(score),
            "f1_macro": float(f1_macro),
            "fold_time": float(time.process_time() - start),
            "very_major_error": float(vme),
            "major_error": float(me),
            "fda_vme_passed": bool(vme < 0.015),
            "fda_me_passed": bool(me < 0.03),
            # "confusion_matrix": wandb.plot.confusion_matrix(
            #     probs=None, y_true=y_test, preds=y_pred, class_names=["Susceptible", "Resistant"]
            # ), # These kinds of artifacts cannot be added through the api, so they've been removed. They can be added within the OG run.
            # "predictions": wandb.Table(data=[[y_test[i], y_pred[i]] for i in range(len(y_test))], columns=["True Label", "Predicted Label"]),
        }
        for key, value in wandb_run_metrics.items():
            fold_wandb_run.summary[key] = value
        
        fold_wandb_run.update()

        # Track best performing fold
        if score > summary_run.summary["best_score"]:
            summary_run.summary["best_score"] = score
            summary_run.summary["best_y_test"] = y_test.copy()
            summary_run.summary["best_y_pred"] = y_pred.copy()
            summary_run.summary["best_fold_idx"] = fold_idx+1
            summary_run.update()

        logging.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f}")
        logging.info(f"Very Major Error (VME): {(vme)*100:.2f}% | Major Error (ME): {(me)*100:.2f}%")


    ### SUMMARY RESULTS ACROSS FOLDS ###
    bal_acc_scores, f1_macro_scores = [], []
    very_major_errors, major_errors = [], []
    fold_times = []

    # Get the ids of all the fold runs from the summary run
    runs = summary_run.summary.get("fold_runs", [])
    
    for run_id in runs:
        run = api.run(f"amr-prediction/{run_id}")
        bal_acc_scores.append(run.summary.get("balanced_accuracy", 0))
        f1_macro_scores.append(run.summary.get("f1_macro", 0))
        very_major_errors.append(run.summary.get("very_major_error", 0))
        major_errors.append(run.summary.get("major_error", 0))
        fold_times.append(run.summary.get("fold_time", 0))

    # Generate detailed report for best model
    best_report = classification_report(summary_run.summary["best_y_test"], summary_run.summary["best_y_pred"], digits=4)

    # Convert classification report to dictionary for easy access/plotting
    best_report_dict = classification_report(summary_run.summary["best_y_test"], summary_run.summary["best_y_pred"], digits=4, output_dict=True)

    logging.info(f"{modelname} CV Summary (also available on wandb for id {summary_run.id}):")
    logging.info(f"\tMean Balanced Accuracy: {np.mean(bal_acc_scores):.3f} ± {np.std(bal_acc_scores):.3f}")
    logging.info(f"\tMean F1 Macro: {np.mean(f1_macro_scores):.3f} ± {np.std(f1_macro_scores):.3f}")
    logging.info(f"\tMean Very Major Error (VME): {np.mean(very_major_errors)*100:.2f}% ± {np.std(very_major_errors)*100:.2f}%")
    logging.info(f"\tMean Major Error (ME): {np.mean(major_errors)*100:.2f}% ± {np.std(major_errors)*100:.2f}%")
    logging.info(f"\tAvg Fold Time: {np.mean(fold_times):.2f}s")
    logging.info(f"Best Model (Fold {summary_run.summary['best_fold_idx']}) - Balanced Accuracy: {summary_run.summary['best_score']:.3f}")
    logging.info(f"Best Model Classification Report:\n{best_report}")

    wandb_tracked_values = {
        "mean_major_error": f"{np.mean(major_errors):.3f}", 
        "std_major_error": f"{np.std(major_errors):.3f}", 
        "mean_very_major_error": f"{np.mean(very_major_errors):.3f}", 
        "std_very_major_error": f"{np.std(very_major_errors):.3f}", 
        "mean_balanced_accuracy": f"{np.mean(bal_acc_scores):.3f}", 
        "std_balanced_accuracy": f"{np.std(bal_acc_scores):.3f}", 
        "mean_f1_macro": f"{np.mean(f1_macro_scores):.3f}", 
        "std_f1_macro": f"{np.std(f1_macro_scores):.3f}",
        "fda_me_passed": True if (np.mean(major_errors) < 0.03) else False,
        "fda_vme_passed": True if (np.mean(very_major_errors) < 0.015) else False
    }

    # Log each summary metric to the summary run
    for key, value in wandb_tracked_values.items():
        summary_run.summary[key] = value
    
    summary_run.update()

    # Save results if path provided
    if path_out:
        results = {
            'model_name': modelname,
            'best_fold_idx': summary_run.summary['best_fold_idx'],
            'best_score': summary_run.summary['best_score'],
            'best_classification_report': best_report_dict,
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
    dataset = f"{args.pathogen}__{args.antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(args.path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")

    now = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    path_out = os.path.join("/home/mila/n/naga-karthik.enamundram/genomicFMs/results", now.split('_')[0], now.split('_')[1])
    os.makedirs(path_out, exist_ok=True)

    # Parameters
    min_kmer_freq = 2   # filters out extremely rare k-mers that don't appear at least twice
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
        path_full_matrix=os.path.join(args.path_csv, f"{patho_short}__{args.antibiotic}__k_{k}__mode_{feature_mode}__feat_10M.npz"),
        path_kmers=os.path.join(args.path_csv, f"{patho_short}__{args.antibiotic}__k_{k}.pkl"),
    )
    
    # Build feature matrix
    X, y = kmer_features.build_feature_matrix(dataset=dataset_list)
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Number of samples: {X.shape[0]}")

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

    # # Initialize wandb
    # # Open the sweep config from the yaml, defining sweep hyperparams and ranges
    # with open("./amr-pred/XGB-sweep.yaml", 'r') as config_file:
    #     sweep_config = yaml.safe_load(config_file)
    
    # Initialize the summary run to collect results from the sweeps across folds
    init_summary_wandb(args, config=fixed_config)

    # Train XGBoost
    # # print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        # verbosity=1
    )

    save_fname = f"results_xgb__{fname}"
    path_out = os.path.join(path_out, save_fname)
    vmes, mes = fitmodel_nestedcv(X, y, xgb_model, wandb_config=None, modelname="xgb", path_out=path_out)#, prescribed_fold=args.fold_idx)



if __name__ == "__main__":
    main()
