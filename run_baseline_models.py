import os
import numpy as np
import argparse
import time
import logging
import xgboost as xgb
import time
import pickle
import wandb
import hydra
from omegaconf import DictConfig

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

logger = logging.getLogger(__name__)

# Global variable to store the fixed parameters for the sweep and summary runs
fixed_config = {}


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

    logger.info(f"Starting CV for {modelname}...")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, pheno)):
        if prescribed_fold is not None and fold_idx != prescribed_fold:
            continue # Skip to the next fold if not the prescribed one
        start = time.process_time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pheno[train_idx], pheno[test_idx]

        if wandb_config is None:
            # If no parameters are provided, use the estimator as is
            logger.info(f"Using default parameters for {modelname} in fold {fold_idx+1}")
            
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
            logger.info(f"Performing grid search for {modelname} in fold {fold_idx+1} using weights and biases")
            wandb_config['name'] = f"{fixed_config['pathogen']}-{fixed_config['antibiotic']}__id_{fixed_config['id']}__fold_{fold_idx+1}__k{fixed_config['k']}__mode_{fixed_config['feat_mode']}__feat_{(fixed_config['max_features']//1000)}K"
            sweep_id = wandb.sweep(wandb_config, project="amr-prediction")

            # Create the trainer obj and run the sweep. 
            trainer = WandbFoldTrainer(estimator, fixed_config, X_train, y_train, fold_idx)
            wandb.agent(sweep_id, function=trainer, count=RUNS_PER_FOLD)

            # Output notification of best model found
            logger.info(f"Best parameters in fold {fold_idx+1}: {trainer.best_params}")
            logger.info(f"Training on full training set with best parameters...")

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

        logger.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f}")
        logger.info(f"Very Major Error (VME): {(vme)*100:.2f}% | Major Error (ME): {(me)*100:.2f}%")


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

    logger.info(f"{modelname} CV Summary (also available on wandb for id {summary_run.id}):")
    logger.info(f"\tMean Balanced Accuracy: {np.mean(bal_acc_scores):.3f} ± {np.std(bal_acc_scores):.3f}")
    logger.info(f"\tMean F1 Macro: {np.mean(f1_macro_scores):.3f} ± {np.std(f1_macro_scores):.3f}")
    logger.info(f"\tMean Very Major Error (VME): {np.mean(very_major_errors)*100:.2f}% ± {np.std(very_major_errors)*100:.2f}%")
    logger.info(f"\tMean Major Error (ME): {np.mean(major_errors)*100:.2f}% ± {np.std(major_errors)*100:.2f}%")
    logger.info(f"\tAvg Fold Time: {np.mean(fold_times):.2f}s")
    logger.info(f"Best Model (Fold {summary_run.summary['best_fold_idx']}) - Balanced Accuracy: {summary_run.summary['best_score']:.3f}")
    logger.info(f"Best Model Classification Report:\n{best_report}")

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


# config_path is a directory relative to this file (run_baseline_models.py)
@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    path_root_fna = cfg.dataset.path_dataroot
    path_data_csv = cfg.dataset.path_csv
    pathogen = cfg.dataset.pathogen
    antibiotic = cfg.dataset.antibiotic

    dataset = f"{pathogen}__{antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(cfg.dataset.path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")

    # Parameters
    min_kmer_freq = 2   # filters out extremely rare k-mers that don't appear at least twice
    k = cfg.k
    feature_mode = cfg.feat_mode
    max_features = cfg.max_features

    patho_short = f"{pathogen.split('_')[0][0]}_{pathogen.split('_')[1]}"
    fname = f"{patho_short}__{antibiotic}__k_{k}__mode_{feature_mode}__feat_{(max_features//1000)}K"

    logger.info(f"Training on {pathogen} for {antibiotic} using k-mers of length {k} ...")

    # NOTE: this is a list of dicts pointing for the paths to the genomic sequences (.fna files) and the corresponding S/R labels
    if cfg.debug:
        dataset_list = load_data(path_data_csv, path_root_fna)[:75]  # Load only first 75 samples for debugging
    else:
        dataset_list = load_data(path_data_csv, path_root_fna)
        logger.info(f"Number of samples in the dataset: {len(dataset_list)}")

    logger.info("Initializing K-mer class to build feature matrix ...")
    kmer_features = KmerFeatures(
        k=k,
        mode=feature_mode,                                      # Essential for resistance gene dosage effects
        min_kmer_freq=min_kmer_freq,                            # Remove extremely rare k-mers
        max_features=max_features,                                   # Very wide initial net for diverse mechanisms
        batch_size=256,                                         # Smaller batches for memory efficiency
        use_canonical=True,
        use_sparse=cfg.dataset.use_sparse,                            # Use sparse matrix for k-mer features
        path_full_matrix=os.path.join(cfg.dataset.path_csv, f"{patho_short}__{antibiotic}__k_{k}__mode_{feature_mode}__feat_10M.npz"),
        path_kmers=os.path.join(cfg.dataset.path_csv, f"{patho_short}__{antibiotic}__k_{k}.pkl"),
    )
    
    # Build feature matrix
    X, y = kmer_features.build_feature_matrix(dataset=dataset_list)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of samples: {X.shape[0]}")

    if cfg.debug:
        # plot UMAP of the feature matrix with different metrics
        # metrics = ["jaccard", "rogerstanimoto", "euclidean", "canberra", "cosine"]
        metrics = ["jaccard", "cosine"]
        for metric in metrics:
            logger.info(f"Plotting UMAP with metric: {metric}")
            start_time = time.time()
            plot_umap(X, y, metric=metric,
                    title=f"UMAP of feature matrix; K: {k}; Mode: {feature_mode}; Metric: {metric}", 
                    save_path=os.path.join(output_dir, f"umap__{fname}__{metric}.png"))
            logger.info(f"UMAP plot for {metric} completed in {time.time() - start_time:.2f} seconds")


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
    init_summary_wandb(cfg, config=fixed_config)

    # Train XGBoost
    # # print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        # verbosity=1
    )

    save_fname = f"results_xgb__{fname}"
    output_dir = os.path.join(output_dir, save_fname)
    vmes, mes = fitmodel_nestedcv(X, y, xgb_model, wandb_config=None, modelname="xgb", path_out=output_dir)#, prescribed_fold=args.fold_idx)



if __name__ == "__main__":
    main()
