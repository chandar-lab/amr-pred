import os
import numpy as np
import time
import logging
import wandb
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, classification_report
)
from features.kmer import KmerFeatures
from features.utils import plot_umap
from data.data_utils import load_data
from data.generate_phylogeny_folds import get_genomic_folds
from training.train_utils import compute_major_errors
from training.wandb_utils import WandbFoldTrainer, init_summary_wandb

# Set up logging
logger = logging.getLogger(__name__)


def fitmodel_nestedcv(X, pheno, cv_splits, estimator, config,use_sweeps=False, sweep_config=None, fixed_run_config=None):
    modelname = config.model.name
    
    logger.info(f"Starting {config.num_folds}-Fold CV for {modelname}...")

    # --- Logic for use_sweeps=True ---
    if use_sweeps:
        api = wandb.Api()
        summary_run = api.run(f"{config.wandb.project}/{fixed_run_config['summary_wandb_id']}")
        summary_run.summary["best_score"] = -1
        summary_run.summary["best_fold_idx"] = -1
        summary_run.update()

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            start = time.process_time()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = pheno[train_idx], pheno[test_idx]

            logger.info(f"Performing grid search for {modelname} in fold {fold_idx+1} using W&B")
            sweep_id = wandb.sweep(sweep_config, project=config.wandb.project)

            # Create the trainer obj and run the sweep. 
            trainer = WandbFoldTrainer(estimator, fixed_run_config, X_train, y_train, fold_idx)
            wandb.agent(sweep_id, function=trainer, count=config.wandb.sweep_runs_per_fold)

            logger.info(f"Best parameters in fold {fold_idx+1}: {trainer.best_params}")
            
            # Use the best hyperparam config to train on full training set and predict on the test set
            logger.info(f"Training on full training set with best parameters...")
            current_model = trainer.best_model
            current_model.fit(X_train, y_train)
            y_pred = current_model.predict(X_test)
            
            wandb.teardown()    # to close the sweep run
            # Update the summary run with the ID of the best performing hyperparam combo
            summary_run.summary[f"fold_runs"].append(trainer.best_id)
            summary_run.update()

            # Log metrics to the best run from the sweep
            fold_wandb_run = api.run(f"{config.wandb.project}/{trainer.best_id}")

            # Compute metrics
            score = balanced_accuracy_score(y_test, y_pred)
            f1_macro = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
            vme, me = compute_major_errors(y_test, y_pred)

            # Log metrics to the fold run
            wandb_run_metrics = {
                "fold_idx": int(fold_idx + 1), "balanced_accuracy": float(score),
                "f1_macro": float(f1_macro), "fold_time": float(time.process_time() - start),
                "very_major_error": float(vme), "major_error": float(me),
                "fda_vme_passed": bool(vme < 0.015), "fda_me_passed": bool(me < 0.03),
            }
            for key, value in wandb_run_metrics.items():
                fold_wandb_run.summary[key] = value
            fold_wandb_run.update()

            if score > summary_run.summary["best_score"]:
                summary_run.summary["best_score"] = score
                summary_run.summary["best_y_test"] = y_test.copy()
                summary_run.summary["best_y_pred"] = y_pred.copy()
                summary_run.summary["best_fold_idx"] = fold_idx+1
                summary_run.update()

            logger.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f} - VME: {vme:.3f} - ME: {me:.3f}")
    
    # --- Logic for use_sweeps=False ---
    else:
        bal_acc_scores, f1_macro_scores, vme_scores, me_scores, fold_times = [], [], [], [], []
        best_score = -1
        best_report_dict = {}
        best_report_str = ""
        
        # Train with fixed parameters across folds, and then log the results
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            start = time.process_time()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = pheno[train_idx], pheno[test_idx]

            logger.info(f"Training {modelname} in fold {fold_idx+1} with fixed/default parameters.")
            current_model = clone(estimator)
            current_model.fit(X_train, y_train)
            y_pred = current_model.predict(X_test)
            
            score = balanced_accuracy_score(y_test, y_pred)
            bal_acc_scores.append(score)
            f1_macro_scores.append(classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score'])
            vme, me = compute_major_errors(y_test, y_pred)
            vme_scores.append(vme)
            me_scores.append(me)
            fold_times.append(time.process_time() - start)

            if score > best_score:
                best_score = score
                best_report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
                best_report_str = classification_report(y_test, y_pred, digits=4)

            logger.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f} - VME: {vme:.3f} - ME: {me:.3f}")

        # After all folds, create one summary run and log the results
        logger.info("CV finished. Logging summary results to W&B...")
        patho_short = fixed_run_config['pathogen_short']
        summary_run_name = f"summary__{patho_short}__{config.dataset.antibiotic}__k_{config.matrix.k}__mode_{config.matrix.feat_mode}__feat_{config.matrix.max_features//1000}K__{modelname}"
        
        with wandb.init(
            project=config.wandb.project, name=summary_run_name, 
            config=fixed_run_config, tags=["summary", "fixed_params", "baseline"]
        ) as summary_run:
            
            mean_me, std_me = np.mean(me_scores), np.std(me_scores)
            mean_vme, std_vme = np.mean(vme_scores), np.std(vme_scores)
            mean_bal_acc, std_bal_acc = np.mean(bal_acc_scores), np.std(bal_acc_scores)
            mean_f1_macro, std_f1_macro = np.mean(f1_macro_scores), np.std(f1_macro_scores)

            wandb_tracked_values = {
                'model_name': modelname,
                "mean_fold_time": np.mean(fold_times),
                "mean_major_error": mean_me, "std_major_error": std_me,
                "mean_very_major_error": mean_vme, "std_very_major_error": std_vme,
                "mean_balanced_accuracy": mean_bal_acc, "std_balanced_accuracy": std_bal_acc,
                "mean_f1_macro": mean_f1_macro, "std_f1_macro": std_f1_macro,
                "fda_me_passed": bool(mean_me < 0.03), "fda_vme_passed": bool(mean_vme < 0.015),
                # "best_fold_accuracy": best_score,
                # "best_classification_report": best_report_str
            }
            summary_run.log(wandb_tracked_values)

            return wandb_tracked_values, best_report_dict

    # --- Summary for use_sweeps=True ---
    api = wandb.Api()
    summary_run = api.run(f"{config.wandb.project}/{fixed_run_config['summary_wandb_id']}")
    bal_acc_scores, f1_macro_scores, vme_scores, me_scores, fold_times = [], [], [], [], []

    for run_id in summary_run.summary.get("fold_runs", []):
        run = api.run(f"{config.wandb.project}/{run_id}")
        bal_acc_scores.append(run.summary.get("balanced_accuracy", 0))
        f1_macro_scores.append(run.summary.get("f1_macro", 0))
        vme_scores.append(run.summary.get("very_major_error", 0))
        me_scores.append(run.summary.get("major_error", 0))
        fold_times.append(run.summary.get("fold_time", 0))

    best_report_dict = classification_report(
        summary_run.summary["best_y_test"], 
        summary_run.summary["best_y_pred"], 
        digits=4, output_dict=True
    )

    mean_me, std_me = np.mean(me_scores), np.std(me_scores)
    mean_vme, std_vme = np.mean(vme_scores), np.std(vme_scores)
    mean_bal_acc, std_bal_acc = np.mean(bal_acc_scores), np.std(bal_acc_scores)
    mean_f1_macro, std_f1_macro = np.mean(f1_macro_scores), np.std(f1_macro_scores)

    wandb_tracked_values = {
        'model_name': modelname, "mean_fold_time": np.mean(fold_times),
        "mean_major_error": mean_me, "std_major_error": std_me,
        "mean_very_major_error": mean_vme, "std_very_major_error": std_vme,
        "mean_balanced_accuracy": mean_bal_acc, "std_balanced_accuracy": std_bal_acc,
        "mean_f1_macro": mean_f1_macro, "std_f1_macro": std_f1_macro,
        "fda_me_passed": bool(mean_me < 0.03), "fda_vme_passed": bool(mean_vme < 0.015),
    }

    for key, value in wandb_tracked_values.items():
        summary_run.summary[key] = value
    summary_run.update()

    return wandb_tracked_values, best_report_dict


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
    k = cfg.matrix.k
    feature_mode = cfg.matrix.feat_mode
    max_features = cfg.matrix.max_features

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
    path_species = os.path.join(cfg.path_out_cache, pathogen)
    kmer_features = KmerFeatures(
        k=k,
        mode=feature_mode,                                      # Essential for resistance gene dosage effects
        min_kmer_freq=min_kmer_freq,                            # Remove extremely rare k-mers
        max_features=max_features,                                   # Very wide initial net for diverse mechanisms
        batch_size=256,                                         # Smaller batches for memory efficiency
        use_canonical=True,
        use_sparse=cfg.dataset.use_sparse,                            # Use sparse matrix for k-mer features
        path_full_matrix=os.path.join(path_species, "feature_matrices", f"{patho_short}__{antibiotic}__k_{k}__mode_{feature_mode}__feat_5M.npz"),
        path_kmers=os.path.join(path_species, "unique_kmers", f"{patho_short}__{antibiotic}__k_{k}.pkl"),
    )
    
    if cfg.build_only:
        logger.info("Only building the feature matrix ...")
        # Build the feature matrix and save it
        kmer_features.build_feature_matrix(dataset=dataset_list)
        # logger.info(f"Feature matrix saved at {kmer_features.path_full_matrix}")
        return
    
    else:
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


        # --- UPDATED: W&B and Sweep Configuration ---
        fixed_config = {}
        sweep_config = None
        
        # Determine if we are running sweeps based on hydra config
        # use_sweeps = "sweep" in cfg.model and cfg.model.sweep is not None
        use_sweeps = True if cfg.wandb.use_sweeps else False

        if use_sweeps:
            logger.info("Mode: Hyperparameter sweep using W&B.")
            # Create a summary run on WandB to track sweep results across folds
            fixed_config = init_summary_wandb(cfg)
            
            # Convert hydra sweep config to a wandb-compatible dict
            sweep_config = OmegaConf.to_container(cfg.model.sweep, resolve=True)
            
            # Add a descriptive name for the sweep entity on the W&B dashboard
            patho_short = fixed_config['pathogen_short']
            sweep_name = f"SWEEP__{patho_short}__{cfg.dataset.antibiotic}__id_{fixed_config['id']}"
            sweep_config['name'] = sweep_name

        else:
            logger.info("Mode: Training with fixed/default parameters.")
            # For non-sweep runs, we'll create the config dict directly
            patho_short = f"{cfg.dataset.pathogen.split('_')[0][0]}_{cfg.dataset.pathogen.split('_')[1]}"
            fixed_config = {
                "pathogen": cfg.dataset.pathogen, "antibiotic": cfg.dataset.antibiotic,
                "pathogen_short": patho_short, "k": cfg.matrix.k,
                "feat_mode": cfg.matrix.feat_mode, "max_features": f"{cfg.matrix.max_features//1000}K",
                "use_sparse": cfg.dataset.use_sparse,
            }
        
        # --- Setup train/test splits ---
        cv_splits = []
        if cfg.use_genomic_splits == True:
            logger.info("Using phylogeny-based splits for training and testing.")
            # Get the dictionary for the split indices
            genomic_dict = get_genomic_folds(
                pathogen=cfg.dataset.pathogen, antibiotic=cfg.dataset.antibiotic,
                path_dataroot=cfg.dataset.path_dataroot, path_csv=cfg.dataset.path_csv,
                num_folds=cfg.num_folds, dist_method=cfg.split_method,
            )
            # Use the dict to determine the split indices
            split_indices = []
            for assembly in dataset_list:
                if assembly['assembly'] in genomic_dict:
                    split_indices.append(genomic_dict[assembly['assembly']])
                else:
                    raise ValueError(f"Assembly {assembly['assembly']} not found in genomic_dict. Please check the dataset.")
            # Create the train/test splits based on the indices
            for fold_idx in range(cfg.num_folds):
                train_idx = [i for i, split in enumerate(split_indices) if split != fold_idx]
                test_idx = [i for i, split in enumerate(split_indices) if split == fold_idx]
                cv_splits.append((train_idx, test_idx))
        else:
            logger.info("Using random splits for training and testing.")
            # Use StratifiedKFold to create the split indices
            skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)
            for train_idx, test_idx in skf.split(X, y):
                cv_splits.append((train_idx, test_idx))
                

        # --- Train Model ---
        model = hydra.utils.instantiate(cfg.model)
        logger.info(f"Training {model.__class__.__name__} model...")
        results_dict, best_classification_report = fitmodel_nestedcv(
            X, y, cv_splits, model, config=cfg, 
            use_sweeps=use_sweeps, 
            sweep_config=sweep_config,
            fixed_run_config=fixed_config
        )

        # Save the results to a CSV file
        save_fname = f"results_{cfg.model.name}__{fname}"
        path_out = os.path.join(output_dir, save_fname)
        logger.info(f"Results saved to {path_out} ...")
        results_df = pd.DataFrame([results_dict])
        results_df.to_csv(f"{path_out}.csv", index=False)

        

if __name__ == "__main__":
    main()
