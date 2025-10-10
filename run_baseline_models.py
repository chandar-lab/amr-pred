import os
import time
import logging
import joblib
import wandb
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import scipy.sparse as sp_sparse
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, classification_report
)
from sklearn.feature_extraction.text import TfidfTransformer
from features.kmer import KmerFeatures
from features.utils import plot_umap
from data.data_utils import load_data
from data.generate_phylogeny_folds import get_genomic_folds
from training.train_utils import compute_major_errors
from training.wandb_utils import WandbFoldTrainer

# Set up logging
logger = logging.getLogger(__name__)


def fitmodel_nestedcv(X, pheno, cv_splits, estimator, config,use_sweeps=False, sweep_config=None, fixed_run_config=None):
    modelname = config.model.name
    
    logger.info(f"Starting {config.num_folds}-Fold CV for {modelname}...")

    fold_results = []
    predictions_per_fold = []            # <-- store y_true / y_pred per fold
    best_overall_score = -1
    best_classification_report = ""

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        start = time.process_time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pheno[train_idx], pheno[test_idx]
        
        current_model = clone(estimator)

        if use_sweeps:
            logger.info(f"Running sweep for fold {fold_idx+1}...")
            sweep_id = wandb.sweep(sweep_config, project=config.wandb.project)
            trainer = WandbFoldTrainer(current_model, fixed_run_config, X_train, y_train, fold_idx)
            wandb.agent(sweep_id, function=trainer, count=config.wandb.sweep_runs_per_fold)
            
            logger.info(f"Best parameters in fold {fold_idx+1}: {trainer.best_params}")
            current_model = trainer.best_model # Get the best model from the sweep
            # Clean up wandb state for the next fold's sweep
            wandb.teardown()
        else:
             logger.info(f"Training {modelname} in fold {fold_idx+1} with fixed parameters.")

        # Fit the best model (from sweep) or the standard model on the full training set for the fold
        current_model.fit(X_train, y_train)
        y_pred = current_model.predict(X_test)
        
        # Evaluate
        score = balanced_accuracy_score(y_test, y_pred)
        f1_macro = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
        vme, me = compute_major_errors(y_test, y_pred)

        # Save per-fold predictions and true labels (convert to python types for serialization)
        def _to_pylist(arr):
            try:
                return [a.item() if hasattr(a, "item") else a for a in list(arr)]
            except Exception:
                return list(arr)

        predictions_per_fold.append({
            "fold": fold_idx,
            "indices": list(test_idx),
            "y_true": _to_pylist(y_test),
            "y_pred": _to_pylist(y_pred)
        })

        fold_results.append({
            "model_name": modelname, "pathogen": fixed_run_config['pathogen'],
            "pathogen_short": fixed_run_config['pathogen_short'],
            "antibiotic": fixed_run_config['antibiotic'], 
            "k": fixed_run_config['k'], "feat_mode": fixed_run_config['feat_mode'],
            "max_features": fixed_run_config['max_features'],
            "seed": fixed_run_config['seed'], 
            "split_method": fixed_run_config['split_method'],
            "clustering": fixed_run_config['clustering'],
            "fold": fold_idx,
            "balanced_accuracy": score,
            "f1_macro": f1_macro,
            "very_major_error": vme,
            "major_error": me,
            "fold_time": time.process_time() - start
        })
        
        if score > best_overall_score:
            best_overall_score = score
            best_classification_report = classification_report(y_test, y_pred, digits=4)
            best_model = current_model

        logger.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f} - VME: {vme:.3f} - ME: {me:.3f} - Time: {time.process_time() - start:.2f}s")

    # --- After all folds, create a summary and log to the main wandb run ---
    logger.info("CV finished. Logging summary results to W&B...")
    
    # Create a pandas DataFrame for easy calculation and table logging
    results_df = pd.DataFrame(fold_results)
    
    # Calculate and log summary statistics
    summary_stats = {
        "mean_balanced_accuracy": results_df['balanced_accuracy'].mean(),
        "std_balanced_accuracy": results_df['balanced_accuracy'].std(),
        "mean_f1_macro": results_df['f1_macro'].mean(),
        "std_f1_macro": results_df['f1_macro'].std(),
        "mean_very_major_error": results_df['very_major_error'].mean(),
        "mean_major_error": results_df['major_error'].mean(),
        "mean_fold_time": results_df['fold_time'].mean(),
        "fda_me_passed": bool(results_df['major_error'].mean() < 0.03),
        "fda_vme_passed": bool(results_df['very_major_error'].mean() < 0.015),
        # "best_classification_report": best_classification_report,
        # "best_fold_accuracy": best_overall_score
    }
    
    return summary_stats, best_classification_report, results_df, best_model, predictions_per_fold


# config_path is a directory relative to this file (run_baseline_models.py)
@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    path_root_fna = cfg.dataset.path_dataroot
    path_data_csv = cfg.dataset.path_csv
    pathogen = cfg.dataset.pathogen
    antibiotic = cfg.dataset.antibiotic
    split = "phylo" if cfg.use_genomic_splits else "random"

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
    fname = f"{patho_short}__{antibiotic}__k_{k}__mode_{feature_mode}__split_{split}__feat_{(max_features//1000)}K"

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
        path_full_matrix=os.path.join(path_species, "feature_matrices", f"{patho_short}__{antibiotic}__k_{k}__mode_{feature_mode}__feat_10M.npz"),
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
        # NOTE: when using the logistic classifier, normalize the feature matrix using TF-IDF 
        # because LGR is sensitive to feature scaling
        if cfg.model.name == "LGR":
            logger.info(f"Normalizing feature matrix using TF-IDF for {cfg.model.name} model ...")
            tfidf_transformer = TfidfTransformer(use_idf=True, norm='l2')
            X = tfidf_transformer.fit_transform(X)
            # check if X is sparse
            if sp_sparse.issparse(X):
                logger.info("Feature matrix is sparse after TF-IDF transformation.")
                logger.info("Converting to dense array ...")
                X = X.toarray()

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

        
        # Determine if we are running sweeps based on hydra config
        use_sweeps = True if cfg.wandb.use_sweeps else False
        patho_short = f"{cfg.dataset.pathogen.split('_')[0][0]}_{cfg.dataset.pathogen.split('_')[1]}"
            
        # Define fixed parameters for the run
        fixed_config = {
            "model_name": cfg.model.name,
            "pathogen": cfg.dataset.pathogen, "antibiotic": cfg.dataset.antibiotic,
            "pathogen_short": patho_short, "k": cfg.matrix.k,
            "feat_mode": cfg.matrix.feat_mode, "max_features": f"{cfg.matrix.max_features//1000}K",
            "use_sparse": cfg.dataset.use_sparse, "seed": cfg.seed, 
            "split_method": split, "clustering": cfg.clustering_method,
        }
        
        sweep_config = None
        if use_sweeps:
            sweep_config = OmegaConf.to_container(cfg.model.sweep, resolve=True)
            sweep_name = f"SWEEP__{cfg.model.name}__{patho_short}__{antibiotic}__k_{k}__{split}"
            sweep_config['name'] = sweep_name

        
        # --- Setup train/test splits ---
        cv_splits = []
        if cfg.use_genomic_splits == True:
            logger.info(f"Using phylogeny-based splits for training and testing with {cfg.clustering_method} folds.")
            # Get the dictionary for the split indices
            genomic_dict, _ = get_genomic_folds(logger=logger,
                pathogen=cfg.dataset.pathogen, antibiotic=cfg.dataset.antibiotic,
                path_dataroot=cfg.dataset.path_dataroot, path_csv=cfg.dataset.path_csv,
                path_out=os.path.join(path_species, "phylogeny"),
                num_folds=cfg.num_folds, dist_method=cfg.split_method, 
                clustering_method=cfg.clustering_method,
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
        model = hydra.utils.instantiate(cfg.model.params)

        logger.info(f"Training {model.__class__.__name__} model...")
        summary_stats, best_classification_report, fold_results_df, best_model, predictions = fitmodel_nestedcv(
            X, y, cv_splits, model, config=cfg, 
            use_sweeps=use_sweeps, 
            sweep_config=sweep_config,
            fixed_run_config=fixed_config
        )
        # ensure target dirs exist
        os.makedirs(os.path.join(path_species, "models"), exist_ok=True)
        os.makedirs(os.path.join(path_species, "results"), exist_ok=True)
        
        # save the best model
        save_fname_model = f"model_{cfg.model.name}__{fname}.pkl"
        path_out_model = os.path.join(path_species, "models", save_fname_model)
        logger.info(f"Saving the best model to {path_out_model} ...")
        joblib.dump(best_model, path_out_model)

        # initialize a single wandb run for the whole experiment 
        run_name = f"{cfg.model.name}__{fname}"
        with wandb.init(
            project=cfg.wandb.project, name=run_name, 
            config=fixed_config, tags=["summary", "baseline", "cv"]
            ) as run:

            logger.info("Logging summary results to W&B...")
            run.log(summary_stats)
            run.log({"fold_results": wandb.Table(dataframe=fold_results_df)})

            # Save the results to a CSV file
            save_fname = f"results_{cfg.model.name}__{fname}"
            path_out = os.path.join(path_species, "results", save_fname)
            logger.info(f"Results saved to {path_out} ...")
            # results_df = pd.DataFrame(results)
            fold_results_df.to_csv(f"{path_out}.csv", index=False)

            # Also save per-fold predictions (y_true, y_pred) so downstream metrics can be computed easily
            preds_rows = []
            for p in predictions:
                for idx, y_t, y_p in zip(p["indices"], p["y_true"], p["y_pred"]):
                    preds_rows.append({"fold": p["fold"], "sample_index": idx, "y_true": y_t, "y_pred": y_p})
            preds_df = pd.DataFrame(preds_rows)
            preds_path = os.path.join(path_species, "results", f"preds_{cfg.model.name}__{fname}.csv")
            preds_df.to_csv(preds_path, index=False)
            logger.info(f"Per-fold predictions saved to {preds_path}")

        

if __name__ == "__main__":
    main()
