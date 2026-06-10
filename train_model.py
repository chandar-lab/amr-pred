import os
import time
import logging
import random
import joblib
import wandb
import gc
import hydra
import pandas as pd
import pickle
from omegaconf import DictConfig, OmegaConf

import scipy.sparse as sp_sparse
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, classification_report
)
from sklearn.feature_extraction.text import TfidfTransformer
from features.kmer import KmerFeatures
from features.utils import plot_umap, efficient_svd, train_val_svd
from data.data_utils import load_data
from data.generate_phylogeny_folds import get_genomic_folds
from training.train_utils import compute_major_errors, load_fold_checkpoint, save_fold_checkpoint
from training.sweep_utils import OptunaFoldTrainer

# Set up logging
logger = logging.getLogger(__name__)


def fitmodel_nestedcv(X, pheno, cv_splits, estimator, config,use_sweeps=False, sweep_config=None, fixed_run_config=None, checkpoint_dir=None):
    modelname = config.model.name if not config.matrix.reduce_dimensionality else f"{config.model.name}_svd"
    modelname += f"_sub{int(config.matrix.subsample_frac*100)}" if config.matrix.subsample_frac < 1.0 else ""
    
    logger.info(f"Starting {config.num_folds}-Fold CV for {modelname}...")

    fold_results = []
    predictions_per_fold = []            # <-- store y_true / y_pred per fold
    best_overall_score = -1
    best_classification_report = ""

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        if checkpoint_dir:
            checkpoint = load_fold_checkpoint(checkpoint_dir, fold_idx)
            if checkpoint is not None:
                result_record, prediction_record = checkpoint
                fold_results.append(result_record)
                predictions_per_fold.append(prediction_record)
                logger.info(f"Fold {fold_idx+1} checkpoint found; loading saved results and skipping training.")
                continue

        start = time.process_time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pheno[train_idx], pheno[test_idx]

        if config.matrix.reduce_dimensionality:
            logger.info("Reducing dimensionality of feature matrix using efficient SVD...")
            X_train, X_test = train_val_svd(X_train, X_test, k=163)

        
        current_model = clone(estimator)

        if use_sweeps:
            logger.info(f"Running Optuna sweep for fold {fold_idx+1}...")
            trainer = OptunaFoldTrainer(current_model, fixed_run_config, sweep_config, X_train, y_train, fold_idx)
            trainer.optimize(
                n_trials=config.sweep_runs_per_fold,
                study_name=f"{sweep_config['name']}__fold_{fold_idx+1}",
            )
            
            logger.info(
                f"Best parameters in fold {fold_idx+1}: {trainer.best_params} "
                f"(val_balanced_accuracy={trainer.best_val_acc:.3f}, trial={trainer.best_trial_number})"
            )
            current_model = trainer.best_model # Get the best model from the sweep
        else:
            logger.info(f"Training {modelname} in fold {fold_idx+1} with fixed parameters.")

        # Fit the best model (from sweep) or the standard model on the full training set for the fold
        current_model.fit(X_train, y_train)
        y_pred = current_model.predict(X_test)
        # Get probabilities, in case we want to use them later for AUC or similar.
        y_probs = current_model.predict_proba(X_test)[:, 1] if hasattr(current_model, 'predict_proba') else None
        
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

        prediction_record = {
            "fold": fold_idx,
            "indices": _to_pylist(test_idx),
            "y_true": _to_pylist(y_test),
            "y_pred": _to_pylist(y_pred),
            "y_prob": _to_pylist(y_probs) if y_probs is not None else None,
        }
        predictions_per_fold.append(prediction_record)

        result_record = {
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
        }
        fold_results.append(result_record)
        if checkpoint_dir:
            checkpoint_path = save_fold_checkpoint(checkpoint_dir, fold_idx, result_record, prediction_record)
            logger.info(f"Fold {fold_idx+1} checkpoint saved to {checkpoint_path}.")
        
        if score > best_overall_score:
            best_overall_score = score
            best_classification_report = classification_report(y_test, y_pred, digits=4)
            # best_model = current_model

        logger.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f} - VME: {vme:.3f} - ME: {me:.3f} - Time: {time.process_time() - start:.2f}s")
        
        # Free memory at the end of the fold
        del X_train, X_test, y_train, y_test, current_model, y_pred, y_probs
        if use_sweeps and 'trainer' in locals():
            del trainer
        gc.collect()

    # --- Train final model on the full dataset, for testing on alternate datasets ---
    current_model = clone(estimator)
    if use_sweeps:
            logger.info(f"Running Optuna sweep for full dataset...")
            trainer = OptunaFoldTrainer(current_model, fixed_run_config, sweep_config, X, pheno, "full") # fold_idx of "full" indicates this is the final model training on the full dataset
            trainer.optimize(
                n_trials=config.sweep_runs_per_fold,
                study_name=f"{sweep_config['name']}__full",
            )
            
            logger.info(
                f"Best parameters on full data: {trainer.best_params} "
                f"(val_balanced_accuracy={trainer.best_val_acc:.3f}, trial={trainer.best_trial_number})"
            )
            current_model = trainer.best_model # Get the best model from the sweep
    else:
        logger.info(f"Training {modelname} on full dataset with fixed parameters.")
    full_train_model = current_model.fit(X, pheno)

    # Free up memory after final training    
    if use_sweeps and 'trainer' in locals():
        del trainer
    gc.collect()

    # --- After all folds, create a summary and log to the main wandb run ---
    logger.info("CV finished. Logging summary results to W&B...")
    
    # Create a pandas DataFrame for easy calculation and table logging
    fold_results = sorted(fold_results, key=lambda row: int(row["fold"]))
    predictions_per_fold = sorted(predictions_per_fold, key=lambda row: int(row["fold"]))
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
    }
    
    return summary_stats, best_classification_report, results_df, full_train_model, predictions_per_fold


# config_path is a directory relative to this file (train_model.py)
@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.seed)
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
    model_name = cfg.model.name if not cfg.matrix.reduce_dimensionality else f"{cfg.model.name}_svd"
    if cfg.matrix.subsample_frac < 1.0:
        model_name += f"_sub{int(cfg.matrix.subsample_frac*100)}"
    min_kmer_freq = 2   # filters out extremely rare k-mers that don't appear at least twice
    k = cfg.matrix.k
    feature_mode = cfg.matrix.feat_mode
    max_features = None if cfg.matrix.max_features == "None" else int(cfg.matrix.max_features)
    

    # Generate the output filename based on the parameters for the run
    patho_short = f"{pathogen.split('_')[0][0]}_{pathogen.split('_')[1]}"
    feat =  "full" if max_features is None else f"{(max_features//1000)}K"
    fname = f"{patho_short}__{antibiotic}__k_{k}__mode_{feature_mode}"


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
        max_features=max_features,                              # Very wide initial net for diverse mechanisms
        batch_size=256,                                         # Smaller batches for memory efficiency
        use_canonical=True,
        use_sparse=cfg.dataset.use_sparse,                      # Use sparse matrix for k-mer features
        reduce_dimensionality=False,                            # This is the "leaky" SVD that acts on both train and test, so it's not currently used.
        path_full_matrix=os.path.join(path_species, "feature_matrices", f"{fname}__feat_{'10M' if max_features is not None and not (cfg.dataset.use_sparse) else 'full'}{'__sparse' if cfg.dataset.use_sparse else ''}.npz"),
        path_kmers=os.path.join(path_species, "unique_kmers", f"{patho_short}__{antibiotic}__k_{k}.pkl"),
    )
    

    # Build feature matrix
    X, y = kmer_features.build_feature_matrix(dataset=dataset_list)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of samples: {X.shape[0]}")

    if cfg.build_only:
        # Exit early but save the feature matrix and k-mers, to load them later.
        logger.info("Only building the feature matrix ...")
        # Build the feature matrix and save it
        kmer_features.build_feature_matrix(dataset=dataset_list)

        # save kmers for this size
        save_fname_kmers = f"kmers_{patho_short}__{antibiotic}__k_{k}__feat_{feat}.pkl"
        path_out_kmers = os.path.join(path_species, "models", save_fname_kmers)
        # Use the kmer_features to get kmer names
        kmers_columns = kmer_features.get_feature_names() # K-mer feature names in the order of the feature matrix columns
        # Build a dict of kmer: count when the count starts at max_features and decrements to 1
        kmer_counts_dict = {kmer: count for kmer, count in zip(kmers_columns, range(max_features, 0, -1))}
        logger.info(f"Saving the k-mers used in the model to {path_out_kmers} ...")
        # Save the truncated kmer features dict
        with open(path_out_kmers, "wb") as f:
            pickle.dump(kmer_counts_dict, f)
        # logger.info(f"Feature matrix saved at {kmer_features.path_full_matrix}")
        return
    

    
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

    
        
    # Define fixed parameters for the run
    fixed_config = {
        "model_name": model_name,
        "pathogen": pathogen, "antibiotic": antibiotic,
        "pathogen_short": patho_short, "k": cfg.matrix.k,
        "feat_mode": cfg.matrix.feat_mode, "max_features": feat,
        "use_sparse": cfg.dataset.use_sparse, "seed": cfg.seed, 
        "split_method": split, "clustering": cfg.clustering_method,
    }
    
    # Create the sweep configuration if sweeps are enabled, to pass to the fold trainer
    sweep_config = None
    if cfg.use_sweeps:
        sweep_config = OmegaConf.to_container(cfg.model.sweep, resolve=True)
        sweep_name = f"SWEEP__{model_name}__{patho_short}__{antibiotic}__k_{k}__{split}"
        sweep_config['name'] = sweep_name

    
    # --- Setup train/test splits ---
    cv_splits = []
    if cfg.use_genomic_splits == True:
        logger.info(f"Using phylogeny-based splits for training and testing with {cfg.clustering_method} folds.")
        # Get the dictionary for the split indices
        genomic_dict, _ = get_genomic_folds(logger=logger,
            pathogen=pathogen, antibiotic=antibiotic,
            path_dataroot=cfg.dataset.path_dataroot, path_csv=cfg.dataset.path_csv,
            path_out=os.path.join(path_species, "phylogeny"),
            num_folds=cfg.num_folds, dist_method=cfg.split_method, 
            clustering_method=cfg.clustering_method,
        )
        # Use the dict to determine the split indices
        split_indices = []
        for assembly in dataset_list:
            if assembly['assembly'] in genomic_dict:
                # Split indices is the fold number (0 to num_folds-1) for each assembly, following the order of the CSV
                split_indices.append(genomic_dict[assembly['assembly']]) 
            else:
                raise ValueError(f"Assembly {assembly['assembly']} not found in genomic_dict. Please check the dataset.")
        # Create the train/test splits based on the indices
        for fold_idx in range(cfg.num_folds):
            train_idx = [i for i, split in enumerate(split_indices) if split != fold_idx]
            test_idx = [i for i, split in enumerate(split_indices) if split == fold_idx]
            cv_splits.append((train_idx, test_idx)) # CV Splits is a list of num_folds tuples of (train_idx, test_idx)
    else:
        logger.info("Using random splits for training and testing.")
        # Use StratifiedKFold to create the split indices
        skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)
        for train_idx, test_idx in skf.split(X, y):
            cv_splits.append((train_idx, test_idx))

    # Apply genome subsampling if specified in config
    if cfg.matrix.subsample_frac < 1.0:
        logger.info(f"Subsampling the dataset to {cfg.matrix.subsample_frac*100}% of the genomes for training ...")
        # Subsample uniformly from each training fold
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            n_subsample = int(len(train_idx) * cfg.matrix.subsample_frac)
            train_idx_subsampled = random.sample(train_idx, n_subsample)
            cv_splits[fold_idx] = (train_idx_subsampled, test_idx)
        logger.info("New training fold sizes after subsampling:")
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            logger.info(f"Fold {fold_idx}: {len(train_idx)}")

    # --- Train Model ---
    model = hydra.utils.instantiate(cfg.model.params)

    # initialize a single wandb run for the whole experiment
    run_name = f"{model_name}__{fname}__split_{split}__feat_{feat}{'__no_sweep' if not cfg.use_sweeps else ''}"
    checkpoint_dir = os.path.join(path_species, "results", "checkpoints", run_name)
    with wandb.init(
        project=cfg.wandb.project, name=run_name, 
        config=fixed_config, tags=["summary", "baseline", "cv"]
        ) as run:

        logger.info(f"Training {model.__class__.__name__} model...")
        summary_stats, best_classification_report, fold_results_df, full_train_model, predictions = fitmodel_nestedcv(
            X, y, cv_splits, model, config=cfg, 
            use_sweeps=cfg.use_sweeps, 
            sweep_config=sweep_config,
            fixed_run_config=fixed_config,
            checkpoint_dir=checkpoint_dir,
        )

        logger.info("Logging summary results to W&B...")
        run.log(summary_stats)
        run.log({"fold_results": wandb.Table(dataframe=fold_results_df)})

        # ensure target dirs exist
        os.makedirs(os.path.join(path_species, "models"), exist_ok=True)
        os.makedirs(os.path.join(path_species, "results"), exist_ok=True)
        
        # save the best model
        save_fname_model = f"model_{model_name}__{fname}__feat_{feat}{'__no_sweep' if not cfg.use_sweeps else ''}.pkl"
        path_out_model = os.path.join(path_species, "models", save_fname_model)
        logger.info(f"Saving the best model to {path_out_model} ...")
        joblib.dump(full_train_model, path_out_model)

        # save kmers for this size
        save_fname_kmers = f"kmers_{patho_short}__{antibiotic}__k_{k}__feat_{feat}.pkl"
        path_out_kmers = os.path.join(path_species, "models", save_fname_kmers)
        # Use the kmer_features to get kmer names
        kmers_columns = kmer_features.get_feature_names() # K-mer feature names in the order of the feature matrix columns
        # Build a dict of kmer: count when the count starts at max_features and decrements to 1
        num_kmers = X.shape[1] if max_features is None else max_features
        kmer_counts_dict = {kmer: count for kmer, count in zip(kmers_columns, range(num_kmers, 0, -1))}
        logger.info(f"Saving the k-mers used in the model to {path_out_kmers} ...")
        # Save the truncated kmer features dict
        with open(path_out_kmers, "wb") as f:
            pickle.dump(kmer_counts_dict, f)

        # Create the output results directory
        os.makedirs(os.path.join(path_species, "results"), exist_ok=True)

        # Save the results to a CSV file
        save_fname = f"results_{run_name}"
        path_out = os.path.join(path_species, "results", save_fname)
        # results_df = pd.DataFrame(results)
        fold_results_df.to_csv(f"{path_out}.csv", index=False)
        
        # Also save per-fold predictions (y_true, y_pred) so downstream metrics can be computed easily
        preds_rows = []
        for p in predictions:
            for idx, y_t, y_p, y_prob in zip(p["indices"], p["y_true"], p["y_pred"], p["y_prob"]):
                preds_rows.append({"fold": p["fold"], "sample_index": idx, "y_true": y_t, "y_pred": y_p, "y_prob": y_prob})
        preds_df = pd.DataFrame(preds_rows)
        preds_path = os.path.join(path_species, "results", f"preds_{model_name}__{fname}.csv")
        preds_df.to_csv(preds_path, index=False)
        logger.info(f"Per-fold predictions saved to {preds_path}")

        

if __name__ == "__main__":
    main()
