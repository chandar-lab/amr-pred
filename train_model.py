import os
import json
import time
import logging
import random
import subprocess
import joblib
import wandb
import gc
import hydra
import numpy as np
import pandas as pd
import pickle
from omegaconf import DictConfig, OmegaConf

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, classification_report
)
from features.kmer import KmerFeatures
from features.utils import plot_umap, efficient_svd, train_val_svd, compute_idf, apply_tfidf
from data.data_utils import load_data
from data.generate_phylogeny_folds import get_genomic_folds
from training.train_utils import (
    compute_major_errors, load_fold_checkpoint, save_fold_checkpoint,
    load_full_checkpoint, save_full_checkpoint, FoldFeatures,
)
from training.sweep_utils import OptunaFoldTrainer

# Set up logging
logger = logging.getLogger(__name__)

# Columns of the per-fold results CSV, in order
RESULT_COLUMNS = [
    "model_name", "pathogen", "pathogen_short", "antibiotic", "k", "feat_mode", "max_features", "seed",
    "split_method", "clustering", "fold", "balanced_accuracy", "f1_macro", "very_major_error", "major_error", "fold_time",
    "swept", "n_sweep_trials", "library_version", "n_train", "n_test", "n_features", "model_params",
]

# Result columns that describe how the model was trained; a checkpoint whose values differ was trained differently
TRAINING_COLUMNS = ["swept", "n_sweep_trials", "model_params"]

# Runs with a feature limit share one cached matrix of the top CACHE_FEATURES k-mers, and take their top max_features from it
CACHE_FEATURES = 10_000_000


def save_model_kmers(kmer_features, path_models, patho_short, antibiotic, k, mode, feat):
    """Save the k-mers needed at inference: the k-mer groups for grouped runs, else the k-mer list."""
    os.makedirs(path_models, exist_ok=True)
    if kmer_features.group_columns:
        # List in column order; each entry lists the column's member k-mers, representative first
        payload = kmer_features.get_column_groups()
        path_out_kmers = os.path.join(path_models, f"kmer_groups_{patho_short}__{antibiotic}__k_{k}__mode_{mode}__feat_{feat}.pkl")
    else:
        kmers_columns = kmer_features.get_feature_names() # K-mer feature names in the order of the feature matrix columns
        # Build a dict of kmer: count when the count starts at the number of k-mers and decrements to 1
        payload = {kmer: count for kmer, count in zip(kmers_columns, range(len(kmers_columns), 0, -1))}
        path_out_kmers = os.path.join(path_models, f"kmers_{patho_short}__{antibiotic}__k_{k}__mode_{mode}__feat_{feat}.pkl")
    logger.info(f"Saving the k-mers used in the model to {path_out_kmers} ...")
    with open(path_out_kmers, "wb") as f:
        pickle.dump(payload, f)


def model_params_json(model, config, extra_keys=()):
    """JSON of the model parameters set in the model config (plus extra_keys, e.g. swept ones), with the values the model uses."""
    keys = {k for k in config.model.params if k != "_target_"} | set(extra_keys)
    params = model.get_params()
    if "device" in params:
        keys.add("device")
    if config.model.get("balance_classes"):
        keys.add("scale_pos_weight")
    return json.dumps({k: params[k] for k in sorted(keys) if k in params}, sort_keys=True, default=str)


def library_version(model):
    """Name and version of the library that provides the model, e.g. "xgboost 3.0.2"."""
    import importlib
    lib = type(model).__module__.split(".")[0]
    return f"{lib} {getattr(importlib.import_module(lib), '__version__', 'unknown')}"


def check_checkpoint_training(record, expected, where):
    """Refuse to reuse a checkpoint trained with different settings than this run (old checkpoints lack them)."""
    stored = {c: record[c] for c in TRAINING_COLUMNS if c in record}
    if not stored:
        logger.warning(f"{where} checkpoint predates the training-parameter columns; can't check it matches this run.")
        return

    def comparable(column, value):
        # The device doesn't change what is trained (much), and collecting runs may be on another node type
        if column == "model_params":
            return {k: v for k, v in json.loads(value).items() if k != "device"}
        return value

    diff = {c: (stored[c], expected[c]) for c in stored
            if comparable(c, stored[c]) != comparable(c, expected[c])}
    if diff:
        raise ValueError(f"{where} checkpoint was trained with different settings (checkpoint, this run): {diff}. "
                         f"Delete the checkpoint directory to retrain.")


def load_fold_params(path):
    """Per-part parameter overrides from fold_params_file, keyed by part ("0".."n-1", "full"); {} without a file."""
    if not path:
        return {}
    with open(path) as f:
        return {str(part): params for part, params in json.load(f).items() if not str(part).startswith("_")}


def model_for_part(estimator, fold_params, part):
    """A fresh copy of estimator with the fold_params overrides of this part (a fold index or "full"), and their keys."""
    overrides = fold_params.get(str(part), {})
    if fold_params and not overrides:
        raise ValueError(f"fold_params_file has no parameters for part {part!r}")
    return clone(estimator).set_params(**overrides), tuple(overrides)


def balance_classes(model, config, y_train):
    """With model.balance_classes, set XGBoost's scale_pos_weight to n_negative / n_positive of y_train (0/1 labels).

    This is XGBoost's counterpart of class_weight="balanced". It is set from the training labels of each fold
    (or of the full dataset), and sweeps keep it: their stratified inner split has the same class ratio.
    """
    if config.model.get("balance_classes"):
        if "scale_pos_weight" not in model.get_params():
            raise ValueError(f"balance_classes needs a model with scale_pos_weight, got {type(model).__name__}")
        n_pos = int(np.sum(y_train == 1))
        if n_pos == 0 or n_pos == len(y_train):
            raise ValueError(f"balance_classes needs both classes in the training labels, got {n_pos} of {len(y_train)} positive")
        model.set_params(scale_pos_weight=(len(y_train) - n_pos) / n_pos)
    return model


LIBLINEAR_MAX_NONZERO = 2**31 - 1


def solver_for_matrix(model, config, X):
    """With model.large_matrix_params, use those parameters (e.g. saga) when liblinear can't train on X.

    liblinear counts the nonzero values in a 32-bit int, and adds up to 2 per row (the intercept and an end marker),
    so it overflows above 2^31 - 1 of them. The choice is made once per part (fold training rows or the full dataset),
    and its sweep trials use the same solver.
    """
    params = config.model.get("large_matrix_params")
    if params and model.get_params().get("solver") == "liblinear":
        n_nonzero = int(np.count_nonzero(X)) + 2 * X.shape[0]
        if n_nonzero > LIBLINEAR_MAX_NONZERO:
            params = OmegaConf.to_container(params, resolve=True)
            logger.info(f"{n_nonzero:,} nonzero values are more than liblinear can index ({LIBLINEAR_MAX_NONZERO:,}); "
                        f"using {params} instead.")
            model.set_params(**params)
    return model


def expected_solver(model, config, stored_params):
    """For checkpoint checks: the expected model, with model.large_matrix_params if the checkpoint was trained with them."""
    params = config.model.get("large_matrix_params")
    if params and stored_params:
        params = OmegaConf.to_container(params, resolve=True)
        stored = json.loads(stored_params)
        if all(stored.get(k) == v for k, v in params.items()):
            model.set_params(**params)
    return model


def resolve_device(device):
    """Device for models that support one (XGBoost): "auto" is "cuda" when a GPU is visible, else "cpu"."""
    if device != "auto":
        return device
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "cpu"
    return "cuda" if out.returncode == 0 and "GPU" in out.stdout else "cpu"


def fitmodel_nestedcv(X, pheno, cv_splits, estimator, config,use_sweeps=False, sweep_config=None, fixed_run_config=None, checkpoint_dir=None, fold_features=None, only_fold=None):
    """
    Cross-validate the estimator and train the final model on the full dataset.

    With only_fold (a fold index, or "full" for the final model), train just that part, save its
    checkpoint and return None. A later run without only_fold loads the checkpoints and writes the results.
    """
    modelname = config.model.name if not config.matrix.reduce_dimensionality else f"{config.model.name}_svd"
    modelname += f"_sub{int(config.matrix.subsample_frac*100)}" if config.matrix.subsample_frac < 1.0 else ""
    modelname += "_grp" if config.matrix.group_columns else ""
    
    logger.info(f"Starting {config.num_folds}-Fold CV for {modelname}...")

    fold_results = []
    predictions_per_fold = []            # <-- store y_true / y_pred per fold
    best_overall_score = -1
    best_classification_report = ""
    parts_from_checkpoint = 0            # folds (and full model) loaded instead of trained, which total_time doesn't cover

    sweep_n_jobs = config.model.get("sweep_n_jobs", 1)
    n_sweep_trials = config.sweep_runs_per_fold if use_sweeps else 0
    lib_version = library_version(estimator)
    fold_params = load_fold_params(config.get("fold_params_file")) if not use_sweeps else {}
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        if only_fold is not None and fold_idx != only_fold:
            continue
        if checkpoint_dir:
            checkpoint = load_fold_checkpoint(checkpoint_dir, fold_idx)
            if checkpoint is not None:
                result_record, prediction_record = checkpoint
                if not use_sweeps:
                    expected_model, extra = model_for_part(estimator, fold_params, fold_idx)
                    balance_classes(expected_model, config, pheno[train_idx])
                    expected_solver(expected_model, config, result_record.get("model_params"))
                    check_checkpoint_training(result_record, {"swept": False, "n_sweep_trials": 0,
                                              "model_params": model_params_json(expected_model, config, extra)},
                                              f"Fold {fold_idx+1}")
                elif "swept" in result_record:
                    # Swept parameters vary by fold, so only the sweep settings can be checked
                    check_checkpoint_training({c: result_record[c] for c in ("swept", "n_sweep_trials")},
                                              {"swept": True, "n_sweep_trials": n_sweep_trials}, f"Fold {fold_idx+1}")
                fold_results.append(result_record)
                predictions_per_fold.append(prediction_record)
                parts_from_checkpoint += 1
                logger.info(f"Fold {fold_idx+1} checkpoint found; loading saved results and skipping training.")
                continue

        start = time.perf_counter()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = pheno[train_idx], pheno[test_idx]

        if fold_features is not None:
            # Group columns on the training rows only; test columns are group medians, as at inference
            X_train, X_test = fold_features.fold(X_train, X_test)
            logger.info(f"Fold {fold_idx+1}: {X_train.shape[1]:,} columns after grouping on the training rows")

        if config.matrix.reduce_dimensionality:
            logger.info("Reducing dimensionality of feature matrix using efficient SVD...")
            X_train, X_test = train_val_svd(X_train, X_test, k=163)

        
        current_model, extra = model_for_part(estimator, fold_params, fold_idx)
        balance_classes(current_model, config, y_train)
        solver_for_matrix(current_model, config, X_train)

        if use_sweeps:
            logger.info(f"Running Optuna sweep for fold {fold_idx+1}...")
            trainer = OptunaFoldTrainer(current_model, fixed_run_config, sweep_config, X_train, y_train, fold_idx)
            trainer.optimize(
                n_trials=config.sweep_runs_per_fold,
                study_name=f"{sweep_config['name']}__fold_{fold_idx+1}",
                n_jobs=sweep_n_jobs,
            )
            
            logger.info(
                f"Best parameters in fold {fold_idx+1}: {trainer.best_params} "
                f"(val_balanced_accuracy={trainer.best_val_acc:.3f}, trial={trainer.best_trial_number})"
            )
            current_model = trainer.best_model # Get the best model from the sweep
            part_params = model_params_json(current_model, config, extra_keys=trainer.best_params or {})
        else:
            logger.info(f"Training {modelname} in fold {fold_idx+1} with fixed parameters"
                        f"{f' (fold_params_file: {dict(fold_params[str(fold_idx)])})' if fold_params else ''}.")
            part_params = model_params_json(current_model, config, extra)

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
            "fold_time": time.perf_counter() - start,
            "swept": bool(use_sweeps),
            "n_sweep_trials": n_sweep_trials,
            "model_params": part_params,
            "library_version": lib_version,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
        }
        fold_results.append(result_record)
        if checkpoint_dir:
            checkpoint_path = save_fold_checkpoint(checkpoint_dir, fold_idx, result_record, prediction_record)
            logger.info(f"Fold {fold_idx+1} checkpoint saved to {checkpoint_path}.")
        
        if score > best_overall_score:
            best_overall_score = score
            best_classification_report = classification_report(y_test, y_pred, digits=4)
            # best_model = current_model

        logger.info(f"Fold {fold_idx+1} - Balanced Accuracy: {score:.3f} - VME: {vme:.3f} - ME: {me:.3f} - Time: {time.perf_counter() - start:.2f}s")
        
        # Free memory at the end of the fold
        del X_train, X_test, y_train, y_test, current_model, y_pred, y_probs
        if use_sweeps and 'trainer' in locals():
            del trainer
        gc.collect()

    if only_fold is not None and only_fold != "full":
        return None

    # --- Train final model on the full dataset, for testing on alternate datasets ---
    # With SVD, the full-dataset model is skipped: the fitted SVD isn't saved, so it couldn't be applied to new data
    reduce_dim = config.matrix.reduce_dimensionality
    checkpoint = load_full_checkpoint(checkpoint_dir) if checkpoint_dir and not reduce_dim else None
    if reduce_dim:
        full_train_model = None
        logger.info("SVD is on: skipping the full-dataset model.")
    elif checkpoint is not None:
        full_train_model, idf = checkpoint
        if not use_sweeps:
            expected_model, extra = model_for_part(estimator, fold_params, "full")
            balance_classes(expected_model, config, pheno)
            expected_solver(expected_model, config, model_params_json(full_train_model, config, extra))
            check_checkpoint_training({"model_params": model_params_json(full_train_model, config, extra)},
                                      {"model_params": model_params_json(expected_model, config, extra)},
                                      "Full-dataset model")
        parts_from_checkpoint += 1
        if fold_features is not None:
            fold_features.idf = idf
        logger.info("Full-dataset model checkpoint found; loading it and skipping training.")
    else:
        X_full = fold_features.full(X) if fold_features is not None else X
        current_model, _ = model_for_part(estimator, fold_params, "full")
        balance_classes(current_model, config, pheno)
        solver_for_matrix(current_model, config, X_full)
        if use_sweeps:
                logger.info(f"Running Optuna sweep for full dataset...")
                trainer = OptunaFoldTrainer(current_model, fixed_run_config, sweep_config, X_full, pheno, "full") # fold_idx of "full" indicates this is the final model training on the full dataset
                trainer.optimize(
                    n_trials=config.sweep_runs_per_fold,
                    study_name=f"{sweep_config['name']}__full",
                    n_jobs=sweep_n_jobs,
                )
                
                logger.info(
                    f"Best parameters on full data: {trainer.best_params} "
                    f"(val_balanced_accuracy={trainer.best_val_acc:.3f}, trial={trainer.best_trial_number})"
                )
                current_model = trainer.best_model # Get the best model from the sweep
        else:
            logger.info(f"Training {modelname} on full dataset with fixed parameters.")
        full_train_model = current_model.fit(X_full, pheno)
        del X_full
        if checkpoint_dir:
            checkpoint_path = save_full_checkpoint(checkpoint_dir, full_train_model,
                                                   fold_features.idf if fold_features is not None else None)
            logger.info(f"Full-dataset model checkpoint saved to {checkpoint_path}.")

        # Free up memory after final training    
        if use_sweeps and 'trainer' in locals():
            del trainer
        gc.collect()

    if only_fold is not None:
        return None

    # --- After all folds, create a summary and log to the main wandb run ---
    logger.info("CV finished. Logging summary results to W&B...")
    
    # Create a pandas DataFrame for easy calculation and table logging
    fold_results = sorted(fold_results, key=lambda row: int(row["fold"]))
    predictions_per_fold = sorted(predictions_per_fold, key=lambda row: int(row["fold"]))
    results_df = pd.DataFrame(fold_results)
    # Fold checkpoints store their keys sorted; keep the column order of freshly trained folds
    # (reindex: checkpoints saved before a column existed lack it, and get NaN)
    results_df = results_df.reindex(columns=RESULT_COLUMNS + [c for c in results_df.columns if c not in RESULT_COLUMNS])
    
    # Calculate and log summary statistics
    summary_stats = {
        "mean_balanced_accuracy": results_df['balanced_accuracy'].mean(),
        "std_balanced_accuracy": results_df['balanced_accuracy'].std(),
        "mean_f1_macro": results_df['f1_macro'].mean(),
        "std_f1_macro": results_df['f1_macro'].std(),
        "mean_very_major_error": results_df['very_major_error'].mean(),
        "mean_major_error": results_df['major_error'].mean(),
        "mean_fold_time": results_df['fold_time'].mean(),
        "parts_from_checkpoint": parts_from_checkpoint,
        "fda_me_passed": bool(results_df['major_error'].mean() < 0.03),
        "fda_vme_passed": bool(results_df['very_major_error'].mean() < 0.015),
    }
    
    return summary_stats, best_classification_report, results_df, full_train_model, predictions_per_fold


# config_path is a directory relative to this file (train_model.py)
@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    run_start = time.perf_counter()
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
    if cfg.matrix.group_columns:
        model_name += "_grp"
    min_kmer_freq = 2   # filters out extremely rare k-mers that don't appear at least twice
    k = cfg.matrix.k
    feature_mode = cfg.matrix.feat_mode
    max_features = None if cfg.matrix.max_features == "None" else int(cfg.matrix.max_features)
    

    # Generate the output filename based on the parameters for the run
    patho_short = f"{pathogen.split('_')[0][0]}_{pathogen.split('_')[1]}"
    feat =  "full" if max_features is None else f"{(max_features//1000)}K"
    fname = f"{patho_short}__{antibiotic}__k_{k}__mode_{feature_mode}"
    cache_features = None if max_features is None else CACHE_FEATURES
    cache_feat = "full" if cache_features is None else f"{cache_features // 1_000_000}M"


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
        path_full_matrix=os.path.join(path_species, "feature_matrices", f"{fname}__feat_{cache_feat}{'__sparse' if cfg.dataset.use_sparse else ''}.npz"),
        cache_features=cache_features,                          # Build the shared cache at its full size
        path_kmers=os.path.join(path_species, "unique_kmers", f"{patho_short}__{antibiotic}__k_{k}.npz"),
        group_columns=cfg.matrix.group_columns,                 # Merge identical columns, drop constant ones
    )
    

    # Build feature matrix
    X, y = kmer_features.build_feature_matrix(dataset=dataset_list)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of samples: {X.shape[0]}")

    if cfg.build_only:
        # Exit early but save the feature matrix and k-mers, to load them later.
        logger.info("Only building the feature matrix ...")

        # save kmers for this size
        save_model_kmers(kmer_features, os.path.join(path_species, "models"), patho_short, antibiotic, k, feature_mode, feat)
        return

    if isinstance(X, np.memmap):
        # Read the memory-mapped cache into RAM now, so the first fold's time doesn't include reading it from disk
        load_start = time.perf_counter()
        X = np.array(X)
        logger.info(f"Read the feature matrix into memory in {time.perf_counter() - load_start:.1f}s")
    

    
    # NOTE: when using the logistic classifier, normalize the feature matrix using TF-IDF 
    # because LGR is sensitive to feature scaling
    # The idf is saved with the model so that inference applies the same transformation.
    # With grouped columns, CV groups each fold's columns on its training rows and builds the test fold
    # like inference does (group medians), and TF-IDF is fit per fold. For grouped columns, each column
    # counts once per merged k-mer in the row norms, plus the dropped constant columns.
    tfidf_idf = None
    fold_features = None
    if cfg.matrix.group_columns:
        fold_features = FoldFeatures(kmer_features.group_sizes, kmer_features.constant_sq_norm,
                                     use_tfidf=cfg.model.name == "LGR")
    elif cfg.model.name == "LGR":
        logger.info(f"Normalizing feature matrix using TF-IDF for {cfg.model.name} model ...")
        tfidf_idf = compute_idf(X)
        X = apply_tfidf(X, tfidf_idf)

    
        
    # Define fixed parameters for the run
    fixed_config = {
        "model_name": model_name,
        "pathogen": pathogen, "antibiotic": antibiotic,
        "pathogen_short": patho_short, "k": cfg.matrix.k,
        "feat_mode": cfg.matrix.feat_mode, "max_features": feat,
        "use_sparse": cfg.dataset.use_sparse, "seed": cfg.seed, 
        "split_method": split, "clustering": cfg.clustering_method,
        "use_sweeps": cfg.use_sweeps,
        "fold_params_file": cfg.get("fold_params_file"),
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
    if "device" in model.get_params():
        device = resolve_device(cfg.device)
        model.set_params(device=device)
        fixed_config["device"] = device
        logger.info(f"Training {cfg.model.name} on device {device}")

    # initialize a single wandb run for the whole experiment
    run_name = f"{model_name}__{fname}__split_{split}__feat_{feat}{'__no_sweep' if not cfg.use_sweeps else ''}"
    if cfg.get("fold_params_file"):
        if cfg.use_sweeps:
            raise ValueError("fold_params_file only applies to runs without sweeps (use_sweeps=False)")
        run_name += "__fold_params"
    checkpoint_dir = os.path.join(path_species, "results", "checkpoints", run_name)

    # Train a single fold (or the full-dataset model) as its own job; a later run without only_fold collects the results
    only_fold = cfg.only_fold
    if only_fold is not None:
        if only_fold != "full" and not (isinstance(only_fold, int) and 0 <= only_fold < cfg.num_folds):
            raise ValueError(f"only_fold must be a fold index in [0, {cfg.num_folds}) or 'full', got {only_fold!r}")
        logger.info(f"Training only {'the full-dataset model' if only_fold == 'full' else f'fold {only_fold}'} "
                    f"(checkpoints in {checkpoint_dir})")
        fitmodel_nestedcv(
            X, y, cv_splits, model, config=cfg,
            use_sweeps=cfg.use_sweeps,
            sweep_config=sweep_config,
            fixed_run_config=fixed_config,
            checkpoint_dir=checkpoint_dir,
            fold_features=fold_features,
            only_fold=only_fold,
        )
        return

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
            fold_features=fold_features,
        )
        if fold_features is not None and fold_features.idf is not None:
            tfidf_idf = fold_features.idf

        # Wall-clock time of the run up to here: loading the data, CV and the full-dataset model (not saving outputs).
        # Parts loaded from checkpoints were trained in other runs, so the time is then unknown (NaN).
        if summary_stats["parts_from_checkpoint"]:
            total_time = float("nan")
            logger.warning(f"{summary_stats['parts_from_checkpoint']} part(s) were loaded from checkpoints; total_time is NaN.")
        else:
            total_time = time.perf_counter() - run_start
            logger.info(f"Total time: {total_time:.1f}s")
        summary_stats["total_time"] = total_time
        fold_results_df["total_time"] = total_time

        logger.info("Logging summary results to W&B...")
        run.log(summary_stats)
        run.log({"fold_results": wandb.Table(dataframe=fold_results_df)})

        # ensure target dirs exist
        os.makedirs(os.path.join(path_species, "models"), exist_ok=True)
        os.makedirs(os.path.join(path_species, "results"), exist_ok=True)
        
        # save the best model
        if full_train_model is not None:  # None with SVD, which skips the full-dataset model
            save_fname_model = f"model_{run_name}.pkl"
            path_out_model = os.path.join(path_species, "models", save_fname_model)
            logger.info(f"Saving the best model to {path_out_model} ...")
            joblib.dump(full_train_model, path_out_model)

        # save the TF-IDF parameters next to the model, for inference
        if tfidf_idf is not None:
            tfidf_params = {"idf": tfidf_idf}
            if kmer_features.group_sizes is not None:
                tfidf_params["group_sizes"] = kmer_features.group_sizes
                tfidf_params["constant_sq_norm"] = kmer_features.constant_sq_norm
            path_out_tfidf = os.path.join(path_species, "models", f"tfidf_{run_name}.npz")
            logger.info(f"Saving the TF-IDF parameters to {path_out_tfidf} ...")
            np.savez(path_out_tfidf, **tfidf_params)

        # save kmers for this size
        save_model_kmers(kmer_features, os.path.join(path_species, "models"), patho_short, antibiotic, k, feature_mode, feat)

        # Create the output results directory
        os.makedirs(os.path.join(path_species, "results"), exist_ok=True)

        # Save the results to a CSV file
        save_fname = f"results_{run_name}"
        path_out = os.path.join(path_species, "results", save_fname)
        # results_df = pd.DataFrame(results)
        # model_params last, as it is long
        fold_results_df = fold_results_df[[c for c in fold_results_df.columns if c != "model_params"] + ["model_params"]]
        fold_results_df.to_csv(f"{path_out}.csv", index=False)
        
        # Also save per-fold predictions (y_true, y_pred) so downstream metrics can be computed easily
        preds_rows = []
        for p in predictions:
            for idx, y_t, y_p, y_prob in zip(p["indices"], p["y_true"], p["y_pred"], p["y_prob"]):
                preds_rows.append({"fold": p["fold"], "sample_index": idx, "assembly": dataset_list[idx]["assembly"],
                                   "y_true": y_t, "y_pred": y_p, "y_prob": y_prob})
        preds_df = pd.DataFrame(preds_rows)
        preds_path = os.path.join(path_species, "results", f"preds_{run_name}.csv")
        preds_df.to_csv(preds_path, index=False)
        logger.info(f"Per-fold predictions saved to {preds_path}")

        

if __name__ == "__main__":
    main()
