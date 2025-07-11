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
import wandb
import yaml

from sklearn.model_selection import train_test_split
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
RUNS_PER_FOLD = 2 # Number of runs per fold for hyperparameter tuning, must be fixed
NUM_FOLDS = 5  # Number of folds for cross-validation

# Global variable to store the fixed parameters for the sweep and summary runs
fixed_config = {}

class WandbFoldTrainer:
    ''' Class to handle training of the model. One instance made per fold. Called by wandb.agent to run the sweep.'''
    def __init__(self, model, X_train, y_train, fold_idx):
        # Split incoming train data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED
        )
        self.model = model
        self.fold_idx = fold_idx
        self.best_model = None
        self.best_val_acc = 0
        self.best_params = None
        self.best_id = None
        # self.wandb_config = wandb_config

    def __call__(self):
        with wandb.init(name=f"fold_{self.fold_idx + 1}__{fixed_config['pathogen']}_{fixed_config['antibiotic']}__id_{fixed_config['id']}__run", tags=["fold", "sweep", "baseline"]) as run:
            # Add the fixed config parameters from the summary run to the sweep config
            # This allows us to track parameters easily for each sweep
            wandb.config.update(fixed_config)
            config = wandb.config
            # Apply the config to the model
            if isinstance(self.model, xgb.XGBClassifier):
                self.model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    max_depth=config.tree_depth,
                    learning_rate=config.learning_rate,
                    subsample=config.subsample,
                    colsample_bytree=config.colsample_bytree,
                    gamma=config.gamma,
                )
            self.model.fit(self.X_train, self.y_train)
            val_preds = self.model.predict(self.X_val)
            val_acc = balanced_accuracy_score(self.y_val, val_preds)
            wandb.log({"val_balanced_accuracy": val_acc})
            # Determine if this is the best model so far, and save it if so
            # That way it can be used on the test set
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model = clone(self.model)
                self.best_params = wandb.config
                self.best_id = run.id # This is so metrics can be accessed later

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

def init_summary_wandb(args):
    ''' Create the overarching summary run to collect results from the sweeps across folds.
    This run will be used to track the fixed parameters for the sweep and the summary results.'''
    global fixed_config
    
    # Get an ID to uniquely identify runs with identical setups
    # Use the weights and biases api to find runs in the same project which contain the same pathogen and antibiotic
    api = wandb.Api()
    entity = api.default_entity  # Automatically set entity from wandb params, must run 'wandb login' first.
    runs = api.runs(f"{entity}/amr-prediction", {"$and": [
        {"config.pathogen": args.pathogen},
        {"config.antibiotic": args.antibiotic}
    ]})
    # Find the highest ID among the runs
    id = 0
    if runs:
        id = max([run.config.get("id", 0) for run in runs]) + 1

    # Append the general config to the sweep_config
    wandb_config = {
        "pathogen": args.pathogen,
        "antibiotic": args.antibiotic,
        "k": args.k,
        "feat_mode": args.feat_mode,
        "max_features": args.max_features,
        "use_sparse": args.use_sparse,
        "id": id,
    }

    ### The following is for an array of sweeps, which currently is not working ###

    # summary_id = args.summary_wandb_id
    # Create a summary wandb project for the pathogen-antibiotic combination
    # if summary_id is not None:
    #     # If a summary wandb ID is provided, use the api to get info about the run
    #     api = wandb.Api()
    #     try:
    #         summary_run = api.run(f"amr-prediction/{summary_id}")
    #     except Exception as e:
    #         raise ValueError(f"Invalid summary wandb ID provided: {summary_id}. Please check the ID and try again.") from e
        
    #     logging.info(f"Using existing summary wandb run with ID: {summary_id}")
    #     expected_config = wandb_config.copy()  # Copy the expected config to compare later
    #     wandb_config = summary_run.config  # Load the existing config

    #     # Ensure that the specified config matches the existing summary run config.
    #     ### NOTE: In the future, it may be good to change this to throw out the passed in config, and instead
    #     ### assume the user wants the config from the existing summary run. This would be especially true with hydra
    #     if wandb_config["pathogen"] != args.pathogen or wandb_config["antibiotic"] != args.antibiotic \
    #         or wandb_config["k"] != args.k or wandb_config["feat_mode"] != args.feat_mode or wandb_config["max_features"] != args.max_features \
    #         or wandb_config["use_sparse"] != args.use_sparse:
    #         # Create a string comparing the two configs
    #         config_str = f"Provided: {wandb_config}, Expected: {expected_config}"
    #         raise ValueError("The provided summary wandb ID does not match the specified config parameters. " + config_str)
    # else:
    summary_run = wandb.init(
        project="amr-prediction",
        name=f"{args.pathogen}-{args.antibiotic}__id_{id}",
        config=wandb_config,
        tags=["summary", "baseline"]
    )
    summary_run.summary["fold_runs"] = []  # Initialize an empty list to store fold run IDs
    logging.info(f"Created new summary wandb run with ID: {summary_run.id}")

    # Get the summary_id
    summary_id = summary_run.id
    summary_run.finish() # Close the run for now, as the sweep will override it. Reopen later for logging.
    

    # Update the sweep config with the wandb_config and summary run id to link them
    fixed_config.update(wandb_config)
    fixed_config['summary_wandb_id'] = summary_id
    
    return fixed_config['summary_wandb_id'] # This is used for generating the summary wandb run separately



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
        temp_dict["assembly"] = assembly # ADDED FOR PHYLOGENY-BASED SPLITS
        dataset_list.append(temp_dict)
    
    print("Number of samples in the dataset: ", len(dataset_list))
    
    return dataset_list


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
                name=f"{fixed_config['pathogen']}-{fixed_config['antibiotic']}__id_{fixed_config['id']}__fold_{fold_idx+1}__k{fixed_config['k']}__mode_{fixed_config['feat_mode']}__feat_{(fixed_config['max_features']//1000)}K",
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
            trainer = WandbFoldTrainer(estimator, X_train, y_train, fold_idx)
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

        # Log to the wandb run
        wandb_run_metrics = {
            "fold_idx": int(fold_idx + 1),
            "balanced_accuracy": float(score),
            "f1_macro": float(f1_macro),
            # "confusion_matrix": wandb.plot.confusion_matrix(
            #     probs=None, y_true=y_test, preds=y_pred, class_names=["Susceptible", "Resistant"]
            # ), # These kinds of artifacts cannot be added through the api, so they've been removed. They can be added within the OG run.
            "fold_time": float(time.process_time() - start),
            "very_major_error": float(vme),
            "major_error": float(me),
            "fda_vme_passed": bool(vme < 0.015),
            "fda_me_passed": bool(me < 0.03),
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
            summary_run.summary["best_fold_idx"] = fold_idx
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
    logging.info(f"Best Model (Fold {summary_run.summary['best_fold_idx']+1}) - Balanced Accuracy: {summary_run.summary['best_score']:.3f}")
    logging.info(f"Best Model Classification Report:\n{best_report}")

    wandb_tracked_values = {"mean_major_error": np.mean(major_errors), 
        "std_major_error": np.std(major_errors), 
        "mean_very_major_error": np.mean(very_major_errors), 
        "std_very_major_error": np.std(very_major_errors), 
        "mean_balanced_accuracy": np.mean(bal_acc_scores), 
        "std_balanced_accuracy": np.std(bal_acc_scores), 
        "mean_f1_macro": np.mean(f1_macro_scores), 
        "std_f1_macro": np.std(f1_macro_scores),
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
    dataset = f"{args.pathogen}/filtered/{args.pathogen}__{args.antibiotic}_SR_filt.csv"
    path_data_csv = os.path.join(args.path_csv, dataset)
    if not os.path.isfile(path_data_csv):
        raise FileNotFoundError(f"CSV file not found: {path_data_csv}. Please check the name of the pathogen and drug.")

    now = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    path_out = os.path.join("/home/mila/a/aselstya/scratch/amr/results", now.split('_')[0], now.split('_')[1])
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

    # # save the matrix to a file
    # if args.path_precomputed is None:
    #     save_fname = os.path.join(args.path_csv, f"{fname}.npz")
    #     kmer_features.save_matrix(X, y, save_fname)
    #     logging.info(f"Feature matrix saved to {save_fname}")

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

    # Initialize wandb
    # Open the sweep config from the yaml, defining sweep hyperparams and ranges
    with open("./amr-pred/XGB-sweep.yaml", 'r') as config_file:
        sweep_config = yaml.safe_load(config_file)
    
    # Initialize the summary run to collect results from the sweeps across folds
    init_summary_wandb(args)

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
    vmes, mes = fitmodel_nestedcv(X, y, xgb_model, sweep_config, "xgb", path_out=path_out)#, prescribed_fold=args.fold_idx)

    # # Evaluation
    # print("Evaluating model...")
    # y_pred = model.predict(X_test)
    # print("\nClassification Report:")
    # target_names = ["Susceptible (0)", "Resistant (1)"]
    # print(classification_report(y_test, y_pred, target_names=target_names))



if __name__ == "__main__":
    main()
