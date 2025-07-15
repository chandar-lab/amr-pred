import wandb
import xgboost as xgb
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone


SEED = 42  # Set a seed for reproducibility

class WandbFoldTrainer:
    ''' Class to handle training of the model. One instance made per fold. Called by wandb.agent to run the sweep.'''
    def __init__(self, model, config, X_train, y_train, fold_idx):
        # Split incoming train data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )

        self.config = config
        self.model = model
        self.fold_idx = fold_idx
        self.best_model = None
        self.best_val_acc = 0
        self.best_params = None
        self.best_id = None
        # self.wandb_config = wandb_config

    def __call__(self):
        with wandb.init(name=f"fold_{self.fold_idx + 1}__{self.config['pathogen']}_{self.config['antibiotic']}__id_{self.config['id']}__run", tags=["fold", "sweep", "baseline"]) as run:
            # Add the fixed config parameters from the summary run to the sweep config
            # This allows us to track parameters easily for each sweep
            wandb.config.update(self.config)
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


def init_summary_wandb(cfg, config):
    ''' Create the overarching summary run to collect results from the sweeps across folds.
    This run will be used to track the fixed parameters for the sweep and the summary results.'''

    patho_short = f"{cfg.dataset.pathogen.split('_')[0][0]}_{cfg.dataset.pathogen.split('_')[1]}"
    
    # Get an ID to uniquely identify runs with identical setups
    # Use the weights and biases api to find runs in the same project which contain the same pathogen and antibiotic
    api = wandb.Api()
    entity = api.default_entity  # Automatically set entity from wandb params, must run 'wandb login' first.
    runs = api.runs(f"{entity}/amr-prediction", {"$and": [
        {"config.pathogen": cfg.dataset.pathogen},
        {"config.antibiotic": cfg.dataset.antibiotic}
    ]})
    # Find the highest ID among the runs
    id = 0
    if runs:
        id = max([run.config.get("id", 0) for run in runs]) + 1

    # Append the general config to the sweep_config
    wandb_config = {
        "pathogen": cfg.dataset.pathogen,
        "antibiotic": cfg.dataset.antibiotic,
        "pathogen_short": patho_short,
        "k": cfg.k,
        "feat_mode": cfg.feat_mode,
        "max_features": f"{cfg.max_features//1000}K",
        "use_sparse": cfg.dataset.use_sparse,
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
        name=f"summary__{patho_short}__{cfg.dataset.antibiotic}__id_{id}__k_{cfg.k}__mode_{cfg.feat_mode}__feat_{cfg.max_features//1000}K",
        config=wandb_config,
        tags=["summary", "baseline"]
    )
    summary_run.summary["fold_runs"] = []  # Initialize an empty list to store fold run IDs
    logging.info(f"Created new summary wandb run with ID: {summary_run.id}")

    # Get the summary_id
    summary_id = summary_run.id
    summary_run.finish() # Close the run for now, as the sweep will override it. Reopen later for logging.
    
    # Update the sweep config with the wandb_config and summary run id to link them
    config.update(wandb_config)
    config['summary_wandb_id'] = summary_id
    
    return config['summary_wandb_id'] # This is used for generating the summary wandb run separately
