import wandb
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone


class WandbFoldTrainer:
    ''' Class to handle training of the model. One instance made per fold. Called by wandb.agent to run the sweep.'''
    def __init__(self, model, config, X_train, y_train, fold_idx):
        self.config = config
        self.model = model
        self.fold_idx = fold_idx
        self.best_model = None
        self.best_val_acc = 0
        self.best_params = None
        self.best_id = None
        # Split incoming train data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.config["seed"], stratify=y_train
        )


    def __call__(self):
        with wandb.init(
            name=f"run__{self.config['pathogen_short']}_{self.config['antibiotic']}__id_{self.config['id']}__fold_{self.fold_idx+1}", 
            tags=["fold", "sweep", "baseline"]
            ) as run:
            # Add the fixed config parameters from the summary run to the sweep config
            wandb.config.update(self.config)
            config = wandb.config
            
            # Model Agnostic Parameter Setting
            # Create a dictionary of parameters from the sweep config that the model accepts
            model_params = {
                key: config[key] 
                for key in config.keys() 
                if key in self.model.get_params().keys()
            }
            
            # Apply the config to the model in a generic way
            if model_params:
                self.model.set_params(**model_params)

            self.model.fit(self.X_train, self.y_train)
            val_preds = self.model.predict(self.X_val)
            val_acc = balanced_accuracy_score(self.y_val, val_preds)
            wandb.log({"val_balanced_accuracy": val_acc})

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model = clone(self.model)
                self.best_params = wandb.config
                self.best_id = run.id


def init_summary_wandb(cfg):
    ''' Create the overarching summary run to collect results from the sweeps across folds.
    This run will be used to track the fixed parameters for the sweep and the summary results.'''

    patho_short = f"{cfg.dataset.pathogen.split('_')[0][0]}_{cfg.dataset.pathogen.split('_')[1]}"
    
    api = wandb.Api()
    entity = api.default_entity
    project = cfg.wandb.project
    runs = api.runs(f"{entity}/{project}", {"$and": [
        {"config.pathogen": cfg.dataset.pathogen},
        {"config.antibiotic": cfg.dataset.antibiotic}
    ]})
    
    id = 0
    if runs:
        id = max([run.config.get("id", 0) for run in runs]) + 1

    wandb_config = {
        "pathogen": cfg.dataset.pathogen,
        "antibiotic": cfg.dataset.antibiotic,
        "pathogen_short": patho_short,
        "k": cfg.matrix.k,
        "feat_mode": cfg.matrix.feat_mode,
        "max_features": f"{cfg.matrix.max_features//1000}K",
        "use_sparse": cfg.dataset.use_sparse,
        "id": id,
        "seed": cfg.seed,
    }

    summary_run = wandb.init(
        project=cfg.wandb.project,
        name=f"summary__{patho_short}__{cfg.dataset.antibiotic}__id_{id}__k_{cfg.matrix.k}__mode_{cfg.matrix.feat_mode}__feat_{cfg.matrix.max_features//1000}K",
        config=wandb_config,
        tags=["summary", "baseline"]
    )
    summary_run.summary["fold_runs"] = []
    logging.info(f"Created new summary wandb run with ID: {summary_run.id}")

    summary_id = summary_run.id
    summary_run.finish()
    
    # eturn the full config
    wandb_config['summary_wandb_id'] = summary_id
    
    return wandb_config