import wandb
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
        name = f"fold_{self.fold_idx}__{self.config['pathogen_short']}_{self.config['antibiotic']}_run"
        
        with wandb.init(name=name, tags=["fold", "sweep_run", "baseline"]) as run:
            # Add the fixed config parameters from the summary run to the sweep config
            wandb.config.update(self.config)
            config = wandb.config
            
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
