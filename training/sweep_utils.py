import math
import gc
import os

import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone


class OptunaFoldTrainer:
    """Train and tune a model for one fold using Optuna."""

    def __init__(self, model, config, sweep_config, X_train, y_train, fold_idx):
        self.config = config
        self.sweep_config = sweep_config or {}
        self.model = model
        self.fold_idx = fold_idx
        self.best_model = None
        self.best_val_acc = -math.inf
        self.best_params = None
        self.best_trial_number = None
        # Split incoming train data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.config["seed"], stratify=y_train
        )
        self._dtrain = None     # XGBoost training data, quantized once and reused by every trial

    # Sweeping one of these changes how the training data is quantized, so it cannot be reused across trials
    _XGB_DATA_PARAMS = {"max_bin", "missing", "enable_categorical", "feature_types"}

    def _reuse_xgb_data(self):
        """Whether trials can share one QuantileDMatrix (plain XGBClassifier fits, as in XGBClassifier.fit)."""
        return (isinstance(self.model, xgb.XGBClassifier)
                and self.model.early_stopping_rounds is None and self.model.callbacks is None
                and not self._XGB_DATA_PARAMS & set(self._valid_sweep_parameters()))

    def _xgb_trial_predict(self, model):
        """Train and predict like XGBClassifier.fit/predict on binary 0/1 labels, but on the shared training data."""
        if self._dtrain is None:
            self._dtrain = xgb.QuantileDMatrix(self.X_train, self.y_train, missing=self.model.missing,
                                               max_bin=self.model.max_bin, nthread=self.model.n_jobs)
        with xgb.config_context(verbosity=model.verbosity):
            booster = xgb.train(model.get_xgb_params(), self._dtrain, model.get_num_boosting_rounds())
            probs = booster.inplace_predict(self.X_val)
        return (probs > 0.5).astype(np.int64)

    @property
    def direction(self):
        goal = self.sweep_config.get("metric", {}).get("goal", "maximize")
        if goal not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported sweep metric goal: {goal}")
        return goal

    def _valid_sweep_parameters(self):
        valid_model_params = self.model.get_params().keys()
        sweep_parameters = self.sweep_config.get("parameters", {})
        return {
            key: spec
            for key, spec in sweep_parameters.items()
            if key in valid_model_params
        }

    def _suggest_model_params(self, trial):
        model_params = {}
        for name, spec in self._valid_sweep_parameters().items():
            if "value" in spec:
                model_params[name] = spec["value"]
            elif "values" in spec:
                model_params[name] = trial.suggest_categorical(name, spec["values"])
            elif spec.get("distribution") == "uniform":
                model_params[name] = trial.suggest_float(name, spec["min"], spec["max"])
            elif spec.get("distribution") == "log_uniform_values":
                model_params[name] = trial.suggest_float(
                    name, spec["min"], spec["max"], log=True
                )
            else:
                raise ValueError(
                    f"Unsupported sweep parameter specification for {name}: {spec}"
                )
        return model_params

    def _best_model_params(self, best_trial):
        model_params = {}
        for name, spec in self._valid_sweep_parameters().items():
            if "value" in spec:
                model_params[name] = spec["value"]
            elif name in best_trial.params:
                model_params[name] = best_trial.params[name]
        return model_params

    def _make_sampler(self):
        method = self.sweep_config.get("method", "bayes")
        seed = self.config["seed"]
        if method == "bayes":
            return optuna.samplers.TPESampler(seed=seed)
        if method == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        if method == "grid":
            search_space = {}
            for name, spec in self._valid_sweep_parameters().items():
                if "values" in spec:
                    search_space[name] = spec["values"]
                elif "value" not in spec:
                    raise ValueError(
                        "Grid search requires all tuned parameters to define "
                        f"'values'; {name} uses {spec}"
                    )
            return optuna.samplers.GridSampler(search_space)
        raise ValueError(f"Unsupported sweep method: {method}")

    def _objective(self, trial):
        model = clone(self.model)
        model_params = self._suggest_model_params(trial)
        if model_params:
            model.set_params(**model_params)

        if self._reuse_xgb_data():
            val_preds = self._xgb_trial_predict(model)
        else:
            model.fit(self.X_train, self.y_train)
            val_preds = model.predict(self.X_val)
        val_acc = balanced_accuracy_score(self.y_val, val_preds)
        trial.set_user_attr("val_balanced_accuracy", val_acc)
        
        del model, val_preds
        return val_acc

    def optimize(self, n_trials, study_name=None, n_jobs=1):
        """n_jobs trials run at once in threads (-1: all CPUs of the job). Use it for single-threaded models."""
        if n_jobs == -1:
            n_jobs = len(os.sched_getaffinity(0))
        study = optuna.create_study(
            direction=self.direction,
            sampler=self._make_sampler(),
            study_name=study_name,
        )
        study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs)

        best_trial = study.best_trial
        self.best_val_acc = best_trial.value
        self.best_params = self._best_model_params(best_trial)
        self.best_trial_number = best_trial.number
        self.best_model = clone(self.model)
        if self.best_params:
            self.best_model.set_params(**self.best_params)
            
        # Free validation splits from memory once optimization finishes
        if hasattr(self, 'X_train'):
            del self.X_train, self.X_val, self.y_train, self.y_val
        self._dtrain = None
        gc.collect()
        
        return study
