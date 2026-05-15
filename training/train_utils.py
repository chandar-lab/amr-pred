import json
import os

from sklearn.metrics import confusion_matrix


def compute_major_errors(y_true, y_pred):
    """Compute Very Major Error (VME) and Major Error (ME) from the confusion matrix."""
    # Compute Very Major Error and Major Error
    # NOTE: # To be cleared for marketing in the USA, the FDA requires that VMEs attributable to a test device should be <1.5% for individual species/drug comparisons, 
    # MEs should not exceed 3%, and an overall essential MIC agreement of >90% of device MICs within one doubling dilution of a CLSI reference MIC.
    # More info: https://academic.oup.com/cid/article-abstract/49/11/1749/344384

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # VME is the false negative rate; proportion of resistant samples misclassified as susceptible; aka Type II error
    vme = fn / (fn + tp) if (fn + tp) > 0 else 0
    # ME is the false positive rate; proportion of susceptible samples misclassified as resistant; aka Type I error
    me = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return vme, me


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def fold_checkpoint_path(checkpoint_dir, fold_idx):
    return os.path.join(checkpoint_dir, f"fold_{fold_idx}.json")


def save_fold_checkpoint(checkpoint_dir, fold_idx, result_record, prediction_record):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = fold_checkpoint_path(checkpoint_dir, fold_idx)
    tmp_path = f"{path}.tmp"
    payload = {
        "result": _to_jsonable(result_record),
        "predictions": _to_jsonable(prediction_record),
    }
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    return path


def load_fold_checkpoint(checkpoint_dir, fold_idx):
    path = fold_checkpoint_path(checkpoint_dir, fold_idx)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    return payload["result"], payload["predictions"]
