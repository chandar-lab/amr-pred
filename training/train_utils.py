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