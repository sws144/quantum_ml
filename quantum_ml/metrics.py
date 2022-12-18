import numpy as np

def gini(y_true, y_pred, sample_weight=None, split=5):
    """
        sklearn-type ginierror
    
        y_truearray-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

        y_predarray-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

        sample_weightarray-like of shape (n_samples,), default=None
        Sample weights.
    
        source: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    """
    
        # The rest of the code requires numpy arrays.
    y_true = np.asarray(y_true)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        sorted_indices = np.argsort(y_pred)
        sorted_x = y_true[sorted_indices]
        sorted_w = sample_weight[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        
        min_weight = min(sample_weight)
        total_weight = sum(sample_weight)
        adjustment_factor = total_weight / (total_weight - min_weight)
        
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1])) * adjustment_factor
    else:
        sorted_indices = np.argsort(y_pred)
        sorted_x = y_true[sorted_indices]
        n = len(y_true)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        adjustment_factor = n / (n - 1) # for bias
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n * adjustment_factor
    