nimport numpy as np
from sklearn.metrics import f1_score

def compute_fmax_score(y_true, y_pred_prob, thresholds=np.arange(0.0, 1.05, 0.05)):
    max_f1_avg = 0
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1_avg = f1_score(y_true, y_pred, average='samples')
        if f1_avg > max_f1_avg:
            max_f1_avg = f1_avg
    
    return max_f1_avg

