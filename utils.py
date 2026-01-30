import random
import torch
import numpy as np
from sklearn.metrics import roc_curve


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)         # for single-GPU
    torch.cuda.manual_seed_all(seed)     # for multi-GPU (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_usage2(counts, row, what):
    cohort = row['cohort']
    return counts[f'{cohort},{what}']


def update_usage1(row):
    usage = row['usage']
    setting = row['setting']
    out = 'Ab_raw'
    
    if setting.strip().lower() == 'binary':
        out += '+ Ab_binary'
        
    if usage.strip().lower() == 'demographic':
        out += '+ demographic'
    
    if setting.strip().lower() == 'no-cutoff':
        pass
    
    return out


def at_95(y_scores, y_true, t=0.9):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Specificity = 1 - FPR
    specificity = 1 - fpr
    sensitivity = tpr
    
    # ---- Sensitivity at 95% specificity ----
    target_spec = t
    idx_spec = np.argmin(np.abs(specificity - target_spec))
    sensitivity_at_95_spec = sensitivity[idx_spec]
    
    # ---- Specificity at 95% sensitivity ----
    target_sens = t
    idx_sens = np.argmin(np.abs(sensitivity - target_sens))
    specificity_at_95_sens = specificity[idx_sens]

    return sensitivity_at_95_spec, specificity_at_95_sens