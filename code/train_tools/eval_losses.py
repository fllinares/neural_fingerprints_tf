import numpy as np
from sklearn.metrics import roc_auc_score

def mean_squared_error(targets, predictions):
    return np.mean((targets - predictions)**2)


def root_mean_squared_error(targets, predictions):
    return np.sqrt(mean_squared_error(targets, predictions))

# Actually, 1.0 - pearson_r, to treat it as a loss to be minimised
def pearson_r(targets, predictions):
    norm_targets = (targets - np.mean(targets, axis=0))/np.std(targets, axis=0)
    norm_predictions = (predictions - np.mean(predictions, axis=0)) / np.std(predictions, axis=0)
    return 1.0 - np.mean(norm_targets*norm_predictions)


def zero_one(targets, predictions):
    return np.mean(np.argmax(targets, axis=1) != np.argmax(predictions, axis=1))

# Actually, 1.0 - AUC, to treat it as a loss to be minimised
def auc(targets, predictions):
    n = targets.shape[0]
    pred_prob = np.exp(predictions - np.reshape(np.max(predictions, axis=1), (n, 1)))
    pred_prob = pred_prob/np.reshape(np.sum(pred_prob, axis=1), (n, 1))
    return 1.0 - roc_auc_score(targets, pred_prob)

def eval_losses_dict():
    EVAL_LOSSES = {'MSE': mean_squared_error,
                   'RMSE': root_mean_squared_error,
                   'Corr': pearson_r,
                   '0/1': zero_one,
                   'AUC': auc}
    return EVAL_LOSSES
