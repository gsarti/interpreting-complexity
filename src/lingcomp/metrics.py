from math import sqrt

import numpy as np
from scipy.stats import t
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(preds, labels):
    max_err = max_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = sqrt(mse)
    r2 = r2_score(labels, preds)
    err_std = np.std([abs(x - y) for x, y in zip(preds, labels)])
    return {"max_err": max_err, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "err_std": err_std}


def classification_metrics(preds, labels):
    confusion = confusion_matrix(labels, preds)
    report = classification_report(labels, preds)
    return {"confusion_matrix": confusion, "classification_report": report}


def token_level_regression_metrics(preds, labels):
    max_err = token_level_metric(labels, preds, name="max_err")
    mae = token_level_metric(labels, preds, name="mae")
    mse = token_level_metric(labels, preds, name="mse")
    rmse = sqrt(mse)
    return {"max_err": max_err, "mae": mae, "mse": mse, "rmse": rmse}


def token_level_metric(preds, labels, name="mse"):
    """
    Both preds and labels are list of lists of scores.
    Compute average metric value across sentences.
    We use this since normal metrics would normally compute element-wise metric
    across vectors, thus cannot support inner vectors of variable length.
    e.g. preds = [[0.5, 1],[-1, 1, 3],[7, -6]]
         labels = [[0, 2],[-1, 2, 2],[8, -5]]
    """
    score_list = []
    try:
        for pred_list, label_list in zip(preds, labels):
            score_list.append(DICT_METRICS[name](pred_list, label_list))
    except TypeError:
        raise TypeError("Problem in constructing token-level regression metrics.")
    return sum(score_list) / len(score_list)


def confidence_intervals(d, conf_interval=0.95):
    """ Given a dict containing mean, std and count computes upper and lower confidence bounds """
    st_interval = t.interval(conf_interval, d["count"] - 1, loc=d["mean"], scale=d["sem"])
    return st_interval


DICT_METRICS = {
    "max_err": max_error,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
}
