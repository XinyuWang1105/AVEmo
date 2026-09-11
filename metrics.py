import warnings

import numpy as np
from sklearn import metrics


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(y_true, y_score, threshold=0.5):
    """Compute binary classification metrics for fake-probability scores.

    Label convention:
        0 = real
        1 = fake

    The prediction-distribution fields make single-class collapse obvious.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    if y_true.size != y_score.size:
        raise ValueError(
            f"y_true and y_score must have the same length, got "
            f"{y_true.size} and {y_score.size}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("binary_metrics expects labels containing only 0 and 1")

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    true_real = int((y_true == 0).sum())
    true_fake = int((y_true == 1).sum())
    pred_real = int((y_pred == 0).sum())
    pred_fake = int((y_pred == 1).sum())
    total = int(y_true.size)

    result = {
        "acc": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_acc": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _safe_div(tn, tn + fp),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(metrics.matthews_corrcoef(y_true, y_pred)),
        "npv": _safe_div(tn, tn + fn),
        "ap": -1.0,
        "auc": -1.0,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "true_real": true_real,
        "true_fake": true_fake,
        "true_fake_rate": _safe_div(true_fake, total),
        "pred_real": pred_real,
        "pred_fake": pred_fake,
        "pred_fake_rate": _safe_div(pred_fake, total),
        "single_class_prediction": int(pred_real == 0 or pred_fake == 0),
        "predicted_only_class": (
            1 if pred_real == 0 else 0 if pred_fake == 0 else -1
        ),
        "score_mean": float(y_score.mean()),
        "score_std": float(y_score.std()),
        "score_min": float(y_score.min()),
        "score_max": float(y_score.max()),
        "num_samples": total,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result["ap"] = float(metrics.average_precision_score(y_true, y_score))
        except ValueError:
            pass
        try:
            result["auc"] = float(metrics.roc_auc_score(y_true, y_score))
        except ValueError:
            pass

    return result
