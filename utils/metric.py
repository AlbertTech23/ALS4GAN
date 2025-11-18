# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class, exclude_background=True):
    """
    Compute segmentation metrics.
    
    Args:
        label_trues: Ground truth labels
        label_preds: Predicted labels
        n_class: Number of classes
        exclude_background: If True, exclude class 0 from Mean IoU calculation
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    
    # Exclude background (class 0) from mean IoU if specified
    if exclude_background and n_class > 1:
        # Only compute mean over classes 1 to n_class-1
        valid_foreground = valid[1:]  # Exclude class 0
        iu_foreground = iu[1:]  # Exclude class 0
        mean_iu = np.nanmean(iu_foreground[valid_foreground])
    else:
        mean_iu = np.nanmean(iu[valid])
    
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }
