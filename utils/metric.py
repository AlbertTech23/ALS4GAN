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
    
    # Overall Accuracy (Pixel Accuracy)
    acc = np.diag(hist).sum() / hist.sum()
    
    # Producer's Accuracy (Recall) - per class
    # How many of the ground truth pixels were correctly predicted
    producer_acc = np.diag(hist) / hist.sum(axis=1)
    producer_acc_mean = np.nanmean(producer_acc)
    
    # User's Accuracy (Precision) - per class  
    # How many of the predicted pixels were correct
    user_acc = np.diag(hist) / hist.sum(axis=0)
    user_acc_mean = np.nanmean(user_acc)
    
    # F1 Score - per class
    # Harmonic mean of precision and recall
    f1_score = 2 * (producer_acc * user_acc) / (producer_acc + user_acc + 1e-10)
    f1_score_mean = np.nanmean(f1_score)
    
    # Mean Accuracy (original)
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    
    # IoU calculations
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0
    
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
    cls_producer_acc = dict(zip(range(n_class), producer_acc))
    cls_user_acc = dict(zip(range(n_class), user_acc))
    cls_f1 = dict(zip(range(n_class), f1_score))

    return {
        "Overall Accuracy": acc,  # Same as Pixel Accuracy
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Producer Accuracy": producer_acc_mean,  # Mean Recall
        "User Accuracy": user_acc_mean,  # Mean Precision
        "F1 Score": f1_score_mean,  # Mean F1
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
        "Class Producer Accuracy": cls_producer_acc,
        "Class User Accuracy": cls_user_acc,
        "Class F1 Score": cls_f1,
    }
