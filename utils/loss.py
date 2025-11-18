import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    Better for imbalanced classes than CrossEntropy.
    
    Formula: 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    """
    def __init__(self, smooth=1.0, ignore_label=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_label = ignore_label
    
    def forward(self, predict, target):
        """
        Args:
            predict: (N, C, H, W) - predicted logits
            target: (N, H, W) - ground truth labels
        """
        # Get softmax probabilities
        predict = F.softmax(predict, dim=1)  # (N, C, H, W)
        
        n, c, h, w = predict.size()
        
        # Create mask for valid pixels
        target_mask = (target >= 0) * (target != self.ignore_label)  # (N, H, W)
        
        if not target_mask.any():
            return Variable(torch.zeros(1).to(predict.device))
        
        # Clamp target to valid range [0, c-1] to avoid index out of bounds
        target_clamped = target.clone()
        target_clamped = torch.clamp(target_clamped, 0, c - 1)
        
        # Convert target to one-hot encoding
        target_one_hot = torch.zeros(n, c, h, w).to(predict.device)
        target_one_hot.scatter_(1, target_clamped.unsqueeze(1), 1)  # (N, C, H, W)
        
        # Apply mask
        target_mask_expanded = target_mask.unsqueeze(1).expand_as(predict)  # (N, C, H, W)
        predict = predict * target_mask_expanded
        target_one_hot = target_one_hot * target_mask_expanded
        
        # Calculate Dice coefficient
        intersection = (predict * target_one_hot).sum(dim=(2, 3))  # (N, C)
        union = predict.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (N, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (N, C)
        
        # Average over classes and batch
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    Formula: -α * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter >= 0 (higher = more focus on hard examples)
        ignore_label: Label to ignore in loss calculation
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_label=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_label = ignore_label
    
    def forward(self, predict, target):
        """
        Args:
            predict: (N, C, H, W) - predicted logits
            target: (N, H, W) - ground truth labels
        """
        n, c, h, w = predict.size()
        
        # Create mask for valid pixels (also exclude out-of-bounds labels)
        target_mask = (target >= 0) * (target < c) * (target != self.ignore_label)
        target_valid = target[target_mask]
        
        if target_valid.numel() == 0:
            return Variable(torch.zeros(1).to(predict.device))
        
        # Clamp to valid range as safety measure
        target_valid = torch.clamp(target_valid, 0, c - 1)
        
        # Reshape predictions
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()  # (N, H, W, C)
        predict_valid = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)  # (N_valid, C)
        
        # Get softmax probabilities
        p = F.softmax(predict_valid, dim=1)  # (N_valid, C)
        
        # Get probabilities of true class
        p_t = p.gather(1, target_valid.unsqueeze(1)).squeeze(1)  # (N_valid,)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(predict_valid, target_valid, reduction='none')
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combination of CrossEntropy, Dice, and Focal losses.
    Provides better performance for imbalanced segmentation tasks.
    """
    def __init__(self, 
                 ce_weight=0.4, 
                 dice_weight=0.4, 
                 focal_weight=0.2,
                 class_weights=None,
                 ignore_label=255):
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.ce_loss = CrossEntropy2d(ignore_label=ignore_label)
        self.dice_loss = DiceLoss(ignore_label=ignore_label)
        self.focal_loss = FocalLoss(ignore_label=ignore_label)
        
        self.class_weights = class_weights
    
    def forward(self, predict, target):
        """
        Args:
            predict: (N, C, H, W)
            target: (N, H, W)
        """
        # Calculate individual losses
        loss_ce = self.ce_loss(predict, target, weight=self.class_weights)
        loss_dice = self.dice_loss(predict, target)
        loss_focal = self.focal_loss(predict, target)
        
        # Combine losses
        total_loss = (self.ce_weight * loss_ce + 
                      self.dice_weight * loss_dice + 
                      self.focal_weight * loss_focal)
        
        return total_loss, loss_ce, loss_dice, loss_focal