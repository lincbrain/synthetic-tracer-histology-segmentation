import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (0, 2, 3)  # sum over batch and spatial dims

        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice_score.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, pos_weight=1.0):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight  # Weight for positive class to handle imbalance
        self.dice = BinaryDiceLoss()
        # pos_weight: if 0.5, penalize false positives (over-prediction)
        # if 2.0, penalize false negatives (under-prediction)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_bce = self.bce(logits, targets.float())
        return self.dice_weight * loss_dice + self.bce_weight * loss_bce
