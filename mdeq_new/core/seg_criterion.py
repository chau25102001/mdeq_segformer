# Modified based on the HRNet repo.

import torch
import torch.nn as nn
from torch.nn import functional as F

class FocalLossV1(nn.Module):

    def __init__ (self,
                  alpha = 0.25,
                  gamma = 2,
                  reduction = 'mean', ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction = 'none')

    def forward (self, logits, label):
        # compute loss
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha [label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class TverskyLoss(nn.Module):
    def __init__ (self, alpha = 0.5, beta = 0.5, smooth = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward (self, pred, label):
        pred = F.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)
        TP = (pred * label).sum()
        FP = ((1 - label) * pred).sum()
        FN = (label * (1 - pred)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class BCELoss(nn.Module):
    def __init__ (self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward (self, pred, label):
        return self.criterion(pred, label)


def structure_loss (pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size = 31, stride = 1, padding = 15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal * weit).sum(dim = (2, 3)) / weit.sum(dim = (2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim = (2, 3))
    union = ((pred + mask) * weit).sum(dim = (2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wfocal + wiou).mean()


class PolypCriterion(nn.Module):
    def __init__(self, alpha, beta, smooth):
        super(PolypCriterion, self).__init__()
        self.Tversky = TverskyLoss(alpha, beta, smooth)

    def forward(self, preds, targets):
        # targets = torch.cat([(1-targets), targets], dim = 1)
        ph, pw = preds.size(-2), preds.size(-1)
        h, w = targets.size(-2), targets.size(-1)
        assert ph == h and pw == w, "Targets and preds are not at the same size"
        # targets = targets.squeeze(1)
        CE_loss = structure_loss(preds, targets)
        Tverky_loss = self.Tversky(preds, targets.long())
        return CE_loss, Tverky_loss

