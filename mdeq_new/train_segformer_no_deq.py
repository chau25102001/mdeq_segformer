import os
import warnings

from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datasets.polyb import PolybDatset
from utils.utils import increment_path, intersect_dicts, AverageMeter
import importlib
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.nn import BCEWithLogitsLoss
from mmseg.models.segmentors import CaraSegUPer_ver2 as UNet


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


def iou (label, pred, smooth = 1e-9):
    print(torch.max(pred), torch.min(pred))
    segment = torch.where(pred > 0.5, 1, 0)
    intersaction = torch.sum((segment > 0) & (label > 0))
    union = torch.sum((segment > 0) | (label > 0))
    x = intersaction / (union + smooth)
    return x.item()


def dice (label, pred, smooth = 1e-9):
    segment = torch.where(pred > 0.5, 1, 0)
    #    for i, seg in enumerate(segment):
    intersaction = torch.sum((segment > 0) & (label > 0))
    union = torch.sum((segment > 0) | (label > 0))
    x = 2 * intersaction / (intersaction + union + smooth)
    return x.item()


def main ():
    final_output_dir = increment_path('runs/segformer_non_deq_MLPDecoder', False)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    writer_dict = {
        'writer': SummaryWriter(final_output_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    m = importlib.import_module('models.segformer_womdeq')
    model = getattr(m, 'mit_b3')(in_chans = 3, num_classes = 1)
    #load imagenet pretrained weight
    weight = torch.load('pretrained/mit_b3.pth', map_location = device)
    intersec = intersect_dicts(weight, model.state_dict())
    print(len(intersec.keys()))
    model.load_state_dict(intersec, strict = False)
    model = torch.nn.DataParallel(model).to(device)
    BCE = BCEWithLogitsLoss()
    Tversky = TverskyLoss(alpha = 0.4, beta = 0.6)
    trainset = PolybDatset(
        root_path = "TrainDataset/train",
        img_subpath = "image",
        label_subpath = "mask",
        img_size = 352,
        cache_train = True,
        use_aug = True,
        use_cutmix = 0.5
    )
    testset = PolybDatset(
        root_path = ["TestDataset/CVC-300", "TestDataset/CVC-ClinicDB", "TestDataset/CVC-ColonDB",
                     "TestDataset/ETIS-LaribPolypDB", "TestDataset/Kvasir"],
        img_subpath = "images",
        label_subpath = "masks",
        img_size = 352,
        cache_train = True,
        use_aug = False,
        use_cutmix = False
    )
    train_loader = DataLoader(
        trainset,
        # shuffle = False,
        shuffle = True,
        batch_size = 8,
        pin_memory = True,
        drop_last = True
    )
    test_loader = DataLoader(
        testset,
        shuffle = False,
        batch_size = 8,
        pin_memory = True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.01,
        # momentum=config.TRAIN.MOMENTUM,
        # weight_decay=config.TRAIN.WD,
        # nesterov=config.TRAIN.NESTEROV
    )

    end_epoch = 50
    best_score = 0
    current_epoch = 0
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = end_epoch * len(train_loader) * 3,
                                                              eta_min = 1e-6)
    writer = writer_dict ['writer']
    global_steps = 0
    for epoch in range(current_epoch, end_epoch):

        model.train()
        BCE_metric = AverageMeter()
        Tversky_metric = AverageMeter()
        pdar = tqdm(train_loader, desc = f"Training {epoch}/{end_epoch}")
        scales = [0.75, 1, 1.25]
        iou_list = []
        dice_list = []
        for batch in pdar:
            images_, labels_ = batch
            images_ = images_.to(device)
            labels_ = labels_.to(device)

            for scale in scales:
                images, labels = images_, labels_
                if scale != 1:
                    old_size = images_.shape [-1]
                    new_size = int(round(old_size * scale / 32) * 32)
                    images = F.upsample(images_, size = (new_size, new_size), mode = "bilinear", align_corners = True)
                    labels = F.upsample(labels_, size = (new_size, new_size), mode = "bilinear", align_corners = True)
                optimizer.zero_grad()
                preds = model(images)
                preds = F.upsample(preds, size = (labels.size(-1), labels.size(-2)), mode = 'bilinear', align_corners = True)
                labels = torch.where(labels > 0.5, 1.0, 0.0)
                print(preds.shape)
                print(labels.shape)
                bce = BCE(preds,labels)
                tversky = Tversky(pred=preds, label = labels)

                loss = bce + 2 * tversky
                # loss = bce
                # loss = tversky
                loss.backward()
                optimizer.step()
                writer.add_scalar("lr", optimizer.param_groups [0] ['lr'], global_steps)
                lr_scheduler.step()

                iou_ = iou(labels.long(), preds)
                dice_ = dice(labels.long(), preds)

                iou_list.append(iou_)
                dice_list.append(dice_)
                BCE_metric.update(bce.item())
                Tversky_metric.update(tversky.item())
                global_steps += 1
                writer_dict ['train_global_steps'] = global_steps
                pdar.set_postfix({
                    "BCE loss": BCE_metric.avg,
                    "Tversky loss": Tversky_metric.avg,
                    "iou": np.mean(iou_list),
                    "dice": np.mean(dice_list)
                })

        BCE_metric_test = AverageMeter()
        Tversky_metric_test = AverageMeter()
        global_steps = writer_dict ['train_global_steps']
        pdar_test = tqdm(test_loader, desc = "Evaluating")
        iou_list_test = []
        dice_list_test = []

        for batch in pdar_test:
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                pred = model(image)
                pred = F.upsample(pred, size = (label.size(-1), label.size(-2)), mode = 'bilinear', align_corners = True)
                # print(pred.shape)
                # print(label.shape)
                bce_test = BCE(pred,label)
                tversky_test = Tversky(pred = pred, label = label)
                # loss = structure_loss(pred, label)
                iou_list_test.append(iou(label.long(), pred))
                dice_list_test.append(dice(label.long(), pred))

                BCE_metric_test.update(bce_test.item())
                Tversky_metric_test.update(tversky_test.item())
            pdar_test.set_postfix({
                "BCE loss": BCE_metric_test.avg,
                "Tversky loss": Tversky_metric_test.avg,
                "iou": np.mean(iou_list_test),
                "dice": np.mean(dice_list_test)
            })
        checkpoint = {
            "epoch": epoch,
            "test_loss": BCE_metric_test.avg + Tversky_metric_test.avg,
            "test_iou": np.mean(iou_list_test),
            "test_dice": np.mean(dice_list_test),
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        if (checkpoint ['test_iou'] + checkpoint ['test_dice']) > 2 * best_score:
            best_score = (checkpoint ['test_iou'] + checkpoint ['test_dice']) / 2
            torch.save(checkpoint, os.path.join(final_output_dir, "checkpoint_best.pt"))
        torch.save(checkpoint, os.path.join(final_output_dir, "checkpoint_last.pt"))

        tags = ["train/BCE", "train/Tversky", "train/iou", "train/dice", "test/BCE", "train/Tversky",
                "test/iou", "test/dice"]
        values = [BCE_metric.avg, Tversky_metric.avg, np.mean(iou_list), np.mean(dice_list),
                  BCE_metric_test.avg, Tversky_metric_test.avg, np.mean(iou_list_test), np.mean(dice_list_test)]

        for tag, value in zip(tags, values):
            writer_dict ['writer'].add_scalar(tag, value, epoch)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
