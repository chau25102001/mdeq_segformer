import logging
import os
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn import functional as F
from utils.utils import AverageMeter, iou, dice
from utils.utils import get_confusion_matrix


def train(
        config,
        epoch,
        num_epoch,
        trainloader,
        optimizer,
        writer_dict,
        model,
        device,
        logger,
        lr_scheduler,
        weights = [0.5, 0.5]
):
    # Training
    if logger is not None:
        logger.info("Training")
    model.train()
    CE_metric = AverageMeter()
    Tver_metric = AverageMeter()
    jac_metric = AverageMeter()
    # cur_iters = epoch * epoch_\iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    # assert global_steps == cur_iters, "Step counter problem... fix this?"
    update_freq = config.LOSS.JAC_INCREMENTAL
    pdar = tqdm(trainloader, desc=f"Training {epoch}/{num_epoch}")
    scales = [0.75, 1, 1.25]
    iou_lis = []
    dice_lis = []
    for batch in pdar:
        images_, labels_ = batch
        images_ = images_.to(device)
        labels_ = labels_.to(device)

        for scale in scales:
            images, labels = images_, labels_
            if scale != 1:
                old_size = images_.shape[-1]
                new_size = int(round(old_size * scale / 32) * 32)
                images = F.upsample(images_, size=(new_size, new_size), mode="bilinear", align_corners=True)
                labels = F.upsample(labels_, size=(new_size, new_size), mode="bilinear", align_corners=True)
            deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
            if deq_steps < 0:
                factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
            elif config.LOSS.JAC_STOP_EPOCH <= epoch:
                # If are above certain epoch, we may want to stop jacobian regularization training
                # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
                # will be dominating and hurt performance!)
                factor = 0
            else:
                factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
            compute_jac_loss = (np.random.uniform(0, 1) < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
            delta_f_thres = random.randint(-config.DEQ.RAND_F_THRES_DELTA, 1) if (
                    config.DEQ.RAND_F_THRES_DELTA > 0 and compute_jac_loss) else 0
            f_thres = config.DEQ.F_THRES + delta_f_thres
            b_thres = config.DEQ.B_THRES
            # print(labels.shape)
            # exit()
            model.zero_grad()
            losses, jac_loss, preds, _ = model(
                images,
                labels,
                train_step=global_steps,
                compute_jac_loss=compute_jac_loss,
                f_thres=f_thres,
                b_thres=b_thres,
                writer=writer_dict,
                logger = logger)
            
            jac_loss = jac_loss.mean()
            loss = (weights[0] * losses[0] + weights[1] * losses[1])
            
            # compute gradient and do update step

            if factor > 0:
                (loss + factor * jac_loss).backward()
            else:
                loss.backward()

            optimizer.step()
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_steps)
            lr_scheduler.step()
            #preds = preds.squeeze(1)
            iou_lis = iou(labels.long(), preds, iou_lis, config.DATASET.NUM_CLASSES)
            dice_lis = dice(labels.long(), preds, dice_lis, config.DATASET.NUM_CLASSES)

            CE_metric.update(losses[0].item())
            Tver_metric.update(losses[1].item())
            if compute_jac_loss:
                jac_metric.update(jac_loss.item())

            global_steps += 1
            writer_dict['train_global_steps'] = global_steps
            logger.info(f"Step: {global_steps}")
            pdar.set_postfix({
                "CE loss": CE_metric.avg,
                "Tversky loss": Tver_metric.avg,
                "jac loss": jac_metric.avg,
                "iou": np.array(iou_lis).mean(),
                "dice": np.array(dice_lis).mean()
            })
    return CE_metric.avg, Tver_metric.avg, np.array(iou_lis).mean(), np.array(dice_lis).mean()


@torch.no_grad()
def validate(config, testloader, model, epoch, writer_dict, device, spectral_radius_mode=False, logger = None):
    model.eval()
    if logger is not None:
        logger.info("Evaluating")
    CE_metric = AverageMeter()
    Tver_metric = AverageMeter()
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    pdar = tqdm(testloader, desc="Evaluating")
    iou_lis = []
    dice_lis = []
    for batch in pdar:
        image, label = batch
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            losses, _, pred, _ = model(
                image,
                label,
                train_step=(-1 if epoch < 0 else global_steps),
                compute_jac_loss=False,
                spectral_radius_mode=spectral_radius_mode,
                writer=writer,
                logger = logger
            )
            iou_lis = iou(label.long(), pred, iou_lis, config.DATASET.NUM_CLASSES)
            dice_lis = dice(label.long(), pred, dice_lis, config.DATASET.NUM_CLASSES)

            # iou = get_iou(label, pred, num_class=config.DATASET.NUM_CLASSES)
            CE_metric.update(losses[0].item())
            Tver_metric.update(losses[1].item())
        pdar.set_postfix({
            "CE loss": CE_metric.avg,
            "Tversky loss": Tver_metric.avg,
            "iou": np.array(iou_lis).mean(),
            "dice": np.array(dice_lis).mean(),
        })
    return CE_metric.avg, Tver_metric.avg, np.array(iou_lis).mean(), np.array(dice_lis).mean()


@torch.no_grad()
def inference(model, loader, num_class, device):
    model.eval()
    loss_metric = AverageMeter()
    # iou_metric = AverageMeter()
    # dice_metric = AverageMeter()
    iou_lis = []
    dice_lis = []
    pdar = tqdm(loader, desc="Inference")
    out_img = []
    for i, (image, label) in enumerate(pdar):
        size = image.shape[-2:]
        image = image.float().to(device)
        label = label.to(device)
        loss, _, pred, _ = model(
            image,
            label,
            train_step=1,
            compute_jac_loss=False,
            spectral_radius_mode=False
        )
        # iou = get_iou(label, pred, num_class)
        # dice = get_dice_score(label, pred, num_class)
        # dice_metric.update(dice.item())
        iou_lis = iou(label, pred, iou_lis, num_class)
        dice_lis = dice(label, pred, dice_lis, num_class)
#        loss_metric.update(loss.item())
        # iou_metric.update(iou.item())
        pred = F.interpolate(pred, size=(size[0], size[1]), mode="bilinear", align_corners=True)
        image_pred = torch.argmax(pred, dim=1)
        out_img.append((image, image_pred, label))
        pdar.set_postfix({
            "iou": np.array(iou_lis).mean(),
            "loss": loss_metric.avg,
            "dice": np.array(dice_lis).mean()
        })
    return np.mean(iou_lis), np.mean(dice_lis), out_img
