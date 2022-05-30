import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from models.mdeq import get_seg_net
from datasets.polyb import PolybDatset
from core.seg_criterion import PolypCriterion
from core.seg_function import train, validate
from utils.utils import FullModel, increment_path, intersect_dicts
import yaml
import json
import warnings
import logging
class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

    def __getitem__(self, item):
        return self.__dict__[item]


def main():
    config = yaml.load(open("experiments/seg_mdeq_SMALL.yaml", "r"), Loader=yaml.FullLoader)
    config = json.loads(json.dumps(config), object_hook=obj)
    final_output_dir = increment_path(config.OUTPUT_DIR, True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    writer_dict = {
        'writer': SummaryWriter(final_output_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    logging.basicConfig(filename = 'log_solver.txt', level = logging.INFO, format = "%(message)s")
    logger = logging.getLogger("logger")
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    model = get_seg_net(config)

    trainset = PolybDatset(
        root_path="TrainDataset/train",
        img_subpath="image",
        label_subpath="mask",
        img_size=config.TRAIN.IMAGE_SIZE,
        cache_train=True,
        use_aug=True,
        use_cutmix=0.5
    )
    testset = PolybDatset(
        root_path=["TestDataset/CVC-300", "TestDataset/CVC-ClinicDB", "TestDataset/CVC-ColonDB",
                   "TestDataset/ETIS-LaribPolypDB", "TestDataset/Kvasir"],
        img_subpath="images",
        label_subpath="masks",
        img_size=config.TEST.IMAGE_SIZE,
        cache_train=True,
        use_aug=False,
        use_cutmix=False
    )
    train_loader = DataLoader(
        trainset,
        shuffle=config.TRAIN.SHUFFLE,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        testset,
        shuffle=False,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        pin_memory=True
    )

    # criterion
    criterion = PolypCriterion(0.4, 0.6, smooth=1)
    model = FullModel(model, criterion)
    ckpt = torch.load("runs/exp60/checkpoint_best.pt", map_location=device)
    # print(ckpt.keys())
    # model.load_state_dict(torch.load("runs/exp/checkpoint_best.pt"))
    model = torch.nn.DataParallel(model)
    model = model.to(device)
   # state_dict = model.module.model.state_dict()
   # intersec = intersect_dicts(state_dict, ckpt)
   # print(intersec)
    model.module.load_state_dict(ckpt["state_dict"])

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WD,
        nesterov=config.TRAIN.NESTEROV
    )

    # epoch_iters = trainset.__len__() // config.TRAIN.BATCH_SIZE_PER_GPU
    end_epoch = config.TRAIN.END_EPOCH
    # best_mIoU = ckpt["test_iou"]
    best_iou_dice = 0
   # last_epoch = 71
    last_epoch = torch.load("runs/exp60/checkpoint_last.pt", map_location = device)['epoch']
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-8)

    for epoch in range(last_epoch, end_epoch):
       # if epoch < 5:
        #    for g in optimizer.param_groups:
         #       g["lr"] = np.interp(epoch, [1e-6, 5 - 1], [0, config.TRAIN.LR])
        train_ce, train_tver, train_iou, train_dice = train(
            config,
            epoch,
            end_epoch,
            train_loader,
            optimizer,
            writer_dict,
            model,
            device,
            logger
        )
       # if epoch >= 5:
        #    lr_scheduler.step()

        torch.cuda.empty_cache()

        test_ce, test_tver, test_iou, test_dice = validate(
            config,
            test_loader,
            model,
            epoch,
            writer_dict,
            device,
            spectral_radius_mode=config.DEQ.SPECTRAL_RADIUS_MODE)

        checkpoint = {
            "epoch": epoch,
            "test_loss": test_ce + test_dice,
            "test_iou": test_iou,
            "test_dice": test_dice,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }

        if (test_iou + test_dice) > 2 * best_iou_dice:
            best_iou_dice = (test_iou + test_dice) / 2
            torch.save(checkpoint, os.path.join(final_output_dir, "checkpoint_best.pt"))
        torch.save(checkpoint, os.path.join(final_output_dir, "checkpoint_last.pt"))
        tags = ["train/CrossEntropy", "train/Tversky", "train/iou", "train/dice", "test/CrossEntropy", "test/Tversky",
                "test/iou", "test/dice", "lr"]
        values = [train_ce, train_tver, train_iou, train_dice, test_ce, test_tver, test_iou, test_dice,
                  optimizer.param_groups[0]["lr"]]
        for tag, value in zip(tags, values):
            writer_dict["writer"].add_scalar(tag, value, epoch)
        logger.info("End of epoch \n")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
