import argparse
import os
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from models.mdeq_seformer_thaysang import get_seg_net
from datasets.polyb import PolybDatset
from core.seg_criterion import PolypCriterion
from core.seg_function import train, validate
from utils.utils import FullModel, increment_path, intersect_dicts
import logging
import yaml
import json
import warnings
class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

    def __getitem__(self, item):
        return self.__dict__[item]


def main():
    config = yaml.load(open("experiments/segfomer_mdeq_thaysang.yaml", "r"), Loader=yaml.FullLoader)
    config = json.loads(json.dumps(config), object_hook=obj)
    final_output_dir = increment_path(config.OUTPUT_DIR, True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    writer_dict = {
        'writer': SummaryWriter(final_output_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    logging.basicConfig(filename='log_solver_segfomer_thaysang.txt', level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("logger")
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    criterion = PolypCriterion(0.4, 0.6, smooth=1)

    model = get_seg_net(config)
    model = FullModel(model, criterion)
    ckpt = torch.load(os.path.join(config.OUTPUT_DIR,"checkpoint_best.pt"), map_location = device)

    model = torch.nn.DataParallel(model)
    model.module.load_state_dict(ckpt['state_dict'])
    model = model.to(device)


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
        # shuffle = False,
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


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.TRAIN.LR,
        # momentum=config.TRAIN.MOMENTUM,
        # weight_decay=config.TRAIN.WD,
        # nesterov=config.TRAIN.NESTEROV
    )

    # epoch_iters = trainset.__len__() // config.TRAIN.BATCH_SIZE_PER_GPU
    end_epoch = config.TRAIN.END_EPOCH
    best_mIoU = ckpt["test_iou"]
    best_iou_dice = ckpt['test_dice']

    current_epoch = ckpt['epoch']
    # current_epoch = 0
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= (end_epoch-current_epoch - 1) * len(train_loader) * 3, eta_min=1e-8)
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    global_steps = current_epoch * len(train_loader) * 3
    optimizer.load_state_dict(ckpt['optimizer'])
    for epoch in range(current_epoch, end_epoch):

        train_ce, train_tver, train_iou, train_dice = train(
            config,
            epoch,
            end_epoch,
            train_loader,
            optimizer,
            writer_dict,
            model,
            device,
            logger,
            lr_scheduler,
            weights=[0.35, 0.65]
        )
        torch.cuda.empty_cache()
        #lr_scheduler.step()
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
                "test/iou", "test/dice"]
        values = [train_ce, train_tver, train_iou, train_dice, test_ce, test_tver, test_iou, test_dice,
                  ]
        for tag, value in zip(tags, values):
            writer_dict["writer"].add_scalar(tag, value, epoch)
        logger.info("End of epoch \n")


if __name__ == '__main__':


    warnings.filterwarnings("ignore")
    main()
    # import matplotlib.pyplot as plt
    # config = yaml.load(open("experiments/segformer.yaml", "r"), Loader=yaml.FullLoader)
    # config = json.loads(json.dumps(config), object_hook=obj)
    #
    # trainset = PolybDatset(
    #     root_path="TrainDataset/train",
    #     img_subpath="image",
    #     label_subpath="mask",
    #     img_size=config.TRAIN.IMAGE_SIZE,
    #     cache_train=True,
    #     use_aug=False,
    #     use_cutmix=0
    # )
    # img, label = trainset[0]
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(img.permute(1,2,0))
    # ax[1].imshow(label.permute(1,2,0),cmap = 'gray')
    # plt.show()
    # config = yaml.load(open("experiments/seg_mdeq_SMALL.yaml", "r"), Loader=yaml.FullLoader)
    # config = json.loads(json.dumps(config), object_hook=obj)
    # model = get_seg_net(config)
    # criterion = PolypCriterion(0.4, 0.6, smooth=1)
    # model = FullModel(model, criterion)
    # model = torch.nn.DataParallel(model)
    # model = model.to('cpu')
    # x = torch.rand((1, 3, 352, 352))
    # y = torch.rand((1, 352, 352))
    # losses, jac_loss, preds, _ = model(x, y, train_step=-1, compute_jac_loss=True, f_thres=10, b_thres=10)
    # print(losses)
    # print(preds.shape)
