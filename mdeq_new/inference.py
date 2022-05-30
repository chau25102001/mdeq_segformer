import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from core.seg_function import inference
from torch.utils.data import DataLoader
from datasets.polyb import PolybDatset
from seg_train import obj
from models.mdeq import get_seg_net
from core.seg_criterion import PolypCriterion
from utils.utils import FullModel
import json
import yaml
import warnings
warnings.filterwarnings("ignore")

def plot_img(raw, pred, gt, path):
    for i, (r, p, g) in enumerate(zip(raw, pred, gt)):
        r = PolybDatset.denormlize(r.cpu().permute(1, 2, 0).numpy())
        p = p.cpu().numpy()
        g = g.cpu().numpy()
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(r)
        ax[1].imshow(p)
        ax[2].imshow(np.reshape(g, (352,352,1)))
        plt.show()
        plt.savefig(f"{path}_{i}.png")
        plt.close()


# def scale_img(img_raw, img_pred, labels):
#     label_size = labels.shape[-2:]
#     imgs_size = img_pred.shape[-2:]
#     if imgs_size != label_size:
#         img_pred = F.interpolate(img_pred, size=(label_size[0], label_size[1]), mode='bilinear', align_corners=True)
#         img_raw = F.interpolate(img_raw, size=(label_size[0], label_size[1]), mode='bilinear', align_corners=True)
#     img_raw = PolybDatset.denormlize(img_raw).cpu().numpy().transpose(0, 2, 3, 1)
#     img_pred = img_pred.cpu().numpy().squeeze(1)
#     labels = labels.cpu().numpy()
#     return img_raw, img_pred, labels


if __name__ == "__main__":
    device =  "cuda:0"
   # device = "cpu"
    config = yaml.load(open("experiments/seg_mdeq_SMALL.yaml", "r"), Loader=yaml.FullLoader)
    config = json.loads(json.dumps(config), object_hook=obj)
    criterion = PolypCriterion(0.4, 0.6, smooth=1)
    model = get_seg_net(config)
    model = FullModel(model, criterion)
    ckpt = torch.load("runs/exp60/checkpoint_best.pt", map_location=device)
    # model.load_state_dict(torch.load("runs/exp/checkpoint_best.pt"))
    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(ckpt["state_dict"])
    root = "TestDataset"
    for file in ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB",
                   "ETIS-LaribPolypDB", "Kvasir"]:
        path = os.path.join(root, file)
        testset = PolybDatset(
            root_path=path,
            # root_path=["TestDataset/CVC-300"],
            img_subpath="images",
            label_subpath="masks",
            img_size=352,
            cache_train=True,
            use_aug=False,
            use_cutmix=False
        )

        test_loader = DataLoader(
            testset,
            shuffle=False,
            batch_size=30,
            pin_memory=True,
            # drop_last=True
        )
        print(file)
        iou, dice, out =  inference(model, test_loader, 2, device)
        print(file)
        print("iou:", iou)
        print("dice", dice)
    #    if not os.path.exists(os.path.join("out_image", file)):
   #         os.makedirs(os.path.join("out_image", file))
 #       for k, (i, p, l) in enumerate(out):
  #          plot_img(i, p, l, f"out_image/{file}/test_{k}")
