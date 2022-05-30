import yaml
import torch
# from config.default import update_config
from config import config


def update_config(cfg, yaml_file):
    cfg.defrost()
    cfg.merge_from_file(yaml_file)

    # if args.modelDir:
    #     cfg.OUTPUT_DIR = args.modelDir
    #
    # if args.logDir:
    #     cfg.LOG_DIR = args.logDir
    #
    # if args.dataDir:
    #     cfg.DATA_DIR = args.dataDir
    #
    # if args.testModel:
    #     cfg.TEST.MODEL_FILE = args.testModel
    #
    # if args.percent < 1:
    #     cfg.PERCENT = args.percent
    #
    # cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == "__main__":
    p2f = "experiments/seg_mdeq_SMALL.yaml"
    cfg = yaml.load(open(p2f, "r"), Loader=yaml.Loader)
    update_config(config, cfg)
