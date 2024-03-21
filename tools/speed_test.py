"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/tools/condlanenet/speed_test.py
"""
import argparse

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv import DictAction
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Speed test")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--filename", type=str)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--n_iter_warmup", type=int, default=1000)
    parser.add_argument("--n_iter_test", type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # build the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    print("Preparing image", args.filename)
    img = cv2.imread(args.filename)
    cuty = cfg.crop_bbox[1] if "crop_bbox" in cfg else 0
    img = img[cuty:, ...]
    img = cv2.resize(img, cfg.img_scale)
    mean = np.array(cfg.img_norm_cfg.mean)
    std = np.array(cfg.img_norm_cfg.std)
    img = mmcv.imnormalize(img, mean, std, False)
    img = torch.unsqueeze(torch.tensor(img).permute(2, 0, 1), 0).cuda()
    model = model.cuda().eval()
    img_metas = [dict()]

    with torch.no_grad():
        # warm up
        print(f"Warming up for {args.n_iter_warmup} iterations.")
        for i in range(args.n_iter_warmup):
            _ = model(img, img_metas, return_loss=False, rescale=True)

        # test
        print(f"Speed test for {args.n_iter_test} iterations.")
        timer = mmcv.Timer()
        for i in range(args.n_iter_test):
            _ = model(img, img_metas, return_loss=False, rescale=True)
        t = timer.since_last_check()
        fps = args.n_iter_test / t
        print("##########################")
        print(f" Elapsed time = {t:.2f}")
        print(f" total number of frames = {args.n_iter_test}")
        print(f" fps = {fps:.3f}")
        print("##########################")


if __name__ == "__main__":
    main()
