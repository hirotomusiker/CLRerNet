# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import cv2
import numpy as np
from mmcv import Config
from mmcv import DictAction
from mmdet.datasets import build_dataset
from tqdm import tqdm

from libs.datasets.metrics.culane_metric import draw_lane
from libs.datasets.metrics.culane_metric import interp


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--savedir", default="seg_mask", type=str)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    return args


def make_seg_img(dataset, idx, rel_save_dir, width=30):
    """Draw segmentation masks and save the image.
    Args:
        dataset (CurvelanesDataset): dataset module.
        idx (int): data index.
        rel_save_dir (str): relative path for seg images
            from `dataset.data_root`.
    Returns:
        str: image path + seg image path to be written in
            `train_seg.txt`.
    """
    save_dir = Path(dataset.img_prefix) / rel_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    img_info = dataset.img_infos[idx]
    imgname = Path(dataset.img_prefix) / img_info
    img_tmp = cv2.imread(str(imgname))
    ori_shape = img_tmp.shape
    kps, id_classes, id_instances = dataset.load_labels(idx)
    kps = [
        [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
        for lane in kps
    ]
    lanes = np.array(
        [interp(kp, n=5) for kp in kps], dtype=object
    )  # (4, 50, 2)
    img = np.zeros(ori_shape, dtype=np.uint8)
    for lane in lanes:
        # Draw single-class lane segmentation mask
        img = draw_lane(
            lane, img=img, img_shape=ori_shape, width=width, color=(1, 1, 1)
        )
    savename = save_dir / Path(imgname).name
    rel_savename = Path(rel_save_dir) / imgname.name
    line = img_info + " " + str(rel_savename)
    cv2.imwrite(str(savename), img)
    return line


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # build the dataloader
    cfg.data.train.data_list = Path(cfg.data.train.data_root) / "train.txt"
    dataset = build_dataset(cfg.data.train)

    # Draw segmentation mask images
    lines = []
    for idx in tqdm(range(len(dataset))):
        lines.append(make_seg_img(dataset, idx, args.savedir))
    list_path = Path(dataset.img_prefix) / "train_seg.txt"
    with open(list_path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
