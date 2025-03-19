"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/tools/condlanenet/speed_test.py
"""
import argparse

import cv2
import mmengine
import torch
from mmdet.apis import init_detector
from mmengine.config import DictAction
from libs.datasets.pipelines import Compose

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=device)

    print("Preparing image", args.filename)
    img = cv2.imread(args.filename)
    ori_shape = img.shape

    cfg = model.cfg
    model.bbox_head.test_cfg.as_lanes = False
    model.bbox_head.test_cfg.extend_bottom = False

    test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    data = dict(
        filename=args.filename,
        sub_img_name=None,
        img=img,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=ori_shape,
        ori_shape=ori_shape,
    )
    data = test_pipeline(data)
    data = dict(
        inputs=[data["inputs"]],
        data_samples=[data["data_samples"]],
    )
    data_preprocessed = model.data_preprocessor(data, False)

    # forward the model
    with torch.no_grad():
        # warm up
        print(f"Warming up for {args.n_iter_warmup} iterations.")
        for i in range(args.n_iter_warmup):
            _ = model.predict(data_preprocessed["inputs"], data_preprocessed["data_samples"])

        # test
        print(f"Speed test for {args.n_iter_test} iterations.")
        timer = mmengine.Timer()
        for i in range(args.n_iter_test):
            _ = model.predict(data_preprocessed["inputs"], data_preprocessed["data_samples"])
        t = timer.since_last_check()
        fps = args.n_iter_test / t
        print("##########################")
        print(f" Elapsed time = {t:.2f}")
        print(f" total number of frames = {args.n_iter_test}")
        print(f" fps = {fps:.3f}")
        print("##########################")


if __name__ == "__main__":
    main()
