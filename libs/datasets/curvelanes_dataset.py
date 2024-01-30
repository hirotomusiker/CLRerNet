"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/curvelanes_dataset.py
"""
from pathlib import Path

import cv2
import numpy as np
from mmdet.datasets.builder import DATASETS
from tqdm import tqdm
from vega.metrics.pytorch.lane_metric import LaneMetricCore

from .culane_dataset import CulaneDataset


@DATASETS.register_module
class CurvelanesDataset(CulaneDataset):
    def prepare_train_img(self, idx):
        """
        Read and process the image through the transform pipeline for training.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        img_info = self.img_infos[idx]
        imgname = str(Path(self.img_prefix) / img_info)
        sub_img_name = img_info
        img_tmp = cv2.imread(imgname)
        ori_shape = img_tmp.shape
        if ori_shape == (1440, 2560, 3):
            img = np.zeros((800, 2560, 3), np.uint8)
            img[:800, :, :] = img_tmp[640:, ...]
            crop_shape = (800, 2560, 3)
            offset_y = -640
        elif ori_shape == (660, 1570, 3):
            img = np.zeros((480, 1570, 3), np.uint8)
            img[:480, :, :] = img_tmp[180:, ...]
            crop_shape = (480, 1570, 3)
            offset_y = -180
        elif ori_shape == (720, 1280, 3):
            img = np.zeros((352, 1280, 3), np.uint8)
            img[:352, :, :] = img_tmp[368:, ...]
            crop_shape = (352, 1280, 3)
            offset_y = -368
        else:
            return None
        img_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(idx, offset_y)
        eval_shape = (
            crop_shape[0] / ori_shape[0] * 224,
            224,
        )  # Used for LaneIoU calculation.
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=img_shape,
            ori_shape=ori_shape,
            eval_shape=eval_shape,
            crop_shape=crop_shape,
        )
        if self.mask_paths[0]:
            mask = self.load_mask(idx)
            mask = mask[-offset_y:, :, 0]
            mask = np.clip(mask, 0, 1)
            assert mask.shape[:2] == crop_shape[:2]
            results["gt_masks"] = mask

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """
        Read and process the image through the transform pipeline for test.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        imgname = str(Path(self.img_prefix) / self.img_infos[idx])
        sub_img_name = self.img_infos[idx]
        img_tmp = cv2.imread(imgname)
        ori_shape = img_tmp.shape

        if ori_shape == (1440, 2560, 3):
            img = np.zeros((800, 2560, 3), np.uint8)
            img[:800, :, :] = img_tmp[640:, ...]
            crop_shape = (800, 2560, 3)
            crop_offset = [0, 640]
        elif ori_shape == (660, 1570, 3):
            img = np.zeros((480, 1570, 3), np.uint8)
            crop_shape = (480, 1570, 3)
            img[:480, :, :] = img_tmp[180:, ...]
            crop_offset = [0, 180]
        elif ori_shape == (720, 1280, 3):
            img = np.zeros((352, 1280, 3), np.uint8)
            img[:352, :, :] = img_tmp[368:, ...]
            crop_shape = (352, 1280, 3)
            crop_offset = [0, 368]

        else:
            return None

        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=crop_shape,
            ori_shape=ori_shape,
            crop_offset=crop_offset,
            crop_shape=crop_shape,
        )
        return self.pipeline(results)

    @staticmethod
    def convert_coords_formal(lanes):
        res = []
        for lane in lanes:
            lane_coords = []
            for coord in lane:
                lane_coords.append({"x": coord[0], "y": coord[1]})
            res.append(lane_coords)
        return res

    def parse_anno(self, filename, formal=True):
        anno_dir = filename.replace(".jpg", ".lines.txt")
        annos = []
        with open(anno_dir, "r") as anno_f:
            lines = anno_f.readlines()
        for line in lines:
            coords = []
            numbers = line.strip().split(" ")
            coords_tmp = [float(n) for n in numbers]

            for i in range(len(coords_tmp) // 2):
                coords.append((coords_tmp[2 * i], coords_tmp[2 * i + 1]))
            annos.append(coords)
        if formal:
            annos = self.convert_coords_formal(annos)
        return annos

    def evaluate(
        self,
        results,
        metric="F1",
        logger=None,
        eval_width=224,
        eval_height=224,
        iou_thresh=0.5,
        lane_width=5,
    ):
        """
        Evaluate the test results.
        Args:
            results (List[dict]): All inference results containing:
                result (dict): contains 'lanes' and 'scores'.
                meta (dict): contains meta information.
            metric (str): Metric type to evaluate. (not used)
            evel_width (int): image width for IoU calculation.
            evel_height (int): image height for IoU calculation.
            iou_thresh (float): IoU threshold for evaluation.
            lane_width (int): lane virtual width to calculate IoUs.
        Returns:
            dict: Evaluation result dict containing
                F1, precision, recall, etc. on the specified IoU thresholds.

        """
        evaluator = LaneMetricCore(
            eval_width=eval_width,
            eval_height=eval_height,
            iou_thresh=iou_thresh,
            lane_width=lane_width,
        )
        evaluator.reset()
        for result in tqdm(results):
            ori_shape = result["meta"]["ori_shape"]
            filename = result["meta"]["filename"]
            pred = self.convert_coords_laneatt(result["result"], ori_shape)
            anno = self.parse_anno(filename)
            gt_wh = dict(height=ori_shape[0], width=ori_shape[1])
            predict_spec = dict(Lines=pred, Shape=gt_wh)
            target_spec = dict(Lines=anno, Shape=gt_wh)
            evaluator(target_spec, predict_spec)
        result_dict = evaluator.summary()
        print(result_dict)
        return result_dict

    def convert_coords_laneatt(self, result, ori_shape):
        lanes = result["lanes"]
        scores = result["scores"]
        ys = np.arange(0, ori_shape[0], 8) / ori_shape[0]
        out = []
        for lane, score in zip(lanes, scores):
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_shape[1]
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_shape[0]
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_coords = []
            for x, y in zip(lane_xs, lane_ys):
                lane_coords.append({"x": x, "y": y})
            out.append(lane_coords)
        return out
