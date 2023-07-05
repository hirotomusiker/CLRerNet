"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/culane_dataset.py
"""


import os
from pathlib import Path
from tqdm import tqdm
import shutil

import cv2
import numpy as np

from mmdet.datasets.custom import CustomDataset
from mmdet.utils import get_root_logger
from mmdet.datasets.builder import DATASETS

from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import eval_predictions


@DATASETS.register_module
class CulaneDataset(CustomDataset):
    """Culane Dataset class."""
    def __init__(
        self,
        data_root,
        data_list,
        pipeline,
        test_mode=True,
        y_step=2,
    ):
        """
        Args:
            data_root (str): Dataset root path.
            data_list (str): Dataset list file path.
            pipeline (List[mmcv.utils.config.ConfigDict]):
                Data transformation pipeline configs.
            test_mode (bool): Test flag.
            y_step (int): Row interval (in the original image's y scale) to sample
                the predicted lanes for evaluation.

        """

        self.img_prefix = data_root
        self.test_mode = test_mode
        self.ori_w, self.ori_h = 1640, 590
        # read image list
        self.img_infos = self.parse_datalist(data_list)
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # build data pipeline
        self.pipeline = Compose(pipeline)
        self.result_dir = 'tmp'
        self.list_path = data_list
        self.test_categories_dir = str(Path(data_root).joinpath('list/test_split/'))
        self.y_step = y_step

    def parse_datalist(self, data_list):
        """
        Read image data list.
        Args:
            data_list (str): Data list file path.
        Returns:
            img_infos (List[str]): List of image paths.
        """
        img_infos = []
        with open(data_list) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                img_paths = line.strip().split(' ')
                img_infos.append(img_paths[0].lstrip('/'))
        return img_infos

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def prepare_test_img(self, idx):
        """
        Read and process the image through the transform pipeline for test.
        Args:
            idx (int): Data index.
        Returns:
            results (dict): Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        img_name = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
        sub_img_name = self.img_infos[idx]
        img = cv2.imread(img_name)
        ori_shape = img.shape
        results = dict(
            filename=img_name,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=ori_shape,
            ori_shape=ori_shape,
        )
        return self.pipeline(results)

    def evaluate(self, results, metric='F1'):
        """
        Write prediction to txt files for evaluation and
        evaluate them with labels.
        Args:
            results (List[dict]): All inference results containing:
                result (dict): contains 'lanes' and 'scores'.
                meta (dict): contains meta information.
            metric (str): Metric type to evaluate. (not used)
        Returns:
            results (dict): Evaluation result dict containing
                F1, precision, recall, etc. on the specified IoU thresholds.

        """
        for result in tqdm(results):
            lanes = result['result']['lanes']
            dst_path = (
                Path(self.result_dir)
                .joinpath(result['meta']['sub_img_name'])
                .with_suffix('.lines.txt')
            )
            dst_path.parents[0].mkdir(parents=True, exist_ok=True)
            with open(str(dst_path), 'w') as f:
                output = self.get_prediction_string(lanes)
                if len(output) > 0:
                    print(output, file=f)

        results = eval_predictions(
            self.result_dir,
            self.img_prefix,
            self.list_path,
            self.test_categories_dir,
            logger=get_root_logger(log_level='INFO'),
        )
        shutil.rmtree(self.result_dir)
        return results

    def get_prediction_string(self, lanes):
        """
        Convert lane instance structure to prediction strings.
        Args:
            lanes (List[Lane]): List of lane instances in `Lane` structure.
        Returns:
            out_string (str): Output string.
        """
        ys = np.arange(0, self.ori_h, self.y_step) / self.ori_h
        out = []
        for lane in lanes:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            if len(lane_xs) < 2:
                continue
            lane_str = ' '.join(
                ['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)]
            )
            if lane_str != '':
                out.append(lane_str)
        return '\n'.join(out) if len(out) > 0 else ''
