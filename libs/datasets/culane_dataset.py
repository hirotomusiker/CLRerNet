"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/culane_dataset.py
https://github.com/open-mmlab/mmengine/blob/v0.10.4/mmengine/dataset/base_dataset.py
"""


from pathlib import Path

import cv2
import numpy as np
from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmengine.logging import MMLogger
from tqdm import tqdm

from mmengine.dataset import BaseDataset
from torch.utils.data import Dataset
from libs.datasets.metrics.culane_metric import eval_predictions
from libs.datasets.pipelines import Compose


@DATASETS.register_module()
class CulaneDataset(Dataset):
    """Culane Dataset class."""

    def __init__(
        self,
        data_root,
        data_list,
        pipeline,
        diff_file=None,
        diff_thr=15,
        test_mode=True,
        y_step=2,
        **kwargs
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
        # read image list
        self.diffs = np.load(diff_file)["data"] if diff_file is not None else []
        self.diff_thr = diff_thr
        self.img_infos, self.annotations, self.mask_paths = self.parse_datalist(
            data_list
        )
        self.img_infos = self.img_infos
        self.metainfo = {}
        print(len(self.img_infos), "data are loaded")
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # build data pipeline
        self.pipeline = Compose(pipeline)
        self.list_path = data_list

    def parse_datalist(self, data_list):
        """
        Read image data list.
        Args:
            data_list (str): Data list file path.
        Returns:
            List[str]: List of image paths.
        """
        img_infos, annotations, mask_paths = [], [], []
        with open(data_list) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if len(self.diffs) > 0 and self.diffs[i] < self.diff_thr:
                    continue
                img_paths = line.strip().split(" ")
                img_infos.append(img_paths[0].lstrip("/"))
                if not self.test_mode:
                    anno_path = img_paths[0].replace(".jpg", ".lines.txt")
                    annotations.append(anno_path.lstrip("/"))
                    if len(img_paths) > 1:
                        mask_paths.append(img_paths[1].lstrip("/"))
                    else:
                        mask_paths.append(None)
        return img_infos, annotations, mask_paths

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def prepare_train_img(self, idx):
        """
        Read and process the image through the transform pipeline for training.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        imgname = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
        sub_img_name = self.img_infos[idx]
        img = cv2.imread(imgname)
        ori_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(idx)
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=ori_shape,
            ori_shape=ori_shape,
            gt_masks=None,
        )
        if self.mask_paths[0]:
            results["gt_masks"] = self.load_mask(idx)
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

    def load_mask(self, idx):
        """
        Read a segmentation mask for training.
        Args:
            idx (int): Data index.
        Returns:
            numpy.ndarray: segmentation mask.
        """
        maskname = str(Path(self.img_prefix).joinpath(self.mask_paths[idx]))
        mask = cv2.imread(maskname, cv2.IMREAD_UNCHANGED)
        return mask

    def load_labels(self, idx):
        """
        Read a ground-truth lane from an annotation file.
        Args:
            idx (int): Data index.
        Returns:
            List[list]: list of lane point lists.
            list: class id (=1) for lane instances.
            list: instance id (start from 1) for lane instances.
        """
        anno_dir = str(Path(self.img_prefix).joinpath(self.annotations[idx]))
        shapes = []
        with open(anno_dir, "r") as anno_f:
            lines = anno_f.readlines()
            for line in lines:
                coords = []
                coords_str = line.strip().split(" ")
                for i in range(len(coords_str) // 2):
                    coord_x = float(coords_str[2 * i])
                    coord_y = float(coords_str[2 * i + 1])
                    coords.append(coord_x)
                    coords.append(coord_y)
                if len(coords) > 3:
                    shapes.append(coords)
        id_classes = [1 for i in range(len(shapes))]
        id_instances = [i + 1 for i in range(len(shapes))]
        return shapes, id_classes, id_instances

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data