"""
Alaug interface for albumentations transformations.
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/pipelines/alaug.py
"""

import collections
import copy

import albumentations as al
import numpy as np
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Alaug(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        # init as None
        self.__augmentor = None
        # put transforms in a list
        self.transforms = []
        self.bbox_params = None
        self.keypoint_params = None

        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Compose':
                    self.get_al_params(transform['params'])
                else:
                    transform = self.build_transforms(transform)
                    if transform is not None:
                        self.transforms.append(transform)
            else:
                raise TypeError('transform must be a dict')
        self.build()

    def get_al_params(self, compose):
        if compose['bboxes']:
            self.bbox_params = al.BboxParams(
                format='pascal_voc',
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["bbox_labels"],
            )
        if compose['keypoints']:
            self.keypoint_params = al.KeypointParams(
                format='xy', remove_invisible=False
            )

    def build_transforms(self, transform):
        if transform['type'] == 'OneOf':
            transforms = transform['transforms']
            choices = []
            for t in transforms:
                parmas = {key: value for key, value in t.items() if key != 'type'}
                choice = getattr(al, t['type'])(**parmas)
                choices.append(choice)
            return getattr(al, 'OneOf')(transforms=choices, p=transform['p'])

        parmas = {key: value for key, value in transform.items() if key != 'type'}
        return getattr(al, transform['type'])(**parmas)

    def build(self):
        if len(self.transforms) == 0:
            return
        self.__augmentor = al.Compose(
            self.transforms,
            bbox_params=self.bbox_params,
            keypoint_params=self.keypoint_params,
        )

    def cal_sum_list(self, itmes, index):
        sum = 0
        for i in range(index):
            sum += itmes[i]
        return sum

    def is_sorted(self, points):
        for lane in points:
            lane_y = np.array(lane[1::2])
            if not np.all(lane_y[1:] < lane_y[:-1]):
                return False
        return True

    def __call__(self, data):
        data_org = copy.deepcopy(data)
        for i in range(30):
            data_aug = self.aug(data)
            if self.is_sorted(data_aug['gt_points']):
                return data_aug
            data = copy.deepcopy(data_org)
        raise ValueError("lane augmentation failed 30 times. modifying GT..")

    def aug(self, data):
        if self.__augmentor is None:
            return data
        img = data['img']
        bboxes = None
        keypoints = None
        masks = None
        if 'gt_bboxes' in data:
            gt_bboxes = data['gt_bboxes']
            bboxes = []
            bbox_labels = []
            for i in range(np.shape(gt_bboxes)[0]):
                if (gt_bboxes[i, 0] == gt_bboxes[i, 2]) | (
                    gt_bboxes[i, 1] == gt_bboxes[i, 3]
                ):
                    pass
                else:
                    b = gt_bboxes[i, :]
                    b = np.concatenate((b, [i]))
                    bboxes.append(b)
                    bbox_labels.append(data['gt_labels'][i])
        else:
            bboxes = None
            bbox_labels = None
        if 'gt_masks' in data:
            masks = data['gt_masks']
        else:
            masks = None
        if 'gt_keypoints' in data:
            keypoints = data["gt_keypoints"]
            kp_group_num = len(keypoints)
            # run aug
            keypoints_index = []
            for k in keypoints:
                keypoints_index.append(int(len(k) / 2))
            keypoints_val = []
            for kps in keypoints:
                num = int(len(kps) / 2)
                for i in range(num):
                    keypoints_val.append(kps[2 * i : 2 * i + 2])
            num_keypoints = len(kps) // 2
        else:
            keypoints_val = None

        if 'gt_points' in data:
            points = data["gt_points"]
            # run aug
            points_index = []
            for k in points:
                points_index.append(int(len(k) / 2))
            points_val = []
            for pts in points:
                num = int(len(pts) / 2)
                for i in range(num):
                    points_val.append(pts[2 * i : 2 * i + 2])
            num_keypoints = len(points_val) // 2
            if keypoints_val is None:
                keypoints_val = points_val
            else:
                keypoints_val = keypoints_val + points_val

        aug = self.__augmentor(
            image=img,
            keypoints=keypoints_val,
            bboxes=bboxes,
            mask=masks,
            bbox_labels=bbox_labels,
        )
        data['img'] = aug['image']
        data['img_shape'] = data['img'].shape
        if 'gt_bboxes' in data:
            if aug['bboxes']:
                data['gt_bboxes'] = np.array(aug['bboxes'])[:, :4]
                data['gt_labels'] = np.array(aug['bbox_labels'])
            else:
                return None
        if 'gt_masks' in data:
            data['gt_masks'] = [np.array(aug['mask'])]
        if 'gt_keypoints' in data:
            kp_list = [[0 for j in range(i * 2)] for i in keypoints_index]
            for i in range(len(keypoints_index)):
                for j in range(keypoints_index[i]):
                    kp_list[i][2 * j] = aug['keypoints'][
                        self.cal_sum_list(keypoints_index, i) + j
                    ][0]
                    kp_list[i][2 * j + 1] = aug['keypoints'][
                        self.cal_sum_list(keypoints_index, i) + j
                    ][1]
            data['gt_keypoints'] = []
            valid = []
            for i in range(kp_group_num):
                index = int(aug['bboxes'][i][-1])
                valid.append(index)
                data['gt_keypoints'].append(kp_list[index])
            data['gt_keypoints_ignore'] = data['gt_keypoints_ignore'][valid]

        if 'gt_points' in data:
            start_idx = num_keypoints if 'gt_keypoints' in data else 0
            points = aug['keypoints'][start_idx:]
            kp_list = [[0 for j in range(i * 2)] for i in points_index]
            for i in range(len(points_index)):
                for j in range(points_index[i]):
                    kp_list[i][2 * j] = points[self.cal_sum_list(points_index, i) + j][
                        0
                    ]
                    kp_list[i][2 * j + 1] = points[
                        self.cal_sum_list(points_index, i) + j
                    ][1]
            data['gt_points'] = kp_list

        if 'gt_bboxes' in data and kp_group_num == 0:
            return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
