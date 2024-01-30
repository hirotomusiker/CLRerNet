"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/pipelines/compose.py
"""
import collections

from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES

from .alaug import Alaug


@PIPELINES.register_module(force=True)
class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform["type"] == "albumentation":
                    transform = Alaug(
                        transform["pipelines"],
                        cut_unsorted=transform["cut_unsorted"],
                    )
                else:
                    transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
