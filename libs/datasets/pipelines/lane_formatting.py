from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import Collect


@PIPELINES.register_module
class CollectCLRNet(Collect):
    def __init__(
        self,
        keys=None,
        meta_keys=None,
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        if 'lanes' in self.meta_keys:  # training
            raise NotImplementedError("Training is not supported yet!")
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data
