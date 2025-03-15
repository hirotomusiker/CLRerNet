from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS


@MODELS.register_module()
class CLRerNet(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        """CLRerNet detector."""
        super(CLRerNet, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, data_preprocessor, init_cfg
        )

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas)
        return losses

    def predict(self, img, data_samples, **kwargs):
        """
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            img_metas (List[dict]): Meta dict containing image information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        for i in range(len(data_samples)):
            data_samples[i].metainfo["batch_input_shape"] = tuple(img.size()[-2:])

        x = self.extract_feat(img)
        outputs = self.bbox_head.predict(x, data_samples)
        return outputs
