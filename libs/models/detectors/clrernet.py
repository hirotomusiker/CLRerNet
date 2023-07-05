from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import DETECTORS


@DETECTORS.register_module()
class CLRerNet(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        """CLRerNet detector."""
        super(CLRerNet, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg
        )

    def forward_train(self, img, img_metas, **kwargs):
        """Coming soon.."""
        raise NotImplementedError("Training is not supported yet!")

    def forward_test(self, img, img_metas, **kwargs):
        """
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            img_metas (List[dict]): Meta dict containing image information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        assert (
            img.shape[0] == 1 and len(img_metas) == 1
        ), "Only single-image test is supported."
        img_metas[0]['batch_input_shape'] = tuple(img.size()[-2:])

        x = self.extract_feat(img)
        output = self.bbox_head.simple_test(x)
        result_dict = {
            'result': output,
            'meta': img_metas[0],
        }
        return [result_dict]  # assuming batch size is 1
