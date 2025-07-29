"""
    config file of the LaneATT model
    modified based on:
    https://github.com/lucastabelini/LaneATT/blob/c97f5731901c90b87c4b561a68a77ea893268aca/cfgs/laneatt_culane_resnet34.yml
"""

model = dict(
    type="LaneATT",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type="ResNet",
        depth=34,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(3,),
        # frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet34"),
    ),
    bbox_head=dict(
        type="LaneATTHead",
        S=72,
        img_w=640,
        img_h=360,
        anchor_generator=dict(
            type="LaneATTAnchorGenerator",
            S=72,
            img_w=640,
            img_h=360,
            stride=32,
            anchors_freq_path="culane_anchors_freq.pt",
            topk_anchors=1000,
            anchor_feat_channels=64,
        ),
    ),
    train_cfg=dict(
        nms=dict(
            type="LaneATTNMS",
            nms_thres=15,
            nms_topk=3000,
            conf_threshold=0.0,
        ),
        reg_assigner=dict(
            type="LaneATTAssigner",
            t_pos=15.0,
            t_neg=20.0,
            iou_cost=None,
            iou_pos=0.7,
            iou_neg=0.5,
            assign_missing_gt=True,
        ),
        cls_assigner=dict(
            type="LaneATTAssigner",
            t_pos=15.0,
            t_neg=20.0,
            iou_pos=0.7,
            iou_neg=0.5,
            assign_missing_gt=False,
            iou_cost=dict(
                type="LaneIoUCost",
                lane_width=7.5 / 800,
                use_pred_start_end=False,
                use_giou=True,
                img_h=590,
                img_w=1640,
            ),
        ),
    ),
    test_cfg=dict(
        nms=dict(
            type="LaneATTNMS",
            nms_thres=50,
            nms_topk=4,
            conf_threshold=0.40,
        ),
    ),
)
