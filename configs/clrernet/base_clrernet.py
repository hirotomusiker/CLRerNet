model = dict(
    type='CLRerNet',
    backbone=dict(
        type='DLANet',
        dla='dla34',
        pretrained=False,
    ),
    neck=dict(
        type='CLRerNetFPN',
        in_channels=[128, 256, 512],
        out_channels=64,
        num_outs=3,
    ),
    bbox_head=dict(
        type='CLRerHead',
        anchor_generator=dict(
            type='CLRerNetAnchorGenerator',
            num_priors=192,
            num_points=72,
        ),
        img_w=800,
        img_h=320,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        attention=dict(type='ROIGather'),
    ),
    # training and testing settings
    train_cfg=dict(),  # coming soon
    test_cfg=dict(
        # conf threshold is obtained from cross-validation
        # of the train set. The following value is
        # for CLRerNet w/ DLA34 & EMA model.
        conf_threshold=0.43,
        use_nms=True,
        as_lanes=True,
        nms_thres=50,
        nms_topk=4,
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
    ),
)
