"""
    config file of the CULane dataset for CondLaneNet
    modified based on:
    https://github.com/lucastabelini/LaneATT/blob/c97f5731901c90b87c4b561a68a77ea893268aca/cfgs/laneatt_culane_resnet34.yml
"""

dataset_type = "CulaneDataset"
data_root = "dataset/culane"
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False
)
img_scale = (640, 360)

train_compose = dict(bboxes=False, keypoints=True, masks=False)

# data pipeline settings
train_al_pipeline = [
    dict(type="Compose", params=train_compose),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(
        type="IAAAffine",
        p=1.0,
        scale=(0.85, 1.15),
        rotate=(-6.0, 6.0),  # this sometimes breaks lane sorting
        translate_percent=0.04,
    ),
    dict(type="HorizontalFlip", p=0.5),
]

val_al_pipeline = [
    dict(type="Compose", params=train_compose),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="albumentation", pipelines=train_al_pipeline),
    dict(
        type="PackLaneATTInputs",
        # keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "ori_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "lanes",
        ],
    ),
]

val_pipeline = [
    dict(type="albumentation", pipelines=val_al_pipeline),
    dict(
        type="PackLaneATTInputs",
        # keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
        ],
    ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + "/list/train.txt",
        diff_file=data_root + "/list/train_diffs.npz",
        diff_thr=15,
        pipeline=train_pipeline,
        test_mode=False,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + "/list/val.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + "/list/test.txt",
        pipeline=val_pipeline,
        test_mode=True,
        laneatt=True,
    ),
)

val_evaluator = dict(
    type="CULaneMetric",
    data_root=data_root,
    data_list=data_root + "/list/val.txt",
)

test_evaluator = dict(
    type="CULaneMetric",
    data_root=data_root,
    data_list=data_root + "/list/test.txt",
)
