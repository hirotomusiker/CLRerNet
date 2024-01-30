"""
    config file of the CULane dataset for CondLaneNet
    Adapted from:
    https://github.com/aliyun/conditional-lane-detection/blob/master/configs/condlanenet/curvelanes/curvelanes_medium_train.py
"""

dataset_type = "CurvelanesDataset"
data_root = "dataset/curvelanes"
img_scale = (800, 320)

img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False
)

compose_cfg = dict(bboxes=False, keypoints=True, masks=True)

# data pipeline settings
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="RGBShift",
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0,
            ),
            dict(
                type="HueSaturationValue",
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0,
            ),
        ],
        p=0.7,
    ),
    dict(type="JpegCompression", quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=1.0),
            dict(type="MedianBlur", blur_limit=3, p=1.0),
        ],
        p=0.2,
    ),
    dict(type="RandomBrightness", limit=0.2, p=0.6),
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6,
    ),
    dict(
        type="RandomResizedCrop",
        height=img_scale[1],
        width=img_scale[0],
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="albumentation", pipelines=train_al_pipeline, cut_unsorted=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectCLRNet",
        max_lanes=16,
        keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "eval_shape",
            "img_shape",
            "img_norm_cfg",
            "ori_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "lanes",
        ],
    ),
]

val_pipeline = [
    dict(type="albumentation", pipelines=val_al_pipeline, cut_unsorted=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectCLRNet",
        max_lanes=16,
        keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "ori_shape",
            "img_shape",
            "gt_points",
            "crop_shape",
            "crop_offset",
        ],
    ),
]

data = dict(
    samples_per_gpu=32,  # medium
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root + "/train/",
        data_list=data_root + "/train/train_seg.txt",
        diff_thr=0,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root + "/valid/",
        data_list=data_root + "/valid/valid.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root + "/valid/",
        data_list=data_root + "/valid/valid.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
