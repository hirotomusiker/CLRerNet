dataset_type = 'CulaneDataset'
data_root = "dataset/culane"
crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
compose_cfg = dict(bboxes=False, keypoints=True, masks=True)

val_al_pipeline = [
    dict(type='Compose', params=compose_cfg),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

val_pipeline = [
    dict(type='albumentation', pipelines=val_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectCLRNet',
        keys=['img'],
        meta_keys=[
            'filename',
            'sub_img_name',
            'ori_shape',
            'img_shape',
            'img_norm_cfg',
        ],
    ),
]

data = dict(
    workers_per_gpu=0,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/val.txt',
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/test.txt',
        pipeline=val_pipeline,
        test_mode=True,
    )
)
