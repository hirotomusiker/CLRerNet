"""
    config file of the medium version of LaneATT for CULane dataset
    modified based on:
    https://github.com/lucastabelini/LaneATT/blob/c97f5731901c90b87c4b561a68a77ea893268aca/cfgs/laneatt_culane_resnet34.yml
"""
_base_ = [
    "../base_laneatt.py",
    "dataset_culane_laneatt.py",
    "../../_base_/default_runtime.py",
]

default_scope = "mmdet"

# custom imports
custom_imports = dict(
    imports=[
        "libs.models.detectors",
        "libs.datasets",
        "libs.utils",
        "libs.core.bbox.assigners",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

# global settings
batch_size = 8
total_epochs = 15
cfg_name = "laneatt_culane_medium.py"

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="Adam", lr=3e-4, betas=(0.9, 0.999), eps=1e-8),
)

# learning rate policy
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        eta_min=0,
        begin=0,
        T_max=total_epochs,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# seed
randomness = dict(seed=0, deterministic=False)


log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
