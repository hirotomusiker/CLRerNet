_base_ = [
    "../base_clrernet.py",
    "dataset_culane_clrernet.py",
    "../../_base_/default_runtime.py",
]
default_scope = 'mmdet'

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

cfg_name = "clrernet_culane_dla34.py"

model = dict(test_cfg=dict(conf_threshold=0.41))

total_epochs = 15
checkpoint_config = dict(interval=total_epochs)

train_cfg = dict(max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

data = dict(samples_per_gpu=24)  # single GPU setting

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type="AdamW", lr=6e-4),
)


# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.0, by_epoch=False)

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
