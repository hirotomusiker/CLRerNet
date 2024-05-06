_base_ = [
    "../base_clrernet.py",
    "dataset_culane_clrernet.py",
    "../../_base_/default_runtime.py",
]

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

cfg_name = "clrernet_culane_dla34_ema.py"

model = dict(test_cfg=dict(conf_threshold=0.43))


custom_hooks = [dict(type="ExpMomentumEMAHook", momentum=0.0001, priority=49)]

total_epochs = 50
evaluation = dict(interval=total_epochs)
checkpoint_config = dict(interval=total_epochs)

data = dict(samples_per_gpu=24)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=6e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=6e-4, by_epoch=False)

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
