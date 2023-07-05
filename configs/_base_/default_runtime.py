# runtime settings
checkpoint_config = dict(interval=10)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

device_ids = "0"
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(interval=1, metric='F1')
