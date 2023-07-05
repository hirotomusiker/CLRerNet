_base_ = [
    '../base_clrernet.py',
    'dataset_culane_clrernet.py',
    '../../_base_/default_runtime.py',
]

# custom imports
custom_imports = dict(
    imports=[
        'libs.models',
        'libs.datasets',
        'libs.core.anchor',
    ],
    allow_failed_imports=False,
)

cfg_name = 'clrernet_culane_dla34_ema.py'
