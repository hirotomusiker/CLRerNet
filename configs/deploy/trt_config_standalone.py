 
codebase_config = dict(
    type='mmdet',           
    task='ObjectDetection'  
)

 
onnx_config = dict(
    type='onnx',
    export_params=True,    
    keep_initializers_as_inputs=False,
    opset_version=11,       
    save_file='end2end.onnx',  
    input_names=['input'],  
    output_names=['cls_logits', 'anchor_params', 'lengths', 'xs'],  
    optimize=True,          
    dynamic_axes={         
        'input': {
            0: 'batch',
        },
        'cls_logits': {
            0: 'batch',
        },
        'anchor_params': {
            0: 'batch',
        },
        'lengths': {
             0: 'batch',
         },
        'xs': {
            0: 'batch',
        }
    })


backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True,                 
        max_workspace_size=1 * (1 << 30) # 1GB Workspace
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 800],
                    opt_shape=[1, 3, 320, 800],
                    max_shape=[1, 3, 320, 800]
                )
            )
        )
    ],
    tensorrt_cfg=dict(
        # Target => Jetson AGX Orin
        cuda_arch='sm_87'
    )
)
