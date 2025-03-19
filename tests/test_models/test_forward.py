import torch
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample

def test_clerernet_forward():
    config_path = "configs/clrernet/culane/clrernet_culane_dla34.py"
    model = init_detector(config_path)
    x = torch.ones((1, 3, 320, 800)).float().cuda()

    data = dict(
        inputs=x,
        data_samples=[DetDataSample()],
    )

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)
    print(results)
    assert len(results) == 1
    assert "lanes" in results[0]
    assert "scores" in results[0]
