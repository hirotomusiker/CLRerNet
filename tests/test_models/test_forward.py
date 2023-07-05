import torch
from mmdet.apis import init_detector


def test_clerernet_forward():
    config_path = "configs/clrernet/culane/clrernet_culane_dla34.py"
    model = init_detector(config_path)
    x = torch.ones((1, 3, 320, 800)).float().cuda()
    img_meta = dict()
    with torch.no_grad():
        results = model(img=x, img_metas=[img_meta], return_loss=False, rescale=True)
    assert len(results) == 1
    assert "result" in results[0]
    assert "lanes" in results[0]["result"] and "scores" in results[0]["result"]
