import numpy as np
import torch

from libs.core.bbox.match_costs import CLRNetIoUCost
from libs.core.bbox.match_costs import LaneIoUCost


def test_laneiou_cost():
    iou_cost = LaneIoUCost(use_pred_start_end=False, use_giou=True)
    data = np.load("tests/data/test_iou_data.npz")
    pred = torch.Tensor(data["pred"])
    target = torch.Tensor(data["target"])
    start = torch.Tensor(data["start"])
    end = torch.Tensor(data["end"])
    iou = iou_cost(pred, target, eval_shape=(320, 1640), start=None, end=None)
    assert torch.Tensor([-342.6832]).allclose(iou.sum())
    assert torch.all(iou.argmax(dim=0) == torch.tensor([155, 37, 19]))
    assert iou.shape == (192, 3)

    iou_cost = LaneIoUCost(use_pred_start_end=True, use_giou=True)
    iou = iou_cost(pred, target, eval_shape=(320, 1640), start=start, end=end)
    assert torch.Tensor([-288.7381]).allclose(iou.sum())
    assert torch.all(iou.argmax(dim=0) == torch.tensor([155, 37, 23]))
    assert iou.shape == (192, 3)


def test_clrnet_iou_cost():
    iou_cost = CLRNetIoUCost()
    data = np.load("tests/data/test_iou_data.npz")
    pred = torch.Tensor(data["pred"])
    target = torch.Tensor(data["target"])
    iou = iou_cost(pred, target)
    assert torch.Tensor([-366.4133]).allclose(iou.sum())
    assert torch.all(iou.argmax(dim=0) == torch.tensor([155, 37, 19]))
    assert iou.shape == (192, 3)
