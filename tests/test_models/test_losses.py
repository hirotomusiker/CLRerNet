import numpy as np
import torch

from libs.models.losses import CLRNetIoULoss, LaneIoULoss


def test_laneiou_loss():
    iou_loss = LaneIoULoss(img_h=320, img_w=1640)
    data = np.load("tests/data/test_iou_data.npz")
    target = torch.Tensor(data["target"])
    pred = torch.Tensor(data["pred"][: len(target)])
    expected = torch.tensor(1.51706361770)
    loss = iou_loss(pred, target)
    assert expected.allclose(loss)


def test_clrnet_iou_loss():
    iou_loss = CLRNetIoULoss()
    data = np.load("tests/data/test_iou_data.npz")
    target = torch.Tensor(data["target"])
    pred = torch.Tensor(data["pred"][: len(target)])
    expected = torch.tensor(1.63517200946)
    loss = iou_loss(pred, target)
    assert expected.allclose(loss)
