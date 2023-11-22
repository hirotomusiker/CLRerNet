import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module
class CLRNetSegLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, num_classes=5, ignore_label=255, bg_weight=0.5):
        super(CLRNetSegLoss, self).__init__()
        self.loss_weight = loss_weight
        weights = torch.ones(num_classes)
        weights[0] = bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=weights)

    def forward(self, preds, targets):
        loss = self.criterion(F.log_softmax(preds, dim=1), targets.long())
        return loss * self.loss_weight
