import torch
from mmdet.registry import MODELS
from nms import nms
from torch import nn


@MODELS.register_module()
class LaneATTNMS(nn.Module):
    # NMS wrapper for LaneATT proposals.
    # modified based on:
    # https://github.com/lucastabelini/LaneATT/blob/main/lib/models/laneatt.py#L112
    def __init__(
        self, nms_thres, nms_topk, conf_threshold, n_offsets=72, stride=4
    ):
        super(LaneATTNMS, self).__init__()
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk
        self.conf_threshold = conf_threshold
        self.stride = stride
        self.n_offsets = n_offsets
        self.n_strips = self.n_offsets - 1
        self.prop_size = 5 + self.n_offsets
        self.dataset_offset = 0

    def forward(self, batch_proposals, anchors, batch_attention_matrix):
        """
        perform non-maximum suppression on proposals
        Args:
            batch_proposals (torch.tensor): proposals (batch, 1000, 77)
            batch_attention_matrix (torch.tensor):
                attention amtrix (batch, 1000, 77)
            nms_thres (float): NMS IoU threshold
            nms_topk (int): top-k proposals limit
            conf_threshold (float): confidence threshold
        Returns:
            proposals_list (list): filtered list of proposals
        """
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(
            batch_proposals, batch_attention_matrix
        ):
            anchor_inds = torch.arange(
                batch_proposals.shape[1], device=proposals.device
            )
            # The gradients do not have to
            # (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]  # (1000,)
                if self.conf_threshold > 0:
                    # apply confidence threshold
                    above_threshold = scores > self.conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append(
                        (
                            proposals[[]],
                            anchors[[]],
                            attention_matrix[[]],
                            None,
                        )
                    )
                    continue
                keep, num_to_keep, _ = nms(
                    proposals,
                    scores,
                    overlap=self.nms_thres,
                    top_k=self.nms_topk,
                )
                keep = keep[:num_to_keep]

            proposals = proposals[keep]

            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append(
                (proposals, anchors[keep], attention_matrix, anchor_inds)
            )

        return proposals_list
