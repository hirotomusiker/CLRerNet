# modified based on:
# https://github.com/lucastabelini/LaneATT/blob/2f8583ba14eccba05e6779668bc3a38bc751984a/lib/models/laneatt.py
from typing import Tuple

import numpy as np
import torch
from mmdet.registry import MODELS
from mmdet.registry import TASK_UTILS
from mmdet.structures import SampleList
from torch import nn
from torch import Tensor

from libs.models.losses.focal_loss import KorniaFocalLoss
from libs.utils.lane_utils import Lane


@MODELS.register_module()
class LaneATTHead(nn.Module):
    def __init__(
        self,
        anchor_generator,
        S=72,
        img_w=640,
        img_h=360,
        max_lanes=4,  # culane
        cls_loss_weight=10.0,
        reg_loss_weight=1.0,
        stride=32,
        backbone_nb_channels=512,
        train_cfg=None,
        test_cfg=None,
    ):
        super(LaneATTHead, self).__init__()

        # Some definitions
        self.stride = stride
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.strip_size = img_h / self.n_strips
        self.offsets_ys = np.arange(img_h, -1, -self.strip_size)
        self.max_lanes = max_lanes
        self.fmap_h = img_h // self.stride
        self.fmap_w = img_w // self.stride
        self.assigner = (
            TASK_UTILS.build(train_cfg["assigner"])
            if "assigner" in train_cfg
            else None
        )
        self.anchor_generator = TASK_UTILS.build(anchor_generator)
        self.anchor_feat_channels = self.anchor_generator.anchor_feat_channels
        self.anchor_ys = self.anchor_generator.anchor_ys

        # Non-learnable parameters
        self.register_buffer(
            name="anchors", tensor=self.anchor_generator.anchors
        )
        self.nms_train = MODELS.build(train_cfg.nms)
        self.nms_val = MODELS.build(test_cfg.nms)

        attention_cfg = {
            "type": "LaneATTAttention",
            "anchors": self.anchors,
            "anchor_feat_channels": self.anchor_feat_channels,
            "fmap_h": self.fmap_h,
        }
        self.attention = MODELS.build(attention_cfg)

        self.reg_assigner = TASK_UTILS.build(train_cfg["reg_assigner"])
        self.cls_assigner = TASK_UTILS.build(train_cfg["cls_assigner"])

        self.focal_loss = KorniaFocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1_loss = nn.SmoothL1Loss()

        # Setup and initialize layers
        # anchor_feat_channels: 64,
        # fmap_h: 11, len(anchors): 1000, n_offsets: 72
        # conv1: 512 -> 64 ch
        # cls_layer: 1408 -> 2 ch
        # reg_layer: 1408 -> 73 ch
        # attention_layer: 704 -> 999 ch
        self.conv1 = nn.Conv2d(
            backbone_nb_channels, self.anchor_feat_channels, kernel_size=1
        )
        self.cls_layer = nn.Linear(
            2 * self.anchor_feat_channels * self.fmap_h, 2
        )
        self.reg_layer = nn.Linear(
            2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1
        )
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def forward(self, batch_features, train=False, **kwargs):
        """
        Args:
            batch_features (Tuple[Tensor]):
                Features from backbone, shape (N, 512, 12, 20).
            train (bool): Whether the model is in training mode.
            **kwargs:
        Returns:
            proposals_list (List[Tensor]):
                List of proposals for each image in the batch.
        """
        batch_features = self.conv1(batch_features[0])  # -> (N, 64, 12, 20)
        batchsize = len(batch_features)
        batch_anchor_features = self.anchor_generator.cut_anchor_features(
            batch_features
        )  # -> (N, 1000, 64, 11, 1)

        # Join proposals from all images into a single proposals features batch
        # -> (N * 1000, 64 * 11)
        batch_anchor_features = batch_anchor_features.view(
            -1, self.anchor_feat_channels * self.fmap_h
        )

        batch_anchor_features, attention_matrix = self.attention(
            batchsize, batch_anchor_features
        )

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)  # (N * 1000, 2)
        cls_logits = cls_logits.reshape(
            batchsize, -1, cls_logits.shape[1]
        )  # (N, 1000, 2)
        reg = self.reg_layer(batch_anchor_features)  # (N * 1000, 73)
        reg = reg.reshape(batchsize, -1, reg.shape[1])  # (N, 1000, 73)

        # Add offsets to anchors
        reg_proposals = torch.zeros(
            (*cls_logits.shape[:2], 5 + self.n_offsets),
            device=batch_features.device,
        )  # (N, 1000, 77)
        reg_proposals += self.anchors  # (1000, 77)
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg
        if train:
            proposals_list = self.nms_train(
                reg_proposals, self.anchors, attention_matrix
            )
        else:
            proposals_list = self.nms_val(
                reg_proposals, self.anchors, attention_matrix
            )
        return proposals_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone, shape (8, 512, 12, 20).
            batch_data_samples (List[:obj:`DetDataSample`]): The data samples
                that include meta information.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        proposals_list = self(x, train=True)
        out_dict = {"predictions": proposals_list}

        losses = self.loss_by_feat(out_dict, batch_data_samples)
        return losses

    def loss_by_feat(self, out_dict, batch_data_samples):
        """Loss calculation from the network output.

        Args:
            out_dict (dict[torch.Tensor]):
                Output dict from the network containing:
                predictions (List[Tuple[Tensor]]):
                    List of proposals, each proposal is a tuple
                    of (proposals, anchors, attention_matrix, anchor_inds).
            batch_data_samples: (List[:obj:`DetDataSample`]): The data samples
                that include meta information.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        proposals_list = out_dict["predictions"]
        device = proposals_list[0][0].device
        targets = [
            sample.metainfo["lanes"].to(device)
            for sample in batch_data_samples
        ]

        cls_loss = torch.tensor(0.0).to(device)
        reg_loss = torch.tensor(0.0).to(device)
        valid_imgs = len(batch_data_samples)
        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # (1000, 77) (4, 77)
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets,
                # all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += self.focal_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary
            # for the positive & negative matching
            with torch.no_grad():
                # reg matcher
                (
                    positives_mask_reg,
                    _,
                    target_positives_indices_reg,
                ) = self.reg_assigner.assign(anchors.detach().clone(), target)
                # cls matcher, use LaneIoU cost
                (
                    positives_mask_cls,
                    negatives_mask_cls,
                    _,
                ) = self.cls_assigner.assign(
                    proposals.detach().clone(), target
                )

            positives_cls = proposals[positives_mask_cls]
            num_positives_cls = len(positives_cls)
            negatives_cls = proposals[negatives_mask_cls]
            num_negatives_cls = len(negatives_cls)

            positives_reg = proposals[positives_mask_reg]

            # Handle edge case of no positives found.
            # Same losses as no-target case
            if num_positives_cls == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += self.focal_loss(cls_pred, cls_target).sum()
            else:
                # Get classification targets
                all_proposals = torch.cat([positives_cls, negatives_cls], 0)
                cls_target = proposals.new_zeros(
                    num_positives_cls + num_negatives_cls
                ).long()
                cls_target[:num_positives_cls] = 1.0
                cls_pred = all_proposals[:, :2]
                cls_loss += self.focal_loss(cls_pred, cls_target).sum() / len(
                    target
                )

            if len(positives_reg) == 0:
                continue
            # Regression targets
            reg_pred = positives_reg[:, 4:]

            with torch.no_grad():
                target = target[target_positives_indices_reg]
                positive_starts = (
                    (positives_reg[:, 2] * self.n_strips)
                    .round()
                    .long()
                    .to(device)
                )
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(
                    len(positives_reg), dtype=torch.long, device=device
                )
                ends = (
                    (positive_starts + target[:, 4] - 1)
                    .round()
                    .long()
                    .to(device)
                )
                invalid_offsets_mask = torch.zeros(
                    (len(positives_reg), 1 + self.n_offsets + 1),
                    dtype=torch.int,
                    device=device,
                )  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[
                    invalid_offsets_mask
                ]

            # Loss calc
            reg_loss += self.smooth_l1_loss(reg_pred, reg_target)

        # Batch mean
        cls_loss = cls_loss * self.cls_loss_weight / valid_imgs
        reg_loss = reg_loss * self.reg_loss_weight / valid_imgs
        result_dict = {"cls_loss": cls_loss, "reg_loss": reg_loss}

        return result_dict

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        scores = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~(
                (
                    ((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0))
                    .cpu()
                    .numpy()[::-1]
                    .cumprod()[::-1]
                ).astype(np.bool_)
            )
            lane_xs[end + 1 :] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1
            ).squeeze(2)
            lane_obj = Lane(
                points=points.cpu().numpy(),
                metadata={
                    "start_x": lane[3],
                    "start_y": lane[2],
                    "conf": lane[1],
                },
            )
            lanes.append(lane_obj)
            scores.append(lane[1].item())
        return lanes, scores

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append(([], []))
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def predict(
        self, x: Tuple[Tensor], batch_data_samples: SampleList, **kwargs
    ) -> SampleList:
        """Single-image test without augmentation.
        Args:
            x (torch.Tensor):
                Input image tensor of shape (1, 3, height, width).
            batch_data_samples (List[:obj:`DetDataSample`]):
                The data samples that include meta information.
        Returns:
            result_dict (List[dict]):
                Single-image result containing prediction outputs and
                img_metas as 'result' and 'metas' respectively.
        """
        proposals_list = self(x, train=False)
        all_decoded = self.decode(proposals_list, as_lanes=True)
        result_dict = [
            {"lanes": lanes, "scores": scores, "metainfo": ds.metainfo}
            for (lanes, scores), ds in zip(all_decoded, batch_data_samples)
        ]
        return result_dict

    def init_weights(self):
        pass
