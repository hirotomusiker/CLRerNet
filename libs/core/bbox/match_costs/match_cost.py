import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST

from libs.models.losses import LaneIoULoss


@MATCH_COST.register_module()
class FocalCost:
    def __init__(self, weight=1.0, alpha=0.25, gamma=2, eps=1e-12):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value.
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log()
            * self.alpha
            * (1 - cls_pred).pow(self.gamma)
        )
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class DistanceCost:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, predictions, targets):
        """
        repeat predictions and targets to generate all combinations
        use the abs distance as the new distance cost
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/dynamic_assign.py
        """
        num_priors = predictions.shape[0]
        num_targets = targets.shape[0]

        predictions = torch.repeat_interleave(
            predictions, num_targets, dim=0
        )  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b] ((np + nt) * 78)

        targets = torch.cat(
            num_priors * [targets]
        )  # applying this 2 times on [c, d] gives [c, d, c, d]

        invalid_masks = (targets < 0) | (targets >= 1.0)
        lengths = (~invalid_masks).sum(dim=1)
        distances = torch.abs((targets - predictions))
        distances[invalid_masks] = 0.0
        distances = distances.sum(dim=1) / (lengths.float() + 1e-9)
        distances = distances.view(num_priors, num_targets)

        return distances


@MATCH_COST.register_module()
class CLRNetIoUCost:
    def __init__(self, weight=1.0, lane_width=15 / 800):
        """
        LineIoU cost employed in CLRNet.
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/losses/lineiou_loss.py
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
        """
        self.weight = weight
        self.lane_width = lane_width

    def _calc_over_union(self, pred, target, pred_width, target_width):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            pred_width (torch.Tensor): virtual lane half-widths
                for prediction at pre-defined rows, shape (Nl, Nr).
            target_width (torch.Tensor): virtual lane half-widths
                for GT at pre-defined rows, shape (Nl, Nr).
        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        ovr = torch.min(px2[:, None, :], tx2[None, ...]) - torch.max(
            px1[:, None, :], tx1[None, ...]
        )
        union = torch.max(px2[:, None, :], tx2[None, ...]) - torch.min(
            px1[:, None, :], tx1[None, ...]
        )
        return ovr, union

    def __call__(self, pred, target):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        ovr, union = self._calc_over_union(
            pred, target, self.lane_width, self.lane_width
        )
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight


@MATCH_COST.register_module()
class LaneIoUCost(CLRNetIoUCost, LaneIoULoss):
    def __init__(
        self,
        weight=1.0,
        lane_width=7.5 / 800,
        use_pred_start_end=False,
        use_giou=True,
    ):
        """
        Angle- and length-aware LaneIoU employed in CLRerNet.
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
            use_pred_start_end (bool): apply the lane range
                (in horizon indices) for pred lanes
            use_giou (bool): GIoU-style calculation that allows
               negative overlap when the lanes are separated
        """
        super(LaneIoUCost, self).__init__(weight, lane_width)
        self.use_pred_start_end = use_pred_start_end
        self.use_giou = use_giou
        self.max_dx = 1e4

    @staticmethod
    def _set_invalid_with_start_end(
        pred, target, ovr, union, start, end, pred_width, target_width
    ):
        """Set invalid rows for predictions and targets and modify overlaps and unions,
        with using start and end points of prediction lanes.

        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            ovr (torch.Tensor): calculated overlap, shape (Nlp, Nlt, Nr).
            union (torch.Tensor): calculated union, shape (Nlp, Nlt, Nr).
            start (torch.Tensor): start row indices of predictions, shape (Nlp).
            end (torch.Tensor): end row indices of predictions, shape (Nlp).
            pred_width (torch.Tensor): virtual lane half-widths
                for prediction at pre-defined rows, shape (Nlp, Nr).
            target_width (torch.Tensor): virtual lane half-widths
                for GT at pre-defined rows, shape (Nlt, Nr).

        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        num_gt = target.shape[0]
        pred_mask = pred.repeat(num_gt, 1, 1).permute(1, 0, 2)
        invalid_mask_pred = (pred_mask < 0) | (pred_mask >= 1.0)
        target_mask = target.repeat(pred.shape[0], 1, 1)
        invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)

        # set invalid-pred region using start and end
        assert start is not None and end is not None
        yind = torch.ones_like(invalid_mask_pred) * torch.arange(
            0, pred.shape[-1]
        ).float().to(pred.device)
        h = pred.shape[-1] - 1
        start_idx = (start * h).long().view(-1, 1, 1)
        end_idx = (end * h).long().view(-1, 1, 1)
        invalid_mask_pred = (
            invalid_mask_pred | (yind < start_idx) | (yind >= end_idx)
        )

        # set ovr and union to zero at horizon lines where either pred or gt is missing
        invalid_mask_pred_gt = invalid_mask_pred | invalid_mask_gt
        ovr[invalid_mask_pred_gt] = 0
        union[invalid_mask_pred_gt] = 0

        # calculate virtual unions for pred-only or target-only horizon lines
        union_sep_target = target_width.repeat(pred.shape[0], 1, 1) * 2
        union_sep_pred = pred_width.repeat(num_gt, 1, 1).permute(1, 0, 2) * 2
        union[invalid_mask_pred_gt & ~invalid_mask_pred] += union_sep_pred[
            invalid_mask_pred_gt & ~invalid_mask_pred
        ]
        union[invalid_mask_pred_gt & ~invalid_mask_gt] += union_sep_target[
            invalid_mask_pred_gt & ~invalid_mask_gt
        ]
        return ovr, union

    @staticmethod
    def _set_invalid_without_start_end(pred, target, ovr, union):
        """Set invalid rows for predictions and targets
        and modify overlaps and unions, without using start and
        end points of prediction lanes.

        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            ovr (torch.Tensor): calculated overlap, shape (Nlp, Nlt, Nr).
            union (torch.Tensor): calculated union, shape (Nlp, Nlt, Nr).

        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        target_mask = target.repeat(pred.shape[0], 1, 1)
        invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)
        ovr[invalid_mask_gt] = 0.0
        union[invalid_mask_gt] = 0.0
        return ovr, union

    def __call__(
        self, pred, target, eval_shape=(320, 1640), start=None, end=None
    ):
        """
        Calculate LaneIoU between predictions and targets
        Args:
            pred (torch.Tensor): lane predictions, shape: (Nlp, Nr), relative coordinate.
            target (torch.Tensor): ground truth, shape: (Nlt, Nr), relative coordinate.
            eval_shape (tuple): Cropped image shape corresponding to the area in evaluation.
                (320, 1640) for CULane and various shapes for CurveLanes.
            start (torch.Tensor): start row indices of predictions, shape (Nlp).
            end (torch.Tensor): end row indices of predictions, shape (Nlp).
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        pred_width, target_width = self._calc_lane_width(
            pred, target, eval_shape
        )
        ovr, union = self._calc_over_union(
            pred, target, pred_width, target_width
        )
        if self.use_pred_start_end is True:
            ovr, union = self._set_invalid_with_start_end(
                pred, target, ovr, union, start, end, pred_width, target_width
            )
        else:
            ovr, union = self._set_invalid_without_start_end(
                pred, target, ovr, union
            )
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight
