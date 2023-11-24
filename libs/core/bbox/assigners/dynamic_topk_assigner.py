# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost


@BBOX_ASSIGNERS.register_module()
class DynamicTopkAssigner(BaseAssigner):
    """Computes dynamick-to-one lane matching between predictions and ground truth (GT).
    The dynamic k for each GT is computed using Lane(Line)IoU matrix.
    The costs matrix is calculated from:
    1) CLRNet: lane horizontal distance, starting point xy, angle and classification scores.
    2) CLRerNet: LaneIoU and classification scores.
    After the dynamick-to-one matching, the un-matched priors are treated as backgrounds.
    Thus each prior's prediction will be assigned with `0` or a positive integer
    indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_cost (dict): cls cost config
        iou_dynamick (dict): iou cost config for dynamic-k calculation
        iou_cost (dict): iou cost config
        reg_cost (dict): reg cost config
        reg_weight (float): cost weight for regression
        cost_combination (int): cost calculation type. 0: CLRNet, 1: CLRerNet.
        use_pred_length_for_iou (bool): prepare pred lane length for iou calculation.
        max_topk (int): max value for dynamic-k.
        min_topk (int): min value for dynamic-k.
    """

    def __init__(
        self,
        cls_cost=None,
        iou_dynamick=None,
        iou_cost=None,
        reg_cost=None,
        reg_weight=3.0,
        cost_combination=0,
        use_pred_length_for_iou=True,
        max_topk=4,
        min_topk=1,
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_dynamick = build_match_cost(iou_dynamick)
        self.iou_cost = build_match_cost(iou_cost)
        self.use_pred_length_for_iou = use_pred_length_for_iou
        self.max_topk = max_topk
        self.min_topk = min_topk
        self.reg_weight = reg_weight
        self.cost_combination = cost_combination

    def dynamic_k_assign(self, cost, ious_matrix):
        """
        Assign grouth truths with priors dynamically.
        Args:
            cost: the assign cost, shape (Np, Ng).
            ious_matrix: iou of grouth truth and priors, shape (Np, Ng).
        Returns:
            torch.Tensor: the indices of assigned prior.
            torch.Tensor: the corresponding ground truth indices.
        Np: number of priors (anchors), Ng: number of GT lanes.
        """
        matching_matrix = torch.zeros_like(cost)
        ious_matrix[ious_matrix < 0] = 0.0
        topk_ious, _ = torch.topk(ious_matrix, self.max_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=self.min_topk)
        num_gt = cost.shape[1]
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[pos_idx, gt_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        matched_gt = matching_matrix.sum(1)
        if (matched_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
            matching_matrix[matched_gt > 1, 0] *= 0.0
            matching_matrix[matched_gt > 1, cost_argmin] = 1.0

        prior_idx = matching_matrix.sum(1).nonzero()
        gt_idx = matching_matrix[prior_idx].argmax(-1)
        return prior_idx.flatten(), gt_idx.flatten()

    def _clrnet_cost(self, predictions, targets, pred_xs, target_xs, img_w, img_h):
        """_summary_
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/dynamic_assign.py
        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            pred_xs (torch.Tensor): predicted x-coordinates on the predefined rows, shape (Np, Nr).
            target_xs (torch.Tensor): GT x-coordinates on the predefined rows, shape (Ng, Nr).
            img_w (int): network input image width (after crop and resize).
            img_h (int): network input image height (after crop and resize).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.

        Returns:
            torch.Tensor: cost matrix, shape (Np, Ng).
        """
        num_priors = predictions["cls_logits"].shape[0]
        num_targets = targets.shape[0]
        # distances cost
        distances_score = self.reg_cost(pred_xs, target_xs)
        distances_score = (
            1 - (distances_score / torch.max(distances_score)) + 1e-2
        )  # normalize the distance

        target_start_xys = targets[:, 2:4]  # num_targets, 2
        target_start_xys[..., 0] *= img_h - 1
        pred_reg_params = predictions["anchor_params"].detach().clone()
        pred_reg_params[:, 0] *= img_h - 1
        pred_reg_params[:, 1] *= img_w - 1
        start_xys_score = torch.cdist(
            pred_reg_params[:, :2], target_start_xys, p=2
        ).reshape(num_priors, num_targets)
        start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

        pred_thetas = pred_reg_params[:, 2:3]  # (192, 1)
        target_thetas = targets[:, 4:5]  # (4, 1)
        theta_score = (
            torch.cdist(pred_thetas, target_thetas, p=1).reshape(
                num_priors, num_targets
            )
            * 180
        )
        theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

        # classification cost
        cls_score = self.cls_cost(
            predictions["cls_logits"].detach().clone(), targets[:, 1].long()
        )

        cost = (
            -((distances_score * start_xys_score * theta_score) ** 2) * self.reg_weight
            + cls_score
        )
        return cost

    def _clrernet_cost(self, predictions, targets, pred_xs, target_xs):
        """_summary_

        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            pred_xs (torch.Tensor): predicted x-coordinates on the predefined rows, shape (Np, Nr).
            target_xs (torch.Tensor): GT x-coordinates on the predefined rows, shape (Ng, Nr).

        Returns:
            torch.Tensor: cost matrix, shape (Np, Ng).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
        """
        start = end = None
        if self.use_pred_length_for_iou:
            y0 = predictions["anchor_params"][:, 0].detach().clone()
            length = predictions["lengths"][:, 0].detach().clone()
            start = (1 - y0).clamp(min=0, max=1)
            end = (start + length).clamp(min=0, max=1)
        iou_cost = self.iou_cost(
            pred_xs,
            target_xs,
            start,
            end,
        )
        iou_score = 1 - (1 - iou_cost) / torch.max(1 - iou_cost) + 1e-2
        # classification cost
        cls_score = self.cls_cost(
            predictions["cls_logits"].detach().clone(), targets[:, 1].long()
        )
        cost = -iou_score * self.reg_weight + cls_score
        return cost

    def assign(
        self,
        predictions,
        targets,
        img_meta,
    ):
        """
        computes dynamicly matching based on the cost, including cls cost and lane similarity cost
        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            img_meta (dict): meta dict that includes per-image information such as image shape.
        return:
            matched_row_inds (Tensor): matched predictions, shape: (num_targets).
            matched_col_inds (Tensor): matched targets, shape: (num_targets).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
        """
        img_h, img_w, _ = img_meta["img_shape"]

        pred_xs = predictions["xs"].detach().clone()  # relative
        target_xs = targets[:, 6:] / (img_w - 1)  # abs -> relative

        iou_dynamick = self.iou_dynamick(pred_xs, target_xs)

        if self.cost_combination == 0:  # CLRNet
            cost = self._clrnet_cost(
                predictions, targets, pred_xs, target_xs, img_w, img_h
            )
        elif self.cost_combination == 1:  # CLRerNet
            cost = self._clrernet_cost(predictions, targets, pred_xs, target_xs)
        else:
            raise NotImplementedError(
                f"cost_combination {self.cost_combination} is not implemented!"
            )

        matched_row_inds, matched_col_inds = self.dynamic_k_assign(cost, iou_dynamick)

        return matched_row_inds, matched_col_inds
