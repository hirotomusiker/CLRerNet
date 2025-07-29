import torch
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class LaneATTAssigner(BaseAssigner):
    def __init__(
        self,
        t_pos=15.0,
        t_neg=20.0,
        iou_cost=None,
        assign_missing_gt=False,
        iou_pos=0.8,
        iou_neg=0.2,
        n_strips=72,
        img_w=640,
    ):
        """
        Adapted from:
        https://github.com/lucastabelini/LaneATT/blob/main/lib/models/matching.py
        Args:
            t_pos (float): positive distance threshold.
            t_neg (float): negative distance threshold.
            use_iou_cost (bool): whether to use IoU cost.
            assign_missing_gt (bool): whether to assign GT lanes
                that do not have close candidates.
            iou_pos (float): IoU positive threshold.
            iou_neg (float): IoU negative threshold.
            n_strips (int): number of lane y strips to divide the image height.
            img_w (int): width of the model's input image.
        """
        self.t_pos = t_pos
        self.t_neg = t_neg
        self.assign_missing_gt = assign_missing_gt
        self.iou_pos = iou_pos
        self.iou_neg = iou_neg
        self.n_strips = n_strips
        self.infinity = 1e8  # a large number to represent infinity
        self.iou_cost = (
            TASK_UTILS.build(iou_cost) if iou_cost is not None else None
        )
        self.img_w = img_w

    def _calculate_valid_offsets(self, proposals, targets):
        device = targets.device
        # get start and the intersection of offsets
        targets_starts = targets[:, 2] * self.n_strips
        proposals_starts = proposals[:, 2] * self.n_strips
        starts = (
            torch.max(targets_starts, proposals_starts)
            .round()
            .long()
            .to(device)
        )
        ends = (targets_starts + targets[:, 4] - 1).round().long().to(device)
        lengths = ends - starts + 1
        ends[lengths < 0] = starts[lengths < 0] - 1
        # a negative number here means no intersection, thus zero length
        lengths[lengths < 0] = 0
        # generate valid offsets mask, which works like this:
        #   start with mask [0, 0, 0, 0, 0]
        #   suppose start = 1
        #   length = 2
        valid_offsets_mask = targets.new_zeros(targets.shape).to(device)
        all_indices = torch.arange(
            valid_offsets_mask.shape[0], dtype=torch.long, device=device
        )
        #   put a one on index `start`, giving [0, 1, 0, 0, 0]
        valid_offsets_mask[all_indices, 5 + starts] = 1.0
        valid_offsets_mask[all_indices, 5 + ends + 1] -= 1.0
        #   put a -1 on the `end` index, giving [0, 1, 0, -1, 0]
        #   if lenght is zero, the previous line
        #   would put a one where it shouldnt be.
        #   this -=1 (instead of =-1) fixes this
        #   the cumsum gives [0, 1, 1, 0, 0], the correct mask for the offsets

        valid_offsets_mask = valid_offsets_mask.to(
            dtype=torch.float32, device=device
        )
        valid_offsets_mask = torch.cumsum(valid_offsets_mask, dim=1)
        valid_offsets_mask = valid_offsets_mask != 0
        return valid_offsets_mask, lengths

    def _distance_based_assign(
        self,
        proposals,
        targets,
        num_proposals,
        num_targets,
    ):
        valid_offsets_mask, lengths = self._calculate_valid_offsets(
            proposals, targets
        )
        # compute distances
        # this compares [ac, ad, bc, bd], i.e., all combinations
        distances = torch.abs(
            (targets - proposals) * valid_offsets_mask.float()
        ).sum(dim=1) / (
            lengths.float() + 1e-9
        )  # avoid division by zero
        distances[lengths == 0] = self.infinity
        distances = distances.view(
            num_proposals, num_targets
        )  # d[i,j] = distance from proposal i to target j
        positives = distances.min(dim=1)[0] < self.t_pos
        negatives = distances.min(dim=1)[0] > self.t_neg
        if self.assign_missing_gt:
            dist_min_per_gt, _ = distances.min(dim=0)
            missing_gt = dist_min_per_gt > self.t_pos
            if missing_gt.sum() > 0:
                _, indices = distances[:, missing_gt].min(dim=0)
                positives[indices] = True
                negatives[indices] = False
        return positives, negatives, distances

    def _iou_based_assign(self, proposals_raw, targets_raw):
        iou = self.iou_cost(
            proposals_raw[:, 5:-1] / self.img_w,
            targets_raw[:, 5:-1] / self.img_w,
        )
        iou_max, _ = iou.max(dim=1)
        positives = iou_max > self.iou_pos
        negatives = iou_max < self.iou_neg
        if self.assign_missing_gt:
            iou_max_per_gt, _ = iou.max(dim=0)
            missing_gt = iou_max_per_gt < self.iou_pos
            if missing_gt.sum() > 0:
                _, indices = iou[:, missing_gt].max(dim=0)
                positives[indices] = True
                negatives[indices] = False
        return positives, negatives, iou

    def assign(self, proposals, targets):
        """
        Assign proposals to targets.
        Args:
            proposals (torch.Tensor): proposals of shape (N, 5 + n_offsets).
            targets (torch.Tensor): targets of shape (M, 5 + n_offsets).
        Returns:
            tuple: (positives, invalid_offsets_mask,
                    negatives, target_positives_indices)
                - positives (torch.Tensor):
                        boolean tensor indicating positive assignments.
                - invalid_offsets_mask (torch.Tensor):
                        mask for invalid offsets.
                - negatives (torch.Tensor):
                        boolean tensor indicating negative assignments.
                - target_positives_indices (torch.Tensor):
                        indices of positive targets.
        """
        # repeat proposals and targets to generate all combinations
        # pad proposals and target for the valid_offset_mask's trick
        num_proposals = proposals.shape[0]
        num_targets = targets.shape[0]
        proposals_pad = proposals.new_zeros(
            proposals.shape[0], proposals.shape[1] + 1
        )
        proposals_pad[:, :-1] = proposals
        proposals = proposals_pad
        targets_pad = targets.new_zeros(targets.shape[0], targets.shape[1] + 1)
        targets_pad[:, :-1] = targets
        targets = targets_pad
        if self.iou_cost is not None:
            proposals_raw = proposals.detach().clone()
            targets_raw = targets.clone()

        proposals = torch.repeat_interleave(
            proposals, num_targets, dim=0
        )  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b]
        targets = torch.cat(
            num_proposals * [targets]
        )  # applying this 2 times on [c, d] gives [c, d, c, d]

        if self.iou_cost is not None:
            positives, negatives, iou = self._iou_based_assign(
                proposals_raw, targets_raw
            )
        else:
            positives, negatives, distances = self._distance_based_assign(
                proposals,
                targets,
                num_proposals,
                num_targets,
            )

        if positives.sum() == 0:
            target_positives_indices = torch.tensor(
                [], device=positives.device, dtype=torch.long
            )
        else:
            if self.iou_cost is not None:
                target_positives_indices = iou[positives].argmax(dim=1)
            else:
                target_positives_indices = distances[positives].argmin(dim=1)
        return (
            positives,
            negatives,
            target_positives_indices,
        )
