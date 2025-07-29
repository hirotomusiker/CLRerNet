import math

import numpy as np
import torch
from mmdet.registry import TASK_UTILS
from torch import nn


@TASK_UTILS.register_module()
class CLRerNetAnchorGenerator(nn.Module):
    """
    Anchor (prior) generator.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py
    Beforehand, download
    https://github.com/lucastabelini/LaneATT/blob/main/data/culane_anchors_freq.pt
    """

    def __init__(
        self,
        num_priors=192,
        num_points=72,
    ):
        """
        Args:
            num_priors (int): Number of anchors.
            num_points (int): Number of points (rows) in one anchor.
        """
        super(CLRerNetAnchorGenerator, self).__init__()
        self.num_priors = num_priors
        self.num_offsets = num_points
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        self.init_anchors()

    def init_anchors(self):
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0], 1 - (i // 2) * strip_size
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 2],
                0.16 if i % 2 == 0 else 0.32,
            )

        for i in range(
            left_priors_nums, left_priors_nums + bottom_priors_nums
        ):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 1.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 1],
                ((i - left_priors_nums) // 4 + 1) * bottom_strip_size,
            )
            nn.init.constant_(
                self.prior_embeddings.weight[i, 2], 0.2 * (i % 4 + 1)
            )

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                1
                - ((i - left_priors_nums - bottom_priors_nums) // 2)
                * strip_size,
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 2],
                0.68 if i % 2 == 0 else 0.84,
            )

    def generate_anchors(
        self, anchor_params, prior_ys, sample_x_indices, img_w, img_h
    ):
        """
        Calculate anchor x coordinates for pre-defined y coordinates.
        Args:
            anchor_params (torch.nn.parameter.Parameter):
                Anchor parameters y0, x0 and theta, shape (Np, 3).
            prior_ys (torch.Tensor): Fixed anchor row coordinates, shape (Nr,).
            sample_x_indices (torch.Tensor): Fixed row coordinates
                to sample when pooling, shape (Ns,).
            img_w (int): Input image width.
            img_h (int): Input image height.
        Returns:
            anchor_xs (torch.Tensor):
                Generated anchor x coordinates, shape (Np, Nr).
            sampled_xs (torch.Tensor):
                Sampled x coordinates, shape (Np, Ns).
                Np: number of priors
                Nr: number of points (rows)
                Ns: number of sample points.
        """
        num_anchors = anchor_params.shape[0]
        x0 = anchor_params[..., 1].unsqueeze(-1).clone().repeat(
            1, self.num_offsets
        ) * (
            img_w - 1
        )  # (192, 72)
        y0 = (
            anchor_params[..., 0]
            .unsqueeze(-1)
            .clone()
            .repeat(1, self.num_offsets)
        )  # (192, 72)
        theta = anchor_params[..., 2].unsqueeze(-1).clone()
        theta = theta.repeat(1, self.num_offsets) * np.pi
        tan_lane = torch.tan(theta + 1e-5)  # (192, 72)
        ys = prior_ys.repeat(num_anchors, 1)  # (192, 72)
        anchor_xs = x0 - ((ys - y0) * img_h / tan_lane)
        anchor_xs /= img_w - 1
        sampled_xs = anchor_xs[..., sample_x_indices]
        return anchor_xs, sampled_xs


@TASK_UTILS.register_module()
class LaneATTAnchorGenerator(nn.Module):
    def __init__(
        self,
        S=72,
        img_w=640,
        img_h=360,
        stride=32,
        max_lanes=4,
        anchors_freq_path=None,
        topk_anchors=None,
        anchor_feat_channels=64,
    ):
        super(LaneATTAnchorGenerator, self).__init__()
        self.stride = stride
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.strip_size = img_h / self.n_strips
        self.offsets_ys = np.arange(img_h, -1, -self.strip_size)
        self.max_lanes = max_lanes
        self.fmap_h = img_h // self.stride  # 11
        self.fmap_w = img_w // self.stride  # 20
        self.anchor_ys = torch.linspace(
            1, 0, steps=self.n_offsets, dtype=torch.float32
        )
        self.anchor_cut_ys = torch.linspace(
            1, 0, steps=self.fmap_h, dtype=torch.float32
        )
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72.0, 60.0, 49.0, 39.0, 30.0, 22.0]
        self.right_angles = [108.0, 120.0, 131.0, 141.0, 150.0, 158.0]
        self.bottom_angles = [
            165.0,
            150.0,
            141.0,
            131.0,
            120.0,
            108.0,
            100.0,
            90.0,
            80.0,
            72.0,
            60.0,
            49.0,
            39.0,
            30.0,
            15.0,
        ]

        # Generate anchors
        # number of anchors - left and right:
        #       72 * 6 = 432, bottom: 128 * 15 = 1920
        # anchor length - anchors:
        #       5 + 72 (n_offsets) = 77, anchors_cut: 5 + 11 (fmap_h) = 16
        # first 5 values = 2 scores, 1 start_y, start_x, 1 length
        self.anchors, self.anchors_cut = self.generate_anchors(
            lateral_n=72, bottom_n=128
        )

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()  # size: 2784
            assert topk_anchors is not None  # e.g. 1000
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]  # filter anchors, 2784 -> 1000
            self.anchors_cut = self.anchors_cut[ind]

        # Pre compute indices for the anchor pooling
        (
            self.cut_zs,
            self.cut_ys,
            self.cut_xs,
            self.invalid_mask,
        ) = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, self.fmap_w, self.fmap_h
        )

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        unclamped_xs = torch.flip(
            (self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,)
        )
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(
            unclamped_xs, n_fmaps, dim=0
        ).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(
            n_proposals, n_fmaps, fmaps_h
        )
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = (
            torch.arange(n_fmaps)
            .repeat_interleave(fmaps_h)
            .repeat(n_proposals)[:, None]
        )

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros(
            (batch_size, n_proposals, n_fmaps, self.fmap_h, 1),
            device=features.device,
        )
        # (8, 1000, 64, 11, 1)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[  # (64, 12, 20)
                self.cut_zs, self.cut_ys, self.cut_xs
            ].view(
                n_proposals, n_fmaps, self.fmap_h, 1
            )  # (1000, 64, 11, 1)
            # cut_zs, ys, xs: (1000*64*11, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(
            self.left_angles, x=0.0, nb_origins=lateral_n
        )
        right_anchors, right_cut = self.generate_side_anchors(
            self.right_angles, x=1.0, nb_origins=lateral_n
        )
        bottom_anchors, bottom_cut = self.generate_side_anchors(
            self.bottom_angles, y=1.0, nb_origins=bottom_n
        )

        return torch.cat(
            [left_anchors, bottom_anchors, right_anchors]
        ), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1.0, 0.0, num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1.0, 0.0, num=nb_origins)]
        else:
            raise Exception(
                "Please define exactly one of "
                "`x` or `y` (not neither nor both)"
            )

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 length, S coordinates
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.0  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (
            start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)
        ) * self.img_w

        return anchor
