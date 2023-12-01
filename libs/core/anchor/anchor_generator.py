import numpy as np
import torch
from mmdet.core.anchor.builder import PRIOR_GENERATORS
from torch import nn


@PRIOR_GENERATORS.register_module()
class CLRerNetAnchorGenerator(nn.Module):
    """
    Anchor (prior) generator.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py
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
                self.prior_embeddings.weight[i, 2], 0.16 if i % 2 == 0 else 0.32
            )

        for i in range(left_priors_nums, left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 1.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 1],
                ((i - left_priors_nums) // 4 + 1) * bottom_strip_size,
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                1 - ((i - left_priors_nums - bottom_priors_nums) // 2) * strip_size,
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 2], 0.68 if i % 2 == 0 else 0.84
            )

    def generate_anchors(self, anchor_params, prior_ys, sample_x_indices, img_w, img_h):
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
            anchor_xs (torch.Tensor): Generated anchor x coordinates, shape (Np, Nr).
            sampled_xs (torch.Tensor): Sampled x coordinates, shape (Np, Ns).
        Np: number of priors, Nr: number of points (rows), Ns: number of sample points.
        """
        num_anchors = anchor_params.shape[0]
        x0 = anchor_params[..., 1].unsqueeze(-1).clone().repeat(1, self.num_offsets) * (
            img_w - 1
        )  # (192, 72)
        y0 = (
            anchor_params[..., 0].unsqueeze(-1).clone().repeat(1, self.num_offsets)
        )  # (192, 72)
        theta = anchor_params[..., 2].unsqueeze(-1).clone()
        theta = theta.repeat(1, self.num_offsets) * np.pi
        tan_lane = torch.tan(theta + 1e-5)  # (192, 72)
        ys = prior_ys.repeat(num_anchors, 1)  # (192, 72)
        anchor_xs = x0 - ((ys - y0) * img_h / tan_lane)
        anchor_xs /= img_w - 1
        sampled_xs = anchor_xs[..., sample_x_indices]
        return anchor_xs, sampled_xs
