"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import build_prior_generator
from mmdet.models.builder import HEADS
from mmcv.cnn.bricks.transformer import build_attention
from libs.utils.lane_utils import Lane
from nms import nms


@HEADS.register_module
class CLRerHead(nn.Module):
    def __init__(
        self,
        anchor_generator,
        img_w=800,
        img_h=320,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        attention=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super(CLRerHead, self).__init__()
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.img_w = img_w
        self.img_h = img_h
        self.n_offsets = self.anchor_generator.num_offsets
        self.n_strips = self.n_offsets - 1
        self.strip_size = self.img_h / self.n_strips
        self.num_priors = attention.num_priors = self.anchor_generator.num_priors
        self.sample_points = attention.sample_points = sample_points
        self.refine_layers = attention.refine_layers = refine_layers
        self.fc_hidden_dim = attention.fc_hidden_dim = fc_hidden_dim
        self.prior_feat_channels = attention.in_channels = prior_feat_channels
        self.attention = build_attention(attention)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Non-learnable parameters
        self.register_buffer(
            name='sample_x_indices',
            tensor=(
                torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32)
                * self.n_strips
            ).long(),
        )
        self.register_buffer(
            name='prior_feat_ys',
            tensor=torch.flip(
                (self.sample_x_indices.float() / self.n_strips), dims=[-1]
            ),
        )
        self.register_buffer(
            name='prior_ys',
            tensor=torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32),
        )

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
            ]
            cls_modules += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
            ]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.reg_layers = nn.Linear(self.fc_hidden_dim, self.n_offsets + 4)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        self.init_weights()

    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)

    def pool_prior_features(self, batch_features, prior_xs):
        """
        Pool features from the feature map along the prior points.
        Args:
            batch_features (torch.Tensor): Input feature maps, shape: (B, C, H, W)
            prior_xs (torch.Tensor):. Prior points, shape (B, Np, Ns)
                where Np is the number of priors and Ns is the number of sample points.
        Returns:
            feature (torch.Tensor): Pooled features with shape (B * Np, C, Ns, 1).
        """

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, self.num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * self.num_priors).view(
            batch_size, self.num_priors, -1, 1
        )

        prior_xs = prior_xs * 2.0 - 1.0
        prior_ys = prior_ys * 2.0 - 1.0
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid, align_corners=True).permute(
            0, 2, 1, 3
        )
        feature = feature.reshape(
            batch_size * self.num_priors,
            self.prior_feat_channels,
            self.sample_points,
            1,
        )
        return feature

    def forward(self, x, **kwargs):
        '''
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: Input features (list[Tensor]). Each tensor has a shape (B, C, H_i, W_i),
                where i is the pyramid level.
                Example of shapes: ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        Return:
            pred_dict (dict): prediction dict containing multiple lanes.
                cls_logits (torch.Tensor): 2-class logits with shape (B, Np, 2).
                anchor_params (torch.Tensor): anchor parameters with shape (B, Np, 3).
                lengths (torch.Tensor): lane lengths in row numbers with shape (B, Np, 1).
                xs (torch.Tensor): x coordinates of the lane points with shape (B, Np, Nr).

        B: batch size, Np: number of priors (anchors), Nr: num_points (rows).
        '''
        batch_size = x[0].shape[0]
        feature_pyramid = list(x[len(x) - self.refine_layers :])
        feature_pyramid.reverse()
        # e.g. [1, 64, 10, 25], [1, 64, 20, 50] [1, 64, 40, 100]

        _, sampled_xs = self.anchor_generator.generate_anchors(
            self.anchor_generator.prior_embeddings.weight,
            self.prior_ys,
            self.sample_x_indices,
            self.img_w,
            self.img_h,
        )

        anchor_params = self.anchor_generator.prior_embeddings.weight.clone().repeat(
            batch_size, 1, 1
        )  # [B, Np, 3]
        priors_on_featmap = sampled_xs.repeat(batch_size, 1, 1)

        predictions_lists = []

        # iterative refine
        pooled_features_stages = []
        for stage in range(self.refine_layers):
            prior_xs = priors_on_featmap  # torch.flip(priors_on_featmap, dims=[2])  # [24, 192, 36]
            # 1. anchor ROI pooling
            # [B, C, H, W] X [B, Np, Ns] => [B * Np, C, Ns, 1]
            pooled_features = self.pool_prior_features(feature_pyramid[stage], prior_xs)
            pooled_features_stages.append(pooled_features)

            # 2. ROI gather
            # pooled features [B * Np, C, Ns, 1] * stages
            # feature pyramid: [B, C, Hs, Ws] (s = 0, 1, 2)
            fc_features = self.attention(
                pooled_features_stages, feature_pyramid, stage
            )  # [B, Np, Ch], Ch: fc_hidden_dim
            fc_features = fc_features.view(self.num_priors, batch_size, -1).reshape(
                batch_size * self.num_priors, self.fc_hidden_dim
            )  # [B * Np, Ch]

            # 3. cls and reg heads
            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1]
            )  # (B, Np, 2)

            reg = self.reg_layers(reg_features)
            reg = reg.reshape(batch_size, -1, reg.shape[1])  # (B, Np, 4 + Nr)

            # 4. reg processing
            anchor_params += reg[:, :, :3]  # y0, x0, theta
            updated_anchor_xs, _ = self.anchor_generator.generate_anchors(
                anchor_params.view(-1, 3),
                self.prior_ys,
                self.sample_x_indices,
                self.img_w,
                self.img_h,
            )
            updated_anchor_xs = updated_anchor_xs.view(batch_size, self.num_priors, -1)
            reg_xs = updated_anchor_xs + reg[..., 4:]

            pred_dict = {
                'cls_logits': cls_logits,
                'anchor_params': anchor_params,
                'lengths': reg[:, :, 3:4],
                'xs': reg_xs,
            }

            predictions_lists.append(pred_dict)

            if stage != self.refine_layers - 1:
                anchor_params = anchor_params.detach().clone()
                priors_on_featmap = updated_anchor_xs.detach().clone()[
                    ..., self.sample_x_indices
                ]

        return predictions_lists[-1]

    def forward_train(self, x, img_metas, **kwargs):
        # Coming Soon...
        raise NotImplementedError("Training is not supported yet!")

    def get_lanes(self, pred_dict, as_lanes=True):
        """
        Convert model output to lane instances.
        Args:
            pred_dict (dict): prediction dict containing multiple lanes.
                cls_logits (torch.Tensor): 2-class logits with shape (B, Np, 2).
                anchor_params (torch.Tensor): anchor parameters with shape (B, Np, 3).
                lengths (torch.Tensor): lane lengths in row numbers with shape (B, Np, 1).
                xs (torch.Tensor): x coordinates of the lane points with shape (B, Np, Nr).
            as_lanes (bool): transform to the Lane instance for interpolation.
        Returns:
            pred (List[torch.Tensor]): List of lane tensors (shape: (N, 2))
                or `Lane` objects, where N is the number of rows.
            scores (torch.Tensor): Confidence scores of the lanes.

        B: batch size, Np: num_priors, Nr: num_points (rows).
        """
        '''
        Args:
            pred_dict (dict): cls_logits, anchor_params, length and xs
            as_lanes (bool): return output as lane format
        '''
        softmax = nn.Softmax(dim=1)
        assert (
            len(pred_dict['cls_logits']) == 1
        ), "Only single-image prediction is available!"
        # filter out the conf lower than conf threshold
        threshold = self.test_cfg.conf_threshold
        scores = softmax(pred_dict["cls_logits"][0])[:, 1]
        keep_inds = scores >= threshold
        scores = scores[keep_inds]
        xs = pred_dict["xs"][0, keep_inds]
        lengths = pred_dict["lengths"][0, keep_inds]
        anchor_params = pred_dict["anchor_params"][0, keep_inds]
        if xs.shape[0] == 0:
            return [], []

        if self.test_cfg.use_nms:
            nms_anchor_params = anchor_params[..., :2].detach().clone()
            nms_anchor_params[..., 0] = 1 - nms_anchor_params[..., 0]
            nms_predictions = torch.cat(
                [
                    pred_dict["cls_logits"][0, keep_inds].detach().clone(),
                    nms_anchor_params[..., :2],
                    lengths.detach().clone() * self.n_strips,
                    xs.detach().clone() * (self.img_w - 1),
                ],
                dim=-1,
            )  # [N, 77]
            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.test_cfg.nms_thres,
                top_k=self.test_cfg.nms_topk,
            )
            keep = keep[:num_to_keep]
            xs = xs[keep]
            scores = scores[keep]
            lengths = lengths[keep]
            anchor_params = anchor_params[keep]

        lengths = torch.round(lengths * self.n_strips)
        pred = self.predictions_to_lanes(xs, anchor_params, lengths, scores, as_lanes)

        return pred, scores

    def predictions_to_lanes(
        self, pred_xs, anchor_params, lengths, scores, as_lanes=True, extend_bottom=True
    ):
        """
        Convert predictions to the lane segment instances.
        Args:
            pred_xs (torch.Tensor): x coordinates of the lane points with shape (Nl, Nr).
            anchor_params (torch.Tensor): anchor parameters with shape (Nl, 3).
            lengths (torch.Tensor): lane lengths in row numbers with shape (Nl, 1).
            scores (torch.Tensor): confidence scores with shape (Nl,).
            as_lanes (bool): transform to the Lane instance for interpolation.
            extend_bottom (bool): if the prediction does not start at the bottom of the image,
                extend its prediction until the x is outside the image.
        Returns:
            lanes (List[torch.Tensor]): List of lane tensors (shape: (N, 2))
                or `Lane` objects, where N is the number of rows.

        B: batch size, Nl: number of lanes after NMS, Nr: num_points (rows).
        """
        prior_ys = self.prior_ys.to(pred_xs.device).double()
        lanes = []
        for lane_xs, lane_param, length, score in zip(
            pred_xs, anchor_params, lengths, scores
        ):
            start = min(
                max(0, int(round((1 - lane_param[0].item()) * self.n_strips))),
                self.n_strips,
            )
            length = int(round(length.item()))
            end = start + length - 1
            end = min(end, len(prior_ys) - 1)
            if extend_bottom:
                edge = (lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)
                start -= edge.flip(0).cumprod(dim=0).sum()
            lane_ys = prior_ys[start : end + 1]
            lane_xs = lane_xs[start : end + 1]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (
                lane_ys * (self.test_cfg.ori_img_h - self.test_cfg.cut_height)
                + self.test_cfg.cut_height
            ) / self.test_cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1
            ).squeeze(2)
            if as_lanes:
                lane = Lane(
                    points=points.cpu().numpy(),
                    metadata={
                        'start_x': lane_param[1],
                        'start_y': lane_param[0],
                        'conf': score,
                    },
                )
            else:
                lane = points
            lanes.append(lane)
        return lanes

    def simple_test(self, feats):
        """Test function without test-time augmentation.
        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the FPN.
        Returns:
            result_dict (dict): Inference result containing
                lanes (List[torch.Tensor]): List of lane tensors (shape: (N, 2))
                    or `Lane` objects, where N is the number of rows.
                scores (torch.Tensor): Confidence scores of the lanes.
        """
        pred_dict = self(feats)
        lanes, scores = self.get_lanes(pred_dict, as_lanes=self.test_cfg.as_lanes)
        result_dict = {
            'lanes': lanes,
            'scores': scores,
        }
        return result_dict
