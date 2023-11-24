"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_attention
from mmdet.core import build_assigner, build_prior_generator
from mmdet.models.builder import HEADS, build_loss
from nms import nms

from libs.models.dense_heads.seg_decoder import SegDecoder
from libs.utils.lane_utils import Lane


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
        loss_cls=None,
        loss_bbox=None,
        loss_iou=None,
        loss_seg=None,
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
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_seg = build_loss(loss_seg) if loss_seg["loss_weight"] > 0 else None
        self.loss_iou = build_loss(loss_iou)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])

        # Non-learnable parameters
        self.register_buffer(
            name="sample_x_indices",
            tensor=(
                torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32)
                * self.n_strips
            ).long(),
        )
        self.register_buffer(
            name="prior_feat_ys",
            tensor=torch.flip(
                (self.sample_x_indices.float() / self.n_strips), dims=[-1]
            ),
        )
        self.register_buffer(
            name="prior_ys",
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
        self.attention = build_attention(attention)

        # Auxiliary head
        if self.loss_seg:
            self.seg_decoder = SegDecoder(
                self.img_h,
                self.img_w,
                loss_seg.num_classes,
                self.prior_feat_channels,
                self.refine_layers,
            )

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
        """
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: Input features (list[Tensor]). Each tensor has a shape (B, C, H_i, W_i),
                where i is the pyramid level.
                Example of shapes: ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        Returns:
            pred_dict (List[dict]): List of prediction dicts each of which containins multiple lane predictions.
                cls_logits (torch.Tensor): 2-class logits with shape (B, Np, 2).
                anchor_params (torch.Tensor): anchor parameters with shape (B, Np, 3).
                lengths (torch.Tensor): lane lengths in row numbers with shape (B, Np, 1).
                xs (torch.Tensor): x coordinates of the lane points with shape (B, Np, Nr).

        B: batch size, Np: number of priors (anchors), Nr: num_points (rows).
        """
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

        predictions_list = []

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
                "cls_logits": cls_logits,
                "anchor_params": anchor_params,
                "lengths": reg[:, :, 3:4],
                "xs": reg_xs,
            }

            predictions_list.append(pred_dict)

            if stage != self.refine_layers - 1:
                anchor_params = anchor_params.detach().clone()
                priors_on_featmap = updated_anchor_xs.detach().clone()[
                    ..., self.sample_x_indices
                ]

        return predictions_list

    def loss(self, out_dict, img_metas):
        """Loss calculation from the network output.

        Args:
            out_dict (dict[torch.Tensor]): Output dict from the network containing:
                predictions (List[dict]): 3-layer prediction dicts each of which contains:
                    cls_logits: shape (B, Np, 2), anchor_params: shape (B, Np, 3),
                    lengths: shape (B, Np, 1) and xs: shape (B, Np, Nr).
                seg (torch.Tensor): segmentation maps, shape (B, C, H, W).
                where
                B: batch size, Np: number of priors (anchors), Nr: number of rows,
                C: segmentation channels, H and W: the largest feature's spatial shape.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = len(img_metas)
        device = out_dict["predictions"][0]["cls_logits"].device
        cls_loss = torch.tensor(0.0).to(device)
        reg_xytl_loss = torch.tensor(0.0).to(device)
        iou_loss = torch.tensor(0.0).to(device)

        for stage in range(self.refine_layers):
            for b, img_meta in enumerate(img_metas):
                pred_dict = {k: v[b] for k, v in out_dict["predictions"][stage].items()}
                cls_pred = pred_dict["cls_logits"]
                target = img_meta["lanes"].clone().to(device)  # [n_lanes, 78]
                target = target[target[:, 1] == 1]
                cls_target = cls_pred.new_zeros(cls_pred.shape[0]).long()

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_loss = cls_loss + self.loss_cls(cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    (matched_row_inds, matched_col_inds) = self.assigner.assign(
                        pred_dict, target.clone(), img_meta
                    )

                # classification targets
                cls_target[matched_row_inds] = 1
                cls_loss = (
                    cls_loss
                    + self.loss_cls(cls_pred, cls_target).sum() / target.shape[0]
                )

                # regression targets -> [start_y, start_x, theta]
                # (all transformed to absolute values), only on matched pairs
                reg_yxtl = torch.cat(
                    (pred_dict["anchor_params"], pred_dict["lengths"]), dim=1
                )
                reg_yxtl = reg_yxtl[matched_row_inds]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= self.img_w - 1
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                pred_xs = pred_dict["xs"][matched_row_inds]
                target_xs = target[matched_col_inds, 6:].clone()

                # adjust target length by start point difference
                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        reg_yxtl[:, 0].round().long(), 0, self.n_strips
                    )  # ensure the predictions starts is valid
                    target_starts = (
                        (target[matched_col_inds, 2] * self.n_strips).round().long()
                    )
                    target_yxtl[:, -1] -= predictions_starts - target_starts

                # Loss calculation
                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180

                reg_xytl_loss = (
                    reg_xytl_loss + self.loss_bbox(reg_yxtl, target_yxtl).mean()
                )

                iou_loss = iou_loss + self.loss_iou(
                    pred_xs * (self.img_w - 1) / self.img_w, target_xs / self.img_w
                )

        cls_loss /= batch_size * self.refine_layers

        reg_xytl_loss /= batch_size * self.refine_layers
        iou_loss /= batch_size * self.refine_layers

        loss_dict = {
            "loss_cls": cls_loss,
            "loss_reg_xytl": reg_xytl_loss,
            "loss_iou": iou_loss,
        }

        # extra segmentation loss
        if self.loss_seg:
            tgt_masks = np.array([t["gt_masks"].data[0] for t in img_metas])
            tgt_masks = torch.tensor(tgt_masks).long().to(device)  # (B, H, W)
            loss_dict["loss_seg"] = self.loss_seg(out_dict["seg"], tgt_masks)

        return loss_dict

    def forward_train(self, x, img_metas, **kwargs):
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        predictions = self(x)
        out_dict = {"predictions": predictions}
        if self.loss_seg:
            out_dict["seg"] = self.forward_seg(x)

        losses = self.loss(out_dict, img_metas)
        return losses

    def forward_seg(self, x):
        """Forward function for training mode.
        Args:
            x (list[torch.tensor]): Features from backbone.
        Returns:
            torch.tensor: segmentation maps, shape (B, C, H, W), where
            B: batch size, C: segmentation channels, H and W: the largest feature's spatial shape.
        """
        batch_features = list(x[len(x) - self.refine_layers :])
        batch_features.reverse()
        seg_features = torch.cat(
            [
                F.interpolate(
                    feature,
                    size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                    mode="bilinear",
                    align_corners=False,
                )
                for feature in batch_features
            ],
            dim=1,
        )
        seg = self.seg_decoder(seg_features)
        return seg

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
        softmax = nn.Softmax(dim=1)
        assert (
            len(pred_dict["cls_logits"]) == 1
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
                        "start_x": lane_param[1],
                        "start_y": lane_param[0],
                        "conf": score,
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
        pred_dict = self(feats)[-1]
        lanes, scores = self.get_lanes(pred_dict, as_lanes=self.test_cfg.as_lanes)
        result_dict = {
            "lanes": lanes,
            "scores": scores,
        }
        return result_dict
