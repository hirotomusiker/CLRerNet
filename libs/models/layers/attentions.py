import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS


@MODELS.register_module()
class ROIGather(nn.Module):
    """
    CLRNet ROIGather module to process pooled features
    and make them interact with global information.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/roi_gather.py
    """

    def __init__(
        self,
        in_channels,
        num_priors,
        sample_points,
        fc_hidden_dim,
        refine_layers,
        mid_channels=48,
        cross_attention_weight=1.0,
    ):
        """
        Args:
            in_channels (int): Number of input feature channels.
            num_priors (int): Number of priors (anchors).
            sample_points (int): Number of pooling sample points (rows).
            fc_hidden_dim (int): FC middle channel number.
            refine_layers (int): The number of refine levels.
            mid_channels (int): The number of input channels to catconv.
            cross_attention_weight (float):
                Weight to fuse cross attention result.
        """
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.cross_attention_weight = cross_attention_weight

        if self.cross_attention_weight > 0:
            self.attention = AnchorVecFeatureMapAttention(
                num_priors, in_channels
            )

        # learnable layers
        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(
                    in_channels,
                    mid_channels,
                    (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm_cfg=dict(type="BN"),
                )
            )

            self.catconv.append(
                ConvModule(
                    mid_channels * (i + 1),
                    in_channels,
                    (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm_cfg=dict(type="BN"),
                )
            )

        self.fc = nn.Linear(sample_points * in_channels, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        """
        Args:
            x (List[torch.Tensor]): List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, C, Ns, 1).
            layer_index (int): Current refine layer index.
        Returns:
            cat_feat (torch.Tensor):
                Fused feature tensor, shape (B * Np, C, Ns, 1).
            B: batch size, Np: number of priors (anchors),
            Ns: number of sample points (rows).
        """
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward_roi(self, roi_features, layer_index, bs):
        """
        Gather the ROI (pooled) features of the multiple refine levels
        and fuse them into the output tensor.
        Args:
            roi_features (List[torch.Tensor]):
                List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, Cin, Ns, 1).
            layer_index (int): The current refine layer index.
            bs (int): Batchsize B.
        Returns:
            roi (torch.Tensor): Output features, shape (B, Np, Ch).
            B: batch size, Np: number of priors (anchors),
            Ns: number of sample points (rows),
            Cin: input channel number, Ch: hidden channel number.
        """
        # [B * Np, Cin, Ns, 1] * N -> [B * Np, Cin, Ns, 1]
        roi = self.roi_fea(roi_features, layer_index)
        # [B * Np, Cin, Ns, 1] -> [B * Np, Cin * Ns]
        roi = roi.contiguous().view(bs * self.num_priors, -1)

        # [B * Np, Cin * Ns] -> [B * Np, Ch]
        roi = F.relu(self.fc_norm(self.fc(roi)))
        # [B * Np, Ch] -> [B, Np, Ch]
        roi = roi.view(bs, self.num_priors, -1)
        return roi

    def forward(self, roi_features, fmap_pyramid, layer_index):
        """
        ROIGather forward function.
        Args:
            roi_features (List[torch.Tensor]):
                List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, Cin, Ns, 1).
            fmap_pyramid (List[torch.Tensor]):
                Multi-level feature pyramid.
                Each tensor has a shape (B, Cin, H_i, W_i)
                where i is the pyramid level.
            layer_index (int): The current refine layer index.
        Returns:
            roi (torch.Tensor): Output feature tensors, shape (B, Np, Ch).
                B: batch size, Np: number of priors (anchors),
                Ns: number of sample points (rows),
                Cin: input channel number, Ch: hidden channel number.
        """
        fmap = fmap_pyramid[layer_index]
        roi = self.forward_roi(
            roi_features, layer_index, fmap.size(0)
        )  # [B, Np, Ch]

        if self.cross_attention_weight > 0:
            context = self.attention(roi, fmap)
            roi = roi + self.cross_attention_weight * F.dropout(
                context, p=0.1, training=self.training
            )

        return roi


class AnchorVecFeatureMapAttention(nn.Module):
    def __init__(self, n_query, dim):
        """
        Args:
            n_query (int): Number of queries (priors, anchors).
            dim (int): Key and Value dim.
        """
        super(AnchorVecFeatureMapAttention, self).__init__()
        self.dim = dim
        self.resize = FeatureResize()
        self.f_key = ConvModule(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type="BN"),
        )

        self.f_query = nn.Sequential(
            nn.Conv1d(
                in_channels=n_query,
                out_channels=n_query,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=n_query,
            ),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W = nn.Conv1d(
            in_channels=n_query,
            out_channels=n_query,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=n_query,
        )
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, roi, fmap):
        """
        Forward function for cross attention.
        Args:
            roi (torch.Tensor):
                Features pooled by priors, shape (B, Np, C).
            fmap (torch.Tensor):
                Feature maps at the current refine level, shape (B, C, H, W).
        Returns:
            context (torch.Tensor): Output global context, shape (B, Np, C).
        B: batch size, Np: number of priors (anchors)
        """
        query = self.f_query(roi)  # [B, Np, C] ->  [B, Np, C]
        key = self.f_key(fmap)  # [B, C, H, W]
        key = self.resize(key)  # [B, C, H'W']
        value = self.f_value(fmap)  # [B, C, H, W]
        value = self.resize(value)  # [B, C, H'W']
        value = value.permute(0, 2, 1)  # [B, H'W', C]

        # attention: [B, Np, C] x [B, C, H'W'] -> [B, Np, H'W']
        sim_map = torch.matmul(query, key)
        sim_map = (self.dim**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)  # [B, Np, C]
        context = self.W(context)  # [B, Np, C]
        return context


class FeatureResize(nn.Module):
    """Resize the feature map by interpolation."""

    def __init__(self, size=(10, 25)):
        """
        Args:
            size (tuple): Target size (H', W').
        """
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        """
        Forward function.
        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W).
        Returns:
            out (torch.Tensor): Resized tensor with shape (B, C, H'W').
        """
        x = F.interpolate(x, self.size)
        return x.flatten(2)


@MODELS.register_module()
class LaneATTAttention(nn.Module):
    def __init__(self, anchors, anchor_feat_channels, fmap_h):
        """
        Args:
            anchors (list): List of anchor points.
            anchor_feat_channels (int): Number of feature channels per anchor.
            fmap_h (int): Height of the feature map.
        """
        super(LaneATTAttention, self).__init__()
        self.anchors = anchors
        self.anchor_feat_channels = anchor_feat_channels
        self.fmap_h = fmap_h

        # Attention layer to compute attention scores
        self.attention_layer = nn.Linear(
            self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1
        )
        self.initialize_layer(self.attention_layer)

    def forward(self, batchsize, batch_anchor_features):
        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(
            batch_anchor_features
        )  # (N * 1000, 64 * 11) -> (N * 1000, 999)
        attention = softmax(scores).reshape(
            batchsize, len(self.anchors), -1
        )  # (N, 1000, 999)

        attention_matrix = torch.eye(
            attention.shape[1], device=batch_anchor_features.device
        ).repeat(
            batchsize, 1, 1
        )  # (N, 1000, 1000)
        non_diag_inds = torch.nonzero(attention_matrix == 0.0, as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[
            non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]
        ] = attention.flatten()  # (N * 1000 * 999)

        batch_anchor_features = batch_anchor_features.reshape(
            batchsize, len(self.anchors), -1
        )  # (N, 1000, 704)
        attention_features = torch.bmm(
            torch.transpose(batch_anchor_features, 1, 2),
            torch.transpose(attention_matrix, 1, 2),
        ).transpose(
            1, 2
        )  # (N, 1000, 704)
        attention_features = attention_features.reshape(
            -1, self.anchor_feat_channels * self.fmap_h
        )  # (N * 1000, 704)
        batch_anchor_features = batch_anchor_features.reshape(
            -1, self.anchor_feat_channels * self.fmap_h
        )  # (N * 1000, 704)
        batch_anchor_features = torch.cat(
            (attention_features, batch_anchor_features), dim=1
        )  # (N * 1000, 1408)
        return batch_anchor_features, attention_matrix

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)
