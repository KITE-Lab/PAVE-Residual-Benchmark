"""BEV reference planner and compatibility loader for existing checkpoints."""

from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tv_models
from . import config
from .utils import log


class ResNet50FeatureBackbone(nn.Module):
    def __init__(self, use_imagenet_weights: bool) -> None:
        super().__init__()
        if tv_models is None:
            raise RuntimeError(
                "torchvision is required for the ResNet50 BEVFormer backbone."
            )
        weights_enum = getattr(tv_models, "ResNet50_Weights", None)
        weights = (
            weights_enum.IMAGENET1K_V2
            if use_imagenet_weights and weights_enum is not None
            else None
        )
        model = tv_models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class SingleScaleFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.lateral(x))


def build_image_backbone() -> nn.Module:
    if config.BACKBONE_NAME != "resnet50":
        raise ValueError(
            f"Unsupported backbone for BEVFormer planner: {config.BACKBONE_NAME}"
        )
    return ResNet50FeatureBackbone(config.USE_IMAGENET_WEIGHTS)


def build_transformer_encoder(
    embed_dim: int, num_heads: int, num_layers: int, dropout: float
) -> nn.Module:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    encoder_kwargs: Dict[str, Any] = {}
    init_varnames = getattr(nn.TransformerEncoder.__init__, "__code__", None)
    if (
        init_varnames is not None
        and "enable_nested_tensor" in init_varnames.co_varnames
    ):
        encoder_kwargs["enable_nested_tensor"] = False
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers, **encoder_kwargs)


class TrajectoryEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=2, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

    def forward(self, past_xy: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(past_xy)
        return hidden[-1]


class LearnedBEVPositionalEncoding(nn.Module):
    def __init__(self, num_feats: int, bev_h: int, bev_w: int) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(bev_h, num_feats)
        self.col_embed = nn.Embedding(bev_w, num_feats)
        self.bev_h = bev_h
        self.bev_w = bev_w
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        rows = torch.arange(self.bev_h, device=device)
        cols = torch.arange(self.bev_w, device=device)
        y_embed = self.row_embed(rows)
        x_embed = self.col_embed(cols)
        pos = torch.cat(
            [
                x_embed.unsqueeze(0).expand(self.bev_h, -1, -1),
                y_embed.unsqueeze(1).expand(-1, self.bev_w, -1),
            ],
            dim=-1,
        )
        return (
            pos.reshape(1, self.bev_h * self.bev_w, -1)
            .to(dtype=dtype)
            .repeat(batch_size, 1, 1)
        )


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, num_heads, head_dim = value.shape
    num_query = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]
    output = value.new_zeros(batch_size, num_query, num_heads, head_dim)
    level_start = 0
    for level_idx in range(spatial_shapes.shape[0]):
        height = int(spatial_shapes[level_idx, 0].item())
        width = int(spatial_shapes[level_idx, 1].item())
        value_level = value[:, level_start : level_start + height * width]
        value_level = value_level.view(
            batch_size, height, width, num_heads, head_dim
        ).permute(0, 3, 4, 1, 2)
        value_level = value_level.reshape(
            batch_size * num_heads, head_dim, height, width
        )
        grid = sampling_locations[:, :, :, level_idx]
        grid = grid.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_heads, num_query, num_points, 2
        )
        grid = grid.mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            value_level,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(
            batch_size, num_heads, head_dim, num_query, num_points
        ).permute(0, 3, 1, 4, 2)
        attn = attention_weights[:, :, :, level_idx].unsqueeze(-1)
        output = output + (sampled * attn).sum(dim=3)
        level_start += height * width
    return output.reshape(batch_size, num_query, num_heads * head_dim)


class TemporalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = 2
        self.dropout = nn.Dropout(dropout)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_bev_queue,
            self.num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dims * self.num_bev_queue,
            self.num_bev_queue * num_heads * num_levels * num_points,
        )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        identity = query
        query_with_pos = query if query_pos is None else query + query_pos
        if prev_bev is None:
            value = torch.cat([query, query], dim=0)
        else:
            value = torch.cat([prev_bev, query], dim=0)
        batch_size, num_query, _ = query.shape
        head_dim = self.embed_dims // self.num_heads
        query_cat = torch.cat([value[:batch_size], query_with_pos], dim=-1)
        value = self.value_proj(value).view(
            batch_size * self.num_bev_queue, num_query, self.num_heads, head_dim
        )
        sampling_offsets = self.sampling_offsets(query_cat).view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query_cat).view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
        )
        attention_weights = attention_weights.softmax(dim=-1)
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            batch_size * self.num_bev_queue,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
            batch_size * self.num_bev_queue,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1].float(), spatial_shapes[..., 0].float()], dim=-1
        )
        repeated_reference = (
            reference_points.unsqueeze(1)
            .repeat(1, self.num_bev_queue, 1, 1, 1)
            .reshape(batch_size * self.num_bev_queue, num_query, self.num_levels, 2)
        )
        sampling_locations = (
            repeated_reference[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = output.view(
            batch_size, self.num_bev_queue, num_query, self.embed_dims
        ).mean(dim=1)
        return identity + self.dropout(self.output_proj(output))


class MSDeformableAttention3D(nn.Module):
    def __init__(
        self, embed_dims: int, num_heads: int, num_levels: int, num_points: int
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_query, _ = query.shape
        head_dim = self.embed_dims // self.num_heads
        value = self.value_proj(value).view(
            batch_size, value.shape[1], self.num_heads, head_dim
        )
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points
        )
        attention_weights = attention_weights.softmax(dim=-1)
        num_z_anchors = reference_points.shape[2]
        if self.num_points % max(num_z_anchors, 1) != 0:
            raise ValueError(
                f"num_points={self.num_points} must be divisible by num_z_anchors={num_z_anchors}"
            )
        points_per_anchor = max(self.num_points // max(num_z_anchors, 1), 1)
        sampling_offsets = (
            sampling_offsets
            / torch.stack(
                [spatial_shapes[..., 1].float(), spatial_shapes[..., 0].float()], dim=-1
            )[None, None, None, :, None, :]
        )
        sampling_offsets = sampling_offsets.view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_levels,
            points_per_anchor,
            num_z_anchors,
            2,
        )
        sampling_locations = (
            reference_points[:, :, None, None, None, :, :] + sampling_offsets
        )
        sampling_locations = sampling_locations.view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_levels,
            points_per_anchor * num_z_anchors,
            2,
        )
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        return output


class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_cams: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_cams = num_cams
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.deformable_attention = MSDeformableAttention3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        camera_mask: torch.Tensor,
    ) -> torch.Tensor:
        identity = query
        slots = torch.zeros_like(query)
        counts = torch.zeros_like(query[..., :1])
        for cam_idx in range(self.num_cams):
            visible = bev_mask[cam_idx].any(dim=-1) & camera_mask[
                :, cam_idx
            ].bool().unsqueeze(-1)
            if not bool(visible.any().item()):
                continue
            camera_output = self.deformable_attention(
                query=query,
                value=value[:, cam_idx],
                reference_points=reference_points_cam[cam_idx],
                spatial_shapes=spatial_shapes,
            )
            visible_f = visible.unsqueeze(-1).to(camera_output.dtype)
            slots = slots + camera_output * visible_f
            counts = counts + visible_f
        slots = slots / counts.clamp_min(1.0)
        return identity + self.dropout(self.output_proj(slots))


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dims: int, hidden_dims: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class BEVFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_self_attention = TemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=4,
            dropout=dropout,
        )
        self.spatial_cross_attention = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=len(config.CAMERA_ORDER),
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.ffn = FeedForwardNetwork(embed_dims, config.BEVFORMER_FFN_DIM, dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        bev_pos: torch.Tensor,
        bev_ref_2d: torch.Tensor,
        bev_spatial_shapes: torch.Tensor,
        feat_tokens: torch.Tensor,
        feat_spatial_shapes: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        camera_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.temporal_self_attention(
            query=query,
            reference_points=bev_ref_2d,
            spatial_shapes=bev_spatial_shapes,
            query_pos=bev_pos,
            prev_bev=None,
        )
        query = self.norm1(query)
        query = self.spatial_cross_attention(
            query=query,
            value=feat_tokens,
            spatial_shapes=feat_spatial_shapes,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            camera_mask=camera_mask,
        )
        query = self.norm2(query)
        query = self.ffn(query)
        query = self.norm3(query)
        return query


class BEVFormerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bev_h = config.BEV_HEIGHT
        self.bev_w = config.BEV_WIDTH
        self.embed_dims = config.BEV_ENCODER_OUT_DIM
        self.pc_range = config.PC_RANGE
        self.num_points_in_pillar = config.BEVFORMER_NUM_POINTS_IN_PILLAR
        self.bev_queries = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.bev_positional_encoding = LearnedBEVPositionalEncoding(
            self.embed_dims // 2, self.bev_h, self.bev_w
        )
        self.camera_embeddings = nn.Embedding(len(config.CAMERA_ORDER), self.embed_dims)
        self.level_embeddings = nn.Parameter(
            torch.zeros(config.BEVFORMER_NUM_LEVELS, self.embed_dims)
        )
        self.layers = nn.ModuleList(
            [
                BEVFormerEncoderLayer(
                    embed_dims=self.embed_dims,
                    num_heads=config.BEVFORMER_NUM_HEADS,
                    num_levels=config.BEVFORMER_NUM_LEVELS,
                    num_points=config.BEVFORMER_NUM_POINTS,
                    dropout=config.TRANSFORMER_DROPOUT,
                )
                for _ in range(config.BEVFORMER_ENCODER_LAYERS)
            ]
        )
        nn.init.normal_(self.level_embeddings)

    def get_reference_points_3d(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        zs = torch.linspace(
            0.5,
            self.num_points_in_pillar - 0.5,
            self.num_points_in_pillar,
            dtype=dtype,
            device=device,
        ).view(-1, 1, 1).expand(
            self.num_points_in_pillar, self.bev_h, self.bev_w
        ) / float(self.num_points_in_pillar)
        xs = torch.linspace(
            0.5, self.bev_w - 0.5, self.bev_w, dtype=dtype, device=device
        ).view(1, 1, self.bev_w).expand(
            self.num_points_in_pillar, self.bev_h, self.bev_w
        ) / float(self.bev_w)
        ys = torch.linspace(
            0.5, self.bev_h - 0.5, self.bev_h, dtype=dtype, device=device
        ).view(1, self.bev_h, 1).expand(
            self.num_points_in_pillar, self.bev_h, self.bev_w
        ) / float(self.bev_h)
        ref_3d = torch.stack((xs, ys, zs), dim=-1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        return ref_3d.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def get_reference_points_2d(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, self.bev_h - 0.5, self.bev_h, dtype=dtype, device=device
            ),
            torch.linspace(
                0.5, self.bev_w - 0.5, self.bev_w, dtype=dtype, device=device
            ),
            indexing="ij",
        )
        ref = torch.stack(
            (
                ref_x.reshape(-1) / float(self.bev_w),
                ref_y.reshape(-1) / float(self.bev_h),
            ),
            dim=-1,
        )
        return ref.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)

    def point_sampling(
        self,
        reference_points: torch.Tensor,
        lidar2img: torch.Tensor,
        img_shape: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = reference_points.shape[0]
        num_cams = lidar2img.shape[1]
        dtype = reference_points.dtype
        device = reference_points.device
        reference_points = reference_points.clone()
        reference_points[..., 0] = (
            reference_points[..., 0] * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        reference_points[..., 1] = (
            reference_points[..., 1] * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        reference_points[..., 2] = (
            reference_points[..., 2] * (self.pc_range[5] - self.pc_range[2])
            + self.pc_range[2]
        )
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), dim=-1
        )
        reference_points = reference_points.permute(1, 0, 2, 3)
        num_depth = reference_points.shape[0]
        num_query = reference_points.shape[2]
        reference_points = reference_points.view(
            num_depth, batch_size, 1, num_query, 4, 1
        ).repeat(1, 1, num_cams, 1, 1, 1)
        lidar2img = (
            lidar2img.to(dtype=dtype, device=device)
            .view(1, batch_size, num_cams, 1, 4, 4)
            .repeat(num_depth, 1, 1, num_query, 1, 1)
        )
        reference_points_cam = torch.matmul(
            lidar2img.float(), reference_points.float()
        ).squeeze(-1)
        depth = reference_points_cam[..., 2:3]
        bev_mask = depth > config.MIN_LIDAR2IMG_DEPTH
        safe_depth = torch.clamp(depth, min=config.MIN_LIDAR2IMG_DEPTH)
        reference_points_cam = reference_points_cam[..., 0:2] / safe_depth
        widths = (
            img_shape[:, :, 1]
            .to(dtype=dtype, device=device)
            .view(1, batch_size, num_cams, 1, 1)
        )
        heights = (
            img_shape[:, :, 0]
            .to(dtype=dtype, device=device)
            .view(1, batch_size, num_cams, 1, 1)
        )
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / widths
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / heights
        bev_mask = bev_mask & (reference_points_cam[..., 0:1] > 0.0)
        bev_mask = bev_mask & (reference_points_cam[..., 0:1] < 1.0)
        bev_mask = bev_mask & (reference_points_cam[..., 1:2] > 0.0)
        bev_mask = bev_mask & (reference_points_cam[..., 1:2] < 1.0)
        bev_mask = torch.nan_to_num(bev_mask.float()).bool()
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return (reference_points_cam, bev_mask)

    def forward(
        self,
        feat_map: torch.Tensor,
        lidar2img: torch.Tensor,
        img_shape: torch.Tensor,
        camera_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_cams, _, feat_h, feat_w = feat_map.shape
        dtype = feat_map.dtype
        device = feat_map.device
        feat_tokens = feat_map.flatten(3).permute(0, 1, 3, 2)
        camera_ids = torch.arange(num_cams, device=device)
        feat_tokens = feat_tokens + self.camera_embeddings(camera_ids).view(
            1, num_cams, 1, self.embed_dims
        ).to(dtype=dtype)
        feat_tokens = feat_tokens + self.level_embeddings[0].view(
            1, 1, 1, self.embed_dims
        ).to(dtype=dtype)
        feat_tokens = feat_tokens * camera_mask[:, :, None, None].to(dtype)
        bev_queries = (
            self.bev_queries.weight.to(dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        bev_pos = self.bev_positional_encoding(batch_size, dtype, device)
        ref_3d = self.get_reference_points_3d(batch_size, dtype, device)
        ref_2d = self.get_reference_points_2d(batch_size, dtype, device)
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, lidar2img, img_shape
        )
        bev_spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]], dtype=torch.long, device=device
        )
        feat_spatial_shapes = torch.tensor(
            [[feat_h, feat_w]], dtype=torch.long, device=device
        )
        bev_embed = bev_queries
        for layer in self.layers:
            bev_embed = layer(
                query=bev_embed,
                bev_pos=bev_pos,
                bev_ref_2d=ref_2d,
                bev_spatial_shapes=bev_spatial_shapes,
                feat_tokens=feat_tokens,
                feat_spatial_shapes=feat_spatial_shapes,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                camera_mask=camera_mask,
            )
        return bev_embed


class BEVPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_image_backbone()
        self.neck = SingleScaleFPN(
            config.BACKBONE_OUT_CHANNELS, config.FPN_OUT_CHANNELS
        )
        self.encoder = BEVFormerEncoder()
        self.traj_encoder = TrajectoryEncoder(config.TRAJECTORY_HIDDEN_DIM)
        self.bev_to_modality = nn.Linear(
            config.BEV_ENCODER_OUT_DIM, config.MODALITY_TOKEN_DIM
        )
        self.traj_to_modality = nn.Linear(
            config.TRAJECTORY_HIDDEN_DIM, config.MODALITY_TOKEN_DIM
        )
        self.modality_embeddings = nn.Embedding(2, config.MODALITY_TOKEN_DIM)
        self.modality_transformer = build_transformer_encoder(
            embed_dim=config.MODALITY_TOKEN_DIM,
            num_heads=config.MODALITY_TRANSFORMER_HEADS,
            num_layers=config.MODALITY_TRANSFORMER_LAYERS,
            dropout=config.TRANSFORMER_DROPOUT,
        )
        self.head = nn.Sequential(
            nn.Linear(config.MODALITY_TOKEN_DIM * 2, config.FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.FUTURE_STEPS * 2),
        )

    def forward_features(
        self,
        images: torch.Tensor,
        lidar2img: torch.Tensor,
        img_shape: torch.Tensor,
        pad_shape: torch.Tensor,
        camera_mask: torch.Tensor,
        past_xy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del pad_shape
        batch_size, num_views, _, _, _ = images.shape
        image_features = self.backbone(
            images.view(
                batch_size * num_views, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH
            )
        )
        stable_context = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if images.device.type == "cuda"
            else nullcontext()
        )
        with stable_context:
            image_features = image_features.float()
            image_features = self.neck(image_features)
            _, feat_channels, _, _ = image_features.shape
            image_features = image_features.view(
                batch_size,
                num_views,
                feat_channels,
                image_features.shape[-2],
                image_features.shape[-1],
            )
            bev_embed = self.encoder(
                feat_map=image_features,
                lidar2img=lidar2img.float(),
                img_shape=img_shape.float(),
                camera_mask=camera_mask.float(),
            )
            bev_feat = bev_embed.transpose(1, 2).reshape(
                batch_size,
                config.BEV_ENCODER_OUT_DIM,
                config.BEV_HEIGHT,
                config.BEV_WIDTH,
            )
            bev_feat = F.adaptive_avg_pool2d(bev_feat, (1, 1)).flatten(1)
            traj_feat = self.traj_encoder(past_xy.float())
            modality_tokens = torch.stack(
                [self.bev_to_modality(bev_feat), self.traj_to_modality(traj_feat)],
                dim=1,
            )
            modality_ids = torch.arange(modality_tokens.shape[1], device=images.device)
            modality_tokens = (
                modality_tokens
                + self.modality_embeddings(modality_ids)
                .view(1, -1, config.MODALITY_TOKEN_DIM)
                .float()
            )
            modality_tokens = self.modality_transformer(modality_tokens)
            fused_feat = modality_tokens.reshape(batch_size, -1)
            future_xy = self.head(fused_feat).view(batch_size, config.FUTURE_STEPS, 2)
        return {
            "pred_future_xy": future_xy,
            "bev_pooled_feat": bev_feat,
            "traj_feat": traj_feat,
            "fused_feat": fused_feat,
        }

    def forward(
        self,
        images: torch.Tensor,
        lidar2img: torch.Tensor,
        img_shape: torch.Tensor,
        pad_shape: torch.Tensor,
        camera_mask: torch.Tensor,
        past_xy: torch.Tensor,
    ) -> torch.Tensor:
        features = self.forward_features(
            images=images,
            lidar2img=lidar2img,
            img_shape=img_shape,
            pad_shape=pad_shape,
            camera_mask=camera_mask,
            past_xy=past_xy,
        )
        return features["pred_future_xy"]


def remap_camera_embedding_weight(
    weight: torch.Tensor, checkpoint_camera_order: Sequence[str]
) -> torch.Tensor:
    source_order = [str(name) for name in checkpoint_camera_order]
    target_weight = weight.new_zeros((len(config.CAMERA_ORDER), weight.shape[1]))
    source_index = {name: idx for idx, name in enumerate(source_order)}
    for target_idx, camera_name in enumerate(config.CAMERA_ORDER):
        source_idx = source_index.get(camera_name)
        if source_idx is None and camera_name == "rear_tele":
            source_idx = source_index.get("front_tele")
        if source_idx is not None and 0 <= source_idx < weight.shape[0]:
            target_weight[target_idx] = weight[source_idx]
    return target_weight


def infer_checkpoint_camera_order(
    checkpoint_obj: Mapping[str, Any], state_dict: Mapping[str, Any]
) -> Tuple[str, ...]:
    raw_camera_order = checkpoint_obj.get("camera_order")
    if isinstance(raw_camera_order, (list, tuple)) and raw_camera_order:
        return tuple((str(name) for name in raw_camera_order))
    camera_weight = state_dict.get("encoder.camera_embeddings.weight")
    if not torch.is_tensor(camera_weight):
        camera_weight = state_dict.get("camera_embeddings.weight")
    if torch.is_tensor(camera_weight):
        if camera_weight.shape[0] == len(config.CAMERA_ORDER):
            return config.CAMERA_ORDER
        if camera_weight.shape[0] == 5:
            return ("front_wide", "front_tele", "left_wide", "right_wide", "rear_tele")
        if camera_weight.shape[0] == 4:
            return ("front_wide", "front_tele", "left_wide", "right_wide")
    return config.CAMERA_ORDER


def load_model_checkpoint_compat(
    model: nn.Module, checkpoint_obj: Mapping[str, Any]
) -> None:
    raw_state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    if not isinstance(raw_state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a valid state dict.")
    checkpoint_camera_order = infer_checkpoint_camera_order(
        checkpoint_obj, raw_state_dict
    )
    model_state = model.state_dict()
    compatible_state: Dict[str, torch.Tensor] = {}
    skipped_keys: List[str] = []
    for key, value in raw_state_dict.items():
        if not torch.is_tensor(value):
            continue
        expected = model_state.get(key)
        if expected is None:
            continue
        tensor = value
        if (
            key in {"camera_embeddings.weight", "encoder.camera_embeddings.weight"}
            and tensor.shape != expected.shape
        ):
            tensor = remap_camera_embedding_weight(tensor, checkpoint_camera_order)
        if tensor.shape == expected.shape:
            compatible_state[key] = tensor
        else:
            skipped_keys.append(key)
    missing_keys, unexpected_keys = model.load_state_dict(
        compatible_state, strict=False
    )
    if skipped_keys:
        log(f"Skipped incompatible checkpoint tensors: {sorted(skipped_keys)}")
    if unexpected_keys:
        log(f"Ignored unexpected checkpoint tensors: {sorted(unexpected_keys)}")
    missing_report = [key for key in missing_keys if key not in skipped_keys]
    if missing_report:
        log(f"Checkpoint left model tensors at init values: {sorted(missing_report)}")
