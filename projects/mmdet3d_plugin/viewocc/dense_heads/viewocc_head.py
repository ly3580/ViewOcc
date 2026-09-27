"""Lift multi-view features and predict semantic occupancy."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (
    build_conv_layer, build_norm_layer, build_upsample_layer)
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer

from ..loss.viewocc_loss import (
    multiscale_soft_supervision, soft_geo_scal_loss, soft_sem_scal_loss)


PRETRAINED_TO_TARGET = torch.tensor(
    [0, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 1, 1, 4, 3, 5],
    dtype=torch.long)
TARGET_NUM_CLASSES = 6
SOURCE_PC_RANGE = (-25.0, -25.0, -5.0, 25.0, 25.0, 3.0)
SOURCE_OCC_SIZE = (100.0, 100.0, 16.0)
TARGET_FOCAL_ALPHA = (1.0, 1.0, 5.0, 2.0, 2.0, 1.5)


def _voxel_to_world(coords, pc_range, occ_size):
    voxel_size = (pc_range[3:] - pc_range[:3]) / occ_size
    return (coords + 0.5) * voxel_size + pc_range[:3]


def _world_to_voxel(coords, pc_range, occ_size):
    voxel_size = (pc_range[3:] - pc_range[:3]) / occ_size
    return torch.floor((coords - pc_range[:3]) / voxel_size)


def _pad_sparse_gt(gt_occ):
    if isinstance(gt_occ, torch.Tensor):
        return gt_occ
    if not isinstance(gt_occ, (list, tuple)) or not gt_occ:
        raise TypeError('gt_occ must be a tensor or a non-empty list.')
    max_points = max(int(sample.shape[0]) for sample in gt_occ)
    num_cols = int(gt_occ[0].shape[1])
    padded = gt_occ[0].new_zeros((len(gt_occ), max_points, num_cols))
    padded[:, :, 3] = 255
    for sample_idx, sample in enumerate(gt_occ):
        padded[sample_idx, :sample.shape[0]] = sample
    return padded


def _remap_target_gt(gt_occ, img_metas):
    remapped = gt_occ.clone()
    source_range = gt_occ.new_tensor(SOURCE_PC_RANGE)
    source_size = gt_occ.new_tensor(SOURCE_OCC_SIZE)
    for sample_idx, img_meta in enumerate(img_metas):
        target_range = gt_occ.new_tensor(img_meta['pc_range'])
        target_size = gt_occ.new_tensor(img_meta['occ_size'])
        world = _voxel_to_world(
            gt_occ[sample_idx, :, :3], source_range, source_size)
        coords = _world_to_voxel(world, target_range, target_size)
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < target_size[0]) &
            (coords[:, 1] >= 0) & (coords[:, 1] < target_size[1]) &
            (coords[:, 2] >= 0) & (coords[:, 2] < target_size[2]))
        remapped[sample_idx, :, :3] = 0
        remapped[sample_idx, :, 3] = 255
        remapped[sample_idx, valid, :3] = coords[valid]
        remapped[sample_idx, valid, 3] = gt_occ[sample_idx, valid, 3]
    return remapped


def aggregate_target_logits(pred):
    """Map 17 pretrained logits to the six target classes."""
    if pred.shape[1] == TARGET_NUM_CLASSES:
        return pred
    if pred.shape[1] != len(PRETRAINED_TO_TARGET):
        raise ValueError(
            f'Expected {len(PRETRAINED_TO_TARGET)} logits, got {pred.shape[1]}.')
    mapping = PRETRAINED_TO_TARGET.to(pred.device)
    outputs = []
    for target_label in range(TARGET_NUM_CLASSES):
        class_ids = torch.nonzero(
            mapping == target_label, as_tuple=False).squeeze(-1)
        outputs.append(torch.logsumexp(pred[:, class_ids], dim=1))
    return torch.stack(outputs, dim=1)


def _soft_focal_loss(pred, target):
    probability = F.softmax(pred, dim=1)
    log_probability = F.log_softmax(pred, dim=1)
    alpha = pred.new_tensor(TARGET_FOCAL_ALPHA).view(1, -1, 1, 1, 1)
    loss = (
        -target * alpha * ((1 - probability) ** 1.5) * log_probability)
    return loss.sum(dim=1).mean()


@HEADS.register_module()
class ViewOccHead(nn.Module):
    """Decode image features into multi-scale occupancy logits."""

    def __init__(self,
                 transformer_template,
                 num_classes,
                 volume_h,
                 volume_w,
                 volume_z,
                 upsample_strides,
                 out_indices,
                 conv_input,
                 conv_output,
                 embed_dims,
                 img_channels,
                 volume_embedding_load_src_shapes=None,
                 volume_embedding_load_src_pc_range=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.num_classes = int(num_classes)
        if self.num_classes != len(PRETRAINED_TO_TARGET):
            raise ValueError('ViewOccHead requires 17 pretrained classes.')
        self.volume_h = list(volume_h)
        self.volume_w = list(volume_w)
        self.volume_z = list(volume_z)
        self.upsample_strides = list(upsample_strides)
        self.out_indices = list(out_indices)
        self.conv_input = list(conv_input)
        self.conv_output = list(conv_output)
        self.embed_dims = list(embed_dims)
        self.img_channels = list(img_channels)
        self.transformer_template = transformer_template
        self.volume_embedding_load_src_shapes = (
            volume_embedding_load_src_shapes)
        self.volume_embedding_load_src_pc_range = (
            volume_embedding_load_src_pc_range)
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        # Build one lifting transformer per feature scale.
        self.transformer = nn.ModuleList()
        for level in range(len(self.embed_dims)):
            cfg = copy.deepcopy(self.transformer_template)
            cfg.embed_dims = cfg.embed_dims[level]
            layer = cfg.encoder.transformerlayers
            layer.attn_cfgs[0].deformable_attention.num_points = (
                self.transformer_template.encoder.transformerlayers
                .attn_cfgs[0].deformable_attention.num_points[level])
            layer.feedforward_channels = (
                self.transformer_template.encoder.transformerlayers
                .feedforward_channels[level])
            layer.embed_dims = (
                self.transformer_template.encoder.transformerlayers
                .embed_dims[level])
            layer.attn_cfgs[0].embed_dims = (
                self.transformer_template.encoder.transformerlayers
                .attn_cfgs[0].embed_dims[level])
            layer.attn_cfgs[0].deformable_attention.embed_dims = (
                self.transformer_template.encoder.transformerlayers
                .attn_cfgs[0].deformable_attention.embed_dims[level])
            cfg.encoder.num_layers = (
                self.transformer_template.encoder.num_layers[level])
            self.transformer.append(build_transformer(cfg))

        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        self.deblocks = nn.ModuleList()
        for index, out_channels in enumerate(self.conv_output):
            stride = self.upsample_strides[index]
            if stride > 1:
                layer = build_upsample_layer(
                    dict(type='deconv3d', bias=False),
                    in_channels=self.conv_input[index],
                    out_channels=out_channels,
                    kernel_size=stride,
                    stride=stride)
            else:
                layer = build_conv_layer(
                    dict(type='Conv3d', bias=False),
                    in_channels=self.conv_input[index],
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            self.deblocks.append(nn.Sequential(
                layer,
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True)))

        self.occ = nn.ModuleList([
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=self.conv_output[index],
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
            for index in self.out_indices
        ])
        self.volume_embedding = nn.ModuleList([
            nn.Embedding(
                self.volume_h[level] *
                self.volume_w[level] *
                self.volume_z[level],
                self.embed_dims[level])
            for level in range(len(self.embed_dims))
        ])
        self.transfer_conv = nn.ModuleList([
            nn.Sequential(
                build_conv_layer(
                    dict(type='Conv2d', bias=True),
                    in_channels=self.img_channels[level],
                    out_channels=self.embed_dims[level],
                    kernel_size=1,
                    stride=1),
                nn.ReLU(inplace=True))
            for level in range(len(self.embed_dims))
        ])

    def init_weights(self):
        for transformer in self.transformer:
            transformer.init_weights()
        for module in self.modules():
            if hasattr(module, 'conv_offset'):
                constant_init(module.conv_offset, 0)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        own_state = self.state_dict()
        if (self.volume_embedding_load_src_shapes is not None and
                self.volume_embedding_load_src_pc_range is not None):
            target_range = self.transformer_template.encoder.pc_range
            for level, source_shape in enumerate(
                    self.volume_embedding_load_src_shapes):
                key = prefix + f'volume_embedding.{level}.weight'
                local_key = key[len(prefix):]
                if key not in state_dict or local_key not in own_state:
                    continue
                if state_dict[key].shape == own_state[local_key].shape:
                    continue
                target_shape = (
                    self.volume_h[level],
                    self.volume_w[level],
                    self.volume_z[level])
                state_dict[key] = self._resample_volume_embedding(
                    state_dict[key],
                    source_shape,
                    self.volume_embedding_load_src_pc_range,
                    target_shape,
                    target_range)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    @staticmethod
    def _resample_volume_embedding(weight, source_shape, source_pc_range,
                                   target_shape, target_pc_range):
        source_h, source_w, source_z = [int(size) for size in source_shape]
        target_h, target_w, target_z = [int(size) for size in target_shape]
        if weight.shape[0] != source_h * source_w * source_z:
            raise ValueError('Volume embedding source shape is inconsistent.')
        source_range = weight.new_tensor(
            source_pc_range, dtype=torch.float32)
        target_range = weight.new_tensor(
            target_pc_range, dtype=torch.float32)
        x = (
            (torch.arange(target_w, device=weight.device) + 0.5) /
            target_w * (target_range[3] - target_range[0]) + target_range[0])
        y = (
            (torch.arange(target_h, device=weight.device) + 0.5) /
            target_h * (target_range[4] - target_range[1]) + target_range[1])
        z = (
            (torch.arange(target_z, device=weight.device) + 0.5) /
            target_z * (target_range[5] - target_range[2]) + target_range[2])
        norm_x = (
            2 * (x - source_range[0]) /
            (source_range[3] - source_range[0]) - 1)
        norm_y = (
            2 * (y - source_range[1]) /
            (source_range[4] - source_range[1]) - 1)
        norm_z = (
            2 * (z - source_range[2]) /
            (source_range[5] - source_range[2]) - 1)
        grid_z, grid_y, grid_x = torch.meshgrid(norm_z, norm_y, norm_x)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)
        channels = int(weight.shape[1])
        volume = weight.float().reshape(
            source_z, source_h, source_w, channels).permute(
                3, 0, 1, 2).unsqueeze(0)
        resized = F.grid_sample(
            volume, grid, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        return resized.squeeze(0).permute(
            1, 2, 3, 0).reshape(-1, channels).to(weight.dtype)

    @auto_fp16(apply_to=('mlvl_feats',))
    def forward(self, mlvl_feats, img_metas):
        # Lift each image scale into a 3D volume.
        batch_size, num_cams = mlvl_feats[0].shape[:2]
        dtype = mlvl_feats[0].dtype
        volumes = []
        for level in range(len(self.embed_dims)):
            queries = self.volume_embedding[level].weight.to(dtype)
            _, _, channels, height, width = mlvl_feats[level].shape
            image = mlvl_feats[level].reshape(
                batch_size * num_cams, channels, height, width)
            weight_dtype = self.transfer_conv[level][0].weight.dtype
            image = self.transfer_conv[level](
                image.to(weight_dtype)).reshape(
                    batch_size, num_cams, -1, height, width)
            volumes.append(self.transformer[level](
                [image],
                queries,
                volume_h=self.volume_h[level],
                volume_w=self.volume_w[level],
                volume_z=self.volume_z[level],
                img_metas=img_metas))

        volume_stack = [
            volumes[level].reshape(
                batch_size,
                self.volume_z[level],
                self.volume_h[level],
                self.volume_w[level],
                -1).permute(0, 4, 3, 2, 1)
            for level in range(len(volumes))
        ]
        outputs = []
        result = volume_stack.pop()
        for index, deblock in enumerate(self.deblocks):
            result = deblock(result)
            if index in self.out_indices:
                outputs.append(result)
            elif volume_stack:
                result = result + volume_stack.pop()
        occ_preds = [
            layer(output) for layer, output in zip(self.occ, outputs)]
        return dict(volume_embed=volumes, occ_preds=occ_preds)

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_occ, preds_dicts, img_metas):
        # Supervise the target six-class grid at every output scale.
        gt_occ = _remap_target_gt(_pad_sparse_gt(gt_occ), img_metas)
        losses = {}
        predictions = [
            aggregate_target_logits(pred)
            for pred in preds_dicts['occ_preds']
        ]
        for level, pred in enumerate(predictions):
            ratio = 2 ** (len(predictions) - 1 - level)
            target = multiscale_soft_supervision(
                gt_occ, ratio, pred.shape, TARGET_NUM_CLASSES)
            loss = (
                _soft_focal_loss(pred, target) +
                soft_sem_scal_loss(pred, target) +
                soft_geo_scal_loss(pred, target))
            losses[f'loss_occ_{level}'] = (
                loss * (0.5 ** (len(predictions) - 1 - level)))
        return losses
