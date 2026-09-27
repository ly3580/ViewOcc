"""Core ViewOcc geometry conditioning and consistency implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS

from ..dense_heads.viewocc_head import (
    _remap_target_gt, aggregate_target_logits)
from .viewocc_base import ViewOccBase


def _quat_to_rotmat(quat):
    quat = F.normalize(quat, p=2, dim=-1)
    x, y, z, w = quat.unbind(dim=-1)
    two = quat.new_tensor(2.0)
    rot = quat.new_zeros(quat.shape[:-1] + (3, 3))
    rot[..., 0, 0] = 1 - two * (y * y + z * z)
    rot[..., 0, 1] = two * (x * y - z * w)
    rot[..., 0, 2] = two * (x * z + y * w)
    rot[..., 1, 0] = two * (x * y + z * w)
    rot[..., 1, 1] = 1 - two * (x * x + z * z)
    rot[..., 1, 2] = two * (y * z - x * w)
    rot[..., 2, 0] = two * (x * z - y * w)
    rot[..., 2, 1] = two * (y * z + x * w)
    rot[..., 2, 2] = 1 - two * (x * x + y * y)
    return rot


class PluckerRayMap(nn.Module):
    """Build per-camera Plucker ray maps for image feature levels."""

    def __init__(self,
                 image_size=(640, 640),
                 intrinsic=((320.0, 0.0, 320.0),
                            (0.0, 320.0, 320.0),
                            (0.0, 0.0, 1.0)),
                 num_cams=6,
                 pixel_center=True,
                 normalize_dirs=True,
                 origin_scale=1.0):
        super().__init__()
        self.image_size = tuple(image_size)
        self.num_cams = int(num_cams)
        self.pixel_center = bool(pixel_center)
        self.normalize_dirs = bool(normalize_dirs)
        self.origin_scale = float(origin_scale)
        if self.origin_scale <= 0:
            raise ValueError('origin_scale must be positive.')
        self.register_buffer(
            'intrinsic',
            torch.tensor(intrinsic, dtype=torch.float32),
            persistent=False)

    def _camera_dirs(self, feat_h, feat_w, device):
        image_h, image_w = self.image_size
        offset = 0.5 if self.pixel_center else 0.0
        y = (torch.arange(feat_h, device=device, dtype=torch.float32) +
             offset) * (float(image_h) / float(feat_h))
        x = (torch.arange(feat_w, device=device, dtype=torch.float32) +
             offset) * (float(image_w) / float(feat_w))
        yy, xx = torch.meshgrid(y, x)
        intrinsic = self.intrinsic.to(device=device)
        dirs = torch.stack(
            [(xx - intrinsic[0, 2]) / intrinsic[0, 0],
             (yy - intrinsic[1, 2]) / intrinsic[1, 1],
             torch.ones_like(xx)],
            dim=-1)
        if self.normalize_dirs:
            dirs = F.normalize(dirs, p=2, dim=-1)
        return dirs

    def forward(self, img_metas, feature_shapes, ref_tensor):
        if not all('camera2global_pose' in meta for meta in img_metas):
            raise KeyError('camera2global_pose is required in img_metas.')
        camera_pose = ref_tensor.new_tensor(
            [meta['camera2global_pose'] for meta in img_metas],
            dtype=torch.float32)
        if (camera_pose.shape[1] != self.num_cams or
                camera_pose.shape[-1] != 7):
            raise ValueError(
                'camera2global_pose must have shape '
                f'[batch, {self.num_cams}, 7].')
        rotations = _quat_to_rotmat(camera_pose[..., 3:7])
        origins = camera_pose[..., :3] / self.origin_scale

        ray_maps = []
        for feat_h, feat_w in feature_shapes:
            dirs_camera = self._camera_dirs(
                feat_h, feat_w, ref_tensor.device)
            dirs_global = torch.einsum(
                'bnij,hwj->bnhwi', rotations, dirs_camera)
            if self.normalize_dirs:
                dirs_global = F.normalize(dirs_global, p=2, dim=-1)
            origins_global = origins[:, :, None, None, :].expand_as(
                dirs_global)
            moments = torch.cross(origins_global, dirs_global, dim=-1)
            plucker = torch.cat([dirs_global, moments], dim=-1)
            ray_maps.append(
                plucker.permute(0, 1, 4, 2, 3).to(ref_tensor.dtype))
        return ray_maps


@DETECTORS.register_module()
class ViewOcc(ViewOccBase):
    """Combine Plucker FiLM with class-peak pair consistency."""

    def __init__(self,
                 plucker_raymap_cfg=None,
                 plucker_fusion_channels=(512, 512, 512),
                 plucker_hidden_channels=64,
                 plucker_film_gamma_init=(0.15, 0.10, 0.05),
                 plucker_film_beta_init=(0.10, 0.08, 0.05),
                 ray_consistency_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        if plucker_raymap_cfg is None:
            raise ValueError('plucker_raymap_cfg is required.')
        self.plucker_raymap = PluckerRayMap(**plucker_raymap_cfg)
        self.plucker_fusion_channels = tuple(plucker_fusion_channels)
        self.ray_consistency_cfg = dict(ray_consistency_cfg or {})
        num_levels = len(self.plucker_fusion_channels)
        if (len(plucker_film_gamma_init) != num_levels or
                len(plucker_film_beta_init) != num_levels):
            raise ValueError(
                'Plucker FiLM initial values must match feature levels.')

        self.lambda_gamma = nn.Parameter(torch.tensor(
            plucker_film_gamma_init, dtype=torch.float32))
        self.lambda_beta = nn.Parameter(torch.tensor(
            plucker_film_beta_init, dtype=torch.float32))
        self.plucker_ray_encoders = nn.ModuleList()
        self.plucker_film_heads = nn.ModuleList()
        for channels in self.plucker_fusion_channels:
            self.plucker_ray_encoders.append(nn.Sequential(
                nn.Conv2d(6, plucker_hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    plucker_hidden_channels,
                    plucker_hidden_channels,
                    3,
                    padding=1),
                nn.ReLU(inplace=True)))
            film_head = nn.Conv2d(
                plucker_hidden_channels, 2 * channels, kernel_size=1)
            nn.init.zeros_(film_head.weight)
            nn.init.zeros_(film_head.bias)
            self.plucker_film_heads.append(film_head)

    def _fuse_plucker_features(self, img_feats, img_metas):
        # Condition image features on camera rays.
        if len(img_feats) != len(self.plucker_fusion_channels):
            raise ValueError(
                'Image and Plucker feature level counts do not match.')
        feature_shapes = [feat.shape[-2:] for feat in img_feats]
        ray_maps = self.plucker_raymap(
            img_metas, feature_shapes, img_feats[0])
        fused_feats = []
        for level, img_feat in enumerate(img_feats):
            bs, num_cams, channels, feat_h, feat_w = img_feat.shape
            if channels != self.plucker_fusion_channels[level]:
                raise ValueError(
                    f'Unexpected channels at feature level {level}: '
                    f'{channels}.')
            conv_dtype = self.plucker_ray_encoders[level][0].weight.dtype
            img_input = img_feat.reshape(
                bs * num_cams, channels, feat_h, feat_w).to(conv_dtype)
            ray_input = ray_maps[level].reshape(
                bs * num_cams, 6, feat_h, feat_w).to(conv_dtype)
            ray_feat = self.plucker_ray_encoders[level](ray_input)
            gamma, beta = self.plucker_film_heads[level](ray_feat).chunk(
                2, dim=1)
            gamma = self.lambda_gamma[level] * gamma
            beta = self.lambda_beta[level] * beta
            fused = (1.0 + gamma) * img_input + beta
            fused_feats.append(fused.reshape(
                bs, num_cams, channels, feat_h, feat_w).to(img_feat.dtype))
        return fused_feats

    def extract_img_feat(self, img, img_metas, len_queue=None):
        img_feats = super().extract_img_feat(
            img, img_metas, len_queue=len_queue)
        if img_feats is None:
            return None
        if len_queue is not None:
            raise ValueError('Plucker FiLM does not support len_queue.')
        return self._fuse_plucker_features(img_feats, img_metas)

    def _forward_pts(self, pts_feats, img_metas):
        return self.pts_bbox_head(pts_feats, img_metas)

    @staticmethod
    def _world_to_ego(points_world, img_meta):
        world_rot = _quat_to_rotmat(
            points_world.new_tensor(img_meta['ego2global_rotation']))
        world_trans = points_world.new_tensor(
            img_meta['ego2global_translation'])
        return torch.matmul(points_world - world_trans, world_rot)

    def _camera_world_rays(self, img_meta, ref_tensor, ray_grid_size):
        camera_pose = ref_tensor.new_tensor(img_meta['camera2global_pose'])
        if camera_pose.dim() != 2 or camera_pose.shape[-1] != 7:
            raise ValueError(
                'camera2global_pose must have shape [num_cams, 7].')
        grid_h, grid_w = ray_grid_size
        dirs_camera = self.plucker_raymap._camera_dirs(
            grid_h, grid_w, ref_tensor.device).to(ref_tensor.dtype)
        camera_rot = _quat_to_rotmat(camera_pose[:, 3:7])
        dirs_world = torch.einsum(
            'nij,hwj->nhwi', camera_rot, dirs_camera)
        dirs_world = F.normalize(dirs_world, p=2, dim=-1)
        origins_world = camera_pose[:, None, None, :3].expand_as(dirs_world)
        return origins_world.reshape(-1, 3), dirs_world.reshape(-1, 3)

    @staticmethod
    def _ray_box_interval(origins_world, dirs_world, img_meta, eps):
        ego_rot = _quat_to_rotmat(
            origins_world.new_tensor(img_meta['ego2global_rotation']))
        ego_trans = origins_world.new_tensor(
            img_meta['ego2global_translation'])
        origins_ego = torch.matmul(origins_world - ego_trans, ego_rot)
        dirs_ego = torch.matmul(dirs_world, ego_rot)
        pc_range = origins_world.new_tensor(img_meta['pc_range'])
        box_min, box_max = pc_range[:3], pc_range[3:]
        parallel = dirs_ego.abs() < eps
        outside_parallel = parallel & (
            (origins_ego < box_min) | (origins_ego > box_max))
        safe_dirs = torch.where(parallel, torch.ones_like(dirs_ego), dirs_ego)
        t0 = (box_min - origins_ego) / safe_dirs
        t1 = (box_max - origins_ego) / safe_dirs
        axis_near = torch.where(
            parallel,
            origins_world.new_full(t0.shape, float('-inf')),
            torch.minimum(t0, t1))
        axis_far = torch.where(
            parallel,
            origins_world.new_full(t0.shape, float('inf')),
            torch.maximum(t0, t1))
        near = axis_near.max(dim=-1).values.clamp_min(0.0)
        far = axis_far.min(dim=-1).values
        valid = (~outside_parallel.any(dim=-1)) & (far > near)
        return near, far, valid

    def _sample_ray_logits(self, occ_logits, sample_idx, points_world,
                           img_meta):
        points_ego = self._world_to_ego(points_world, img_meta)
        pc_range = occ_logits.new_tensor(img_meta['pc_range'])
        normalized = (
            2.0 * (points_ego - pc_range[:3]) /
            (pc_range[3:] - pc_range[:3]) - 1.0)
        grid = normalized.unsqueeze(0).unsqueeze(3)
        volume = occ_logits[sample_idx].permute(
            0, 3, 2, 1).unsqueeze(0)
        sampled = F.grid_sample(
            volume,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        return sampled.squeeze(0).squeeze(-1).permute(1, 2, 0)

    @staticmethod
    def _points_visible_in_any_camera(points_ego, img_meta):
        lidar2img = points_ego.new_tensor(img_meta['lidar2img'])
        if lidar2img.dim() == 2:
            lidar2img = lidar2img.unsqueeze(0)
        img_shapes = img_meta['img_shape']
        if len(img_shapes) >= 2 and not hasattr(img_shapes[0], '__len__'):
            img_shapes = [img_shapes] * lidar2img.shape[0]
        if len(img_shapes) != lidar2img.shape[0]:
            raise ValueError(
                'img_shape and lidar2img camera counts do not match.')
        points_homo = torch.cat(
            [points_ego, torch.ones_like(points_ego[:, :1])], dim=-1)
        projected = torch.matmul(
            lidar2img[:, None], points_homo[None, :, :, None]).squeeze(-1)
        depth = projected[..., 2]
        eps = points_ego.new_tensor(1e-5)
        image_x = projected[..., 0] / depth.clamp_min(eps)
        image_y = projected[..., 1] / depth.clamp_min(eps)
        heights = points_ego.new_tensor(
            [shape[0] for shape in img_shapes]).view(-1, 1)
        widths = points_ego.new_tensor(
            [shape[1] for shape in img_shapes]).view(-1, 1)
        visible = (
            (depth > eps) &
            (image_x >= 0) & (image_x < widths) &
            (image_y >= 0) & (image_y < heights))
        return visible.any(dim=0)

    @staticmethod
    def _pad_sparse_gt_occ(gt_occ):
        if isinstance(gt_occ, torch.Tensor):
            return gt_occ
        if not isinstance(gt_occ, (list, tuple)) or not gt_occ:
            raise TypeError('gt_occ must be a non-empty tensor or sequence.')
        max_points = max(int(sample.shape[0]) for sample in gt_occ)
        num_cols = int(gt_occ[0].shape[1])
        padded = gt_occ[0].new_zeros((len(gt_occ), max_points, num_cols))
        if num_cols > 3:
            padded[:, :, 3] = 255
        for sample_idx, sample in enumerate(gt_occ):
            padded[sample_idx, :sample.shape[0]] = sample
        return padded

    @staticmethod
    def _remap_target_gt(gt_occ, img_metas):
        return _remap_target_gt(gt_occ, img_metas)

    @staticmethod
    def _foreground_ray_mask(gt_occ, sample_idx, img_meta, ray_grid_size):
        grid_h, grid_w = ray_grid_size
        lidar2img = gt_occ.new_tensor(
            img_meta['lidar2img'], dtype=torch.float32)
        num_cams = lidar2img.shape[0]
        foreground = torch.zeros(
            (num_cams, grid_h, grid_w),
            dtype=torch.bool,
            device=gt_occ.device)
        sample = gt_occ[sample_idx]
        sample = sample[(sample[:, 3] != 255) & (sample[:, 3] != 0)]
        if sample.numel() == 0:
            return foreground.reshape(-1)
        pc_range = gt_occ.new_tensor(
            img_meta['pc_range'], dtype=torch.float32)
        occ_size = gt_occ.new_tensor(
            img_meta['occ_size'], dtype=torch.float32)
        voxel_size = (pc_range[3:] - pc_range[:3]) / occ_size
        points_ego = (
            sample[:, :3].float() + 0.5) * voxel_size + pc_range[:3]
        points_homo = torch.cat(
            [points_ego, torch.ones_like(points_ego[:, :1])], dim=1)
        projected = torch.matmul(
            lidar2img[:, None], points_homo[None, :, :, None]).squeeze(-1)
        depth = projected[..., 2]
        eps = gt_occ.new_tensor(1e-5, dtype=torch.float32)
        image_x = projected[..., 0] / depth.clamp_min(eps)
        image_y = projected[..., 1] / depth.clamp_min(eps)
        img_shapes = img_meta['img_shape']
        if len(img_shapes) >= 2 and not hasattr(img_shapes[0], '__len__'):
            img_shapes = [img_shapes] * num_cams
        heights = gt_occ.new_tensor(
            [shape[0] for shape in img_shapes],
            dtype=torch.float32).view(-1, 1)
        widths = gt_occ.new_tensor(
            [shape[1] for shape in img_shapes],
            dtype=torch.float32).view(-1, 1)
        visible = (
            (depth > eps) &
            (image_x >= 0) & (image_x < widths) &
            (image_y >= 0) & (image_y < heights))
        camera_idx, point_idx = torch.nonzero(visible, as_tuple=True)
        if camera_idx.numel() == 0:
            return foreground.reshape(-1)
        grid_x = torch.floor(
            image_x[camera_idx, point_idx] /
            widths[camera_idx, 0] * float(grid_w)).long().clamp(0, grid_w - 1)
        grid_y = torch.floor(
            image_y[camera_idx, point_idx] /
            heights[camera_idx, 0] * float(grid_h)).long().clamp(0, grid_h - 1)
        foreground[camera_idx, grid_y, grid_x] = True
        return F.max_pool2d(
            foreground[:, None].float(),
            kernel_size=3,
            stride=1,
            padding=1).squeeze(1).bool().reshape(-1)

    @staticmethod
    def _termination_distribution(logits, include_escape_bin, eps):
        empty_prob = F.softmax(logits, dim=-1)[..., 0]
        alpha = (1.0 - empty_prob).clamp(min=eps, max=1.0 - eps)
        survival = torch.cumprod(torch.cat(
            [torch.ones_like(alpha[:, :1]), 1.0 - alpha], dim=1), dim=1)
        termination = survival[:, :-1] * alpha
        if include_escape_bin:
            termination = torch.cat(
                [termination, survival[:, -1:]], dim=1)
        return termination / termination.sum(
            dim=1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _symmetric_kl(student_prob, teacher_prob, eps):
        student_prob = student_prob.clamp_min(eps)
        teacher_prob = teacher_prob.clamp_min(eps)
        student_log = torch.log(student_prob)
        teacher_log = torch.log(teacher_prob)
        student_teacher = (
            student_prob * (student_log - teacher_log)).sum(dim=1)
        teacher_student = (
            teacher_prob * (teacher_log - student_log)).sum(dim=1)
        return 0.5 * (student_teacher + teacher_student)

    @staticmethod
    def _peak_window_class_distribution(
            logits, termination_dist, peak_indices, window_size, eps):
        num_samples = logits.shape[1]
        sample_indices = torch.arange(
            num_samples, device=logits.device)[None, :]
        peak_indices = peak_indices[:, None]
        window_mask = (
            (sample_indices >= peak_indices - window_size) &
            (sample_indices <= peak_indices + window_size))
        weights = termination_dist[:, :num_samples] * window_mask.to(
            termination_dist.dtype)
        window_mass = weights.sum(dim=1)
        class_prob = F.softmax(logits[..., 1:], dim=-1)
        class_dist = (
            weights[:, :, None] * class_prob
        ).sum(dim=1) / window_mass[:, None].clamp_min(eps)
        return class_dist, window_mass

    @staticmethod
    def _aggregate_target_logits(occ_logits):
        return aggregate_target_logits(occ_logits)

    def _directional_ray_consistency(self,
                                     occ_logits,
                                     student_idx,
                                     teacher_idx,
                                     img_metas,
                                     gt_occ,
                                     cfg):
        # Match ray termination and local semantic distributions.
        ray_grid_size = tuple(cfg.get('ray_grid_size', (64, 64)))
        num_rays = int(cfg.get('num_rays', 4096))
        num_samples = int(cfg.get('num_samples_per_ray', 64))
        foreground_ray_ratio = float(cfg.get('foreground_ray_ratio', 0.9))
        min_ray_length = float(cfg.get('min_ray_length', 0.5))
        require_co_visible = bool(cfg.get('require_co_visible', True))
        detach_teacher = bool(cfg.get('detach_teacher', True))
        include_escape_bin = bool(cfg.get('include_escape_bin', True))
        class_loss_weight = float(cfg.get('class_loss_weight', 1.0))
        class_peak_window = int(cfg.get('class_peak_window', 1))
        class_ray_weights = cfg.get(
            'class_ray_weights', (1.0, 2.0, 2.5, 2.0, 2.0))
        if len(class_ray_weights) != 5:
            raise ValueError('class_ray_weights must contain five values.')
        eps = float(cfg.get('eps', 1e-6))

        student_meta = img_metas[student_idx]
        teacher_meta = img_metas[teacher_idx]
        origins, directions = self._camera_world_rays(
            student_meta, occ_logits, ray_grid_size)
        student_near, student_far, student_valid = self._ray_box_interval(
            origins, directions, student_meta, eps)
        teacher_near, teacher_far, teacher_valid = self._ray_box_interval(
            origins, directions, teacher_meta, eps)
        near = torch.maximum(student_near, teacher_near)
        far = torch.minimum(student_far, teacher_far)
        valid = (student_valid & teacher_valid &
                 ((far - near) >= min_ray_length))
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            return None

        foreground_mask = self._foreground_ray_mask(
            gt_occ, student_idx, student_meta, ray_grid_size)
        foreground_indices = valid_indices[foreground_mask[valid_indices]]
        background_indices = valid_indices[~foreground_mask[valid_indices]]

        def _random_subset(indices, count):
            if indices.numel() <= count:
                return indices
            order = torch.randperm(
                indices.numel(), device=indices.device)[:count]
            return indices[order]

        desired_foreground = int(round(num_rays * foreground_ray_ratio))
        selected = torch.cat([
            _random_subset(foreground_indices, desired_foreground),
            _random_subset(
                background_indices, num_rays - desired_foreground)], dim=0)
        remaining_count = min(num_rays, int(valid.sum().item())) - selected.numel()
        if remaining_count > 0:
            selected_mask = torch.zeros_like(valid)
            selected_mask[selected] = True
            remaining_indices = torch.nonzero(
                valid & ~selected_mask, as_tuple=False).squeeze(1)
            selected = torch.cat([
                selected,
                _random_subset(remaining_indices, remaining_count)], dim=0)
        origins, directions = origins[selected], directions[selected]
        near, far = near[selected], far[selected]
        depth_fraction = (
            torch.arange(
                num_samples,
                device=occ_logits.device,
                dtype=occ_logits.dtype) + 0.5) / float(num_samples)
        depths = near[:, None] + (
            far - near)[:, None] * depth_fraction[None]
        points_world = (
            origins[:, None, :] +
            depths[:, :, None] * directions[:, None, :])

        if require_co_visible:
            student_visible = self._points_visible_in_any_camera(
                self._world_to_ego(points_world, student_meta).reshape(-1, 3),
                student_meta).reshape(points_world.shape[:2])
            teacher_visible = self._points_visible_in_any_camera(
                self._world_to_ego(points_world, teacher_meta).reshape(-1, 3),
                teacher_meta).reshape(points_world.shape[:2])
            ray_visible = (student_visible & teacher_visible).all(dim=1)
            if not ray_visible.any():
                return None
            points_world = points_world[ray_visible]
            depths = depths[ray_visible]

        student_logits = self._sample_ray_logits(
            occ_logits, student_idx, points_world, student_meta)
        teacher_logits = self._sample_ray_logits(
            occ_logits, teacher_idx, points_world, teacher_meta)
        if detach_teacher:
            teacher_logits = teacher_logits.detach()
        student_dist = self._termination_distribution(
            student_logits, include_escape_bin, eps)
        teacher_dist = self._termination_distribution(
            teacher_logits, include_escape_bin, eps)
        termination_loss = self._symmetric_kl(
            student_dist, teacher_dist, eps)
        teacher_peak = teacher_dist[:, :num_samples].argmax(dim=1)
        student_class_dist, student_mass = (
            self._peak_window_class_distribution(
                student_logits,
                student_dist,
                teacher_peak,
                class_peak_window,
                eps))
        teacher_class_dist, teacher_mass = (
            self._peak_window_class_distribution(
                teacher_logits,
                teacher_dist,
                teacher_peak,
                class_peak_window,
                eps))
        class_loss = 0.5 * (student_mass + teacher_mass) * self._symmetric_kl(
            student_class_dist, teacher_class_dist, eps)
        teacher_class = teacher_class_dist.argmax(dim=1)
        ray_weights = teacher_class_dist.new_tensor(
            class_ray_weights)[teacher_class]
        class_loss = ray_weights * class_loss
        ray_loss = termination_loss + class_loss_weight * class_loss

        return ray_loss.mean()

    def _compute_ray_consistency(self, gt_occ, occ_preds, img_metas):
        # Average source, target, and cross-domain pair losses.
        cfg = self.ray_consistency_cfg
        level = cfg.get('level', 'last')
        level_idx = len(occ_preds) - 1 if level == 'last' else int(level)
        occ_logits = self._aggregate_target_logits(
            occ_preds[level_idx]).float()
        gt_occ = self._remap_target_gt(
            self._pad_sparse_gt_occ(gt_occ), img_metas)

        pair_to_indices = {}
        for sample_idx, img_meta in enumerate(img_metas):
            if 'pair_id' in img_meta:
                pair_to_indices.setdefault(
                    int(img_meta['pair_id']), []).append(sample_idx)

        def _ordered_pair(indices):
            role_to_index = {
                str(img_metas[index].get('pair_role', '')): index
                for index in indices
            }
            if set(role_to_index) == {'anchor', 'positive'}:
                return [role_to_index['anchor'], role_to_index['positive']]
            return list(indices)

        target_robot_label = int(cfg.get('target_robot_label', 2))
        source_pairs, target_pairs = [], []
        for indices in pair_to_indices.values():
            if len(indices) != 2:
                continue
            labels = [
                int(img_metas[index].get('robot_label', -1))
                for index in indices
            ]
            if all(label == target_robot_label for label in labels):
                target_pairs.append(_ordered_pair(indices))
            elif all(label != target_robot_label for label in labels):
                source_pairs.append(_ordered_pair(indices))

        if (bool(cfg.get('skip_source_only_ray_consistency', False)) and
                source_pairs and not target_pairs):
            return occ_logits.sum() * 0.0

        group_specs = []
        source_internal = []
        for first_idx, second_idx in source_pairs:
            source_internal.append((first_idx, second_idx))
        group_specs.append(('source_', source_internal))
        target_internal = []
        for first_idx, second_idx in target_pairs:
            target_internal.append((first_idx, second_idx))
        group_specs.append(('target_', target_internal))
        source_target = []
        for source_pair, target_pair in zip(source_pairs, target_pairs):
            for source_idx, target_idx in zip(source_pair, target_pair):
                source_target.append((target_idx, source_idx))
        group_specs.append(('source_target_', source_target))

        group_losses = []
        zero = occ_logits.sum() * 0.0
        for _, index_pairs in group_specs:
            losses = []
            for student_idx, teacher_idx in index_pairs:
                loss = self._directional_ray_consistency(
                    occ_logits,
                    student_idx,
                    teacher_idx,
                    img_metas,
                    gt_occ,
                    cfg)
                if loss is not None:
                    losses.append(loss)
            if losses:
                group_losses.append(torch.stack(losses).mean())

        if not group_losses:
            return zero
        return torch.stack(group_losses).mean()

    def forward_pts_train(self, pts_feats, gt_occ, img_metas):
        # Add consistency supervision to the occupancy objective.
        outs = self._forward_pts(pts_feats, img_metas)
        losses = self.pts_bbox_head.loss(
            gt_occ, outs, img_metas=img_metas)
        ray_weight = float(
            self.ray_consistency_cfg.get('loss_weight', 0.5))
        if ray_weight <= 0.0:
            return losses

        ray_loss = self._compute_ray_consistency(
            gt_occ, outs['occ_preds'], img_metas)
        losses['loss_ray_cons'] = ray_weight * ray_loss
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        return self._forward_pts(x, img_metas)
