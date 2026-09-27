"""Loss utilities used by the ViewOcc occupancy head."""

import torch
import torch.nn.functional as F


def _safe_bce(value, target):
    with torch.cuda.amp.autocast(enabled=False):
        return F.binary_cross_entropy(value.float(), target.float())


def multiscale_soft_supervision(gt_occ, ratio, gt_shape, num_classes):
    """Aggregate sparse labels into voxel-wise class fractions."""
    batch_size = int(gt_shape[0])
    out_w, out_h, out_z = [int(size) for size in gt_shape[2:5]]
    block_volume = int(ratio) ** 3
    num_output_voxels = out_w * out_h * out_z
    target = gt_occ.new_zeros(
        (batch_size, num_classes, out_w, out_h, out_z),
        dtype=torch.float32)
    target[:, 0] = float(block_volume)

    fine_w = out_w * int(ratio)
    fine_h = out_h * int(ratio)
    fine_z = out_z * int(ratio)
    for batch_idx in range(batch_size):
        labels = gt_occ[batch_idx, :, 3].long()
        valid = labels != 255
        coords = gt_occ[batch_idx, valid, :3].long()
        labels = labels[valid]
        if coords.numel() == 0:
            continue
        if ((labels <= 0) | (labels >= num_classes)).any():
            invalid = torch.unique(
                labels[(labels <= 0) | (labels >= num_classes)]).tolist()
            raise ValueError(f'Invalid occupancy labels: {invalid}.')

        in_bounds = (
            (coords[:, 0] >= 0) & (coords[:, 0] < fine_w) &
            (coords[:, 1] >= 0) & (coords[:, 1] < fine_h) &
            (coords[:, 2] >= 0) & (coords[:, 2] < fine_z))
        coords = coords[in_bounds]
        labels = labels[in_bounds]
        if coords.numel() == 0:
            continue

        fine_linear = (
            coords[:, 0] * fine_h * fine_z +
            coords[:, 1] * fine_z +
            coords[:, 2])
        coord_label, coord_label_counts = torch.unique(
            fine_linear * num_classes + labels,
            return_counts=True)
        fine_linear = torch.div(
            coord_label, num_classes, rounding_mode='floor')
        labels = coord_label % num_classes
        unique_coords, coord_inverse = torch.unique(
            fine_linear, return_inverse=True)
        coord_label_counts = coord_label_counts.to(target.dtype)
        coord_totals = torch.zeros(
            unique_coords.shape[0],
            dtype=target.dtype,
            device=target.device)
        coord_totals.scatter_add_(0, coord_inverse, coord_label_counts)
        label_weights = coord_label_counts / coord_totals[coord_inverse]

        x = torch.div(
            unique_coords, fine_h * fine_z, rounding_mode='floor')
        yz = unique_coords % (fine_h * fine_z)
        y = torch.div(yz, fine_z, rounding_mode='floor')
        z = yz % fine_z
        coarse_x = torch.div(x, ratio, rounding_mode='floor')
        coarse_y = torch.div(y, ratio, rounding_mode='floor')
        coarse_z = torch.div(z, ratio, rounding_mode='floor')
        coarse_linear = (
            coarse_x * out_h * out_z +
            coarse_y * out_z +
            coarse_z)

        flat_target = target[batch_idx].reshape(
            num_classes, num_output_voxels)
        occupied_counts = torch.bincount(
            coarse_linear, minlength=num_output_voxels).to(target.dtype)
        flat_target[0] -= occupied_counts
        coarse_linear = coarse_linear[coord_inverse]
        class_linear = labels * num_output_voxels + coarse_linear
        class_counts = torch.bincount(
            class_linear,
            weights=label_weights,
            minlength=num_classes * num_output_voxels).reshape(
                num_classes, num_output_voxels)
        flat_target += class_counts

    if (target < 0).any():
        raise ValueError('Sparse labels exceed the target voxel capacity.')
    return target / float(block_volume)


def soft_geo_scal_loss(pred, target_distribution):
    """Match empty and occupied geometry statistics."""
    pred = F.softmax(pred, dim=1)
    empty_prob = pred[:, 0]
    occupied_prob = 1 - empty_prob
    occupied_target = 1 - target_distribution[:, 0]
    intersection = (occupied_target * occupied_prob).sum()
    precision = intersection / occupied_prob.sum().clamp_min(1e-6)
    recall = intersection / occupied_target.sum().clamp_min(1e-6)
    empty_target = target_distribution[:, 0]
    specificity = (
        empty_target * empty_prob).sum() / empty_target.sum().clamp_min(1e-6)
    one = torch.ones_like(precision)
    return (
        _safe_bce(precision, one) +
        _safe_bce(recall, one) +
        _safe_bce(specificity, one))


def soft_sem_scal_loss(pred, target_distribution):
    """Match per-class precision, recall, and specificity."""
    pred = F.softmax(pred, dim=1)
    losses = []
    for class_idx in range(pred.shape[1]):
        target = target_distribution[:, class_idx]
        if target.sum() <= 0:
            continue
        probability = pred[:, class_idx]
        intersection = (probability * target).sum()
        precision = intersection / probability.sum().clamp_min(1e-6)
        recall = intersection / target.sum().clamp_min(1e-6)
        negative_target = 1 - target
        specificity = (
            (1 - probability) * negative_target).sum() / (
                negative_target.sum().clamp_min(1e-6))
        one = torch.ones_like(precision)
        losses.append(
            _safe_bce(precision, one) +
            _safe_bce(recall, one) +
            _safe_bce(specificity, one))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()
