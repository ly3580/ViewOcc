"""Evaluate six-class semantic occupancy."""

import numpy as np


SOURCE_PC_RANGE = np.asarray(
    [-25.0, -25.0, -5.0, 25.0, 25.0, 3.0], dtype=np.float32)
SOURCE_OCC_SIZE = np.asarray([100, 100, 16], dtype=np.float32)


def _sparse_to_dense(gt, pc_range, occ_size):
    """Project sparse source-grid labels onto the prediction grid."""
    voxel_size = (
        SOURCE_PC_RANGE[3:] - SOURCE_PC_RANGE[:3]) / SOURCE_OCC_SIZE
    world = (gt[:, :3] + 0.5) * voxel_size + SOURCE_PC_RANGE[:3]
    target_size = np.asarray(occ_size, dtype=np.int64)
    target_voxel = (
        np.asarray(pc_range[3:]) - np.asarray(pc_range[:3])) / target_size
    xyz = np.floor(
        (world - np.asarray(pc_range[:3])) / target_voxel).astype(np.int64)
    valid = np.all((xyz >= 0) & (xyz < target_size), axis=1)
    dense = np.zeros(target_size, dtype=np.int64)
    dense[tuple(xyz[valid].T)] = gt[valid, 3].astype(np.int64)
    return dense


def evaluation_semantic(pred_occ, gt_occ, img_meta, class_num):
    """Return intersection, target, and prediction counts."""
    results = []
    for prediction, target in zip(pred_occ.cpu().numpy(), gt_occ):
        target = _sparse_to_dense(
            target.cpu().numpy(), img_meta['pc_range'], img_meta['occ_size'])
        valid = target != 255
        score = np.zeros((class_num, 3), dtype=np.float64)
        for label in range(class_num):
            target_mask = target != 0 if label == 0 else target == label
            pred_mask = prediction != 0 if label == 0 else prediction == label
            score[label] = (
                np.logical_and(target_mask, pred_mask)[valid].sum(),
                target_mask[valid].sum(),
                pred_mask[valid].sum())
        results.append(score)
    return np.stack(results)
