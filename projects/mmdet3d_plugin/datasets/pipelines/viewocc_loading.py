"""Load multi-view images and sparse occupancy labels."""

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFilesViewOcc:
    """Load synchronized camera images."""

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        images = [
            mmcv.imread(path, self.color_type)
            for path in results['img_filename']]
        if any(image is None for image in images):
            raise FileNotFoundError('A configured camera image is missing.')
        if len({image.shape for image in images}) != 1:
            raise ValueError('All camera images must have the same shape.')
        if self.to_float32:
            images = [image.astype(np.float32) for image in images]
        results.update(
            filename=results['img_filename'],
            img=images,
            img_shape=images[0].shape,
            ori_shape=images[0].shape,
            pad_shape=images[0].shape,
            scale_factor=1.0,
            img_norm_cfg=dict(
                mean=np.zeros(3, dtype=np.float32),
                std=np.ones(3, dtype=np.float32),
                to_rgb=False),
            img_fields=['img'])
        return results


@PIPELINES.register_module()
class LoadOccupancy:
    """Load sparse ``[x, y, z, label]`` occupancy rows."""

    def __call__(self, results):
        occupancy = np.load(results['occ_path']).astype(np.float32)
        labels = occupancy[:, 3].astype(np.int64)
        invalid = (labels != 255) & ((labels < 1) | (labels > 5))
        if np.any(invalid):
            raise ValueError('Occupancy labels must be in [1, 5] or 255.')
        results['gt_occ'] = occupancy
        return results
