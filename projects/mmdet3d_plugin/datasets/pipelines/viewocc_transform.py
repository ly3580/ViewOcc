"""Transform and collect ViewOcc inputs."""

import mmcv
import numpy as np
from mmcv.parallel import DataContainer
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PadMultiViewImage:
    """Pad each camera image to a common divisor."""

    def __init__(self, size_divisor, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        results['ori_shape'] = [image.shape for image in results['img']]
        results['img'] = [
            mmcv.impad_to_multiple(
                image, self.size_divisor, pad_val=self.pad_val)
            for image in results['img']]
        results['img_shape'] = [image.shape for image in results['img']]
        results['pad_shape'] = results['img_shape']
        return results


@PIPELINES.register_module()
class NormalizeMultiviewImage:
    """Normalize every camera image."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = [
            mmcv.imnormalize(image, self.mean, self.std, self.to_rgb)
            for image in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply compact brightness and contrast augmentation."""

    def __call__(self, results):
        images = []
        for image in results['img']:
            if np.random.randint(2):
                image = image + np.random.uniform(-32, 32)
            if np.random.randint(2):
                image = image * np.random.uniform(0.5, 1.5)
            images.append(image)
        results['img'] = images
        return results


@PIPELINES.register_module()
class CustomCollect3D:
    """Collect tensors and geometry metadata."""

    META_KEYS = (
        'filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam',
        'cam_intrinsic', 'pad_shape', 'img_norm_cfg', 'pc_range', 'occ_size',
        'occ_path', 'token', 'sample_idx', 'robot_label', 'pair_id',
        'pair_role', 'pair_token', 'reference_pair_id', 'reference_index',
        'reference_token', 'reference_distance_xyz',
        'ego2global_translation', 'ego2global_rotation',
        'camera2global_pose', 'camera2global_translation',
        'camera2global_rotation')

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        metadata = {
            key: results[key] for key in self.META_KEYS if key in results}
        data = {'img_metas': DataContainer(metadata, cpu_only=True)}
        data.update({key: results[key] for key in self.keys})
        return data
