"""Load the generic multi-view occupancy annotation format."""

import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset


@DATASETS.register_module()
class ViewOccDataset(Custom3DDataset):
    """Provide images, poses, pairs, and sparse occupancy labels."""

    CLASSES = tuple(
        f'pretrained_{index:02d}' for index in range(1, 17))

    def __init__(self, occ_size, pc_range, camera_names=None,
                 load_interval=1, *args, **kwargs):
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_names = camera_names
        self.load_interval = int(load_interval)
        self.use_semantic = True
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load timestamp-ordered annotation records."""
        annotations = mmcv.load(ann_file)
        infos = sorted(annotations['infos'], key=lambda item: item['timestamp'])
        self.metadata = annotations.get('metadata', {})
        return infos[::self.load_interval]

    def get_data_info(self, index):
        """Convert one annotation record to pipeline inputs."""
        info = self.data_infos[index]
        data = dict(
            occ_path=info['occ_path'],
            occ_size=np.asarray(self.occ_size),
            pc_range=np.asarray(self.pc_range),
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'])
        for key in (
                'token', 'robot_label', 'pair_id', 'pair_role', 'pair_token',
                'reference_pair_id', 'reference_index', 'reference_token',
                'reference_distance_xyz'):
            if key in info:
                data[key] = info[key]
        data['sample_idx'] = str(info.get('token', index))

        if self.modality['use_camera']:
            camera_items = info['cams'].items()
            if self.camera_names is not None:
                camera_items = [
                    (name, info['cams'][name]) for name in self.camera_names]
            image_paths, lidar2img, lidar2cam, intrinsics = [], [], [], []
            camera_poses, camera_translations, camera_rotations = [], [], []
            for _, camera in camera_items:
                image_paths.append(camera['data_path'])
                lidar_to_camera = np.asarray(camera['lidar2cam'])
                intrinsic = np.asarray(camera['cam_intrinsic'])
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam.append(lidar_to_camera)
                lidar2img.append(intrinsic_pad @ lidar_to_camera)
                intrinsics.append(intrinsic_pad)
                camera_poses.append(camera['camera2global_pose'])
                camera_translations.append(
                    camera['camera2global_translation'])
                camera_rotations.append(camera['camera2global_rotation'])
            data.update(
                img_filename=image_paths,
                lidar2img=lidar2img,
                lidar2cam=lidar2cam,
                cam_intrinsic=intrinsics,
                camera2global_pose=camera_poses,
                camera2global_translation=camera_translations,
                camera2global_rotation=camera_rotations)
        return data

    def evaluate(self, results, **kwargs):
        """Aggregate occupancy IoU statistics."""
        scores = np.stack(results, axis=0).sum(axis=0)
        ious = scores[:, 0] / np.maximum(
            scores[:, 1] + scores[:, 2] - scores[:, 0], 1)
        names = ['IoU', 'flat', 'agent', 'static', 'terrain', 'vegetation']
        metrics = {name: float(ious[i]) for i, name in enumerate(names)}
        metrics['mIoU'] = float(ious[1:].mean())
        return metrics
