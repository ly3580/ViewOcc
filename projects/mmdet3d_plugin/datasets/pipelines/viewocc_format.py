"""Format ViewOcc tensors for batching."""

from mmcv.parallel import DataContainer
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D


@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Format images and variable-length sparse occupancy."""

    def __call__(self, results):
        results = super().__call__(results)
        results['gt_occ'] = DataContainer(
            to_tensor(results['gt_occ']), stack=False)
        return results
