"""Run distributed ViewOcc evaluation."""

import mmcv
import torch
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu, collect_results_gpu


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Collect per-sample occupancy statistics from all workers."""
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    progress = mmcv.ProgressBar(len(data_loader.dataset)) if rank == 0 else None
    for data in data_loader:
        with torch.no_grad():
            output = model(return_loss=False, rescale=True, **data)
        batch_results = list(output['evaluation'])
        results.extend(batch_results)
        if progress is not None:
            for _ in range(len(batch_results) * world_size):
                progress.update()
    if world_size == 1:
        return results
    if gpu_collect:
        return collect_results_gpu(results, len(data_loader.dataset))
    return collect_results_cpu(results, len(data_loader.dataset), tmpdir)
