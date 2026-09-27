"""Build ViewOcc datasets and data loaders."""

import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset
from mmdet.datasets.samplers import (
    DistributedGroupSampler, DistributedSampler, GroupSampler)
from mmcv.utils import build_from_cfg
from torch.utils.data import DataLoader

from .samplers import ViewOccPairSampler


def custom_build_dataset(cfg, default_args=None):
    """Build a dataset, including two-file source/target training."""
    if isinstance(cfg.get('ann_file'), (list, tuple)):
        return _concat_dataset(cfg, default_args)
    return build_from_cfg(cfg, DATASETS, default_args)


def _worker_init(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1,
                     dist=True, shuffle=True, seed=None,
                     shuffler_sampler=None, **kwargs):
    """Build the training or evaluation data loader."""
    rank, world_size = get_dist_info()
    if dist and shuffle and (
            shuffler_sampler or {}).get('type') == 'ViewOccPairSampler':
        sampler = ViewOccPairSampler(
            dataset, samples_per_gpu, world_size, rank, seed)
    elif dist and shuffle:
        sampler = DistributedGroupSampler(
            dataset, samples_per_gpu, world_size, rank, seed)
    elif dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=False, seed=seed)
    else:
        sampler = GroupSampler(
            dataset, samples_per_gpu) if shuffle else None

    batch_size = samples_per_gpu if dist else num_gpus * samples_per_gpu
    workers = workers_per_gpu if dist else num_gpus * workers_per_gpu
    init_fn = None if seed is None else partial(
        _worker_init, num_workers=workers, rank=rank, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=kwargs.pop('pin_memory', True),
        worker_init_fn=init_fn,
        **kwargs)
