"""Evaluate a ViewOcc checkpoint."""

import argparse
import importlib
import os

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch
from mmdet.models import build_detector

from projects.mmdet3d_plugin.datasets.builder import (
    build_dataloader, custom_build_dataset)
from projects.mmdet3d_plugin.viewocc.apis import custom_multi_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ViewOcc')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--tmpdir')
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    cfg.data.test.test_mode = True
    dataset = custom_build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, 1, cfg.data.workers_per_gpu,
        dist=distributed, shuffle=False)
    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    else:
        model = MMDataParallel(model.cuda(), device_ids=[0])
    results = custom_multi_gpu_test(
        model, loader, args.tmpdir, args.gpu_collect)
    rank, _ = get_dist_info()
    if rank == 0:
        print(dataset.evaluate(results))


if __name__ == '__main__':
    main()
