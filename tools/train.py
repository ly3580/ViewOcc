"""Train ViewOcc from a compact configuration."""

import argparse
import importlib
import os
import time

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import set_random_seed
from mmdet.models import build_detector
from mmdet3d.utils import get_root_logger

from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset
from projects.mmdet3d_plugin.viewocc.apis import custom_train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViewOcc')
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume-from')
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
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
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume_from:
        cfg.resume_from = args.resume_from

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        cfg.gpu_ids = range(1)
    cfg.seed = args.seed
    set_random_seed(args.seed, deterministic=args.deterministic)
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    logger = get_root_logger(log_level=cfg.log_level)

    model = build_detector(cfg.model)
    model.init_weights()
    dataset = custom_build_dataset(cfg.data.train)
    logger.info('ViewOcc training initialized.')
    custom_train_model(
        model, [dataset], cfg, distributed=distributed,
        validate=not args.no_validate, timestamp=timestamp,
        meta=dict(seed=args.seed))


if __name__ == '__main__':
    main()
