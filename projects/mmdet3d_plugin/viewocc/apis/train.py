"""Build the ViewOcc runner and launch training."""

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook, EpochBasedRunner, Fp16OptimizerHook, OptimizerHook,
    build_optimizer, build_runner)
from mmdet.utils import get_root_logger

from ...core.evaluation import ViewOccDistEvalHook, ViewOccEvalHook
from ...datasets.builder import build_dataloader, custom_build_dataset


def custom_train_model(model, dataset, cfg, distributed=False,
                       validate=False, timestamp=None, meta=None):
    """Train ViewOcc with the configured source/target sampler."""
    logger = get_root_logger(cfg.log_level)
    datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    loaders = [
        build_dataloader(
            item,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.get('shuffler_sampler'))
        for item in datasets]
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get(
                'find_unused_parameters', False))
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model, optimizer=optimizer, work_dir=cfg.work_dir,
            logger=logger, meta=meta))
    runner.timestamp = timestamp
    if cfg.get('fp16') is not None:
        optimizer_hook = Fp16OptimizerHook(
            **cfg.optimizer_config, **cfg.fp16, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_hook = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_hook = cfg.optimizer_config
    runner.register_training_hooks(
        cfg.lr_config, optimizer_hook, cfg.checkpoint_config, cfg.log_config)
    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    if validate:
        val_dataset = custom_build_dataset(
            cfg.data.val, dict(test_mode=True))
        val_loader = build_dataloader(
            val_dataset, 1, cfg.data.workers_per_gpu,
            dist=distributed, shuffle=False)
        eval_cfg = cfg.get('evaluation', {}).copy()
        eval_cfg['by_epoch'] = True
        hook = ViewOccDistEvalHook if distributed else ViewOccEvalHook
        runner.register_hook(hook(val_loader, **eval_cfg))
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(loaders, cfg.workflow)
