"""Evaluate occupancy metrics during training."""

import os.path as osp

from mmcv.runner import DistEvalHook, EvalHook


class _ViewOccEvaluation:
    """Share the ViewOcc result collection path between hooks."""

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return
        from ...viewocc.apis.test import custom_multi_gpu_test
        tmpdir = self.tmpdir or osp.join(runner.work_dir, '.evaluation')
        results = custom_multi_gpu_test(
            runner.model, self.dataloader, tmpdir,
            getattr(self, 'gpu_collect', False))
        if runner.rank == 0:
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, score)


class ViewOccEvalHook(_ViewOccEvaluation, EvalHook):
    """Single-process evaluation hook."""


class ViewOccDistEvalHook(_ViewOccEvaluation, DistEvalHook):
    """Distributed evaluation hook."""
