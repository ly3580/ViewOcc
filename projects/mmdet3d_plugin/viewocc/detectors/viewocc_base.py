"""Provide the minimal image-to-occupancy execution path for ViewOcc."""

import torch
from mmcv.runner import auto_fp16
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from ...datasets.viewocc_evaluation import evaluation_semantic
from ...models.utils.grid_mask import GridMask
from ..dense_heads.viewocc_head import (
    TARGET_NUM_CLASSES, aggregate_target_logits)


class ViewOccBase(MVXTwoStageDetector):
    """Run image encoding, occupancy lifting, training, and evaluation."""

    def __init__(self,
                 use_grid_mask,
                 img_backbone,
                 img_neck,
                 pts_bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            **kwargs)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False,
            ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = bool(use_grid_mask)
        self.fp16_enabled = False

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Encode each camera image into multi-scale features."""
        batch_size = img.size(0)
        if img.dim() == 5 and batch_size == 1:
            img = img.squeeze(0)
        elif img.dim() == 5:
            batch_size, num_cams, channels, height, width = img.size()
            img = img.reshape(
                batch_size * num_cams, channels, height, width)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        features = self.img_backbone(img)
        if isinstance(features, dict):
            features = list(features.values())
        if self.with_img_neck:
            features = self.img_neck(features)
        outputs = []
        for feature in features:
            batch_cams, channels, height, width = feature.shape
            if len_queue is not None:
                outputs.append(feature.view(
                    int(batch_size / len_queue),
                    len_queue,
                    int(batch_cams / batch_size),
                    channels,
                    height,
                    width))
            else:
                outputs.append(feature.view(
                    batch_size,
                    int(batch_cams / batch_size),
                    channels,
                    height,
                    width))
        return outputs

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(
            img, img_metas, len_queue=len_queue)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img',))
    def forward_train(self, img_metas=None, gt_occ=None, img=None):
        features = self.extract_feat(img=img, img_metas=img_metas)
        return self.forward_pts_train(features, gt_occ, img_metas)

    def forward_pts_train(self, pts_feats, gt_occ, img_metas):
        outputs = self.pts_bbox_head(pts_feats, img_metas)
        return self.pts_bbox_head.loss(
            gt_occ, outputs, img_metas=img_metas)

    def forward_test(self, img_metas, img=None, gt_occ=None, **kwargs):
        output = self.simple_test(img_metas, img)
        pred = aggregate_target_logits(output['occ_preds'][-1])
        pred = torch.softmax(pred, dim=1).argmax(dim=1)
        metrics = evaluation_semantic(
            pred, gt_occ, img_metas[0], TARGET_NUM_CLASSES)
        return dict(evaluation=metrics)

    def simple_test_pts(self, features, img_metas, rescale=False):
        return self.pts_bbox_head(features, img_metas)

    def simple_test(self, img_metas, img=None, rescale=False):
        features = self.extract_feat(img=img, img_metas=img_metas)
        return self.simple_test_pts(
            features, img_metas, rescale=rescale)
