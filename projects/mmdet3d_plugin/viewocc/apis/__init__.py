"""Expose ViewOcc train and evaluation entry points."""

from .test import custom_multi_gpu_test
from .train import custom_train_model

__all__ = ['custom_train_model', 'custom_multi_gpu_test']
