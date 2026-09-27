"""Expose the ViewOcc occupancy losses."""

from .viewocc_loss import (
    multiscale_soft_supervision, soft_geo_scal_loss, soft_sem_scal_loss)

__all__ = [
    'multiscale_soft_supervision', 'soft_geo_scal_loss', 'soft_sem_scal_loss']
