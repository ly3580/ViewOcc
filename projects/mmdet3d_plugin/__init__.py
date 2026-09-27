"""Register the components required by ViewOcc."""

from .datasets import ViewOccDataset
from .datasets.pipelines import (
    CustomCollect3D, CustomDefaultFormatBundle3D,
    LoadMultiViewImageFromFilesViewOcc, LoadOccupancy,
    NormalizeMultiviewImage, PadMultiViewImage,
    PhotoMetricDistortionMultiViewImage)
from .viewocc import *
