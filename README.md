# ViewOcc

ViewOcc is a camera-based 3D occupancy perception codebase built on MMDetection3D. It includes two experiment stages: Plücker FiLM supervision and class-peak pair consistency.

## Repository structure

- `configs/`: base, stage 1, and stage 2 experiment configurations.
- `projects/mmdet3d_plugin/`: ViewOcc models, datasets, evaluation, and training APIs.
- `tools/`: training and evaluation entry points.
- `requirements.txt`: Python dependencies used by this codebase.

## Installation

Create an environment with a Python and CUDA combination compatible with the pinned legacy OpenMMLab packages, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Data and checkpoints

Before running an experiment, replace the placeholder paths in the selected config (for example `<DATA_ROOT>`, annotation paths, occupancy paths, and pretrained checkpoints) with paths for your local setup.

## Training

```bash
# Stage 1
python tools/train.py configs/viewocc_stage1.py

# Stage 2
python tools/train.py configs/viewocc_stage2.py
```

For distributed training:

```bash
bash tools/dist_train.sh CONFIG GPUS WORK_DIR
```

## Evaluation

```bash
python tools/test.py CONFIG CHECKPOINT
```

For distributed evaluation:

```bash
bash tools/dist_test.sh CONFIG CHECKPOINT GPUS
```
