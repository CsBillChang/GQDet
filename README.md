# GQDet

This repository contains the submission-time code snapshot for **GQDet**, an object detection model built on top of DEIM/D-FINE.

GQDet introduces three components in the DEIM pipeline:

- **DSWA**: a deformable self-attention block inserted into the high-level encoder feature.
- **DGMF**: a dynamic gated multi-scale feature fusion neck.
- **GQRC**: a query refinement head for the D-FINE decoder.

The public snapshot keeps the project structure, configuration chain, training entry, and module interfaces used in the paper. Some internal implementation details are simplified for anonymous review and will be updated after the paper process allows a complete release.

## Project Layout

```text
GQDet/
├── train.py
├── configs/
│   ├── base/
│   ├── dataset/
│   ├── dfine/
│   └── ultralytics-yaml/
│       ├── deim_m_substation_dswa_dgmf_gqrc.yml
│       └── cfg-improve/dfine-m-dswa-dgmf-gqrc.yaml
└── engine/
    ├── backbone/
    ├── core/
    ├── data/
    ├── deim/
    ├── extre_module/
    ├── misc/
    ├── optim/
    └── solver/
```

## Environment

```bash
conda create -n gqdet python=3.10 -y
conda activate gqdet
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Dataset

The released configuration expects a COCO-style detection dataset. Edit:

```text
configs/dataset/substation_detection.yml
```

and set the image folders and annotation files for your local dataset.

## Training

```bash
python train.py -c configs/ultralytics-yaml/deim_m_substation_dswa_dgmf_gqrc.yml --use-amp
```

For multi-GPU training, launch the same configuration with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py -c configs/ultralytics-yaml/deim_m_substation_dswa_dgmf_gqrc.yml --use-amp
```
