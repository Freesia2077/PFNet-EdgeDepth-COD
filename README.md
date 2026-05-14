# PFNet-EdgeDepth-COD

Reproduction and enhancement of PFNet for camouflaged object detection, including baseline reproduction, PVTv2 backbone replacement, edge-aware auxiliary learning, and depth-prior fusion.

## Overview

Camouflaged Object Detection (COD) aims to segment objects that are visually blended into their surroundings. This repository reproduces PFNet as the baseline and explores several improvements, including:

- PVTv2-B2 backbone replacement
- Edge-aware auxiliary learning
- RGB-D / depth-prior fusion
- Hyperparameter tuning and experimental analysis

The project is mainly based on PFNet and related works on boundary-guided COD and source-free depth priors.

## Repository Structure

- `CVPR2021_PFNet-main/`: Original PFNet reproduction baseline
- `PVTv2/`: PFNet with PVTv2-B2 backbone
- `BG/`: Edge-aware / boundary-guided variant
- `depth/`: Depth-prior fusion variant
- `README.md`: Project documentation
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment configuration

## Datasets

Experiments are conducted on three commonly used COD datasets:

- CHAMELEON
- CAMO
- COD10K

Please download the datasets from their official sources. The dataset files are not included in this repository.

Recommended dataset structure:

- `datasets/TrainDataset/Image/`
- `datasets/TrainDataset/GT/`
- `datasets/TestDataset/CHAMELEON/Image/`
- `datasets/TestDataset/CHAMELEON/GT/`
- `datasets/TestDataset/CAMO/Image/`
- `datasets/TestDataset/CAMO/GT/`
- `datasets/TestDataset/COD10K/Image/`
- `datasets/TestDataset/COD10K/GT/`
- `datasets/Depth/`

## Installation

Clone this repository:

`git clone https://github.com/Freesia2077/PFNet-EdgeDepth-COD.git`

Enter the repository:

`cd PFNet-EdgeDepth-COD`

Create a conda environment:

`conda env create -f environment.yml`

Activate the environment:

`conda activate pfnet-cod`

Alternatively, install dependencies with pip:

`pip install -r requirements.txt`

## Training

For the original PFNet baseline:

`cd CVPR2021_PFNet-main`

`python train.py`

For the PVTv2 variant:

`cd PVTv2`

`python train.py`

For the edge-aware variant:

`cd BG`

`python train.py`

For the depth-prior variant:

`cd depth`

`python train.py`

Please modify dataset paths, pretrained backbone paths, and checkpoint paths according to your local environment.

## Inference

Example:

`python infer.py`

The prediction maps will be saved to the configured result directory.

## Evaluation

This project uses standard COD evaluation metrics:

- S-measure
- Adaptive E-measure
- Weighted F-measure
- Mean Absolute Error

The original PFNet evaluation code is located in:

`CVPR2021_PFNet-main/eval/`

## Results

| Method | CHAMELEON S↑ | CHAMELEON E↑ | CHAMELEON F↑ | CHAMELEON M↓ | CAMO S↑ | CAMO E↑ | CAMO F↑ | CAMO M↓ | COD10K S↑ | COD10K E↑ | COD10K F↑ | COD10K M↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PFNet official | 0.882 | 0.942 | 0.810 | 0.033 | 0.782 | 0.852 | 0.695 | 0.085 | 0.800 | 0.868 | 0.660 | 0.040 |
| PFNet + Adam | 0.824 | 0.875 | 0.685 | 0.058 | 0.694 | 0.791 | 0.530 | 0.123 | 0.727 | 0.766 | 0.506 | 0.064 |
| PFNet + 512 input | 0.895 | 0.942 | 0.834 | 0.028 | 0.784 | 0.856 | 0.702 | 0.085 | 0.813 | 0.882 | 0.684 | 0.037 |
| PFNet + PVTv2-B2 | 0.901 | 0.934 | 0.842 | 0.027 | 0.852 | 0.908 | 0.796 | 0.055 | 0.853 | 0.908 | 0.749 | 0.027 |
| PFNet + Edge | 0.884 | 0.947 | 0.817 | 0.031 | 0.776 | 0.850 | 0.691 | 0.085 | 0.800 | 0.871 | 0.661 | 0.039 |
| PFNet + Depth | 0.887 | 0.951 | 0.812 | 0.036 | 0.785 | 0.849 | 0.696 | 0.088 | 0.793 | 0.874 | 0.663 | 0.036 |

## Main Findings

- SGD with momentum shows better generalization than Adam in the PFNet reproduction experiments.
- Increasing the input resolution from 416 to 512 improves fine-detail perception.
- Replacing ResNet-50 with PVTv2-B2 strengthens global context modeling.
- Edge-aware auxiliary learning helps improve boundary quality.
- Depth priors can provide useful geometric cues, but noisy depth maps may hurt performance in complex scenes.

## Acknowledgement

This repository builds upon the official PFNet implementation and is inspired by related works on PVTv2, boundary-guided camouflaged object detection, and source-free depth priors.

## Citation

If this project is useful, please cite the original PFNet paper and related works:

Mei, H., Ji, G.-P., Wei, Z., Yang, X., Wei, X., and Fan, D.-P. Camouflaged Object Segmentation with Distraction Mining. CVPR, 2021.
