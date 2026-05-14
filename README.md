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

```text
PFNet-EdgeDepth-COD/
├── CVPR2021_PFNet-main/   # Original PFNet reproduction baseline
├── PVTv2/                 # PFNet with PVTv2-B2 backbone
├── BG/                    # Edge-aware / boundary-guided variant
├── depth/                 # Depth-prior fusion variant
├── README.md
├── requirements.txt
└── environment.yml
Datasets

Experiments are conducted on three commonly used COD datasets:

CHAMELEON
CAMO
COD10K

Please download the datasets from their official sources. The dataset files are not included in this repository.

A possible dataset structure is:

datasets/
├── TrainDataset/
│   ├── Image/
│   └── GT/
├── TestDataset/
│   ├── CHAMELEON/
│   │   ├── Image/
│   │   └── GT/
│   ├── CAMO/
│   │   ├── Image/
│   │   └── GT/
│   └── COD10K/
│       ├── Image/
│       └── GT/
└── Depth/
Installation

Clone this repository:

git clone https://github.com/Freesia2077/PFNet-EdgeDepth-COD.git
cd PFNet-EdgeDepth-COD

Create a conda environment:

conda env create -f environment.yml
conda activate pfnet-cod

Alternatively, install dependencies with pip:

pip install -r requirements.txt
Training

For the original PFNet baseline, enter the baseline directory:

cd CVPR2021_PFNet-main
python train.py

For other variants, please enter the corresponding directory and run the training script:

cd PVTv2
python train.py
cd BG
python train.py
cd depth
python train.py

Please modify dataset paths, pretrained backbone paths, and checkpoint paths according to your local environment.

Inference

Example:

python infer.py

The prediction maps will be saved to the configured result directory.

Evaluation

This project uses standard COD evaluation metrics:

S-measure
Adaptive E-measure
Weighted F-measure
Mean Absolute Error

The original PFNet evaluation code is located in:

CVPR2021_PFNet-main/eval/
Results
Method	CHAMELEON S↑	CHAMELEON F↑	CAMO S↑	CAMO F↑	COD10K S↑	COD10K F↑	COD10K M↓
PFNet official	0.882	0.810	0.782	0.695	0.800	0.660	0.040
PFNet reproduced	TODO	TODO	TODO	TODO	TODO	TODO	TODO
PFNet + 512 input	0.895	0.834	0.784	0.702	0.813	0.684	0.037
PFNet + PVTv2-B2	0.901	0.842	0.852	0.796	0.853	0.749	0.027
PFNet + Edge	TODO	TODO	TODO	TODO	TODO	TODO	TODO
PFNet + Depth	TODO	TODO	TODO	TODO	TODO	TODO	TODO
Main Findings
SGD with momentum shows better generalization than Adam in the PFNet reproduction experiments.
Increasing the input resolution from 416 to 512 improves fine-detail perception.
Replacing ResNet-50 with PVTv2-B2 strengthens global context modeling.
Edge-aware auxiliary learning helps improve boundary quality.
Depth priors can provide useful geometric cues, but noisy depth maps may hurt performance in complex scenes.
Acknowledgement

This repository builds upon the official PFNet implementation and is inspired by related works on PVTv2, boundary-guided camouflaged object detection, and source-free depth priors.

Citation

If this project is useful, please cite the original PFNet paper and related works.
