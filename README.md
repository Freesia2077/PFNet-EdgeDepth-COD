# PFNet-EdgeDepth-COD
PyTorch implementation and experimental study of PFNet-based camouflaged object detection with PVTv2 backbone, edge-aware learning, and RGB-D depth-prior fusion.

## Overview

Camouflaged Object Detection (COD) aims to segment objects that are visually blended into their surroundings. This repository reproduces PFNet as the baseline and explores several architectural and training improvements for more accurate camouflaged object segmentation.

The project includes:

- PFNet reproduction and inference pipeline
- Training and hyperparameter analysis
- PVTv2-B2 backbone replacement
- Edge-aware auxiliary learning
- RGB-D / depth-prior fusion
- Evaluation on CHAMELEON, CAMO, and COD10K
- Qualitative and quantitative result analysis

## Motivation

Camouflaged objects often share highly similar texture, color, and structure with the background, making them difficult to detect using generic object detection or salient object detection methods. PFNet addresses this problem through a positioning-and-focus mechanism, while this project further investigates whether stronger backbones, boundary guidance, and depth cues can improve segmentation quality.

## Method

### Baseline: PFNet

PFNet contains two main components:

- Positioning Module: locates potential camouflaged objects from high-level features.
- Focus Module: mines false-positive and false-negative distractions to refine segmentation results.

### Improvement 1: PVTv2-B2 Backbone

The original ResNet-50 backbone is replaced with PVTv2-B2 to enhance global context modeling and multi-scale feature representation.

### Improvement 2: Edge-Aware Learning

An auxiliary edge branch is introduced to strengthen boundary perception. The design includes edge-aware feature modeling and edge-guided feature enhancement.

### Improvement 3: Depth-Prior Fusion

Depth information is introduced as an additional geometric prior. The model explores RGB-D fusion to improve object localization and boundary refinement in visually ambiguous scenes.

## Dataset

Experiments are conducted on commonly used COD datasets:

- CHAMELEON
- CAMO
- COD10K

Please download the datasets from their official sources and organize them as follows:

```text
datasets/
  CHAMELEON/
    image/
    mask/
  CAMO/
    image/
    mask/
  COD10K/
    image/
    mask/
