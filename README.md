# Exploring Temporal Action Segmentation Techniques for Enhanced Bird Behavior Recognition

This project explores deep learning methods for fine-grained bird behavior recognition in ecological video data. Specifically, it evaluates frame-level and segment-level Temporal Action Segmentation (TAS) pipelines using modern Convolutional Neural Networks (CNNs), Transformers, and tailored data augmentation strategies. All experiments are conducted on the Visual-WetlandBirds dataset, which contains densely annotated bird behaviors recorded in real wetland environments.

This repository contains the full experimental framework developed for the study.

## Frame-level pipeline

- `generate_cropped_masks.py`: Generates segmentation masks using SAM, required for data augmentations.
- `extract_and_train_FL.py`: Extracts features (from the base or augmented dataset) and trains MLP classifiers.
- `evaluate_FL.py`: Evaluates the frame-level model, reporting mAP, accuracy, per-class AP, and confusion matrices.

## Segment-level pipeline

Includes both original PDAN code (`PDAN.py`, `apmeter.py`, `meter.py`) and custom modules:

- `birds_feature_dataset.py`: Loads and manages the dataset.
- `extract_I3D_crops.py`, `extract_mvit_and_r2plus1d.py`: Extract features from I3D, R(2+1)D, and MViT-B backbones.
- `train_PDAN_birds.py`, `evaluate_pdan.py`: Train and evaluate PDAN.
- `train_and_eval_PDAN.sh`: Wrapper script for configuring, training, and evaluating segment-level models.

---

# Setup Environment & Run

To ensure reproducibility, the repository includes a `Dockerfile` and a `launch_docker.sh` script.

### 1. Download the dataset  
The Visual-WetlandBirds dataset is available at [Zenodo](https://zenodo.org/records/14355257).

### 2. Build and launch the Docker container  
./launch_docker.sh

### 3. Run experiments inside the container

Use the scripts based on the task you want to run:

#### Frame-level

- `generate_cropped_masks.py`: Create SAM-based masks for augmentations.
- `extract_and_train_FL.py`: Extract features and train MLP classifiers.
- `evaluate_FL.py`: Evaluate model and report metrics.

#### Segment-level

- `extract_I3D_crops.py`, `extract_mvit_and_r2plus1d.py`: Extract features using video backbones.
- `train_and_eval_PDAN.sh`: Full pipeline for training and evaluating PDAN, with optional feature extraction.


# Author
bruno.sancho@ua.es
