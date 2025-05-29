# Exploring Temporal Action Segmentation Techniques for Enhanced Bird Behavior Recognition
This project investigates deep learning methods for fine-grained bird behavior recognition in ecological video data. Specifically, it explores frame-level and segment-level Temporal Action Segmentation (TAS) pipelines, leveraging modern Convolutional Neural Networks (CNNs), Transformers, and domain-specific data augmentation strategies. All experiments are conducted on the Visual-WetlandBirds dataset, featuring densely annotated bird behaviors in real wetland environments.
This repository provides the complete experimental framework used in our study. For the frame-level pipeline, it includes: `generate_cropped_masks.py` to generate segmentation masks using SAM (required for the dataset augmentation); `extract_and_train_FL.py` to extract features (over the augmented or base dataset) and train MLP classifiers; and `evaluate_FL.py` for computing mAP, accuracy, per-class AP, and confusion matrices. For the segment-level pipeline, we include original PDAN code (`PDAN.py`, `apmeter.py`, `meter.py`) and adapted modules: `birds_feature_dataset.py` to use the dataset; `extract_I3D_crops.py`, and `extract_mvit_and_r2plus1d.py` for feature extraction; `train_PDAN_birds.py`, `evaluate_pdan.py` and `train_and_eval_PDAN.sh` for training and evaluation.


# Setup enviroment

# Author
bruno.sancho@ua.es
