# Machine Learning (ML) and Deep Learning (DL) based methods

This repository includes different projects in the field of Artificial Intelligence (AI).

## MSDA-WJDOT
It is an algorithm based on Optimal Transport that performs multi-source domain adaptation. 
It has been implemented in PyTorch.

## DL4dysarthricASR 
It contains multiple folders related to different Deep Learning strategies to address Automatic Speech Recognition for Dysarthric patients.

- AF_Synthesis: given audio and/or phonetic labels it generates Articulatory Features.
  The codes have been implemented in TensorFlow 1.0.
- SA_MSDA-WJDOT: it applies MSDA-WJDOT algorithm to the ASR systems, in which the target domain is a dysarthric speaker.
  The codes have been implemented in PyTorch.
- EasyCall: it contains information about a collected Dysarthric Speech Corpus called "EasyCall".

## AD_classification
It is a DL pipeline from data processing to model evaluation in the context of Alzheimer's Disease (AD) diagnosis.
The codes have been implemented in PyTorch.

## 3D_CNN_pretrained_model
It provides a 3D-CNN model pretrained on ADNI dataset to diagnose AD. 
The model can be directly used for external validation or as feature extractor.
The codes have been implemented in PyTorch. 
