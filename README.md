# interactive_annotation
An Interactive Annotation Framework for Kidney Tumor Segmentation based on 3D images

## Introduction
This repo contains a 3D implementation of an interactive segmentation framework for identifying the kidney tumors based on CT images. The 2D version of such framework can be found in the following paper:

Sakinis, Tomas, et al. "Interactive segmentation of medical images through fully convolutional neural networks." arXiv preprint arXiv:1903.08205 (2019).

I extended the framework to adapt 3D medical image. In addition, A novel attention 3D UNET is implemented to improve the segmentation accuracy.

Current segmentation resutls: 
3D UNET : 85% dice; 
Attention  UNET with 3 cliks: 89% dice 
