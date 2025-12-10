# HeatMapLBPNET - CK+ Dataset

This repository contains sample code for training and evaluating LBPNet models on the **CK+ (Extended Cohn-Kanade) dataset**.

## Important Note

This codebase is specifically configured for the CK+ dataset. If you want to use other datasets, you will need to make appropriate adjustments to the data loaders, configurations, and preprocessing steps.

## Project Structure

Navigate to the `CK+/Box/` directory where you'll find two main folders:

### 1. **originalLBPNet**
Contains the original LBPNet implementation.

### 2. **HeatmapGuidedLBPNet**
Contains our heatmap-guided version of LBPNet.

## Getting Started

### Training Models

In each folder, you'll find the main training scripts:

- `train_original_model.py` - Train on the full-size images
- `train_original_model_cropped.py` - Train on cropped images

Heatmap version is only meant to run on cropped inputs so you will not find `train_original_model.py` inside HeatmapGuidedLBPNet/.

### Configuration

Both training scripts contain configuration parameters that can be adjusted as needed. You can modify:
- Learning rates
- Batch sizes
- Number of epochs
- Model architecture parameters
- Data augmentation settings
- And other hyperparameters

Simply open the training file you want to use and adjust the configuration section according to your needs.

## Usage

1. Navigate to your desired folder:
   ```
   cd CK+/Box/originalLBPNet
   ```
   or
   ```
   cd CK+/Box/HeatmapGuidedLBPNet
   ```

2. Run the training script:
   ```
   python train_original_model.py
   ```
   or
   ```
   python train_original_model_cropped.py
   ```

3. Adjust configurations within the training files as needed for your specific requirements.


## Dataset

This code is designed for the CK+ dataset. For other datasets, modifications will be required in the data loading and preprocessing pipeline.
You can download CK+ dataset by running CK+\getData.py & CK+\data_loader.py or using Kaggle: https://www.kaggle.com/datasets/davilsena/ckdataset
