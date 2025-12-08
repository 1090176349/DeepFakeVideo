# DeepFakeVideo
Code for Enhancing Deepfake Video Detection via Stochastic Differential Equations and Quantum Uncertainty Fusion.

## Project Name and Introduction
We propose a Deepfake Video Detection framework leveraging 3D Residual Networks combined with Continuous-time Neural Stochastic Differential Filter Modules (CNSDFM), Fast Fractional-Order Temporal Anomaly Detection Modules (FOTADM), and Quantum Uncertainty-Aware Fusion Modules (QUAFM) to enhance the performance of video tampering detection.

## Requirements

To run this project, you'll need to have the following libraries installed:

- Python 3.9+
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- Pillow (PIL)
- numpy
- opencv-python
- dlib
- pandas
- tqdm

You can install the required libraries using pip:

```bash
pip install torch torchvision matplotlib scikit-learn pillow numpy opencv-python
```

The dataset can be downloaded from the links provided below:
1. FaceForensics++: https://github.com/ondyari/FaceForensics
2. Celeb-DF: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
3. DFDC (DeepFake Detection Challenge): https://www.kaggle.com/c/deepfake-detection-challenge


## Dataset Preparation

### Step 1: Convert Videos into Frames
Extract frames from videos using preprocessing scripts. Ensure to remove frames containing incomplete faces, non-face regions (e.g., hands, clothing), or irrelevant background content to focus on facial regions.

### Step 2: Dataset Structure

Organize your dataset in the following structure:

```
video_dataset/
    dataset_name1/
        REAL/
            video1/
                frame1.jpg
                frame2.jpg
                ...
            video2/
                ...
        FAKE/
            video1/
                frame1.jpg
                frame2.jpg
                ...
            video2/
                ...
    dataset_name2/
        ... (same structure as dataset_name1)
```

### Step 3: Split Dataset into Train, Validation, and Test Sets

Configure dataset paths in `config/config.py` (under `TRAIN_DATASET_PATHS` and `TEST_DATASET_PATHS`). The training-validation split ratio can be adjusted via `TRAIN_VAL_SPLIT_RATIO` in the config file.

## Training and Testing the Model

### Configuration Setup
Modify parameters in `config/config.py` as needed, including:
- Dataset paths (`TRAIN_DATASET_PATHS`, `TEST_DATASET_PATHS`)
- Training hyperparameters (`NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`)
- Model parameters (`NUM_FRAMES`, `TRAIN_TRANSFORM`, `TEST_TRANSFORM`)
- Device configuration (`DEVICE`)

### Train the Model
Run the training script to start training and save the best model (based on validation AUC):

```bash
python train.py
```

The best model will be saved in the `./checkpoints` directory with a name generated based on the training datasets.

### Test the Model
Evaluate the trained model on the test set using:

```bash
python test.py
```

This will output test set metrics including Accuracy (ACC) and AUC-ROC.

## Results Visualization

The training script generates and saves the following visualizations after training:
- Training and validation loss curves
- Training and validation accuracy curves
- Training and validation AUC curves

These plots help analyze model performance trends, overfitting, and convergence.

## Conclusion
This project presents a comprehensive workflow for deepfake video detection, integrating 3D residual networks with advanced temporal filtering and uncertainty-aware fusion modules. The framework includes data loading, model training with early stopping, performance evaluation, providing a robust solution for detecting manipulated video content.

Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation

```bibtex
@article{Zhang2025SDEQNet,
  title={Enhancing Deepfake Video Detection via Stochastic Differential Equations and Quantum Uncertainty Fusion},
  author={Ruixing Zhang and Bin Li and Degang Xu},
  journal={},
  year={2025},
  publisher={}
}
```
