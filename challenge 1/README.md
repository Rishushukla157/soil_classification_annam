# Soil-classification-challenge- 1
**Soil Classification with EfficientNet-B3 and Test Time Augmentation**
A comprehensive deep learning solution for soil type classification using EfficientNet-B3 architecture with 5-fold cross-validation and Test Time Augmentation (TTA) for enhanced prediction accuracy.

**Project Overview**
This project implements a robust soil classification pipeline that leverages state-of-the-art computer vision techniques to classify different soil types from images. The solution combines multiple advanced techniques including ensemble learning, cross-validation, and test time augmentation to achieve optimal performance.

Key Features

1.Advanced Architecture: EfficientNet-B3 with ImageNet pre-trained weights for superior
feature extraction

2.Robust Training: 5-fold stratified cross-validation ensuring balanced class distribution

3.Enhanced Inference: Test Time Augmentation (TTA) with multiple transformations

4.Ensemble Learning: Prediction averaging across multiple trained models

5.GPU Optimization: Automatic GPU detection and efficient batch processing

**Technical Architecture
Model Configuration**
Base Model: EfficientNet-B3 with ImageNet pre-trained weights

1.Input Resolution: 224×224 pixels

2.Final classifier layer adapted for soil classification

3.Transfer learning approach for optimal feature extraction

**Training Strategy**
**Cross-Validation Setup:**

1.5-fold stratified cross-validation

2.Each fold trains for 10 epochs

3.Best model weights saved based on validation accuracy

4.Learning rate scheduling with ReduceLROnPlateau

**Optimization Parameters:**

1.Optimizer: Adam with learning rate 1e-4

2.Loss Function: CrossEntropyLoss

3.Batch Size: 32

4.Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)

**Data Processing Pipeline
Training Data Augmentation**

train_transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(10),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

Test Time Augmentation (TTA)**
The model applies three different transformations during inference:

1.Original Transform: Standard resize and ImageNet normalization

2.Horizontal Flip: Images flipped horizontally with probability 1.0

3.Rotation: Random rotation within ±15 degrees

All TTA predictions are averaged to produce the final classification result.

**Dataset Structure**

soil_classification-2025/
├── train/ # Training images directory
├── test/ # Test images directory
├── train_labels.csv # Training labels with image_id and soil_type
└── test_ids.csv # Test image IDs for prediction

**Data Format**
**Training Labels (train_labels.csv):**
text
image_id,soil_type
image_001.jpg,clay
image_002.jpg,sand
...
**Test IDs (test_ids.csv):**
text
image_id
test_001.jpg
test_002.jpg
...

**Implementation Details**
Custom Dataset Classes**

SoilDataset: Handles training and validation data loading with configurable transformations

TTASoilDataset: Specialized dataset for Test Time Augmentation that applies multiple transformations to each test image

**Training Process**

1.Data Preparation: Load training data and encode soil type labels

2.Cross-Validation Loop: Train 5 separate models on different data splits

3.Model Training: Each fold trains with early stopping based on validation accuracy

4.Model Selection: Best weights saved for each fold based on validation performance

**Inference Pipeline**
1.TTA Application: Apply multiple augmentations to each test image

2.Model Ensemble: Run inference with all 5 trained models

3.Prediction Averaging: Average probabilities across TTA transformations and model folds

4.Final Classification: Select class with highest averaged probability

**Requirements**
text
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0

**Usage Instructions**

Running the Complete Pipeline
python
1. Import required libraries and set up paths

2. Load and preprocess training data

3. Train 5-fold cross-validation models

4. Apply TTA inference on test set

5. Generate ensemble predictions

6. Create submission file

**Output Generation**
The notebook automatically generates a submission.csv file with the following format:

text
image_id,soil_type
test_001.jpg,clay
test_002.jpg,sand
test_003.jpg,loam
...

**Performance Optimization**
**Memory Management**
1.Efficient batch processing to handle GPU memory constraints

2.Proper tensor cleanup and garbage collection

3.Optimized data loading with appropriate batch sizes

**Training Efficiency**

1.Early Stopping: Prevents overfitting and reduces training time

2.Learning Rate Scheduling: Adaptive learning rate for optimal convergence

3.GPU Acceleration: Automatic CUDA utilization when available

**Inference Speed**

1.Batch Processing: Efficient test set processing

2.Model Caching: Reuse of trained models across predictions

3.Optimized Transformations: Streamlined TTA pipeline

**Model Ensemble Strategy**
**Cross-Validation Benefits**

1.Reduced Overfitting: Multiple models trained on different data splits

2.Improved Generalization: Ensemble averaging reduces prediction variance

3.Robust Performance: Less sensitive to individual model variations

**TTA Integration**

1.Multiple Perspectives: Different augmentations provide varied viewpoints

2.Prediction Stability: Averaging reduces impact of transformation-specific artifacts

3.Enhanced Accuracy: Combines benefits of data augmentation at inference time


**File Structure**

text
├── fork-of-notebook1a0c5bef1c.ipynb # Main implementation notebook
├── submission.csv # Generated predictions (after execution)
├── README.md # This documentation
└── requirements.txt # Python dependencies
**Performance Characteristics**

**Training Metrics**

1.Training Time: Approximately 15 minutes on GPU for complete 5-fold training

2.Memory Usage: Optimized for standard Kaggle GPU environments

3.Convergence: Stable training with learning rate scheduling

**Prediction Quality**

1.Ensemble Robustness: Multiple model averaging for reliable predictions

2.TTA Enhancement: Improved accuracy through augmentation averaging

3.Cross-Validation Stability: Consistent performance across different data splits

**Configuration Parameters**
Hyperparameters

python
N_FOLDS = 5
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
RANDOM_STATE = 42


**TTA Configuartion**
python
TTA_TRANSFORMS = 3 # Original, HorizontalFlip, Rotation
ROTATION_DEGREES = 15
HORIZONTAL_FLIP_PROB = 1.0


**Reproducibility**
All random operations use fixed seeds for consistent results:

1.PyTorch: Manual seed setting for model initialization

2.NumPy: Random state configuration for data splitting

3.Scikit-learn: Fixed random_state for cross-validation splits

**Troubleshooting**
Common Issues

1.GPU Memory: Reduce batch size if encountering CUDA out of memory errors

2.Data Loading: Ensure correct file paths for Kaggle environment

3.Dependencies: Verify all required packages are installed with correct versions

**Performance Tips**
1.Batch Size: Adjust based on available GPU memory
2.Epochs: Monitor validation accuracy to determine optimal training duration
3.TTA: Balance between accuracy improvement and inference time
