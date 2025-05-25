Soil Image Classification README
Overview
This project implements a deep learning pipeline for binary classification of soil images using PyTorch and EfficientNet-B0. The workflow includes data loading, preprocessing, model training, validation, and generation of predictions for test images, resulting in a submission-ready CSV file.

Dataset Structure
Train Images: /kaggle/input/soil-classification-part-2/soil_competition-2025/train/

Test Images: /kaggle/input/soil-classification-part-2/soil_competition-2025/test/

Train Labels: /kaggle/input/soil-classification-part-2/soil_competition-2025/train_labels.csv

Test IDs: /kaggle/input/soil-classification-part-2/soil_competition-2025/test_ids.csv

Requirements
Python 3.11+

PyTorch

torchvision

pandas

numpy

scikit-learn

PIL (Pillow)

tqdm

matplotlib

Setup
Install dependencies using pip:

bash
pip install torch torchvision pandas numpy scikit-learn pillow tqdm matplotlib
Usage
1. Data Preparation
Ensure the train and test image directories and CSV files are in the correct paths as specified above.

2. Data Loading and Preprocessing
The code loads labels and splits the training data into training and validation sets (80/20 split, stratified by label).

Images are loaded using a custom PyTorch Dataset class.

Data augmentations (resize, random flips, rotations, color jitter) are applied to training images; validation/test images are only resized and normalized.

3. Model Architecture
Pretrained EfficientNet-B0 from torchvision is used.

The classifier head is replaced for binary classification (output size = 1).

Model is moved to GPU if available.

4. Training
Binary cross-entropy with logits loss (nn.BCEWithLogitsLoss) is used.

Adam optimizer with a learning rate of 1e-4.

The model is trained for 5 epochs.

After each epoch, F1 score is computed on the validation set.

5. Inference and Submission
The trained model predicts labels for the test set.

Results are saved in submission.csv with columns: image_id, label.

6. Example Training Output
text
Epoch 1/5, Loss: 13.2084, Val F1: 0.9959
Epoch 2/5, Loss: 4.5811, Val F1: 1.0000
Epoch 3/5, Loss: 1.3426, Val F1: 1.0000
Epoch 4/5, Loss: 0.5913, Val F1: 1.0000
Epoch 5/5, Loss: 0.3838, Val F1: 1.0000
✅ submission.csv saved.
File Structure
notebook.ipynb — Main code (Jupyter Notebook)

submission.csv — Output predictions for test data

Customization
Adjust batch_size, image_size, or number of epochs as needed.

You can swap EfficientNet-B0 for another torchvision model if desired.

Data augmentation strategies can be modified in the train_transform.

Notes
The notebook is designed for use in a Kaggle environment with GPU acceleration.

All labels are binary (0 or 1).

The code is modular and can be adapted for other binary image classification tasks.

Citation
If you use this code, please cite the original notebook and dataset authors.

For questions or issues, please open an issue or contact the maintainer.
