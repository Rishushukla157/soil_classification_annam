**Soil Classification Challenge - IIT Ropar Hackathon**
A comprehensive deep learning solution for binary soil image classification using EfficientNet architectures with advanced ensemble techniques and test-time augmentation.

🏆 Competition Overview
This project was developed for the Soil Image Classification Challenge organized by Annam.ai at IIT Ropar as part of their hackathon selection process. The task involves binary classification to determine whether images contain soil or not.

Competition Details:

Organizer: Annam.ai at IIT Ropar

Task: Binary classification (soil vs non-soil images)

Deadline: May 25, 2025, 11:59 PM IST

Purpose: Initial screening task for hackathon participants

**🚀 Key Features**
Dual Architecture Approach: Both EfficientNet-B3 (multi-class adapted) and EfficientNet-B0 (binary optimized)

Robust Cross-Validation: 5-fold stratified cross-validation for reliable model training

Test Time Augmentation (TTA): Multiple transformations during inference for enhanced accuracy

Ensemble Learning: Prediction averaging across multiple trained models

GPU Optimization: Efficient CUDA utilization with automatic device detection
 **📊 Technical Architecture
Solution 1: EfficientNet-B3 with 5-Fold CV + TTA**


# Model Configuration
Base Model: EfficientNet-B3 (ImageNet pre-trained)
Input Resolution: 224×224 pixels
Training Strategy: 5-fold stratified cross-validation
Epochs per fold: 10
Optimizer: Adam (lr=1e-4)
Loss Function: CrossEntropyLoss


**🚨 Important Notes**
Competition Deadline: May 25, 2025, 11:59 PM IST

Early Submission Recommended: Avoid last-minute server overload

Binary Classification: Task is to classify soil vs non-soil images

GPU Recommended: For efficient training and inference

**🤝 Contributing**
This project was developed for the IIT Ropar hackathon selection process. Feel free to fork and adapt for similar classification tasks.

**📄 License**
This project is open source and available under the MIT License.

**🙏 Acknowledgments**
Annam.ai and IIT Ropar for organizing the competition

EfficientNet authors for the powerful architecture

PyTorch and torchvision teams for excellent deep learning frameworks


