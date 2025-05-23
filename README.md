# Facial Emotion Recognition Using CNNs, Transfer Learning, and Attention Mechanisms

## 📌 Project Overview

This project explores the task of **Facial Emotion Recognition (FER)** using the FER-2013 dataset by implementing and comparing four deep learning architectures. The objective is to evaluate the individual and combined effectiveness of **Convolutional Neural Networks (CNNs)**, **Transfer Learning with ResNet-50**, and **Attention Mechanisms** (CBAM and Spatial Attention).

---

## 📄 Research Paper

[📘 Read: Enhancing Facial Emotion Recognition Using Deep Learning (PDF)](https://github.com/ClutchForce/AI-Research/blob/main/Enhancing%20Facial%20Emotion%20Recognition.pdf)

---

## 🔍 Motivation

Facial expressions are essential non-verbal signals used in human communication. Automatically recognizing these expressions from images is a challenging task due to:

- Low-resolution and noisy images
- Class imbalance (e.g., few 'disgust' samples)
- Subtle inter-class variations (e.g., fear vs. surprise)

---

## 🧠 Model Architectures

| Model ID | Description |
|----------|-------------|
| **Model 1** | Baseline CNN (3 Conv blocks + FC layers) |
| **Model 2** | ResNet-50 pretrained on ImageNet, fine-tuned |
| **Model 3** | Custom CNN + Spatial Attention Module |
| **Model 4** | ResNet-50 + CBAM (Channel + Spatial Attention) |

> All models are trained and evaluated using PyTorch. Training includes grayscale normalization, data augmentation, and consistent hyperparameter settings.

---

## 📂 File Structure

```bash
.
├── model1_2_cnn_and_resnet.ipynb      # Code for Model 1 (CNN) and Model 2 (ResNet-50)
├── model3_cnn_attention.ipynb         # Code for Model 3 (CNN + Spatial Attention)
├── model4_resnet_attention.ipynb      # Code for Model 4 (ResNet-50 + CBAM)
├── 4442_Final_Project.pdf             # Final research report
└── README.md                          # This file
```

---

## 🧪 Dataset: FER-2013

- **Images**: 48×48 grayscale
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Split**:
  - 70% Training
  - 10% Validation
  - 20% Testing

---

## 🛠️ Preprocessing Steps

- **Grayscale to RGB (3 Channels)** for pretrained models
- Resizing to:
  - `128x128` for custom CNNs
  - `224x224` for ResNet-50
- **Normalization**: Pixel values scaled to `[0, 1]`
- **Data Augmentation**: Random flips, rotations, brightness shifts

---

## ⚙️ Training Configuration

- **Framework**: PyTorch
- **Optimizer**: Adam (`lr=0.001`)
- **Scheduler**: Cosine Annealing with Warm Restarts
- **Batch Size**: 32
- **Loss**: CrossEntropyLoss
- **Hardware**: NVIDIA GTX 1060 GPU

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Learning Curves (Accuracy/Loss)**
- **Qualitative Image Predictions**

---

## 🧪 Results Summary

| Model              | Accuracy |
|-------------------|----------|
| Baseline CNN       | 58.9%    |
| ResNet-50 (5 epochs) | 43.5%  |
| CNN + Attention     | 45.4%    |
| ResNet-50 + CBAM (5 epochs) | 66.2% |
| ResNet-50 + CBAM (20 epochs) | **67.4%** |

> **Key Insight**: Model 4 (ResNet-50 + CBAM) achieved high performance early in training—gaining only ~1.2% improvement between 5 and 20 epochs—showing the power of attention-guided transfer learning.

---

## 🤖 Visual Examples

Confusion matrices and training plots for each model are included in the final report (`4442_Final_Project.pdf`).



---

## 🧰 System & Python Requirements

To successfully run the notebooks and train the models, the following system and Python environment setup is recommended:

### 🖥️ Hardware
- **GPU:** NVIDIA GPU with CUDA Compute Capability ≥ 6.1 (e.g., RTX 3060)
- **VRAM:** ≥ 8GB recommended
- **RAM:** ≥ 16GB system memory
- **Disk Space:** At least 2GB free for storing datasets, logs, and plots

### 🧪 Python Environment
- **Python Version:** 3.9 or 3.10 (tested on 3.10)

### 📦 Required Packages
Install via `pip install -r requirements.txt` or manually:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib numpy pandas scikit-learn opencv-python notebook
```

### 🔧 Optional (For GPU Support)
- CUDA Toolkit: **v11.2**
- cuDNN: **v8.1** for CUDA 11.2
- NVIDIA Driver: **≥ v460**

> ✅ Make sure `torch.cuda.is_available()` returns `True` if using GPU acceleration.

--- 


## 💡 Conclusion

- Transfer learning boosts feature extraction for FER.
- Attention mechanisms (especially CBAM) improve focus on facial regions and enhance generalization.
- Combining both leads to faster convergence and state-of-the-art results on FER-2013.
- Future work could explore:
  - Vision Transformers
  - Cross-dataset generalization
  - Self-supervised FER training
  - Mobile deployment

---

## 📚 References

- FER-2013 Dataset (Kaggle)
- ResNet-50 [He et al., 2016]
- CBAM Attention [Woo et al., 2018]
- Hu et al., 2018 (SE-Net)
- PyTorch and torchvision documentation

---

## ✍️ Authors

- Yoosuf Bakhtair – ybakhtai@uwo.ca  
- Hassan Amin – habid22@uwo.ca  
- Paul Gherghel – pgherghe@uwo.ca

University of Western Ontario, Canada


