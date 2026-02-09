# al3oda

## CNN Architectures Implementation

This repository contains implementations of two landmark Convolutional Neural Network (CNN) architectures: **LeNet** and **AlexNet**. These architectures represent important milestones in the development of deep learning for computer vision.

---

## 1. LeNet Architecture

### Overview
LeNet-5, developed by Yann LeCun et al. in 1998, is one of the earliest convolutional neural networks. It was originally designed for handwritten digit recognition and became a pioneering architecture in the field of computer vision.

### Architecture Details
The LeNet-5 CNN architecture consists of **7 layers**:
- **3 Convolutional layers** - Extract features from input images
- **2 Subsampling (Pooling) layers** - Reduce spatial dimensions
- **2 Fully connected layers** - Perform classification

### Layer-by-Layer Breakdown
1. **Input Layer**: 28×28×1 grayscale images (MNIST dataset)
2. **Conv2D Layer 1**: 6 filters, 5×5 kernel, tanh activation
3. **Average Pooling Layer 1**: 2×2 pool size, stride of 2
4. **Conv2D Layer 2**: 16 filters, 5×5 kernel, tanh activation
5. **Average Pooling Layer 2**: 2×2 pool size, stride of 2
6. **Flatten Layer**: Converts 2D feature maps to 1D vector
7. **Dense Layer 1**: 120 units, tanh activation
8. **Dense Layer 2**: 84 units, tanh activation
9. **Output Layer**: 10 units (digits 0-9), softmax activation

### Key Characteristics
- **Dataset**: MNIST (handwritten digits)
- **Activation Function**: Tanh (hyperbolic tangent)
- **Pooling**: Average Pooling
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Total Parameters**: Relatively small, suitable for simpler tasks

### Implementation Highlights
```
- Normalization: Images scaled to [0, 1] range
- Input shape: 28×28×1 (grayscale)
- Training: 10 epochs, batch size 128, 10% validation split
```

---

## 2. AlexNet Architecture

### Overview
AlexNet, developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012, revolutionized computer vision by winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with a significant margin. It demonstrated the power of deep CNNs and popularized the use of GPUs for training neural networks.

### Architecture Details
The AlexNet architecture is significantly deeper and more complex than LeNet, consisting of:
- **5 Convolutional layers** - Deep feature extraction
- **3 Max Pooling layers** - Spatial dimension reduction
- **3 Fully connected layers** - High-level reasoning and classification
- **Dropout layers** - Regularization to prevent overfitting
- **Batch Normalization** - Stabilize training and reduce internal covariate shift

### Layer-by-Layer Breakdown

**Convolutional Layers:**
1. **Conv2D Layer 1**: 96 filters, 11×11 kernel, stride 4, ReLU activation
   - Batch Normalization
   - MaxPooling2D: 2×2, stride 2
   
2. **Conv2D Layer 2**: 256 filters, 5×5 kernel, stride 1, ReLU activation
   - Batch Normalization
   - MaxPooling2D: 2×2, stride 2

3. **Conv2D Layer 3**: 384 filters, 3×3 kernel, stride 1, ReLU activation
   - Batch Normalization

4. **Conv2D Layer 4**: 384 filters, 3×3 kernel, stride 1, ReLU activation
   - Batch Normalization

5. **Conv2D Layer 5**: 256 filters, 3×3 kernel, stride 1, ReLU activation
   - Batch Normalization
   - MaxPooling2D: 2×2, stride 2

**Fully Connected Layers:**
6. **Flatten Layer**: Converts feature maps to 1D vector

7. **Dense Layer 1**: 4096 units, ReLU activation
   - Batch Normalization
   - Dropout (0.4)

8. **Dense Layer 2**: 4096 units, ReLU activation
   - Batch Normalization
   - Dropout (0.4)

9. **Dense Layer 3**: 1000 units, ReLU activation
   - Batch Normalization
   - Dropout (0.4)

10. **Output Layer**: 10 units (CIFAR-10 classes), softmax activation
    - Batch Normalization

### Key Characteristics
- **Dataset**: CIFAR-10 (10 classes of objects)
- **Input Shape**: 32×32×3 (RGB color images)
- **Activation Function**: ReLU (Rectified Linear Unit)
- **Pooling**: Max Pooling
- **Regularization**: 
  - Dropout (40% drop rate)
  - Batch Normalization (addresses internal covariate shift)
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Data Augmentation**: Rotation, horizontal flip, zoom
- **Learning Rate Schedule**: ReduceLROnPlateau callback
- **Total Parameters**: Millions of parameters (much larger than LeNet)

### Implementation Highlights
```
- Data split: 70% training, 30% validation
- Batch size: 100
- Epochs: 5 (can be increased for better accuracy)
- Data augmentation: rotation_range=2, horizontal_flip=True, zoom_range=0.1
- Learning rate reduction: factor=0.1, patience=3
```

### Batch Normalization Importance
Batch Normalization in AlexNet helps solve the **internal covariate shift** problem:
- Each layer produces outputs at different scales (some large, some small)
- Batch Normalization calculates mean and standard deviation of layer outputs
- Normalizes the data to a consistent scale
- **Benefits**: 
  - Minimizes overfitting
  - Increases training speed
  - Allows higher learning rates
  - Reduces sensitivity to initialization

---

## Comparison: LeNet vs AlexNet

| Feature | LeNet | AlexNet |
|---------|-------|---------|
| **Year** | 1998 | 2012 |
| **Depth** | 7 layers | 8 main layers + normalization |
| **Activation** | Tanh | ReLU |
| **Pooling** | Average Pooling | Max Pooling |
| **Regularization** | None | Dropout + Batch Normalization |
| **Dataset** | MNIST (grayscale) | CIFAR-10 (color) |
| **Input Size** | 28×28×1 | 32×32×3 |
| **Parameters** | ~60K | Millions |
| **Use Case** | Simple digit recognition | Complex object classification |
| **Data Augmentation** | No | Yes |
| **Training Complexity** | Low | High |

---

## Repository Structure

```
al3oda/
├── README.md
├── LeNet/
│   ├── LeNet.ipynb       # LeNet implementation
│   └── LeNet.png         # Architecture diagram
└── alexnet/
    └── alexnet.ipynb     # AlexNet implementation
```

---

## Key Takeaways

1. **LeNet** is a foundational architecture, simple yet effective for basic image recognition tasks
2. **AlexNet** demonstrated the power of deep learning and GPUs, ushering in the modern deep learning era
3. Both architectures follow the pattern: **Convolution → Activation → Pooling → Fully Connected**
4. AlexNet's innovations (ReLU, Dropout, Data Augmentation, Batch Normalization) became standard practices
5. The evolution from LeNet to AlexNet shows the progression toward deeper, more complex networks

---

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- scikit-learn
- Jupyter Notebook

---

## Usage

1. Clone the repository
2. Open the desired notebook (LeNet or AlexNet)
3. Run the cells sequentially to train and evaluate the models

---

## References

- LeNet-5: LeCun, Y., et al. "Gradient-based learning applied to document recognition." (1998)
- AlexNet: Krizhevsky, A., Sutskever, I., & Hinton, G. E. "ImageNet classification with deep convolutional neural networks." (2012)
