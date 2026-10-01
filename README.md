# CIFAR-10: Classical Machine Vision vs Deep Learning

A comparative image-classification project investigating **classical machine-vision techniques and modern deep learning** using the CIFAR-10 dataset.

The project was originally developed as part of my university Machine Vision coursework and implements two fundamentally different approaches to image classification:

- **Classical Machine Vision:** SIFT feature extraction → K-Means visual vocabulary → Bag-of-Visual-Words representation → Support Vector Machine classification
- **Deep Learning:** ResNet-18 Convolutional Neural Network using PyTorch

The project also investigates the effect of **visual-vocabulary size** on SVM performance and explores how **data augmentation** affects both classification approaches.

The experiments produced a clear performance difference between the two approaches, with the ResNet-18 model achieving approximately **80.85% classification accuracy**, compared with approximately **24.78% for the strongest SVM configuration** in the recorded experiments.

---

## Project Overview

The project follows two parallel image-classification pipelines:

```text
                         CIFAR-10
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
       CLASSICAL PIPELINE        DEEP LEARNING
                │                       │
           Grayscale                 Images
                │                       │
                ▼                       ▼
              SIFT                  ResNet-18
                │                       │
                ▼                       │
        Feature Descriptors             │
                │                       │
                ▼                       │
             K-Means                    │
                │                       │
                ▼                       │
       Visual Vocabulary                │
                │                       │
                ▼                       │
      Bag-of-Visual-Words               │
                │                       │
                ▼                       │
               SVM                      │
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
                 Performance Evaluation
                            │
                            ▼
          Accuracy / Precision / Recall / F1
```

This allowed the performance of a hand-engineered feature pipeline to be compared directly with a learned convolutional representation.

---

## Technologies

- Python
- OpenCV
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- SIFT
- K-Means clustering
- Bag-of-Visual-Words
- Support Vector Machines
- ResNet-18
- Convolutional Neural Networks
- CUDA / GPU acceleration
- Data augmentation
- Image classification

---

## Dataset

The project uses the **CIFAR-10** image-classification dataset.

CIFAR-10 contains images belonging to ten classes:

```text
0  Airplane
1  Automobile
2  Bird
3  Cat
4  Deer
5  Dog
6  Frog
7  Horse
8  Ship
9  Truck
```

Each image has dimensions:

```text
32 × 32 × 3
```

representing a 32×32 RGB colour image.

The dataset is divided into training and test sets and is used throughout the project for both the classical and deep-learning experiments.

---

## Repository Structure

```text
cifar10-classical-vs-deep-learning/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── results/
│   ├── sift_keypoints.png
│   ├── bovw_histograms.png
│   ├── svm_vs_resnet18.png
│   └── resnet18_augmentation_comparison.png
│
└── src/
    ├── data/
    │   ├── explore_cifar10_dataset.py
    │   └── prepare_grayscale_dataset.py
    │
    ├── features/
    │   ├── visualise_sift_keypoints.py
    │   └── inspect_sift_descriptors.py
    │
    ├── bovw/
    │   ├── generate_bovw_codebook.py
    │   ├── compare_codebook_sizes.py
    │   └── generate_bovw_histograms.py
    │
    ├── classification/
    │   ├── prepare_dataset_splits.py
    │   ├── train_svm_classifier.py
    │   ├── train_resnet18_classifier.py
    │   └── compare_svm_and_cnn.py
    │
    └── augmentation/
        ├── augment_cifar10_dataset.py
        ├── train_augmented_svm.py
        ├── compare_svm_augmentation.py
        ├── train_augmented_resnet18.py
        └── compare_cnn_augmentation.py
```

The original coursework scripts were named according to assignment task numbers. For portfolio presentation, they have been reorganised and renamed according to their actual functionality.

---

# 1. Dataset Exploration & Preparation

The first stage loads and examines the CIFAR-10 dataset.

The initial workflow is:

```text
CIFAR-10
    │
    ▼
Load Dataset
    │
    ▼
Inspect Shape
    │
    ▼
Reshape Images
    │
    ▼
Normalise Pixel Values
    │
    ▼
Visualise Samples
```

The project then creates grayscale versions of the training and test datasets for use by the classical machine-vision pipeline.

Conceptually:

```text
RGB CIFAR-10
      │
      ▼
OpenCV Colour Conversion
      │
      ▼
Grayscale Dataset
      │
      ├── Training Images
      │
      └── Test Images
```

This provides the input representation used for SIFT feature extraction.

---

# 2. SIFT Feature Extraction

The classical pipeline begins with **Scale-Invariant Feature Transform (SIFT)**.

Rather than feeding raw image pixels directly into a classifier, SIFT identifies distinctive local regions within an image and calculates descriptors representing those regions.

```text
Grayscale Image
      │
      ▼
     SIFT
      │
      ├── Keypoints
      │
      └── Descriptors
```

A single image can contain multiple keypoints, with each keypoint represented by a local feature descriptor.

---

## SIFT Keypoint Visualisation

`visualise_sift_keypoints.py` demonstrates the detected SIFT features across examples from the CIFAR-10 classes.

The detected features can be visualised directly over the source images.

```text
Input Image
     │
     ▼
SIFT Detection
     │
     ▼
Keypoints
     │
     ▼
Visualisation
```

Example output:

![SIFT keypoints](results/sift_keypoints.png)

This provides a visual representation of the local image structures being used by the classical classification pipeline.

---

# 3. Bag-of-Visual-Words

Individual images contain varying numbers of SIFT descriptors.

A machine-learning classifier, however, requires a consistent feature representation.

The project therefore uses a **Bag-of-Visual-Words (BoVW)** approach.

```text
Images
   │
   ▼
SIFT Descriptors
   │
   ▼
Descriptor Collection
   │
   ▼
K-Means Clustering
   │
   ▼
Visual Vocabulary
   │
   ▼
Assign Descriptors
to Visual Words
   │
   ▼
Histogram
   │
   ▼
Fixed-Length
Image Representation
```

The approach is conceptually similar to a bag-of-words representation used in text processing, except that local image descriptors are grouped into **visual words**.

---

# 4. Visual Vocabulary Generation

K-Means clustering is applied to the collected SIFT descriptors.

Each cluster centre becomes a visual word.

```text
Thousands of SIFT Descriptors
             │
             ▼
          K-Means
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
      C1    C2    ... CK
       │     │       │
       └─────┴───────┘
             │
             ▼
       Visual Vocabulary
```

The project investigates three vocabulary sizes:

```text
K = 50
K = 100
K = 200
```

This allows the effect of vocabulary size on classification performance to be evaluated.

---

# 5. Bag-of-Visual-Words Histograms

Once the vocabulary has been generated, the SIFT descriptors from each image are assigned to their nearest visual word.

The number of descriptors assigned to each word forms a histogram.

For a vocabulary containing `K` words:

```text
Image
  │
  ▼
SIFT
  │
  ▼
Descriptors
  │
  ▼
Nearest Visual Word
  │
  ▼
┌─────────────────────────────┐
│ 3 │ 0 │ 7 │ 2 │ 1 │ ...   │
└─────────────────────────────┘
             │
             ▼
       BoVW Histogram
```

The resulting histogram has a fixed dimensionality and can therefore be used as the feature vector for an SVM classifier.

Example histograms generated using different vocabulary sizes:

![Bag-of-Visual-Words histograms](results/bovw_histograms.png)

---

# 6. Support Vector Machine Classification

The BoVW histograms are used to train a **Support Vector Machine (SVM)** classifier.

```text
Training Images
      │
      ▼
SIFT
      │
      ▼
BoVW Histograms
      │
      ▼
     SVM
      │
      ▼
CIFAR-10 Class
```

The experiments compare SVM classifiers generated using:

```text
K = 50
K = 100
K = 200
```

visual words.

This allows the relationship between vocabulary size and classification performance to be investigated.

---

# 7. ResNet-18 Deep Learning Pipeline

The second classification approach replaces the hand-engineered SIFT/BoVW pipeline with a convolutional neural network.

The project uses **ResNet-18** through PyTorch.

```text
CIFAR-10 Image
      │
      ▼
   ResNet-18
      │
      ▼
Learned Features
      │
      ▼
Classification Layer
      │
      ▼
10 CIFAR Classes
```

Unlike the classical pipeline, the CNN learns useful visual features directly from the training images.

This removes the need to manually construct:

```text
SIFT
  │
  ▼
K-Means
  │
  ▼
BoVW
```

representations.

---

## GPU Support

The training implementation checks whether CUDA is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The model and training data are then moved to the selected device.

Conceptually:

```text
Start Training
      │
      ▼
CUDA Available?
   /        \
 Yes         No
  │           │
  ▼           ▼
 GPU          CPU
  │           │
  └─────┬─────┘
        │
        ▼
 ResNet Training
```

This allows the training process to take advantage of GPU acceleration when compatible hardware is available.

---

# 8. Model Evaluation

The classifiers are evaluated using four metrics:

```text
Accuracy
Precision
Recall
F1 Score
```

This provides a broader evaluation than classification accuracy alone.

---

## Recorded Classification Results

The original experiments produced the following results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| SVM — K=50 | 22.09% | 21.84% | 22.09% | 21.67% |
| SVM — K=100 | **24.78%** | **24.50%** | **24.78%** | **24.24%** |
| SVM — K=200 | 24.00% | 23.96% | 24.00% | 23.64% |
| ResNet-18 | **80.85%** | **80.90%** | **80.85%** | **80.82%** |

The best-performing classical configuration in these experiments was therefore:

```text
SIFT
  │
  ▼
K-Means
K = 100
  │
  ▼
BoVW
  │
  ▼
SVM

Accuracy = 24.78%
```

while the ResNet-18 implementation achieved:

```text
Accuracy = 80.85%
```

---

## Classical vs Deep Learning Comparison

![SVM vs ResNet-18 performance](results/svm_vs_resnet18.png)

The experiment produced a substantial difference between the two approaches.

```text
Classification Accuracy

SVM K=50       22.09%
███████████

SVM K=100      24.78%
████████████

SVM K=200      24.00%
████████████

ResNet-18      80.85%
████████████████████████████████████████
```

Within this experimental setup, the learned CNN representation was considerably more effective for CIFAR-10 classification than the classical SIFT/BoVW representation.

---

# Why Did the CNN Perform Better?

The result highlights an important difference between classical and deep-learning approaches.

## Classical Pipeline

The classical model relies on a predefined feature extractor:

```text
Image
  │
  ▼
SIFT
  │
  ▼
Local Features
```

SIFT was originally designed to identify distinctive local structures.

CIFAR-10 images, however, are extremely small:

```text
32 × 32 pixels
```

which limits the amount of local structure available for feature extraction.

The BoVW representation also discards much of the spatial relationship between detected features.

---

## CNN Pipeline

The CNN instead learns its own feature hierarchy from the training data:

```text
Pixels
  │
  ▼
Low-Level Features
  │
  ▼
Edges / Textures
  │
  ▼
Intermediate Features
  │
  ▼
Higher-Level Representations
  │
  ▼
Object Class
```

This learned representation proved substantially more effective for this particular dataset and experiment.

---

# 9. Data Augmentation

The final section investigates whether expanding the training data through augmentation improves classification performance.

Augmentation creates modified versions of existing training images.

Conceptually:

```text
                 Original Image
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Flip         Rotation      Modified
                                  Appearance
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                Augmented Dataset
```

The objective is to expose the classifier to greater variation during training.

---

# SVM Augmentation Results

The augmentation experiment did **not** improve the SVM results.

Recorded results included:

| Model | Original Accuracy | Augmented Accuracy |
|---|---:|---:|
| SVM — K=100 | 24.78% | 22.38% |
| SVM — K=200 | 24.00% | 23.18% |

This is an important experimental result: augmentation does not automatically improve classification performance.

Its usefulness depends on factors including:

- The augmentation transformations
- The feature representation
- The dataset
- The classifier
- The amount of training data
- Hyperparameter selection

---

# ResNet-18 Augmentation Results

The same investigation was performed with the CNN.

The recorded results were:

| Metric | Original ResNet-18 | Augmented ResNet-18 |
|---|---:|---:|
| Accuracy | **80.85%** | 73.35% |
| Precision | **80.90%** | 74.01% |
| Recall | **80.85%** | 73.35% |
| F1 Score | **80.82%** | 72.48% |

The augmentation experiment therefore reduced performance in this implementation.

![ResNet-18 augmentation comparison](results/resnet18_augmentation_comparison.png)

---

# Interpreting the Augmentation Experiment

The result is useful because it demonstrates that:

```text
More Training Variation
        ≠
Guaranteed Better Model
```

Data augmentation introduces assumptions about which transformations should preserve an image's semantic class.

The effectiveness of those transformations therefore needs to be evaluated experimentally rather than assumed.

In this project, the selected augmentation pipeline and training configuration did not improve generalisation compared with the original ResNet-18 experiment.

A further investigation could examine whether the reduction resulted from:

- Augmentation strength
- Selected transformations
- Training duration
- Learning rate
- Optimisation configuration
- Dataset preprocessing
- Model convergence
- Hyperparameter selection

The recorded results alone do not establish which of these factors was responsible, so further controlled experiments would be required.

---

# Classical Machine Vision vs Deep Learning

The project demonstrates two very different approaches to computer vision.

## Classical Machine Vision

```text
Image
  │
  ▼
Engineer Feature Extraction
  │
  ▼
SIFT
  │
  ▼
Engineer Representation
  │
  ▼
BoVW
  │
  ▼
Train Classifier
  │
  ▼
SVM
```

The feature representation is largely designed before classifier training.

---

## Deep Learning

```text
Image
  │
  ▼
Neural Network
  │
  ▼
Learn Features
  │
  ▼
Learn Representation
  │
  ▼
Learn Classification
  │
  ▼
Prediction
```

The feature extraction and classification stages are learned together.

---

# Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

```text
numpy
matplotlib
opencv-python
scikit-learn
torch
torchvision
```

---

# Running the Project

The repository contains individual scripts corresponding to different stages of the experiments rather than a single application entry point.

For example:

```bash
python src/data/explore_cifar10_dataset.py
```

can be used for initial dataset exploration.

SIFT feature visualisation can be run using:

```bash
python src/features/visualise_sift_keypoints.py
```

and the ResNet implementation is contained in:

```bash
python src/classification/train_resnet18_classifier.py
```

Some later stages depend on intermediate datasets, descriptors or models generated by earlier stages of the experimental pipeline.

---

# Concepts Demonstrated

This project provided practical experience with:

### Computer Vision

- Image classification
- Image preprocessing
- Grayscale conversion
- SIFT
- Keypoint detection
- Feature descriptors
- Classical machine vision

### Machine Learning

- K-Means clustering
- MiniBatch K-Means
- Visual vocabularies
- Bag-of-Visual-Words
- Support Vector Machines
- Training/test datasets
- Classification metrics
- Model comparison

### Deep Learning

- PyTorch
- Torchvision
- Convolutional Neural Networks
- ResNet-18
- Training loops
- GPU acceleration
- CUDA-aware execution
- Data augmentation

### Experimental Evaluation

- Accuracy
- Precision
- Recall
- F1 score
- Vocabulary-size comparison
- Classical vs deep-learning comparison
- Before/after augmentation comparison
- Visualisation of experimental results

---

# Original Coursework

This repository originates from university Machine Vision coursework.

The original source was organised using assignment task numbers.

During portfolio preparation:

- Scripts were renamed according to their functionality.
- The repository was reorganised into logical processing stages.
- Experimental results were collected into a dedicated `results` directory.
- A descriptor-extraction loop was corrected so that processing did not terminate after encountering all ten CIFAR-10 classes.
- Documentation was added to explain the complete experimental pipeline.

The core implementations and recorded experimental results remain representative of the original project.

---

# Known Limitations

This was an educational machine-vision experiment rather than a production image-classification system.

Several areas could be improved or investigated further.

## Hyperparameter Optimisation

The models were evaluated using a limited selection of configurations.

A more comprehensive investigation could systematically optimise:

```text
SVM Parameters
      │
      ├── C
      ├── Kernel
      └── Gamma

BoVW Parameters
      │
      └── Vocabulary Size

CNN Parameters
      │
      ├── Learning Rate
      ├── Batch Size
      ├── Optimiser
      ├── Epochs
      └── Weight Decay
```

---

## Reproducibility

A modern implementation would define explicit random seeds for:

- NumPy
- scikit-learn
- PyTorch
- CUDA

where appropriate.

This would make repeated experimental runs easier to compare.

---

## Pipeline Integration

The coursework implementation consists of separate experimental scripts.

A more maintainable version could implement reusable components:

```text
                  Experiment
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
ClassicalClassifier          CNNClassifier
        │                           │
        ▼                           ▼
SIFTExtractor              ResNetTrainer
        │
        ▼
BoVWEncoder
        │
        ▼
SVMClassifier
```

A common evaluation module could then calculate metrics for both approaches.

---

# How I Would Extend the Project

A useful next stage would be to compare additional classical and learned representations.

For example:

```text
                    CIFAR-10
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
    SIFT/BoVW          HOG           ResNet
       │               │               │
       ▼               ▼               ▼
      SVM             SVM             CNN
       │               │               │
       └───────────────┼───────────────┘
                       │
                       ▼
              Common Evaluation
```

Other extensions could include:

- Confusion matrices
- Per-class precision and recall
- Alternative CNN architectures
- Transfer learning
- Learning-rate scheduling
- Early stopping
- Cross-validation for the classical pipeline
- Systematic augmentation ablation studies
- Training-time comparison
- Inference-time comparison

---

# Robotics Context

Image classification is one component of the wider robotic-perception problem.

The techniques explored in this project fit into a broader perception pipeline:

```text
Camera
  │
  ▼
Image Acquisition
  │
  ▼
Preprocessing
  │
  ▼
Feature Extraction
  │
  ├───────────────┐
  ▼               ▼
Classical       Learned
Features        Features
  │               │
  ▼               ▼
Classifier     Neural Network
  │               │
  └───────┬───────┘
          │
          ▼
    Object / Scene
     Understanding
          │
          ▼
   Robot Decision
       Making
```

Understanding both classical and deep-learning approaches provides useful context when selecting perception techniques for robotic systems.

---

# Portfolio Context

This project represents a progression from fundamental image processing into **machine learning and deep-learning-based visual perception**.

```text
Image Processing Fundamentals
            │
            ▼
      Feature Extraction
            │
            ▼
     Classical Vision
      SIFT + BoVW
            │
            ▼
     Machine Learning
           SVM
            │
            ▼
       Deep Learning
        ResNet-18
            │
            ▼
     Visual Perception
            │
            ▼
   Autonomous Systems
```

Rather than demonstrating only the use of a neural network, the project explores the evolution from **hand-engineered visual features to learned representations** and evaluates the two approaches quantitatively.

This makes the project particularly relevant to my wider work in **robotics, computer vision and autonomous systems**.
