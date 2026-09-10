# ASL Alphabet Classifier

An end-to-end **American Sign Language (ASL) Alphabet Classification** project built using **Python, TensorFlow/Keras, and MobileNetV2 transfer learning**. The system recognizes **29 classes**, including A–Z, `space`, `delete`, and `nothing`.

The project covers the complete ML workflow, from dataset preparation and augmentation to model training, evaluation, and deployment.

## Features

* Recognizes 29 ASL alphabet classes
* MobileNetV2 transfer learning
* Data augmentation for improved generalization
* Two-phase training with fine-tuning
* Accuracy and Top-3 accuracy evaluation
* Precision, recall, F1-score, and confusion matrix
* Misclassification analysis
* Image upload and webcam-based prediction
* Streamlit interface and JavaScript webcam client
* Top-5 predictions with confidence scores
* Flask-based prediction API

## Tech Stack

**Python | TensorFlow/Keras | MobileNetV2 | OpenCV | NumPy | Pandas | Matplotlib | Streamlit | Flask | JavaScript**

## Dataset

The project uses the **ASL Alphabet Dataset** from Kaggle.

* Approximately **87,000 images**
* **29 classes**
* Around **3,000 images per class**
* RGB images

The dataset is divided into:

```text
70% → Training
15% → Validation
15% → Testing
```

The split is performed per class to maintain class balance.

## Model Architecture

The project uses **MobileNetV2 pretrained on ImageNet** as the feature extractor, followed by a custom classification head.

```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Batch Normalization
     ↓
Dense (512, ReLU)
     ↓
Dropout (0.5)
     ↓
Dense (256, ReLU)
     ↓
Dropout (0.25)
     ↓
Dense (29, Softmax)
```

### Why MobileNetV2?

MobileNetV2 was selected because it is **lightweight and computationally efficient**, making it suitable for webcam-based prediction and systems with limited computational resources.

## Training

Training is performed in two phases:

**Phase 1:** The pretrained MobileNetV2 base is frozen and only the classification head is trained.

**Phase 2:** Later MobileNetV2 layers are unfrozen and fine-tuned using a lower learning rate to adapt the pretrained features to ASL-specific patterns.

Data augmentation includes rotation, shifting, shearing, zooming, horizontal flipping, and brightness variation.

## Evaluation

The model is evaluated using:

* Accuracy
* Top-3 Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Misclassification analysis is used to identify classes that are frequently confused with each other.

## Application

The trained model can be accessed through a **Streamlit application** that supports image-based prediction and provides the top-5 predicted classes with their confidence scores.

A separate **JavaScript webcam client** communicates with a Flask prediction API for webcam-based inference.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ASL-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

### 4. Evaluate the model

```bash
python evaluate.py
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

## Future Improvements

* Add low-confidence prediction handling
* Improve recognition of visually similar signs
* Implement continuous/sequence-based sign recognition
* Optimize real-time webcam inference
* Experiment with alternative lightweight architectures
* Add automated testing and CI/CD

## Author

**Gaytri Tripathi**

Computer Science & Engineering Undergraduate
Interested in **Software Engineering, Machine Learning, and AI**.
