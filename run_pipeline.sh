#!/bin/bash
# run_pipeline.sh - Complete execution pipeline for ASL Recognition System

echo "========================================================================"
echo "ASL Alphabet Recognition - Complete Pipeline Execution"
echo "========================================================================"
echo ""

# Set error handling
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python installation
print_status "Checking Python installation..."
if ! command -v python &> /dev/null; then
    print_error "Python is not installed. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_warning "Virtual environment not detected. It's recommended to use a virtual environment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if data exists
echo ""
echo "========================================================================"
echo "Step 1: Checking Dataset"
echo "========================================================================"

if [ ! -d "data/raw/asl_alphabet_train" ]; then
    print_error "Dataset not found at data/raw/asl_alphabet_train/"
    echo ""
    echo "Please download the dataset using one of these methods:"
    echo ""
    echo "Option 1 - Kaggle API (Recommended):"
    echo "  1. Install: pip install kaggle"
    echo "  2. Setup credentials: place kaggle.json in ~/.kaggle/"
    echo "  3. Download: kaggle datasets download -d grassknoted/asl-alphabet"
    echo "  4. Extract: unzip asl-alphabet.zip -d data/raw/"
    echo ""
    echo "Option 2 - Manual Download:"
    echo "  1. Visit: https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
    echo "  2. Download and extract to data/raw/asl_alphabet_train/"
    echo ""
    exit 1
else
    print_status "Dataset found!"
    NUM_CLASSES=$(ls -d data/raw/asl_alphabet_train/*/ | wc -l)
    print_status "Number of class directories: $NUM_CLASSES"
fi

# Step 2: Data Preprocessing
echo ""
echo "========================================================================"
echo "Step 2: Data Preprocessing"
echo "========================================================================"
print_status "Running data preprocessing..."
python src/data_preprocessing.py

if [ $? -eq 0 ]; then
    print_status "Data preprocessing completed successfully!"
else
    print_error "Data preprocessing failed!"
    exit 1
fi

# Step 3: Model Training
echo ""
echo "========================================================================"
echo "Step 3: Model Training"
echo "========================================================================"
print_warning "This may take 30-60 minutes on CPU..."
print_status "Starting training pipeline..."
python src/train.py

if [ $? -eq 0 ]; then
    print_status "Training completed successfully!"
else
    print_error "Training failed!"
    exit 1
fi

# Step 4: Model Evaluation
echo ""
echo "========================================================================"
echo "Step 4: Model Evaluation"
echo "========================================================================"
print_status "Evaluating model on test set..."
python src/evaluate.py

if [ $? -eq 0 ]; then
    print_status "Evaluation completed successfully!"
else
    print_error "Evaluation failed!"
    exit 1
fi

# Summary
echo ""
echo "========================================================================"
echo "Pipeline Execution Complete! 🎉"
echo "========================================================================"
echo ""
print_status "All steps completed successfully!"
echo ""
echo "Results saved to:"
echo "  - Model: models/saved_models/best_model.h5"
echo "  - Training history: results/training_history.png"
echo "  - Confusion matrix: results/confusion_matrix.png"
echo "  - Classification report: results/classification_report.txt"
echo ""
echo "Next steps:"
echo "  1. View results: ls -lh results/"
echo "  2. Run Streamlit app: streamlit run app/streamlit_app.py"
echo "  3. Run webcam demo: python app/webcam_demo.py"
echo ""
echo "========================================================================"