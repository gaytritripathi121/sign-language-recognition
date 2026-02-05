
import os
import json
import shutil
import random
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import *


class ASLDataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.data_info = {}
        
    def download_dataset(self):
        """Download dataset from Kaggle"""
        print("Downloading dataset from Kaggle...")
        print("Please ensure you have kaggle.json in ~/.kaggle/")
        print(f"Run: kaggle datasets download -d {KAGGLE_DATASET}")
        print(f"Then extract to: {RAW_DATA_DIR}")
        
    def analyze_dataset(self) -> Dict:
        """Analyze the raw dataset structure"""
        print("\n" + "="*60)
        print("Analyzing Dataset Structure")
        print("="*60)
        
        if not RAW_DATA_DIR.exists():
            raise FileNotFoundError(f"Dataset not found at {RAW_DATA_DIR}")
        
        class_distribution = {}
        total_images = 0
        
        for class_name in CLASS_NAMES:
            class_dir = RAW_DATA_DIR / class_name
            if class_dir.exists():
                num_images = len(list(class_dir.glob('*.jpg'))) + \
                             len(list(class_dir.glob('*.png')))
                class_distribution[class_name] = num_images
                total_images += num_images
            else:
                print(f"Warning: Class directory '{class_name}' not found")
                class_distribution[class_name] = 0
        
        self.data_info = {
            'total_images': total_images,
            'num_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES,
            'class_distribution': class_distribution,
            'image_size': IMG_SIZE
        }
        
        print(f"\nTotal Images: {total_images}")
        print(f"Number of Classes: {len(CLASS_NAMES)}")
        print("\nClass Distribution:")
        for cls, count in sorted(class_distribution.items()):
            print(f"  {cls:10s}: {count:5d} images")
        
        with open(DATA_DIR / 'dataset_info.json', 'w') as f:
            json.dump(self.data_info, f, indent=4)
        
        return self.data_info
    
    def create_train_val_test_split(self):
        """Split dataset into train, validation, and test sets"""
        print("\n" + "="*60)
        print("Creating Train/Val/Test Split")
        print("="*60)
        
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        for split in ['train', 'val', 'test']:
            split_dir = PROCESSED_DATA_DIR / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
        
        total_processed = 0
        
        for class_name in tqdm(CLASS_NAMES, desc="Processing classes"):
            class_dir = RAW_DATA_DIR / class_name
            
            if not class_dir.exists():
                print(f"Skipping {class_name} - directory not found")
                continue
            
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png'))
            
            if len(image_files) == 0:
                print(f"Warning: No images found for class {class_name}")
                continue
            
            random.shuffle(image_files)
            
            n_total = len(image_files)
            n_train = int(n_total * TRAIN_SPLIT)
            n_val = int(n_total * VAL_SPLIT)
            
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            for split, files in [('train', train_files), 
                                ('val', val_files), 
                                ('test', test_files)]:
                split_class_dir = PROCESSED_DATA_DIR / split / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img_file in files:
                    shutil.copy2(img_file, split_class_dir / img_file.name)
                    total_processed += 1
        
        print(f"\nTotal images processed: {total_processed}")
        print(f"Train images: {len(list((PROCESSED_DATA_DIR / 'train').rglob('*.jpg')))}")
        print(f"Val images: {len(list((PROCESSED_DATA_DIR / 'val').rglob('*.jpg')))}")
        print(f"Test images: {len(list((PROCESSED_DATA_DIR / 'test').rglob('*.jpg')))}")
        print(f"\nData split completed! Files saved to: {PROCESSED_DATA_DIR}")
    
    def create_data_generators(self) -> Tuple:
        """Create data generators with augmentation"""
        print("\n" + "="*60)
        print("Creating Data Generators")
        print("="*60)
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=ROTATION_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            shear_range=SHEAR_RANGE,
            zoom_range=ZOOM_RANGE,
            horizontal_flip=HORIZONTAL_FLIP,
            brightness_range=BRIGHTNESS_RANGE,
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            PROCESSED_DATA_DIR / 'train',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=RANDOM_SEED
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            PROCESSED_DATA_DIR / 'val',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            PROCESSED_DATA_DIR / 'test',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\nTrain batches: {len(train_generator)}")
        print(f"Validation batches: {len(val_generator)}")
        print(f"Test batches: {len(test_generator)}")
        print(f"Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for inference"""
        img = Image.open(image_path)
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


def main():
    """Main preprocessing pipeline"""
    preprocessor = ASLDataPreprocessor()
    
    preprocessor.analyze_dataset()
    preprocessor.create_train_val_test_split()
    train_gen, val_gen, test_gen = preprocessor.create_data_generators()
    
    print("\n" + "="*60)
    print("Data Preprocessing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
