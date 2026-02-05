"""
Unit tests for ASL Recognition System
"""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.model import ASLModel
from src.inference import ImprovedASLPredictor



class TestASLModel(unittest.TestCase):
    """Test model building and compilation"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.model_builder = ASLModel()
    
    def test_model_creation(self):
        """Test model can be created"""
        model = self.model_builder.build_model()
        self.assertIsNotNone(model)
        
    def test_model_input_shape(self):
        """Test model has correct input shape"""
        model = self.model_builder.build_model()
        expected_shape = (None, *IMG_SIZE, 3)
        self.assertEqual(model.input_shape, expected_shape)
    
    def test_model_output_shape(self):
        """Test model has correct output shape"""
        model = self.model_builder.build_model()
        expected_shape = (None, NUM_CLASSES)
        self.assertEqual(model.output_shape, expected_shape)
    
    def test_model_compilation(self):
        """Test model can be compiled"""
        model = self.model_builder.build_model()
        self.model_builder.compile_model()
        self.assertIsNotNone(self.model_builder.model.optimizer)


class TestASLPredictor(unittest.TestCase):
    """Test inference functionality"""
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (*IMG_SIZE, 3), dtype=np.uint8)
        predictor = ASLPredictor.__new__(ASLPredictor)
        
        processed = predictor.preprocess_image(dummy_img)
        
        # Check shape
        self.assertEqual(processed.shape, (1, *IMG_SIZE, 3))
        
        # Check normalization
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)


if __name__ == '__main__':
    unittest.main()
