import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

from src.config import *


class ASLPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = MODEL_DIR / 'best_model.h5'

        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image, top_k: int = 3):
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)[0]

        top_indices = np.argsort(predictions)[-top_k:][::-1]

        results = {
            'predictions': [],
            'top_prediction': self.class_names[top_indices[0]],
            'top_confidence': float(predictions[top_indices[0]])
        }

        for idx in top_indices:
            results['predictions'].append({
                'class': self.class_names[idx],
                'confidence': float(predictions[idx])
            })

        return results

    def predict_from_webcam_frame(self, frame):
        height, width = frame.shape[:2]
        roi_size = 300
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2

        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

        display_frame = frame.copy()
        cv2.rectangle(
            display_frame,
            (roi_x, roi_y),
            (roi_x + roi_size, roi_y + roi_size),
            (0, 255, 0),
            3
        )

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.predict(roi_rgb, top_k=3)

        y_offset = 30
        for i, pred in enumerate(results['predictions']):
            text = f"{pred['class']}: {pred['confidence']:.2%}"
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(
                display_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )
            y_offset += 35

        cv2.putText(
            display_frame,
            "Position hand in green box",
            (roi_x, roi_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return results, display_frame

    def predict_from_file(self, image_path: str):
        image = Image.open(image_path)
        return self.predict(image)


def test_predictor():
    predictor = ASLPredictor()

    print("\n" + "=" * 70)
    print("ASL Predictor Loaded Successfully")
    print("=" * 70)
    print(f"Model input shape: {predictor.model.input_shape}")
    print(f"Number of classes: {len(predictor.class_names)}")
    print(f"Classes: {', '.join(predictor.class_names)}")
    print("\nReady for inference!")


if __name__ == "__main__":
    test_predictor()
