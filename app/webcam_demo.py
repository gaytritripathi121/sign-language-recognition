"""
Real-time webcam demo for ASL Alphabet Recognition
"""
import cv2
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import ASLPredictor


def main():
    """Run real-time webcam demo"""
    print("\n" + "="*70)
    print("ASL Alphabet Recognition - Webcam Demo")
    print("="*70)
    
    # Load predictor
    try:
        predictor = ASLPredictor()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("Please ensure model is trained. Run: python src/train.py")
        return
    
    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Webcam initialized!")
    print("\n" + "="*70)
    print("Instructions:")
    print("  - Position your hand in the green box")
    print("  - Make an ASL alphabet sign")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'p' to pause/resume")
    print("="*70 + "\n")
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get prediction
            results, display_frame = predictor.predict_from_webcam_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = time.time()
            
            # Add FPS counter
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (display_frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add confidence indicator
            top_conf = results['top_confidence']
            conf_color = (0, 255, 0) if top_conf > 0.8 else \
                        (0, 255, 255) if top_conf > 0.5 else (0, 0, 255)
            
            cv2.rectangle(display_frame, (10, display_frame.shape[0] - 50),
                         (int(10 + top_conf * 300), display_frame.shape[0] - 20),
                         conf_color, -1)
            cv2.putText(display_frame, "Confidence", (10, display_frame.shape[0] - 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Display pause message
            pause_frame = display_frame.copy()
            cv2.putText(pause_frame, "PAUSED - Press 'p' to resume", 
                       (50, pause_frame.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            display_frame = pause_frame
        
        # Show frame
        cv2.imshow('ASL Alphabet Recognition', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('p'):
            # Toggle pause
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Demo ended successfully!")


if __name__ == "__main__":
    main()