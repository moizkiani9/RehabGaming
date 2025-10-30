"""
Pose Detection Demo
Basic demonstration of MediaPipe pose detection capabilities.
Shows detected landmarks and skeleton overlay.
"""

import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from modules.PoseModule import PoseDetector


def main():
    print("=" * 60)
    print("POSE DETECTION DEMO")
    print("=" * 60)
    print("\nThis demo shows real-time pose detection using MediaPipe.")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'd' to toggle drawing landmarks")
    print("  - Press 'a' to show angle calculation")
    print("=" * 60)
    print("\n Starting webcam...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize pose detector
    detector = PoseDetector(
        detection_confidence=0.7,
        tracking_confidence=0.7
    )
    
    # State variables
    show_landmarks = True
    show_angles = False
    frame_count = 0
    
    print("✅ Webcam started successfully!")
    print("📸 Stand in front of the camera...")
    
    while cap.isOpened():
        success, img = cap.read()
        
        if not success:
            print("⚠️ Failed to read frame")
            break
        
        # Mirror the image
        img = cv2.flip(img, 1)
        
        # Detect pose
        img = detector.find_pose(img, draw=show_landmarks)
        lmList = detector.find_positions(img, draw=False)
        
        # Show detection status
        if len(lmList) > 0:
            status = "✓ Person Detected"
            status_color = (0, 255, 0)
            
            # Show angles if enabled
            if show_angles:
                # Right arm angle
                right_angle = detector.find_angle(img, 12, 14, 16, draw=True)
                # Left arm angle
                left_angle = detector.find_angle(img, 11, 13, 15, draw=True)
                
                # Right leg angle
                right_leg = detector.find_angle(img, 24, 26, 28, draw=True)
                # Left leg angle
                left_leg = detector.find_angle(img, 23, 25, 27, draw=True)
        else:
            status = "✗ No Person Detected"
            status_color = (0, 0, 255)
        
        # Draw info panel
        h, w, _ = img.shape
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Status
        cv2.putText(img, status, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Landmarks count
        cv2.putText(img, f"Landmarks: {len(lmList)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        frame_count += 1
        cv2.putText(img, f"Frame: {frame_count}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls help
        cv2.putText(img, "Press 'q' to quit | 'd' landmarks | 'a' angles", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the result
        cv2.imshow("Pose Detection Demo", img)
        
        # Handle keyboard input
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('d'):
            show_landmarks = not show_landmarks
            print(f"🎨 Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('a'):
            show_angles = not show_angles
            print(f"📐 Angles: {'ON' if show_angles else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo completed!")


if __name__ == "__main__":
    main()