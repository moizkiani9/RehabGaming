import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from modules.PoseModule import PoseDetector
except ImportError:
    from PoseModule import PoseDetector


class ArmRaiseExercise:
    """
    Arm Raise Exercise Tracker
    Counts arm raises and provides real-time feedback
    """
    
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initialize the Arm Raise Exercise tracker.
        
        Args:
            detection_confidence: Pose detection confidence threshold
            tracking_confidence: Pose tracking confidence threshold
        """
        # Initialize pose detector with good confidence values
        self.detector = PoseDetector(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence
        )
        
        # Tracking variables
        self.arm_counter = 0
        self.rep_count = 0  # Alias for compatibility
        self.direction = 0  # 0 = down, 1 = up
        self.points = 0
        self.feedback = "Waiting for person..."
        
        # Performance tracking
        self.perfect_reps = 0
        self.good_reps = 0
        self.okay_reps = 0
        
        # Angle thresholds
        self.ANGLE_DOWN = 40  # Arms considered down
        self.ANGLE_UP = 120   # Arms considered raised
        self.ANGLE_PERFECT_MIN = 130
        self.ANGLE_PERFECT_MAX = 150
    
    def process_frame(self, img):
        """
        Process each frame and detect arm raises
        
        Args:
            img: Input frame from webcam
            
        Returns:
            Processed frame with overlays
        """
        # Detect pose
        img = self.detector.find_pose(img, draw=True)
        lmList = self.detector.find_positions(img, draw=False)
        
        # Check if person is detected
        if len(lmList) == 0:
            self.feedback = "No person detected"
            self._draw_ui(img)
            return img
        
        # Calculate shoulder angles (shoulder-elbow-wrist)
        # Right arm: shoulder(12), elbow(14), wrist(16)
        # Left arm: shoulder(11), elbow(13), wrist(15)
        right_shoulder_angle = self.detector.find_angle(img, 12, 14, 16, draw=False)
        left_shoulder_angle = self.detector.find_angle(img, 11, 13, 15, draw=False)
        
        # Handle None values (landmarks not visible)
        if right_shoulder_angle is None or left_shoulder_angle is None:
            self.feedback = "Arms not visible"
            self._draw_ui(img)
            return img
        
        # Display angles for debugging
        cv2.putText(img, f"Right: {int(right_shoulder_angle)}°", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Left: {int(left_shoulder_angle)}°", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Exercise logic - detect arm raises
        # Arms down position
        if right_shoulder_angle < self.ANGLE_DOWN and left_shoulder_angle < self.ANGLE_DOWN:
            self.feedback = "Arms Down - Ready"
            self.direction = 0
        
        # Arms raised position
        elif right_shoulder_angle > self.ANGLE_UP and left_shoulder_angle > self.ANGLE_UP:
            if self.direction == 0:  # Just raised arms (transition from down to up)
                self.arm_counter += 1
                self.rep_count = self.arm_counter  # Keep in sync
                self.direction = 1
                
                # Award points based on form quality
                if (self.ANGLE_PERFECT_MIN <= right_shoulder_angle <= self.ANGLE_PERFECT_MAX and
                    self.ANGLE_PERFECT_MIN <= left_shoulder_angle <= self.ANGLE_PERFECT_MAX):
                    self.points += 10
                    self.perfect_reps += 1
                    self.feedback = "Perfect Form! +10 pts 🌟"
                elif (right_shoulder_angle > 110 and left_shoulder_angle > 110):
                    self.points += 7
                    self.good_reps += 1
                    self.feedback = "Good Form! +7 pts 👍"
                else:
                    self.points += 5
                    self.okay_reps += 1
                    self.feedback = "Keep Going! +5 pts 💪"
            else:
                self.feedback = "Arms Up - Hold"
        
        # In between (transitioning)
        else:
            if self.direction == 0:
                self.feedback = "Raising arms..."
            else:
                self.feedback = "Lowering arms..."
        
        # Draw UI overlay
        self._draw_ui(img)
        return img
    
    def _draw_ui(self, img):
        """Draw counter, points, and feedback overlay"""
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (320, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Repetition count
        cv2.putText(img, f'Reps: {int(self.arm_counter)}', (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Points
        cv2.putText(img, f'Points: {int(self.points)}', (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # Feedback
        cv2.putText(img, f'{self.feedback}', (15, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        # Performance breakdown
        cv2.putText(img, f'Perfect: {self.perfect_reps} Good: {self.good_reps} OK: {self.okay_reps}', 
                   (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def get_summary(self):
        """Return exercise summary"""
        return {
            "total_reps": int(self.arm_counter),
            "total_points": int(self.points),
            "perfect_reps": self.perfect_reps,
            "good_reps": self.good_reps,
            "okay_reps": self.okay_reps,
            "avg_points_per_rep": round(self.points / self.arm_counter, 1) if self.arm_counter > 0 else 0,
            "feedback": self.feedback
        }
    
    def reset(self):
        """Reset all counters"""
        self.arm_counter = 0
        self.rep_count = 0
        self.direction = 0
        self.points = 0
        self.perfect_reps = 0
        self.good_reps = 0
        self.okay_reps = 0
        self.feedback = "Waiting for person..."


def main():
    """Main function to run the arm raise exercise tracker"""
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create exercise tracker
    exercise = ArmRaiseExercise()
    
    print("=" * 50)
    print("ARM RAISE EXERCISE TRACKER")
    print("=" * 50)
    print("\nInstructions:")
    print("- Stand in front of the camera")
    print("- Start with arms down at your sides")
    print("- Raise both arms above shoulder level")
    print("- Lower arms back down to complete one rep")
    print("\nControls:")
    print("- Press 'q' to quit and see summary")
    print("- Press 'r' to reset counters")
    print("=" * 50)
    
    while cap.isOpened():
        success, img = cap.read()
        
        if not success:
            print("Failed to read frame from webcam")
            break
        
        # Mirror image for intuitive movement
        img = cv2.flip(img, 1)
        
        # Process the frame
        img = exercise.process_frame(img)
        
        # Display the result
        cv2.imshow("Arm Raise Exercise Tracker", img)
        
        # Handle keyboard input
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            # Quit and show summary
            print("\n" + "=" * 50)
            print("EXERCISE SUMMARY")
            print("=" * 50)
            summary = exercise.get_summary()
            for k, v in summary.items():
                print(f"{k.replace('_', ' ').title()}: {v}")
            print("=" * 50)
            break
        
        elif key == ord('r'):
            # Reset counters
            exercise.reset()
            print("Counters reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()