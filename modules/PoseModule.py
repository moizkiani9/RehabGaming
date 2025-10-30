import cv2
import mediapipe as mp
import math
from typing import List, Tuple, Optional


class PoseDetector:
    """
    A class for detecting and analyzing human body pose using MediaPipe.
    Provides key landmark detection, angle calculation, and visualization utilities.
    """

    def __init__(self,
                 mode: bool = False,
                 complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        """
        Initializes the MediaPipe Pose model and drawing utilities.
        
        Args:
            mode: Whether to treat input images as a batch of static images
            complexity: Complexity of the pose landmark model (0, 1, or 2)
            smooth_landmarks: Whether to filter landmarks to reduce jitter
            enable_segmentation: Whether to predict segmentation mask
            smooth_segmentation: Whether to filter segmentation to reduce jitter
            detection_confidence: Minimum confidence for pose detection [0.0, 1.0]
            tracking_confidence: Minimum confidence for pose tracking [0.0, 1.0]
        """
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe pose and drawing utils
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.results = None
        self.lmList: List[List[int]] = []

    def find_pose(self, img, draw: bool = True):
        """
        Detects pose landmarks in a given image.
        
        Args:
            img: Input BGR image (numpy array)
            draw: Whether to draw the detected landmarks
            
        Returns:
            Processed image with landmarks drawn (if draw=True)
        """
        if img is None or img.size == 0:
            return img
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )

        return img

    def find_positions(self, img, draw: bool = True) -> List[List[int]]:
        """
        Extracts landmark positions (x, y) in pixels from the detected pose.
        
        Args:
            img: Image used for detection
            draw: Whether to draw keypoints on the image
            
        Returns:
            List of [id, x, y] for each detected landmark
        """
        self.lmList = []
        
        if img is None or img.size == 0:
            return self.lmList
            
        if self.results and self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Check visibility threshold
                if lm.visibility < 0.5:
                    continue
                    
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return self.lmList

    def find_angle(self, img, p1: int, p2: int, p3: int, draw: bool = True) -> Optional[float]:
        """
        Calculates the angle formed by three body landmarks.
        
        Args:
            img: Input image
            p1, p2, p3: Landmark indices (p2 is the vertex)
            draw: Whether to visualize angle lines
            
        Returns:
            Calculated angle in degrees, or None if landmarks not found
        """
        # Validate landmarks exist
        if len(self.lmList) == 0:
            return None
            
        # Find the landmarks in lmList
        landmarks = {lm[0]: lm[1:] for lm in self.lmList}
        
        if p1 not in landmarks or p2 not in landmarks or p3 not in landmarks:
            return None

        x1, y1 = landmarks[p1]
        x2, y2 = landmarks[p2]
        x3, y3 = landmarks[p3]

        # Calculate the angle using atan2
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) -
            math.atan2(y1 - y2, x1 - x2)
        )

        # Normalize the angle to [0, 180]
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle

        # Draw visual elements
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, (x, y), 6, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 12, (0, 0, 255), 2)

            cv2.putText(img, f"{int(angle)}°", (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return angle

    def get_landmark_position(self, landmark_id: int) -> Optional[Tuple[int, int]]:
        """
        Get the (x, y) position of a specific landmark.
        
        Args:
            landmark_id: Index of the landmark
            
        Returns:
            Tuple of (x, y) coordinates, or None if not found
        """
        for lm in self.lmList:
            if lm[0] == landmark_id:
                return (lm[1], lm[2])
        return None

    def is_pose_detected(self) -> bool:
        """Check if a pose was detected in the last processed frame."""
        return self.results is not None and self.results.pose_landmarks is not None

    def close(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """
    Simple webcam test to verify pose detection works.
    Press 'q' to quit.
    """
    with PoseDetector(detection_confidence=0.7, tracking_confidence=0.7) as detector:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Press 'q' to quit")

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Failed to read frame")
                break

            # Detect pose
            img = detector.find_pose(img)
            lmList = detector.find_positions(img, draw=True)

            # Example: Calculate elbow angle if landmarks detected
            if len(lmList) > 0:
                # Right arm: shoulder(12), elbow(14), wrist(16)
                angle = detector.find_angle(img, 12, 14, 16, draw=True)
                if angle:
                    # Display FPS and detection status
                    cv2.putText(img, f"Right Elbow: {int(angle)}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 0, 0), 2)

            cv2.imshow("Pose Detection", img)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()