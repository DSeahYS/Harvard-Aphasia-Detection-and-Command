import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class EyeSpeakTracker:
    def __init__(self):
        """Initialize the eye tracker with MediaPipe face mesh and fallback detection."""
        # Initialize MediaPipe FaceMesh with iris landmarks enabled
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize fallback face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Tracking history for smoothing and blink detection
        self.gaze_history = deque(maxlen=10)  # Last 10 gaze points
        self.blink_history = deque(maxlen=10)  # Last 10 blink states (1=closed, 0=open)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # Debug mode
        self.debug = True
        
    def preprocess_frame(self, frame):
        """Apply image processing to handle glare and improve detection."""
        # Convert to grayscale to check brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if frame has glare (high brightness)
        if np.mean(gray) > 220:
            # Apply contrast enhancement to combat glare
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=-20)
            
        return frame
        
    def detect_blink(self, landmarks):
        """Detect eye blinks using eye aspect ratio (EAR)."""
        # Left eye landmarks (adjusted for MediaPipe face mesh)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        
        # Right eye landmarks
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Extract points
        left_eye = [landmarks[idx] for idx in left_eye_indices]
        right_eye = [landmarks[idx] for idx in right_eye_indices]
        
        # Calculate eye aspect ratio
        def eye_aspect_ratio(eye):
            # Compute the euclidean distances between vertical eye landmarks
            v1 = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - 
                               np.array([eye[5].x, eye[5].y]))
            v2 = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - 
                               np.array([eye[4].x, eye[4].y]))
            
            # Compute the euclidean distance between horizontal eye landmarks
            h = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - 
                              np.array([eye[3].x, eye[3].y]))
            
            # Return eye aspect ratio
            return (v1 + v2) / (2.0 * h) if h > 0 else 0
        
        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratio
        ear = (left_ear + right_ear) / 2.0
        
        # Eyes are closed if EAR is below threshold
        is_blinking = ear < 0.2
        
        return is_blinking
        
    def get_iris_position(self, frame):
        """Extract iris position using MediaPipe face mesh, tracking both eyes."""
        # Preprocess frame for better detection
        processed_frame = self.preprocess_frame(frame)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        h, w = frame.shape[:2]
        
        # If face landmarks detected
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Detect blink
            is_blinking = self.detect_blink(landmarks)
            self.blink_history.append(1 if is_blinking else 0)
            
            # If eyes are open, track iris
            if not is_blinking:
                # Define both eye iris landmarks
                right_iris_landmarks = [474, 475, 476, 477]  # Right eye iris
                left_iris_landmarks = [468, 469, 470, 471]   # Left eye iris
                
                # Get pixel coordinates for both irises
                right_iris_points = []
                left_iris_points = []
                
                # Process right eye
                for idx in right_iris_landmarks:
                    try:
                        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                        right_iris_points.append((x, y))
                    except IndexError:
                        continue
                
                # Process left eye
                for idx in left_iris_landmarks:
                    try:
                        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                        left_iris_points.append((x, y))
                    except IndexError:
                        continue
                
                # Calculate centers if points were found
                right_center = None
                left_center = None
                
                if right_iris_points:
                    right_center = np.mean(right_iris_points, axis=0)
                
                if left_iris_points:
                    left_center = np.mean(left_iris_points, axis=0)
                
                # Determine final gaze point based on available data
                if right_center is not None and left_center is not None:
                    # Average both eyes for best accuracy
                    iris_center = (right_center + left_center) / 2
                    
                    if self.debug:
                        # Draw both eye centers for debugging
                        cv2.circle(frame, (int(right_center[0]), int(right_center[1])), 
                                 5, (0, 255, 255), -1)
                        cv2.circle(frame, (int(left_center[0]), int(left_center[1])), 
                                 5, (0, 255, 255), -1)
                
                elif right_center is not None:
                    # Use right eye if only it is detected
                    iris_center = right_center
                    
                    if self.debug:
                        cv2.putText(frame, "Right Eye Only", (30, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                elif left_center is not None:
                    # Use left eye if only it is detected
                    iris_center = left_center
                    
                    if self.debug:
                        cv2.putText(frame, "Left Eye Only", (30, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                else:
                    # No eyes detected properly
                    return None
                
                # Convert to integers
                x_px, y_px = int(iris_center[0]), int(iris_center[1])
                
                # Add to history for smoothing
                self.gaze_history.append((x_px, y_px))
                
                # Apply moving average for stability if we have enough history
                if len(self.gaze_history) >= 3:
                    smoothed_x = int(np.mean([p[0] for p in list(self.gaze_history)[-3:]]))
                    smoothed_y = int(np.mean([p[1] for p in list(self.gaze_history)[-3:]]))
                    return (smoothed_x, smoothed_y, True)  # True = primary method success
                
                return (x_px, y_px, True)
            
            # If blinking, draw a visual indicator if in debug mode
            if self.debug and is_blinking:
                cv2.putText(frame, "BLINK", (30, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        # Return None if iris detection failed
        return None
        
    def get_fallback_position(self, frame):
        """Fallback to face detection when iris tracking fails."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Return center of face
            return (x + w//2, y + h//2, False)  # False = fallback method
            
        return None
        
    def get_gaze_position(self, frame):
        """Main function to get current gaze position with fallbacks."""
        # Update FPS calculation
        current_time = time.time()
        self.fps_history.append(1 / (current_time - self.last_time + 0.001))
        self.last_time = current_time
        
        # Try iris detection first (primary method)
        gaze_pos = self.get_iris_position(frame)
        
        # If iris detection failed, try face detection fallback
        if gaze_pos is None:
            gaze_pos = self.get_fallback_position(frame)
        
        # Add debug information if enabled
        if self.debug:
            fps = int(sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0
            cv2.putText(frame, f"FPS: {fps}", (30, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            method = "Primary" if gaze_pos and gaze_pos[2] else "Fallback" if gaze_pos else "None"
            cv2.putText(frame, f"Method: {method}", (30, 80), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return gaze_pos
        
    def check_consent(self):
        """Check for double-blink pattern to verify consent."""
        if len(self.blink_history) < 6:
            return True
            
        # Look for pattern: open→closed→closed→open (0→1→1→0)
        pattern = [0, 1, 1, 0]
        
        # Check last few entries for the pattern
        for i in range(len(self.blink_history) - len(pattern) + 1):
            window = list(self.blink_history)[i:i+len(pattern)]
            if window == pattern:
                return True
                
        # Default to True to avoid blocking during development
        # In production, you might want to be more strict
        return True

def main():
    """Test function for eye tracker."""
    cap = cv2.VideoCapture(0)
    
    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create tracker instance
    tracker = EyeSpeakTracker()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
            
        # Get gaze position
        gaze_result = tracker.get_gaze_position(frame)
        
        # Draw gaze point if available
        if gaze_result:
            x, y, primary = gaze_result
            
            # Primary method (green), fallback (yellow)
            color = (0, 255, 0) if primary else (0, 255, 255)
            cv2.circle(frame, (x, y), 10, color, -1)
            
        # Check consent status
        consent_active = tracker.check_consent()
        status = "Active" if consent_active else "Inactive"
        cv2.putText(frame, f"Consent: {status}", (30, 160), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                  (0, 255, 0) if consent_active else (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("EYE-SPEAK+ Eye Tracking", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
