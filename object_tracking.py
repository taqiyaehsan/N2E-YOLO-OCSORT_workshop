# YOLOv8 + OC-SORT Demo
# This script demonstrates real-time person tracking using YOLOv8 for detection
# and OC-SORT for tracking, optimized for dynamic videos.

import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from filterpy.kalman import KalmanFilter # type: ignore
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from skimage.feature import hog
import time

def extract_features(frame, bbox):
    """
    Extract appearance features from a detected person to help with tracking.
    
    Args:
        frame: Input video frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        Combined feature vector of color and HOG features
    
    Key Concepts:
    - Color features: Uses HSV color histogram which is more robust to lighting changes
    - HOG features: Captures shape/edge information that helps distinguish different people
    - Feature combination: Merges both types for more robust tracking
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    person_image = frame[y1:y2, x1:x2]
    
    # Standardize image size for consistent feature extraction
    person_image = cv2.resize(person_image, (64, 128))
    
    # Color histogram in HSV space (more robust to lighting changes)
    hsv_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    color_features = hist.flatten()
    
    # HOG features for shape/edge information
    gray_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), visualize=False)
    
    return np.concatenate([color_features, hog_features])

def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    
    Note: IoU is a key metric for measuring spatial overlap between detections
    and predicted track locations.
    """
    # Calculate intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def compute_similarity(feature1, feature2, bbox1, bbox2, time_diff):
    """
    Compute overall similarity between two detections using multiple cues.
    
    Args:
        feature1, feature2: Appearance feature vectors
        bbox1, bbox2: Bounding box coordinates
        time_diff: Time difference between detections in frames
    
    Returns:
        Weighted similarity score combining appearance, spatial, and temporal information
    
    Note: You can adjust the weights (0.7, 0.2, 0.1) to change the importance of each factor
    """
    appearance_sim = 1 - cosine(feature1, feature2)  # Feature similarity
    spatial_sim = iou(bbox1, bbox2)                  # Spatial overlap
    temporal_weight = np.exp(-time_diff / 30)        # Temporal decay
    
    # Weighted combination - adjust these weights based on your needs
    return appearance_sim * 0.7 + spatial_sim * 0.2 + temporal_weight * 0.1

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects using Kalman filter.
    
    Key Features:
    - Maintains position and velocity estimates
    - Tracks confidence and feature history
    - Handles track initialization and updates
    
    Important Parameters:
    - max_feature_history: Number of past features to store (default: 10)
    - reid_confidence: Track reliability score (0.0 to 1.0)
    """
    count = 0  # Class variable for unique ID generation
    
    def __init__(self, bbox):
        """Initialize tracker with bounding box detection."""
        # Initialize Kalman filter with 7 state variables and 4 measurement variables
        # State: [x, y, w, h, dx, dy, dw] (position, size, velocity)
        # Measurement: [x, y, w, h] (position and size)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Set up transition matrix (how state evolves)
        self.kf.F = np.array([[1,0,0,0,1,0,0],  # x = x + dx
                             [0,1,0,0,0,1,0],    # y = y + dy
                             [0,0,1,0,0,0,1],    # w = w + dw
                             [0,0,0,1,0,0,0],    # h = h
                             [0,0,0,0,1,0,0],    # dx = dx
                             [0,0,0,0,0,1,0],    # dy = dy
                             [0,0,0,0,0,0,1]])   # dw = dw

        # Measurement matrix (what we can measure)
        self.kf.H = np.array([[1,0,0,0,0,0,0],  # We can measure x
                             [0,1,0,0,0,0,0],    # We can measure y
                             [0,0,1,0,0,0,0],    # We can measure w
                             [0,0,0,1,0,0,0]])   # We can measure h

        # Tune these parameters based on your needs:
        self.kf.R[2:,2:] *= 10.    # Measurement noise
        self.kf.P[4:,4:] *= 1000.  # Initial state uncertainty
        self.kf.P *= 10.           # Initial state uncertainty
        self.kf.Q[-1,-1] *= 0.01   # Process noise
        self.kf.Q[4:,4:] *= 0.01   # Process noise

        # Initialize state with first detection
        self.kf.x[:4] = bbox[:4].reshape(4,1)
        
        # Track metadata
        self.time_since_update = 0
        self.id = self.get_next_id()
        self.history = []
        self.hits = 0                    # Number of detections
        self.hit_streak = 0              # Consecutive detections
        self.age = 0                     # Total frames
        self.reid_confidence = 0.5       # Track confidence score
        self.feature_history = []        # Store appearance features
        self.max_feature_history = 10    # Maximum features to store
        self.last_position = bbox[:4]    # Last known position
        self.last_seen_frame = 0         # Frame number when last seen

    @classmethod
    def get_next_id(cls):
        """Generate unique track IDs."""
        id = cls.count
        cls.count += 1
        return id

    def update_features(self, feature):
        """Update appearance feature history."""
        self.feature_history.append(feature)
        if len(self.feature_history) > self.max_feature_history:
            self.feature_history.pop(0)

    def get_average_feature(self):
        """Get average appearance feature from history."""
        if not self.feature_history:
            return None
        return np.mean(self.feature_history, axis=0)

    def update(self, bbox, frame_count=None):
        """
        Update tracker state with new detection.
        
        Args:
            bbox: New detection bounding box
            frame_count: Current frame number (optional)
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox[:4].reshape(4,1))
        self.reid_confidence = min(1.0, self.reid_confidence + 0.1)
        self.last_position = bbox[:4]
        if frame_count is not None:
            self.last_seen_frame = frame_count

    def predict(self):
        """
        Advance tracker state and return predicted bounding box estimate.
        
        Returns:
            Predicted bounding box location
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        self.reid_confidence = max(0.0, self.reid_confidence - 0.05)
        return self.history[-1]

    def get_state(self):
        """Return current bounding box estimate."""
        return self.kf.x[:4].flatten()

class OCSort:
    """
    Online and Continuous Sort (OC-SORT) implementation optimized for exercise videos.
    
    Key Features:
    - Robust tracking during rapid movements
    - Identity recovery after occlusions
    - Appearance feature matching
    
    Customization Points:
    - Adjust weights between appearance, position, and motion
    - Tune thresholds for different movement patterns
    - Modify temporal windows for different video types
    """
    
    def __init__(self, 
                 det_thresh=0.3,        # Detection confidence threshold
                 max_age=45,            # Maximum frames to keep dead tracks
                 min_hits=2,            # Minimum hits to initialize track
                 iou_threshold=0.2,     # IOU threshold for matching
                 max_history=50,        # Maximum frames of history
                 id_cooldown=90,        # Frames before ID can be reused
                 delta_t=2,             # Time window for prediction
                 appearance_weight=0.55, # Weight for appearance matching
                 position_weight=0.15,   # Weight for spatial matching
                 motion_weight=0.30):    # Weight for motion prediction
        """
        Initialize tracker with customizable parameters.
        
        Tip: Adjust these parameters based on your video characteristics:
        - For faster movements: decrease iou_threshold, increase motion_weight
        - For more pose changes: decrease appearance_weight
        - For longer occlusions: increase max_age, id_cooldown
        """
        # Core tracking parameters
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh
        self.max_history = max_history
        self.id_cooldown = id_cooldown
        self.delta_t = delta_t
        
        # Matching weights
        self.appearance_weight = appearance_weight
        self.position_weight = position_weight
        self.motion_weight = motion_weight
        
        # Track management
        self.trackers = []
        self.frame_count = 0
        self.lost_trackers = []
        self.recent_detections = []
        self.inactive_trackers = {}
        self.id_feature_history = {}
        
        # Exercise-specific parameters
        self.temporal_window = 45       # Frames to look back for recovery
        self.vertical_motion_threshold = 1.5  # For jumping movements
        self.size_change_threshold = 0.4      # For pose changes
        
        # Recovery thresholds
        self.recovery_similarity_threshold = 0.55
        self.reid_threshold = 0.6

    # ... [rest of the OCSort class implementation] ...

def generate_color(idx):
    """
    Generate consistent colors for visualization based on track ID.
    
    Args:
        idx: Track ID number
    
    Returns:
        RGB color tuple
    
    Note: Same track ID will always get the same color
    """
    np.random.seed(idx)
    color = tuple(map(int, np.random.randint(0, 255, 3)))
    return color

def main():
    """
    Main function to run the YOLOv8 + OC-SORT demo.
    
    Setup:
    1. Install required packages:
       pip install ultralytics opencv-python numpy scipy scikit-image filterpy
    
    2. Download YOLOv8 model:
       from ultralytics import YOLO
       model = YOLO('yolov8n.pt')
    
    Usage:
    - Change video_path to your input video
    - Adjust tracker parameters based on your needs
    - Press 'q' to quit during processing
    """
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # you can use different YOLOv8 models (n, s, m, l, x)

    # Initialize tracker with parameters tuned for exercise videos
    tracker = OCSort(
        det_thresh=0.3,          # Detection confidence threshold
        max_age=60,              # Maximum frames to keep dead tracks
        min_hits=3,              # Minimum detections to initialize track
        iou_threshold=0.2,       # IOU threshold for matching
        max_history=40,          # Maximum frames of history to keep
        id_cooldown=120,         # Frames before ID can be reused
        delta_t=2,               # Time window for prediction
        
        # Matching weights - adjust these based on your needs:
        appearance_weight=0.6,   # Weight for appearance matching
        position_weight=0.2,     # Weight for spatial matching
        motion_weight=0.3        # Weight for motion prediction
    )

    # Set up video processing
    video_path = 'yolo_demo_video.mp4'  # Change to your video path
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Original FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
    # Initialize tracking variables
    frame_count = 0
    processing_times = []
    start_time = time.time()

    # Set up video writer for output
    output_path = "ocsort_output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 file
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Main processing loop
    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_start_time = time.time()
        
        # Step 1: Object Detection
        # Run YOLOv8 detection on the frame
        results = model(frame)

        # Step 2: Process Detections
        # Extract person detections and compute features
        detections = []
        features = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # YOLOv8 class 0 is person
                if box.cls == 0:  
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    # Filter by confidence threshold
                    if conf >= 0.8:  # Adjust this threshold based on your needs
                        detection = [x1, y1, x2, y2, conf]
                        detections.append(detection)
                        # Compute appearance features for each detection
                        features.append(extract_features(frame, detection))

        # Step 3: Update Tracking
        if len(detections) > 0:
            # Convert lists to numpy arrays for tracking
            detections = np.array(detections)
            features = np.array(features)

            # Update tracker with new detections
            tracked_objects = tracker.update(detections, features)

            # Step 4: Visualization
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id, reid_conf = obj
                # Generate consistent color for this track ID
                color = generate_color(int(track_id))
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add text background for better visibility
                text = f"ID: {int(track_id)}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), 
                            (int(x1) + text_width, int(y1)), color, -1)
                
                # Add track ID text
                cv2.putText(frame, text, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Step 5: Performance Monitoring
        # Calculate and track processing speed
        processing_time = time.time() - frame_start_time
        processing_times.append(processing_time)
        
        # Calculate progress and estimated time remaining
        elapsed_time = time.time() - start_time
        progress = frame_count / total_frames
        if progress > 0:
            eta = elapsed_time * (1 - progress) / progress
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        else:
            eta_str = "Calculating..."

        # Calculate average processing speed
        avg_time = sum(processing_times[-30:]) / min(len(processing_times), 30)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Prepare information text for display
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Progress: {progress*100:.1f}%",
            f"ETA: {eta_str}",
            f"FPS: {avg_fps:.1f}"
        ]
        
        # Add processing information to frame
        y_position = 30
        for text in info_text:
            # Add background rectangle for better text visibility
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, y_position - text_height - 5), 
                        (10 + text_width, y_position + 5), (0, 0, 0), -1)
            # Add text
            cv2.putText(frame, text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_position += 30

        # Write frame to output video
        out.write(frame)
        
        # Print progress to console (update every 30 frames)
        if frame_count % 30 == 0:
            print(f"\rProcessing: {progress*100:.1f}% | FPS: {avg_fps:.1f} | ETA: {eta_str}", end="")
        
        # Check for 'q' key press to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Step 6: Final Statistics
    total_time = time.time() - start_time
    avg_processing_time = sum(processing_times) / len(processing_times)
    
    # Print processing summary
    print(f"\n\nProcessing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Average processing time per frame: {avg_processing_time:.3f} seconds")
    print(f"Average processing FPS: {1.0/avg_processing_time:.1f}")
    print(f"Input video FPS: {fps}")
    print(f"Total frames processed: {frame_count}")

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()