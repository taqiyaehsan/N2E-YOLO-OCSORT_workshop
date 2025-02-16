# **N2E YOLO-OCSORT Workshop**  

This tutorial demonstrates real-time object detection using **YOLOv8** and integrates **OC-SORT** for robust object tracking, with optimizations for dynamic videos. Follow the steps below to set up your environment and run the demo.

## **Set Up the Virtual Environment**  

1. **Ensure Python is installed** on your machine. You can check by running:  
   ```bash
   python --version
   ```
2. **Open VSCode Terminal** and navigate to your project directory:  
   ```bash
   mkdir yolo_workshop && cd yolo_workshop
   ```
3. **Create a virtual environment**:  
   ```bash
   python -m venv yolo_env
   ```
4. **Activate the virtual environment**:  
   - **Windows**:  
     ```bash
     yolo_env\Scripts\activate
     ```
   - **macOS/Linux**:  
     ```bash
     source yolo_env/bin/activate
     ```
5. **Upgrade pip** and install required packages:  
   ```bash
   pip install --upgrade pip  
   pip install -r requirements.txt
   ```

## **Project Structure**

The project consists of two main components:

1. **YOLO Demo**:
   - Basic object detection demonstration
   - Introduction to YOLOv8 capabilities
   - Visualization examples

2. **YOLO+OC-SORT Implementation**:
   - Advanced person tracking system
   - Feature extraction and matching
   - Robust tracking through occlusions
   - Exercise/fitness video optimizations

## **Running the Basic YOLO Demo**

1. **Open** the Jupyter notebook:
   ```bash
   jupyter notebook yolo_demo.ipynb
   ```
2. **Follow the steps** in the notebook to run basic object detection.

## **Running the Advanced Tracking System**

1. **Prepare your video**:
   - Place your input video in the project directory
   - Supported formats: MP4, AVI

2. **Run the tracker**:
   ```bash
   python person_tracker.py --video path/to/your/video.mp4
   ```

3. **Customize tracking parameters** by modifying the OCSort initialization:
   ```python
   tracker = OCSort(
       det_thresh=0.3,          # Detection confidence threshold
       max_age=60,              # Maximum frames to keep dead tracks
       min_hits=3,              # Minimum detections to initialize track
       iou_threshold=0.2,       # IOU threshold for matching
       appearance_weight=0.6,   # Weight for appearance matching
       position_weight=0.2,     # Weight for spatial matching
       motion_weight=0.3        # Weight for motion prediction
   )
   ```

## **Understanding the Code**

### Key Components:

1. **Feature Extraction** (`extract_features`):
   - Combines color and shape information
   - Optimized for person tracking
   - Robust to lighting changes

2. **Kalman Filter Tracking** (`KalmanBoxTracker`):
   - Predicts object motion
   - Handles occlusions
   - Maintains track identity

3. **OC-SORT Tracker** (`OCSort`):
   - Advanced tracking algorithm
   - Feature matching for re-identification
   - Optimized for exercise movements

### Customization Points:

1. **Detection Parameters**:
   ```python
   det_thresh=0.3        # Confidence threshold for detections
   min_hits=3            # Minimum detections to confirm track
   ```

2. **Tracking Weights**:
   ```python
   appearance_weight=0.6  # Weight for feature matching
   position_weight=0.2    # Weight for spatial similarity
   motion_weight=0.3      # Weight for motion prediction
   ```

3. **Dynamic Motion Parameters**:
   ```python
   vertical_motion_threshold=1.5  # For jumping movements
   size_change_threshold=0.4      # For pose changes
   ```

## **Visualization and Output**

The system provides:
- Real-time tracking visualization
- Performance metrics (FPS, processing time)
- Track ID persistence
- Output video with annotations

## **Common Issues and Solutions**

1. **Low FPS**:
   - Reduce input video resolution
   - Use a smaller YOLOv8 model (e.g., yolov8n.pt)
   - Adjust processing parameters

2. **Track ID Switches**:
   - Increase `appearance_weight`
   - Adjust `reid_threshold`
   - Modify `id_cooldown` period

3. **Missing Detections**:
   - Lower `det_thresh`
   - Decrease `min_hits`
   - Adjust lighting conditions