# **N2E-YOLO-OCSORT Workshop**  

This workshop covers real-time object detection using **YOLOv8** and integrates **OC-SORT** for object tracking. Follow the steps below to set up your environment and run the demo.  

## **Step 1: Set Up the Virtual Environment**  

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
   pip install ultralytics opencv-python notebook matplotlib  
   ```

## **Step 2: Running the YOLO Demo Notebook**  

1. **Open** the Jupyter notebook named **`yolo_demo.ipynb`**.
3. **Follow the steps** in the notebook to run the demo on a sample video.  
