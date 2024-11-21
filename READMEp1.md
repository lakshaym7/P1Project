
# REAL-TIME FALL DETECTION USING AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (ARIMA)

## Overview  
This project presents a robust real-time fall detection system using OpenPifPaf for human pose estimation and ARIMA for temporal trend analysis. The system processes video inputs to extract pose keypoints, analyzes sequential data, and identifies falls in dynamic, multi-camera environments with high accuracy.  

---

## Features  
- **Real-time Detection:** Processes live video streams or recorded footage to detect falls instantly.  
- **Pose Estimation:** Utilizes OpenPifPaf to extract human keypoints from video frames.  
- **Time-Series Analysis:** Applies ARIMA to analyze temporal patterns and detect fall anomalies.  
- **Multi-Camera Support:** Handles inputs from multiple camera views simultaneously.  
- **Scalable and Adaptable:** Designed for diverse environments like homes, healthcare facilities, and workplaces.  

---

## Requirements  
### Libraries  
The following Python libraries are required:  
- OpenPifPaf  
- OpenCV  
- Pandas  
- NumPy  
- Statsmodels (for ARIMA)  
- Matplotlib (for visualization)  

### Hardware Requirements  
- A system with GPU support is recommended for pose estimation.  
- Minimum 8 GB RAM for smooth processing of video data.  

---

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repository/fall-detection-arima.git
   cd fall-detection-arima
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Download OpenPifPaf pretrained models:  
   ```bash
   python -m openpifpaf.download
   ```  

---

## How to Use  
### Step 1: Initialize the System  
- Define video source path and frame rate in the configuration file or as input parameters.  

### Step 2: Extract Frames  
- Run `extract_frames(video_path)` to capture frames from the video.  

### Step 3: Pose Estimation  
- Use `extract_keypoints(frame)` to process each frame and store pose keypoints in a CSV file.  

### Step 4: Data Processing  
- Apply `process_data()` to prepare sequential data for ARIMA analysis.  

### Step 5: Fall Detection  
- Use the trained ARIMA model to classify frames as "fall" or "non-fall" in real-time.  

---

## Project Structure  
- `frames/` - Directory to store extracted frames.  
- `keypoints/` - CSV files storing pose keypoints for each frame.  
- `models/` - Pretrained OpenPifPaf model files.  
- `scripts/` - Python scripts for frame extraction, pose estimation, and ARIMA training.  

---

## Future Work  
- Integrating GRU networks for improved temporal modeling and trend detection.  
- Enhancing system performance in highly dynamic or occluded scenarios.  
- Expanding dataset coverage for better generalization.  

---

## Contributors  
- **[Your Name]** - Project Lead  
- **[Contributor Names]** - Research and Development  

---

## License  
This project is licensed under the MIT License.  

---

## Contact  
For questions or feedback, reach out at **your-email@example.com**.  
