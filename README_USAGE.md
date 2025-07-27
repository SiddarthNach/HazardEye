# 🛣️ HazardEye - Dual Road Safety Analysis

## Overview
This Streamlit app provides comprehensive road safety analysis by combining advanced lane detection with AI-powered pothole detection to generate safety scores and risk assessments.

## Features
✅ **User Authentication** - Secure login and signup system  
✅ **Video Upload** - Support for MP4, AVI, MOV files  
✅ **Lane Detection** - Advanced lane detection using OpenCV  
✅ **Pothole Detection** - YOLOv8-based pothole detection AI  
✅ **Dual Analysis** - Combined lane and pothole safety scoring  
✅ **Safety Scoring** - Comprehensive safety metrics (0-100 scale)  
✅ **Risk Assessment** - LOW/MEDIUM/HIGH/CRITICAL risk levels  
✅ **Visual Overlays** - Processed video with lane detection and pothole annotations  
✅ **Detailed Analytics** - Curve warnings, position deviation, and pothole count analysis  

## How to Use

### 1. Start the Application
```bash
cd /Users/siddarthnachannagari/HazardEye
streamlit run app.py
```
The app will be available at: http://localhost:8502

### 2. Create Account or Login
- Select "SignUp" to create a new account
- Or select "Login" to access existing account

### 3. Upload Dashcam Video
- Upload a dashcam video file (MP4, AVI, MOV)
- Supported formats ensure maximum compatibility
- The original video will be displayed for preview

### 4. Run Dual Analysis
- Click "🔍 Start Road Safety Analysis"
- The system will process your video frame by frame
- **Lane Detection**: Analyzes lane positions, curves, and driving patterns
- **Pothole Detection**: Uses YOLOv8 AI to identify potholes in real-time
- Progress indicator shows real-time processing status

### 5. Review Comprehensive Results

#### Safety Score (0-100)
- **80-100**: ✅ LOW RISK - Excellent driving, no major hazards
- **60-79**: ⚠️ MEDIUM RISK - Some concerns detected
- **40-59**: 🚨 HIGH RISK - Multiple issues or hazards  
- **0-39**: 🆘 CRITICAL RISK - Serious safety concerns

#### Key Metrics
- **Lane Detection Success Rate**: Percentage of frames with successful lane detection
- **🕳️ Potholes Detected**: Total number of potholes identified
- **Curve Warnings**: Percentage of frames with sharp curves (< 500m radius)
- **Center Deviation**: Percentage of frames with significant lane position deviation (> 1.0m)
- **Pothole Risk Frames**: Percentage of frames containing potholes

#### Analyzed Video
- Download the processed video with lane detection overlays and pothole annotations
- Visual indicators show detected lanes, curves, position information, and pothole locations

## Dual Safety Scoring Algorithm

The app uses a weighted scoring system prioritizing road hazards over driving patterns:

**Scoring Weights:**
- **🕳️ Pothole Detection: 70%** (Primary safety factor)
- **🛣️ Lane Detection: 30%** (Secondary driving factor)

### Pothole Component (70% weight):
- **Base Penalty**: 50 points for any pothole detection
- **Density Penalties**:
  - >5% frames with potholes: Additional 40 points (90 total)
  - >2% frames with potholes: Additional 30 points (80 total)  
  - >1% frames with potholes: Additional 20 points (70 total)
  - <1% frames with potholes: Additional 10 points (60 total)
- **Absolute Count**: 0.5 points per pothole (max 30 points)
- **Risk Levels**: Based primarily on pothole count
  - 20+ potholes: CRITICAL
  - 10+ potholes: HIGH
  - 5+ potholes: MEDIUM

### Lane Component (30% weight):
- **Lane Detection Success**: Base score multiplied by success rate
- **Curve Analysis**: 30-point penalty scale for sharp curves
- **Position Deviation**: 25-point penalty scale for off-center driving
- **Consistency Factors**: Additional penalties for poor driving patterns

## Technical Details

### Dual Analysis Pipeline
1. **Perspective Transformation** - Bird's eye view conversion
2. **Color Thresholding** - HLS/HSV color space analysis  
3. **Lane Line Fitting** - Polynomial curve fitting
4. **YOLOv8 Pothole Detection** - AI-powered pothole identification
5. **Curvature Calculation** - Real-world curve radius measurement
6. **Position Tracking** - Vehicle position relative to lane center
7. **Hazard Annotation** - Visual marking of detected potholes

### File Structure
```
HazardEye/
├── app.py                               # Main Streamlit application
├── utils.py                             # Safety analysis and scoring functions
├── lane_analysis.py                     # Lane detection pipeline
├── my_model.pt                          # YOLOv8 pothole detection model
├── train/weights/best.pt                # Best trained pothole model
├── Pothole_Detection_Model_Training.ipynb # Training notebook
└── temp/                                # Temporary files and processed videos
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install streamlit opencv-python numpy sqlite3
   ```

2. **Video Upload Issues**
   - Check file format (MP4, AVI, MOV)
   - Ensure file size is reasonable (< 100MB recommended)
   - Verify file is not corrupted

3. **Lane Detection Problems**
   - Works best with clear lane markings
   - Adequate lighting conditions improve results
   - Highway/road videos work better than city traffic

4. **Pothole Detection Issues**
   - Model works best on clear road surface videos
   - Lighting and resolution affect detection accuracy
   - Some false positives may occur with shadows or road marks

5. **Performance Issues**
   - Processing time depends on video length and resolution
   - Dual analysis (lane + pothole) takes longer than lane-only
   - Consider using shorter clips for testing

### Test the Installation
```bash
python test_app.py
```

## Example Workflows

### Basic Analysis
1. Login to app
2. Upload dashcam video
3. Click "Start Analysis"
4. Review safety score and metrics
5. Download processed video

### Detailed Review
1. Run basic analysis
2. Expand "Detailed Analysis" section
3. Review frame-by-frame metrics
4. Check curve and deviation percentages
5. Read safety recommendations

## API Integration (Future)
The app is designed to be easily extended with:
- Pothole detection (YOLOv8 integration ready)
- Traffic sign recognition
- Vehicle detection and tracking
- Speed estimation
- Weather condition analysis

## Support
For issues or questions:
1. Check the troubleshooting section
2. Run `python test_app.py` to verify installation
3. Review console logs for error messages
4. Ensure all dependencies are properly installed

---
*Built with Streamlit, OpenCV, and NumPy for advanced road safety analysis.*
