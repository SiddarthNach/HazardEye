import os
import cv2
import hashlib
import numpy as np
import sys
import tempfile

# Try to import AWS config
try:
    from aws_config import AWSConfig
    AWS_AVAILABLE = True
except ImportError:
    AWSConfig = None
    AWS_AVAILABLE = False

# Try to import ML libraries - Disable YOLO for now due to signal handler issues
YOLO_AVAILABLE = False
YOLO = None

def get_yolo_model():
    """YOLO disabled due to signal handler conflicts in Streamlit"""
    return None, False

# Add the current directory to Python path to import lane_analysis
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from lane_analysis import FindLaneLines
    LANE_ANALYSIS_AVAILABLE = True
except ImportError:
    FindLaneLines = None
    LANE_ANALYSIS_AVAILABLE = False

def create_user_table(c):
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)')

def add_user(c, username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    c.execute('INSERT INTO users(username, password) VALUES (?, ?)', (username, hashed))

def login_user(c, username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed))
    return c.fetchone()

def save_uploaded_file(uploadedfile):
    """Save uploaded file locally and optionally to S3"""
    import streamlit as st
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    local_file_path = os.path.join("temp", uploadedfile.name)
    
    # Save locally first
    with open(local_file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    
    # Upload to S3 if configured
    if AWS_AVAILABLE and AWSConfig:
        try:
            aws_config = AWSConfig()
            s3_key = f"uploads/{uploadedfile.name}"
            s3_url = aws_config.upload_video_to_s3(local_file_path, s3_key)
            st.info(f"✅ Video uploaded to S3: {s3_key}")
        except Exception as e:
            st.warning(f"⚠️ S3 upload failed: {str(e)}. Using local storage.")
    else:
        st.info("📁 File saved locally (S3 not configured)")
    
    return local_file_path

def load_pothole_model():
    """
    Load the trained pothole detection model
    """
    import streamlit as st
    
    YOLO, YOLO_AVAILABLE = get_yolo_model()
    
    if not YOLO_AVAILABLE:
        st.error("YOLO not available. Cannot load pothole detection model.")
        return None
        
    try:
        # Try to load the best model first, fallback to the main model
        model_paths = [
            "train/weights/best.pt",
            "train/weights/last.pt", 
            "my_model.pt"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                st.info(f"Loading pothole model from: {model_path}")
                model = YOLO(model_path)
                return model
        
        st.error("No pothole detection model found!")
        return None
    except Exception as e:
        st.error(f"Error loading pothole model: {str(e)}")
        return None

def analyze_road_safety(video_path):
    """
    Analyze road safety from video with fallback for missing dependencies
    """
    import streamlit as st
    
    try:
        st.info("🎬 Starting road safety analysis...")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            return {
                'error': f"Video file not found: {video_path}",
                'analysis_complete': False
            }
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'error': f"Could not open video file: {video_path}",
                'analysis_complete': False
            }
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.info(f"📊 Video info: {total_frames} frames at {fps} FPS")
        
        # Initialize counters
        processed_frames = 0
        lane_success_count = 0
        pothole_count = 0
        pothole_frames = 0
        curve_warnings = 0
        deviation_warnings = 0
        
        # Progress bar
        progress_bar = st.progress(0)
        
        # Process frames (enhanced analysis without heavy ML)
        frame_skip = max(1, fps * 3)  # Process 1 frame every 3 seconds for efficiency
        
        # Initialize lane detector once
        lane_detector = None
        if LANE_ANALYSIS_AVAILABLE and FindLaneLines:
            lane_detector = FindLaneLines()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frames += 1
            
            # Skip frames for performance
            if processed_frames % frame_skip != 0:
                continue
            
            # Update progress
            progress = processed_frames / total_frames
            progress_bar.progress(progress)
            
            # Enhanced frame analysis
            if lane_detector:
                try:
                    # Lane detection
                    lane_result = lane_detector.forward(frame)
                    if lane_result is not None:
                        lane_success_count += 1
                        
                        # Simulate additional analysis based on frame properties
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Check for potential curves (edge density analysis)
                        edges = cv2.Canny(frame_gray, 50, 150)
                        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                        if edge_density > 0.15:  # High edge density suggests curves
                            curve_warnings += 1
                        
                        # Check for lane deviation (using image moments)
                        moments = cv2.moments(edges)
                        if moments['m00'] > 0:
                            cx = int(moments['m10'] / moments['m00'])
                            frame_center = frame.shape[1] // 2
                            deviation = abs(cx - frame_center) / frame_center
                            if deviation > 0.2:  # Significant deviation from center
                                deviation_warnings += 1
                        
                except Exception as e:
                    st.warning(f"Lane analysis error: {str(e)}")
            
            # Simulate pothole detection using basic computer vision
            try:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Look for dark circular regions (potential potholes)
                blurred = cv2.GaussianBlur(frame_gray, (9, 9), 2)
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                                         param1=50, param2=30, minRadius=10, maxRadius=50)
                
                if circles is not None:
                    # Filter circles that are significantly darker than surroundings
                    for circle in circles[0]:
                        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                        if 0 <= x-r and x+r < frame_gray.shape[1] and 0 <= y-r and y+r < frame_gray.shape[0]:
                            roi = frame_gray[y-r:y+r, x-r:x+r]
                            surrounding = frame_gray[max(0,y-2*r):min(frame_gray.shape[0],y+2*r), 
                                                   max(0,x-2*r):min(frame_gray.shape[1],x+2*r)]
                            
                            if np.mean(roi) < np.mean(surrounding) * 0.7:  # Significantly darker
                                pothole_count += 1
                                pothole_frames += 1
                                break  # Only count one per frame
            except Exception as e:
                pass  # Silently continue if pothole detection fails
        
        cap.release()
        progress_bar.progress(1.0)
        
        # Calculate metrics
        analyzed_frames = processed_frames // frame_skip
        lane_success_rate = (lane_success_count / max(analyzed_frames, 1)) * 100 if LANE_ANALYSIS_AVAILABLE else 0
        pothole_frames_percent = (pothole_frames / max(analyzed_frames, 1)) * 100
        curve_warnings_percent = (curve_warnings / max(analyzed_frames, 1)) * 100
        deviation_warnings_percent = (deviation_warnings / max(analyzed_frames, 1)) * 100
        
        # Enhanced safety score calculation
        lane_component = lane_success_rate * 0.4  # Increased weight for lane detection
        curve_penalty = curve_warnings_percent * 0.3
        deviation_penalty = deviation_warnings_percent * 0.2
        pothole_penalty = min(pothole_count * 5, 30)  # Cap penalty at 30 points
        
        safety_score = max(0, 100 - curve_penalty - deviation_penalty - pothole_penalty + (lane_component * 0.3))
        
        # Determine risk level with more nuanced criteria
        total_warnings = curve_warnings_percent + deviation_warnings_percent
        
        if safety_score >= 85 and pothole_count == 0 and total_warnings < 10:
            risk_level = "LOW"
        elif safety_score >= 70 and pothole_count <= 2 and total_warnings < 25:
            risk_level = "MEDIUM"
        elif safety_score >= 50 and pothole_count <= 5 and total_warnings < 40:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'analysis_complete': True,
            'safety_score': {
                'score': round(safety_score, 1),
                'risk_level': risk_level,
                'lane_success_rate': round(lane_success_rate, 1),
                'pothole_count': pothole_count,
                'pothole_frames_percent': round(pothole_frames_percent, 1),
                'curve_warnings_percent': round(curve_warnings_percent, 1),
                'deviation_warnings_percent': round(deviation_warnings_percent, 1),
                'lane_component_score': round(lane_component, 1),
                'pothole_component_score': round(max(0, 100 - pothole_penalty), 1)
            },
            'metrics': {
                'total_frames': total_frames,
                'processed_frames': analyzed_frames,
                'lane_success_frames': lane_success_count,
                'pothole_frames': pothole_frames,
                'curve_warnings': curve_warnings,
                'center_deviation_warnings': deviation_warnings,
                'average_curvature': [],  # Placeholder
                'average_position_deviation': []  # Placeholder
            }
        }
        
    except Exception as e:
        return {
            'error': f"Critical error during analysis: {str(e)}",
            'analysis_complete': False,
            'traceback': str(e)
        }
