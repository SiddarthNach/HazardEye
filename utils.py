import os
import cv2
import hashlib
import numpy as np
import sys
import tempfile
import streamlit as st

# Try to import ultralytics with fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not available - pothole detection disabled")

# Try to import AWS config with fallback
try:
    from aws_config import AWSConfig
    AWS_AVAILABLE = True
except ImportError:
    AWSConfig = None
    AWS_AVAILABLE = False
    print("⚠️ AWS config not available - S3 upload disabled")

# Add the current directory to Python path to import lane_analysis
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from lane_analysis import FindLaneLines
except ImportError:
    st.error("Lane analysis module not found. Please ensure lane_analysis.py is in the same directory.")
    FindLaneLines = None

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
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    local_file_path = os.path.join("temp", uploadedfile.name)
    
    # Save locally first
    with open(local_file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    
    # Upload to S3 if configured
    if AWS_AVAILABLE:
        try:
            aws_config = AWSConfig()
            s3_key = f"uploads/{uploadedfile.name}"
            s3_url = aws_config.upload_video_to_s3(local_file_path, s3_key)
            st.info(f"✅ Video uploaded to S3: {s3_key}")
        except Exception as e:
            st.warning(f"⚠️ S3 upload failed: {str(e)}. Using local storage.")
    else:
        st.info("📁 File saved locally (AWS S3 not configured)")
    
    return local_file_path

def load_pothole_model():
    """
    Load the trained pothole detection model
    """
    if not YOLO_AVAILABLE:
        print("⚠️ YOLO not available - pothole detection disabled")
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
                model = YOLO(model_path)
                print(f"✅ Loaded pothole model from: {model_path}")
                return model
        
        print("❌ No pothole model found")
        return None
        
    except Exception as e:
        print(f"❌ Error loading pothole model: {e}")
        return None

def detect_potholes_in_frame(frame, model, confidence_threshold=0.5):
    """
    Detect potholes in a single frame using YOLO model
    Returns number of potholes detected and annotated frame
    """
    try:
        if model is None or not YOLO_AVAILABLE:
            return 0, frame
            
        # Run inference
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        # Count detections
        pothole_count = 0
        annotated_frame = frame.copy()
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                pothole_count = len(boxes)
                
                # Draw bounding boxes on frame
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add label
                    label = f"Pothole {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return pothole_count, annotated_frame
        
    except Exception as e:
        print(f"Error in pothole detection: {e}")
        return 0, frame

def analyze_road_safety(video_path):
    """
    Comprehensive road safety analysis combining lane detection and pothole detection
    Returns a detailed safety score and analysis
    """
    if FindLaneLines is None:
        return {
            'output_video_path': None,
            'safety_score': {'score': 0, 'risk_level': 'UNKNOWN'},
            'metrics': {},
            'analysis_complete': False,
            'error': 'Lane analysis module not available'
        }
    
    try:
        # Initialize lane detection and pothole detection
        lane_detector = FindLaneLines()
        pothole_model = load_pothole_model()
        
        # Video processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join("temp", f"{base_name}_analyzed.mp4")
        
        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Analysis metrics
        safety_metrics = {
            'lane_detection_score': 0,
            'curve_warnings': 0,
            'center_deviation_warnings': 0,
            'pothole_detections': 0,
            'pothole_frames': 0,
            'total_frames': total_frames,
            'processed_frames': 0,
            'average_curvature': [],
            'average_position_deviation': [],
            'risk_level': 'LOW'
        }
        
        frame_count = 0
        
        # Create progress indicators with try/catch for non-Streamlit environments
        progress_bar = None
        status_text = None
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
        except:
            # Running outside Streamlit environment
            pass
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with lane detection
            try:
                processed_frame = lane_detector.forward(frame_rgb)
                
                # Extract metrics from lane detection
                if lane_detector.lanelines.left_fit is not None and lane_detector.lanelines.right_fit is not None:
                    # Calculate curvature and position
                    left_curve, right_curve, position = lane_detector.lanelines.measure_curvature()
                    
                    # Store metrics
                    min_curvature = min(left_curve, right_curve)
                    safety_metrics['average_curvature'].append(min_curvature)
                    safety_metrics['average_position_deviation'].append(abs(position))
                    
                    # Warning thresholds
                    if min_curvature < 500:  # Sharp curve warning
                        safety_metrics['curve_warnings'] += 1
                        
                    if abs(position) > 1.0:  # Off-center warning
                        safety_metrics['center_deviation_warnings'] += 1
                    
                    safety_metrics['lane_detection_score'] += 1
                
            except Exception as e:
                # Frame processing failed, use original frame
                processed_frame = frame_rgb
                print(f"Frame {frame_count} lane detection failed: {e}")
            
            # Pothole detection on processed frame
            try:
                pothole_count, processed_frame = detect_potholes_in_frame(processed_frame, pothole_model)
                if pothole_count > 0:
                    safety_metrics['pothole_detections'] += pothole_count
                    safety_metrics['pothole_frames'] += 1
                    print(f"🕳️ Frame {frame_count}: {pothole_count} potholes detected")
                    
            except Exception as e:
                print(f"Frame {frame_count} pothole detection failed: {e}")
            
            # Convert back to BGR for video output
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            out.write(processed_frame_bgr)
            
            frame_count += 1
            safety_metrics['processed_frames'] = frame_count
            
            # Update progress (with error handling)
            try:
                if progress_bar is not None and status_text is not None:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
            except:
                pass
        
        cap.release()
        out.release()
        
        # Clean up progress indicators
        try:
            if progress_bar is not None:
                progress_bar.empty()
            if status_text is not None:
                status_text.empty()
        except:
            pass
        
        # Calculate final scores
        safety_score = calculate_safety_score(safety_metrics)
        
        print(f"📊 Analysis completed: {frame_count} frames processed")
        print(f"🏆 Safety score: {safety_score['score']}")
        print(f"⚠️ Risk level: {safety_score['risk_level']}")
        
        return {
            'output_video_path': output_path,
            'safety_score': safety_score,
            'metrics': safety_metrics,
            'analysis_complete': True
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to use st.error only if in Streamlit context
        try:
            st.error(f"Error during analysis: {str(e)}")
        except:
            pass
            
        return {
            'output_video_path': None,
            'safety_score': {'score': 0, 'risk_level': 'ERROR'},
            'metrics': {},
            'analysis_complete': False,
            'error': str(e)
        }

def calculate_safety_score(metrics):
    """
    Calculate overall safety score based on lane detection and pothole detection metrics
    Score ranges from 0-100 where 100 is safest
    Weighting: Potholes 70%, Lane Detection 30%
    """
    # Initialize components
    lane_score = 100
    pothole_score = 100
    
    # LANE DETECTION COMPONENT (30% weight)
    lane_success_rate = 0
    if metrics['total_frames'] > 0:
        lane_success_rate = metrics['lane_detection_score'] / metrics['total_frames']
        lane_score *= lane_success_rate
        
        # Penalty for curve warnings
        curve_penalty = (metrics['curve_warnings'] / metrics['total_frames']) * 30
        lane_score -= curve_penalty
        
        # Penalty for center deviation
        deviation_penalty = (metrics['center_deviation_warnings'] / metrics['total_frames']) * 25
        lane_score -= deviation_penalty
        
        # Average curvature factor
        if metrics['average_curvature']:
            avg_curvature = np.mean(metrics['average_curvature'])
            if avg_curvature < 300:  # Very sharp curves
                lane_score -= 15
            elif avg_curvature < 500:  # Moderate curves
                lane_score -= 8
        
        # Average position deviation factor
        if metrics['average_position_deviation']:
            avg_deviation = np.mean(metrics['average_position_deviation'])
            if avg_deviation > 1.5:  # Significantly off-center
                lane_score -= 20
            elif avg_deviation > 1.0:  # Moderately off-center
                lane_score -= 12
    else:
        lane_score = 0
    
    # POTHOLE DETECTION COMPONENT (70% weight)
    if metrics['total_frames'] > 0:
        pothole_density = metrics['pothole_detections'] / metrics['total_frames']
        pothole_frame_rate = metrics['pothole_frames'] / metrics['total_frames']
        
        # Heavy penalty for pothole presence
        if pothole_density > 0:
            # Base penalty: 50 points for any potholes
            pothole_score -= 50
            
            # Additional penalty based on density
            if pothole_density > 0.05:  # More than 5% of frames have potholes
                pothole_score -= 40  # Total: 90 point penalty
            elif pothole_density > 0.02:  # More than 2% of frames have potholes
                pothole_score -= 30  # Total: 80 point penalty
            elif pothole_density > 0.01:  # More than 1% of frames have potholes
                pothole_score -= 20  # Total: 70 point penalty
            else:  # Less than 1% but still present
                pothole_score -= 10  # Total: 60 point penalty
            
            # Additional penalty for absolute count
            absolute_penalty = min(30, metrics['pothole_detections'] * 0.5)  # 0.5 points per pothole, max 30
            pothole_score -= absolute_penalty
    
    # Ensure individual scores are between 0 and 100
    lane_score = max(0, min(100, lane_score))
    pothole_score = max(0, min(100, pothole_score))
    
    # Calculate weighted final score: 30% lane + 70% pothole
    final_score = (lane_score * 0.3) + (pothole_score * 0.7)
    
    # Determine risk level with heavy pothole consideration
    pothole_risk_factor = metrics['pothole_detections'] / metrics['total_frames'] if metrics['total_frames'] > 0 else 0
    
    if metrics['pothole_detections'] > 20:  # High pothole count
        risk_level = 'CRITICAL'
    elif metrics['pothole_detections'] > 10:  # Moderate pothole count
        risk_level = 'HIGH'
    elif metrics['pothole_detections'] > 5:  # Low pothole count
        risk_level = 'MEDIUM'
    elif final_score >= 80 and pothole_risk_factor < 0.005:  # Very few or no potholes
        risk_level = 'LOW'
    elif final_score >= 60:
        risk_level = 'MEDIUM'
    elif final_score >= 40:
        risk_level = 'HIGH'
    else:
        risk_level = 'CRITICAL'
    
    return {
        'score': round(final_score, 1),
        'risk_level': risk_level,
        'lane_success_rate': round(lane_success_rate * 100, 1) if metrics['total_frames'] > 0 else 0,
        'curve_warnings_percent': round((metrics['curve_warnings'] / metrics['total_frames']) * 100, 1) if metrics['total_frames'] > 0 else 0,
        'deviation_warnings_percent': round((metrics['center_deviation_warnings'] / metrics['total_frames']) * 100, 1) if metrics['total_frames'] > 0 else 0,
        'pothole_count': metrics['pothole_detections'],
        'pothole_frames_percent': round((metrics['pothole_frames'] / metrics['total_frames']) * 100, 1) if metrics['total_frames'] > 0 else 0,
        'lane_component_score': round(lane_score, 1),
        'pothole_component_score': round(pothole_score, 1)
    }

def detect_potholes_and_lanes(video_path):
    """
    Legacy function name for backwards compatibility
    """
    return analyze_road_safety(video_path)
