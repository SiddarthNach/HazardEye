import streamlit as st
import cv2
import os
import boto3
from utils_fallback import analyze_road_safety, save_uploaded_file, create_user_table, add_user, login_user

# Initialize session state for analysis
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False

# AWS S3 setup (configure your credentials)
s3 = boto3.client('s3')  # This would require proper AWS config

# Initialize database
import sqlite3
conn = sqlite3.connect('users.db')
c = conn.cursor()
create_user_table(c)

st.set_page_config(
    page_title="HazardEye - Road Safety AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">👁️ HazardEye - AI Road Safety Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# User Authentication
st.sidebar.header("🔐 User Authentication")

auth_choice = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

if auth_choice == "Register":
    st.sidebar.subheader("Create New Account")
    new_username = st.sidebar.text_input("Username")
    new_password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Register"):
        if new_username and new_password:
            try:
                add_user(c, new_username, new_password)
                conn.commit()
                st.sidebar.success("✅ Account created successfully! Please login.")
            except Exception as e:
                st.sidebar.error(f"❌ Registration failed: {e}")
        else:
            st.sidebar.error("Please fill in all fields")

else:  # Login
    st.sidebar.subheader("Login to Your Account")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if username and password:
            result = login_user(c, username, password)
            if result:
                st.sidebar.success(f"✅ Welcome back, {username}!")
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.sidebar.error("❌ Invalid credentials")
        else:
            st.sidebar.error("Please enter both username and password")

# Main app functionality (only if logged in)
if st.session_state.get('logged_in', False):
    with st.container():
        st.header("📹 Video Upload & Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a dashcam video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a dashcam video for road safety analysis"
        )

        if uploaded_file is not None:
            # Save uploaded file
            file_path = save_uploaded_file(uploaded_file)
            
            # Display uploaded video
            st.subheader("📺 Uploaded Video")
            st.video(file_path)

            # Analysis section
            if st.button("🔍 Start Road Safety Analysis", type="primary"):
                st.session_state.analysis_in_progress = True
                st.session_state.analysis_result = None
            
            if st.session_state.analysis_in_progress and st.session_state.analysis_result is None:
                st.header("🎯 Analyzing Road Safety...")
                
                try:
                    st.info(f"📹 Starting analysis of: {uploaded_file.name}")
                    st.info(f"📁 File path: {file_path}")
                    
                    if not os.path.exists(file_path):
                        st.error(f"❌ File does not exist: {file_path}")
                        st.session_state.analysis_in_progress = False
                    else:
                        with st.spinner("Processing video for lane detection and safety analysis..."):
                            st.write("🔄 Calling analyze_road_safety function...")
                            analysis_result = analyze_road_safety(file_path)
                            st.write("✅ analyze_road_safety function completed")
                            
                            # Store result in session state
                            st.session_state.analysis_result = analysis_result
                            st.session_state.analysis_in_progress = False
                
                except Exception as e:
                    st.error(f"❌ Critical error during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.analysis_in_progress = False
            
            # Display results if available
            if st.session_state.analysis_result is not None:
                analysis_result = st.session_state.analysis_result
                
                if 'error' in analysis_result:
                    st.error(f"�� Error found: {analysis_result['error']}")
                    st.json(analysis_result)
                elif not analysis_result.get('analysis_complete', False):
                    st.error("❌ Analysis did not complete successfully")
                    st.json(analysis_result)
                else:
                    # Analysis was successful
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    # Safety score display
                    score_data = analysis_result['safety_score']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="🏆 Safety Score", 
                            value=f"{score_data['score']}/100",
                            delta=f"Risk Level: {score_data['risk_level']}"
                        )
                    
                    with col2:
                        st.metric(
                            label="🛣️ Lane Detection Success", 
                            value=f"{score_data['lane_success_rate']}%"
                        )
                    
                    with col3:
                        st.metric(
                            label="🕳️ Potholes Detected", 
                            value=f"{score_data['pothole_count']}"
                        )
                    
                    # Additional metrics row
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        st.metric(
                            label="⚠️ Warning Events", 
                            value=f"{score_data['curve_warnings_percent'] + score_data['deviation_warnings_percent']:.1f}%"
                        )
                    
                    with col5:
                        st.metric(
                            label="🛣️ Pothole Risk Frames", 
                            value=f"{score_data['pothole_frames_percent']:.1f}%"
                        )
                    
                                        # Risk level indicator with pothole consideration
                    pothole_warning = ""
                    if score_data['pothole_count'] > 0:
                        pothole_warning = f" | {score_data['pothole_count']} potholes detected!"
                    
                    if score_data['risk_level'] == 'LOW':
                        st.success(f"✅ **{score_data['risk_level']} RISK** - Good driving conditions detected{pothole_warning}")
                    elif score_data['risk_level'] == 'MEDIUM':
                        st.warning(f"⚠️ **{score_data['risk_level']} RISK** - Some safety concerns detected{pothole_warning}")
                    elif score_data['risk_level'] == 'HIGH':
                        st.error(f"🚨 **{score_data['risk_level']} RISK** - Multiple safety issues detected{pothole_warning}")
                    else:
                        st.error(f"🆘 **{score_data['risk_level']} RISK** - Serious safety concerns!{pothole_warning}")
                    
                    # Detailed metrics
                    with st.expander("📈 Detailed Analysis"):
                        metrics = analysis_result['metrics']
                        st.write(f"**Total Frames Processed:** {metrics['processed_frames']}/{metrics['total_frames']}")
                        
                        # Component scores
                        st.write("### 🏗️ Score Breakdown")
                        col_comp1, col_comp2 = st.columns(2)
                        with col_comp1:
                            st.metric("🛣️ Lane Component (30%)", f"{score_data['lane_component_score']}/100")
                        with col_comp2:
                            st.metric("🕳️ Pothole Component (70%)", f"{score_data['pothole_component_score']}/100")
                        
                        st.write("### 📊 Detection Metrics")
                        st.write(f"**Curve Warnings:** {metrics['curve_warnings']} ({score_data['curve_warnings_percent']}%)")
                        st.write(f"**Center Deviation Warnings:** {metrics['center_deviation_warnings']} ({score_data['deviation_warnings_percent']}%)")
                        st.write(f"**🕳️ Total Potholes Detected:** {score_data['pothole_count']}")
                        st.write(f"**🛣️ Frames with Potholes:** {metrics['pothole_frames']} ({score_data['pothole_frames_percent']}%)")
                        
                        if metrics['average_curvature']:
                            import numpy as np
                            avg_curve = np.mean(metrics['average_curvature'])
                            st.write(f"**Average Road Curvature:** {avg_curve:.0f}m radius")
                        
                        if metrics['average_position_deviation']:
                            avg_dev = np.mean(metrics['average_position_deviation'])
                            st.write(f"**Average Lane Position Deviation:** {avg_dev:.2f}m from center")
                    
                    # Safety recommendations
                    st.header("💡 Safety Recommendations")
                    if score_data['score'] >= 80 and score_data['pothole_count'] == 0:
                        st.info("✅ Excellent driving! Continue maintaining good lane discipline and road awareness.")
                    elif score_data['score'] >= 60:
                        recommendations = ["⚠️ Consider improving lane positioning and reducing sharp maneuvers."]
                        if score_data['pothole_count'] > 0:
                            recommendations.append(f"🕳️ {score_data['pothole_count']} potholes detected - exercise caution and report to authorities.")
                        st.warning(" ".join(recommendations))
                    else:
                        recommendations = ["🚨 Important safety concerns detected. Please review your driving patterns."]
                        if score_data['pothole_count'] > 0:
                            recommendations.append(f"🕳️ {score_data['pothole_count']} potholes detected - avoid these areas and report immediately.")
                        st.error(" ".join(recommendations))
                    
                    # Clear analysis button
                    if st.button("🔄 Clear Analysis Results", type="secondary"):
                        st.session_state.analysis_result = None
                        st.session_state.analysis_in_progress = False
                        st.rerun()

else:
    st.error("🔒 Please login to access the road safety analysis features.")

# Close database connection
conn.close()
