import streamlit as st
import cv2
import tempfile
import numpy as np
import os
import base64
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# Page config with enhanced title and description
st.set_page_config(page_title="Crowd Detection", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .step-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f2f6;
        color: #757575;
    }
    </style>
    """, unsafe_allow_html=True)

# Application title and description
st.markdown("<h1 class='main-title'>👥 Crowd Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced video analysis for crowd monitoring and management</p>", unsafe_allow_html=True)

# Removed top expander - will be moved to bottom

# Add sidebar for configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-bottom: 1rem; border-bottom: 1px solid #e0e0e0;">
        <h2 style="color: #1E88E5;">⚙️ Detection Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for organization
    setting_tabs = st.tabs(["Detection", "Visualization"])
    
    with setting_tabs[0]:
        # Model confidence
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detection"
        )
        
        # Proximity settings
        proximity = st.slider(
            "Proximity Threshold",
            min_value=20,
            max_value=150,
            value=50,
            help="Maximum distance (pixels) between people to be considered in the same group"
        )
        
        # Min people for crowd
        min_crowd = st.slider(
            "Minimum Crowd Size",
            min_value=2,
            max_value=10,
            value=3,
            help="Minimum number of people to be considered a crowd"
        )
    
    with setting_tabs[1]:
        # Display options with icons
        show_boxes = st.checkbox("🔲 Show Bounding Boxes", value=True)
        show_confidence = st.checkbox("🔢 Show Confidence Scores", value=False)
        show_crowd_areas = st.checkbox("🔴 Highlight Crowd Areas", value=True)
        
    # Add model info box
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 4px solid #1E88E5;">
    <h4 style="margin-top: 0;">Model Information</h4>
    <p style="margin-bottom: 0;">Using YOLOv8n for person detection</p>
    </div>
    """, unsafe_allow_html=True)

# Load YOLOv8 model
@st.cache_resource
def load_model(model_path="yolov8n.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        return None

model = load_model()  # Load the default YOLOv8n model

# Function to group people based on proximity
def group_people_by_distance(centers, eps=50, min_samples=1):
    """
    Groups people into clusters based on spatial proximity using DBSCAN.
    
    Args:
        centers: List of center points [x,y] for each detected person
        eps: Maximum distance between two points to be considered in the same neighborhood
        min_samples: Minimum number of points to form a dense region
    
    Returns:
        List of arrays containing indices of people in each group
    """
    if len(centers) == 0:
        return []
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    
    # Organize people by group
    groups = []
    for label in set(labels):
        if label == -1:  # Skip noise points
            continue
        group_indices = np.where(labels == label)[0]
        groups.append(group_indices)
    
    return groups

# File upload section with better styling
st.markdown("""
<div style="background-color: #f5f7fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h3 style="margin-top: 0;">📤 Upload Video</h3>
    <p>Select a video file for crowd detection analysis</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Create columns for video info
    col1, col2, col3 = st.columns(3)
    
    # Display a progress message in a styled info box
    progress_text = st.empty()
    progress_text.info("⏳ Processing video. This may take a moment...")
    
    # Create a progress bar with better visibility
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
    
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Open video file with OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Display video information in columns with icons
    with col1:
        st.metric("🖼️ Resolution", f"{width}x{height}")
    with col2:
        st.metric("⏱️ Duration", f"{duration:.1f} sec")
    with col3:
        st.metric("🎞️ Frame Rate", f"{fps:.1f} FPS")
    
    # Setup for video output
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # Create a placeholder for the current frame
    stframe = st.empty()
    
    # Process the video frame by frame
    frame_count = 0
    max_crowd_size = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        
        # Run YOLOv8 detection on the frame
        results = model(frame, conf=confidence)[0]
        
        person_boxes = []
        centers = []
        
        # Extract person detections (class 0 is person in COCO dataset)
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            if cls_id == 0:  # Person class
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
                conf = float(box.conf.cpu().numpy())
                
                person_boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
                
                # Calculate center point of each person for clustering
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                centers.append([center_x, center_y])
        
        # Convert to numpy array for processing
        centers_np = np.array(centers)
        
        # Group people into crowds based on proximity
        groups = group_people_by_distance(centers_np, eps=proximity, min_samples=1)
        
        # Identify crowds (groups with at least min_crowd people)
        crowds = [g for g in groups if len(g) >= min_crowd]
        
        # Track statistics
        for crowd in crowds:
            max_crowd_size = max(max_crowd_size, len(crowd))
        
        # Visualize the results on the frame
        if show_boxes:
            for i, box in enumerate(person_boxes):
                x1, y1, x2, y2, conf = box
                
                # Determine color: green for individuals, red for crowd members
                if any(i in crowd for crowd in crowds):
                    color = (0, 0, 255)  # Red for crowd members
                else:
                    color = (0, 255, 0)  # Green for individuals
                
                # Draw rectangle for person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Optionally show confidence score
                if show_confidence:
                    conf_text = f"{conf:.2f}"
                    cv2.putText(frame, conf_text, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Highlight crowd areas with convex hulls
        if show_crowd_areas and len(centers) > 0:
            for crowd in crowds:
                if len(crowd) >= 3:  # Need at least 3 points for a convex hull
                    crowd_points = np.array([centers[i] for i in crowd])
                    hull = cv2.convexHull(crowd_points)
                    cv2.polylines(frame, [hull], True, (0, 0, 255), 2)
                    
                    # Optional: Add semi-transparent fill
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [hull], (0, 0, 255))
                    alpha = 0.2  # Transparency factor
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add statistics overlay
        cv2.putText(frame, f"People detected: {len(person_boxes)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
        cv2.putText(frame, f"Crowds detected: {len(crowds)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Show details for each crowd
        for idx, crowd in enumerate(crowds):
            cv2.putText(frame, f"Crowd {idx+1}: {len(crowd)} people", (20, 120 + 30*idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Write the processed frame to the output video
        out.write(frame)
        
        # Display the processed frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                      use_container_width=True,
                      caption=f"Processing: Frame {frame_count}/{total_frames}")
        
        frame_count += 1
    
    # Clean up resources
    cap.release()
    out.release()
    
    # Update the progress message with styled success box
    progress_text.success("✅ Video processing complete!")
    
    # Display statistics in a clean card format
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4CAF50;">
        <h3 style="margin-top: 0; color: #4CAF50;">📊 Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 3 columns for metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Frames Processed", f"{frame_count}")
    with metric_col2:
        st.metric("Max People Count", f"{max([len(centers) for centers in centers_np]) if len(centers_np) > 0 else 0}")
    with metric_col3:
        st.metric("Largest Crowd", f"{max_crowd_size} people")
    
    # Create download button for the processed video with better styling
    st.markdown("### Download Results")
    with open(out_path, "rb") as file:
        video_bytes = file.read()
        b64 = base64.b64encode(video_bytes).decode()
        download_button = f'''
        <a href="data:video/mp4;base64,{b64}" download="crowd_detected.mp4" 
           style="display:inline-block; padding:0.5em 1.5em; color:white; 
                  background-color:#1E88E5; text-decoration:none; 
                  border-radius:5px; margin:1em 0; font-weight:bold;
                  text-align:center; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            📥 Download Processed Video
        </a>
        '''
        st.markdown(download_button, unsafe_allow_html=True)

    # Clean up the temporary file
    os.unlink(tfile.name)

# Add explanatory section at the bottom in an expander
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    <div class="step-box">
    <h3>Processing Pipeline:</h3>
    <ol>
        <li><strong>Video Input:</strong> The video is processed frame-by-frame using OpenCV</li>
        <li><strong>Person Detection:</strong> YOLOv8 AI identifies people and their bounding boxes</li>
        <li><strong>Crowd Analysis:</strong> DBSCAN algorithm clusters people into groups based on proximity</li>
        <li><strong>Visualization:</strong> Green boxes for individuals, red for crowd members</li>
        <li><strong>Metrics:</strong> Real-time statistics about crowd sizes are displayed</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer

# Footer
st.markdown("""
<footer class="footer">
    <p>This application is designed for crowd monitoring, social distancing analysis, and public space management.</p>
    <p>Developed with YOLOv8, OpenCV, and Streamlit</p>
</footer>
""", unsafe_allow_html=True)