"""
Skeleton visualization module for web application
Draws skeleton keypoints and activity labels on video frames
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# COCO 17 keypoint connections (skeleton structure)
SKELETON_CONNECTIONS = [
    # Head and torso
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head to shoulders
    (1, 2),  # Shoulders
    (1, 5), (2, 6),  # Shoulders to hips
    (5, 6),  # Hips
    # Arms
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    # Legs
    (5, 11), (11, 13),  # Left leg
    (6, 12), (12, 14),  # Right leg
    # Feet
    (13, 15), (14, 16)  # Ankles to feet
]

# Keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    confidence_threshold: float = 0.3,
    scale_factor: float = 1.0
) -> np.ndarray:
    """
    Draw skeleton keypoints and connections on a single frame.
    
    Args:
        frame: Video frame (H, W, 3) BGR format
        keypoints: Keypoints array (17, 2) - x, y coordinates
        confidence_threshold: Minimum confidence to draw keypoint
        scale_factor: Scale factor for line thickness and circle size
        
    Returns:
        Frame with skeleton drawn
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Scale line thickness and circle size based on resolution
    # Make thinner for low-resolution videos
    if w <= 320 or h <= 240:
        # Very small images - very thin lines
        base_line_thickness = 1
        base_circle_radius = 2
        base_circle_outline = 1
    elif w <= 640 or h <= 480:
        # Small images - thin lines
        base_line_thickness = 1.5
        base_circle_radius = 3
        base_circle_outline = 1
    else:
        # Normal to large images
        base_line_thickness = 2
        base_circle_radius = 4
        base_circle_outline = 2
    
    line_thickness = max(1, int(base_line_thickness * scale_factor))
    circle_radius = max(1, int(base_circle_radius * scale_factor))
    circle_outline = max(1, int(base_circle_outline * scale_factor))
    
    # Draw connections (bones)
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            
            # Check if both points are valid (non-zero)
            if (pt1[0] > 0 and pt1[1] > 0 and 
                pt2[0] > 0 and pt2[1] > 0 and
                pt1[0] < w and pt1[1] < h and
                pt2[0] < w and pt2[1] < h):
                cv2.line(frame, pt1, pt2, (0, 255, 0), line_thickness)  # Green lines
    
    # Draw keypoints (joints)
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0 and x < w and y < h:
            cv2.circle(frame, (int(x), int(y)), circle_radius, (0, 0, 255), -1)  # Red circles
            cv2.circle(frame, (int(x), int(y)), circle_radius + circle_outline, (255, 255, 255), circle_outline)  # White outline
    
    return frame


def draw_activity_labels(
    frame: np.ndarray,
    activities: Dict[str, float],
    top_n: int = 3,
    scale_factor: float = 1.0,
    window_info: Optional[Dict] = None
) -> np.ndarray:
    """
    Draw activity labels and probabilities on frame.
    
    Args:
        frame: Video frame
        activities: Dictionary mapping activity names to probabilities
        top_n: Number of top activities to display
        scale_factor: Scale factor for text and UI elements
        window_info: Dictionary with 'window_num', 'frame_num', 'window_start', 'window_end'
        
    Returns:
        Frame with labels drawn
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Scale UI elements based on video resolution
    # Compact sizes to prevent covering entire frame
    if w <= 320 or h <= 240:
        # Very small images - compact sizes but readable
        base_text_height = 15
        base_padding = 4
        base_font_scale = 0.4  # Slightly larger for better visibility
        base_line_thickness = 1
        base_box_width = 140
        min_box_width = 100
        max_box_width_ratio = 0.4  # Max 40% of image width
    elif w <= 640 or h <= 480:
        # Small images - compact sizes but readable
        base_text_height = 17
        base_padding = 5
        base_font_scale = 0.45  # Slightly larger for better visibility
        base_line_thickness = 1
        base_box_width = 180
        min_box_width = 120
        max_box_width_ratio = 0.35  # Max 35% of image width
    else:
        # Normal to large images - compact sizes but readable
        base_text_height = 22
        base_padding = 6
        base_font_scale = 0.55  # Slightly larger for better visibility
        base_line_thickness = 1
        base_box_width = 220
        min_box_width = 150
        max_box_width_ratio = 0.3  # Max 30% of image width
    
    text_height = max(13, int(base_text_height * scale_factor))  # Slightly larger minimum
    padding = max(3, int(base_padding * scale_factor))
    font_scale = max(0.35, base_font_scale * scale_factor)  # Slightly larger minimum for readability
    line_thickness = max(1, int(base_line_thickness * scale_factor))
    
    # Calculate box width - ensure it doesn't exceed max ratio of image width
    calculated_box_width = int(base_box_width * scale_factor)
    max_box_width = int(w * max_box_width_ratio)
    box_width = max(min_box_width, min(calculated_box_width, max_box_width))
    
    # Ensure box doesn't overflow horizontally
    if padding + box_width > w:
        box_width = w - (padding * 2)
        box_width = max(min_box_width, box_width)
    
    # Sort activities by probability
    sorted_activities = sorted(
        activities.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    # Calculate box height (including window info if provided)
    num_items = len(sorted_activities)
    if window_info:
        num_items += 1  # Add space for window info
    
    rect_height = (text_height + 3) * num_items + padding * 2  # Reduced spacing between items
    
    # Ensure box doesn't overflow vertically - more restrictive
    max_box_height_ratio = 0.3 if (w <= 320 or h <= 240) else 0.35  # Reduced coverage
    max_box_height = int(h * max_box_height_ratio)
    if rect_height > max_box_height:
        # Reduce text height and padding proportionally
        scale_down = max_box_height / rect_height
        min_text_height = 10 if (w <= 320 or h <= 240) else 12
        min_padding = 2 if (w <= 320 or h <= 240) else 3
        text_height = max(min_text_height, int(text_height * scale_down))
        padding = max(min_padding, int(padding * scale_down))
        rect_height = (text_height + 3) * num_items + padding * 2  # Reduced spacing
        # Also reduce font scale
        font_scale = max(0.25, font_scale * scale_down)
    
    # Create overlay for semi-transparent background
    overlay = frame.copy()
    
    # # Draw semi-transparent background rectangle (dark with transparency)
    # cv2.rectangle(
    #     overlay,
    #     (padding, padding),
    #     (padding + box_width, padding + rect_height),
    #     (0, 0, 0),  # Black color
    #     -1
    # )
    
    # Blend overlay with original frame - increased opacity for better text contrast
    alpha = 0.8 if (w <= 320 or h <= 240) else 0.85  # More opaque for better text visibility
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Border removed - no white outline rectangle
    
    y_offset = padding + text_height
    
    # Draw window information if provided
    if window_info:
        window_num = window_info.get('window_num', 0)
        frame_num = window_info.get('frame_num', 0)
        window_start = window_info.get('window_start', 0)
        window_end = window_info.get('window_end', 0)
        
        # Shorter text format to fit in narrower box
        window_text = f"W:{window_num} F:{frame_num} ({window_start}-{window_end})"
        # Check text width and truncate if needed
        (text_width, _), _ = cv2.getTextSize(window_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
        max_text_width = box_width - (padding * 4)
        if text_width > max_text_width:
            # Further shorten if needed
            window_text = f"W:{window_num} F:{frame_num}"
            (text_width, _), _ = cv2.getTextSize(window_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
            if text_width > max_text_width:
                window_text = f"W:{window_num}"
        
        # Draw text with black outline for better visibility
        # Draw black outline first
        cv2.putText(
            frame,
            window_text,
            (padding * 2, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black outline
            line_thickness + 2,  # Thicker outline
            cv2.LINE_AA
        )
        # Draw main text on top
        cv2.putText(
            frame,
            window_text,
            (padding * 2, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 0),  # Bright yellow color
            line_thickness,
            cv2.LINE_AA
        )
        y_offset += text_height + 3  # Reduced spacing
    
    # Draw activity labels
    for i, (activity, prob) in enumerate(sorted_activities):
        # Abbreviate activity names to make text narrower
        activity_abbrev = {
            'walking': 'Walk',
            'jogging': 'Jog',
            'running': 'Run',
            'boxing': 'Box',
            'handwaving': 'Wave',
            'handclapping': 'Clap'
        }
        activity_text = activity_abbrev.get(activity.lower(), activity.replace('_', ' ').title()[:6])
        prob_text = f"{prob * 100:.0f}%"  # No decimal for narrower text
        
        # Color based on probability - use brighter colors
        if prob > 0.5:
            color = (0, 255, 0)  # Bright green for high confidence
        elif prob > 0.3:
            color = (0, 255, 255)  # Bright yellow/cyan for medium confidence
        else:
            color = (0, 200, 255)  # Bright orange for low confidence
        
        # Shorter text format to fit in narrower box
        text = f"{i+1}. {activity_text}: {prob_text}"
        
        # Check text width and truncate if needed
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
        max_text_width = box_width - (padding * 4)
        if text_width > max_text_width:
            # Try shorter format
            text = f"{i+1}. {activity_text} {prob_text}"
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
            if text_width > max_text_width:
                # Just number and percentage
                text = f"{i+1}. {prob_text}"
        
        # Draw text with black outline for better visibility
        # Draw black outline first
        cv2.putText(
            frame,
            text,
            (padding * 2, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black outline
            line_thickness + 2,  # Thicker outline
            cv2.LINE_AA
        )
        # Draw main text on top
        cv2.putText(
            frame,
            text,
            (padding * 2, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            line_thickness,
            cv2.LINE_AA
        )
        
        # Draw progress bar
        bar_width = box_width - padding * 4
        bar_height = max(3, int(5 * scale_factor))  # Smaller bar
        bar_x = padding * 2
        bar_y = y_offset + int(5 * scale_factor)
        
        # Background bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Filled bar
        filled_width = int(bar_width * prob)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + filled_width, bar_y + bar_height),
            color,
            -1
        )
        
        y_offset += text_height + 3  # Reduced spacing
    
    return frame


def create_annotated_video(
    video_path: str,
    keypoints: np.ndarray,
    activities: Dict[str, float],
    output_path: str,
    confidence_threshold: float = 0.3,
    per_window_probs: Optional[List[Dict]] = None,
    clip_length: int = 32,
    overlap: float = 0.5,
    step: int = 16
) -> bool:
    """
    Create an annotated video with skeleton and activity labels.
    
    Args:
        video_path: Path to input video
        keypoints: Keypoints array (T, 17, 2) where T is number of frames
        activities: Dictionary mapping activity names to probabilities
        output_path: Path to save annotated video
        confidence_threshold: Minimum confidence to draw keypoint
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25  # Default FPS
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate scale factor based on video resolution
        # Use smaller reference resolution for better scaling on low-res images
        # For very small images, use more aggressive scaling
        if width <= 320 or height <= 240:
            # Very small images - use 160x120 as reference
            ref_width, ref_height = 160, 120
            scale_factor = min(width / ref_width, height / ref_height)
            scale_factor = max(0.2, min(1.0, scale_factor))  # Very aggressive for tiny images
        elif width <= 640 or height <= 480:
            # Small images - use 320x240 as reference
            ref_width, ref_height = 320, 240
            scale_factor = min(width / ref_width, height / ref_height)
            scale_factor = max(0.3, min(1.2, scale_factor))
        else:
            # Normal to large images
            ref_width, ref_height = 640, 480
            scale_factor = min(width / ref_width, height / ref_height)
            scale_factor = max(0.5, min(1.5, scale_factor))
        
        # Determine codec based on output file extension
        file_ext = os.path.splitext(output_path)[1].lower()
        
        if file_ext == '.webm':
            # WebM format - try VP8 or VP9 codec
            codecs_to_try = [
                ('VP80', 'VP8'),   # VP8 for WebM
                ('VP90', 'VP9'),   # VP9 for WebM (better quality)
            ]
        else:
            # MP4 format - try H.264
            codecs_to_try = [
                ('avc1', 'H.264/AVC1'),  # Best browser support for MP4
                ('mp4v', 'MPEG-4'),      # Fallback
            ]
        
        out = None
        used_codec = None
        
        for fourcc_str, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    used_codec = codec_name
                    print(f"Using codec: {codec_name} ({fourcc_str}) for {file_ext}")
                    break
                else:
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                print(f"Failed to use codec {codec_name}: {e}")
                if out:
                    out.release()
                out = None
        
        if not out or not out.isOpened():
            print(f"Error: Could not create video writer for {output_path} with any codec")
            print(f"Tried codecs: {[c[1] for c in codecs_to_try]}")
            cap.release()
            return False
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Creating annotated video: {total_frames} frames...")
        print(f"Scale factor: {scale_factor:.2f} (resolution: {width}x{height})")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get keypoints for this frame
            if frame_idx < len(keypoints):
                frame_keypoints = keypoints[frame_idx]
            else:
                # Use last frame's keypoints if video has more frames
                frame_keypoints = keypoints[-1] if len(keypoints) > 0 else np.zeros((17, 2))
            
            # Determine which window(s) this frame belongs to
            window_info = None
            frame_activities = activities  # Default to averaged activities
            
            if per_window_probs is not None:
                # Find the window(s) that contain this frame
                # With overlap, a frame can belong to multiple windows
                # We'll use the window that contains the frame's center or the most recent window
                for window_data in reversed(per_window_probs):  # Check from most recent
                    window_start = window_data['window_start']
                    window_end = window_data['window_end']
                    
                    if window_start <= frame_idx <= window_end:
                        window_info = {
                            'window_num': window_data['window_num'],
                            'frame_num': frame_idx,
                            'window_start': window_start,
                            'window_end': window_end
                        }
                        frame_activities = window_data['probabilities']
                        break  # Use the most recent window that contains this frame
            
            # Draw skeleton
            frame = draw_skeleton_on_frame(frame, frame_keypoints, confidence_threshold, scale_factor)
            
            # Draw activity labels with window info
            frame = draw_activity_labels(frame, frame_activities, top_n=3, scale_factor=scale_factor, window_info=window_info)
            
            # Write frame
            out.write(frame)
            
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        print(f"Annotated video saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating annotated video: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_skeleton_with_activities(
    video_path: str,
    activities: Dict[str, float],
    output_dir: str,
    model_name: str = "yolo11n-pose.pt",
    confidence: float = 0.25,
    per_window_probs: Optional[List[Dict]] = None,
    clip_length: int = 32,
    overlap: float = 0.5,
    step: int = 16,
    keypoints: Optional[np.ndarray] = None
) -> Optional[str]:
    """
    Extract skeleton from video and create annotated video with activities.
    
    Args:
        video_path: Path to input video
        activities: Dictionary mapping activity names to averaged probabilities
        output_dir: Directory to save annotated video
        model_name: YOLOv11 model name (only used if keypoints not provided)
        confidence: Detection confidence threshold (only used if keypoints not provided)
        per_window_probs: List of dictionaries with per-window probabilities and window info
        clip_length: Length of each clip/window in frames
        overlap: Overlap ratio between windows
        step: Step size between windows (calculated from clip_length and overlap)
        keypoints: Optional pre-extracted keypoints array (T, 17, 2) to avoid re-extraction
        
    Returns:
        Path to annotated video if successful, None otherwise
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Cannot extract skeleton.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename - use WebM for better browser compatibility
    input_filename = Path(video_path).stem
    output_filename = f"{input_filename}_annotated.webm"
    output_path = os.path.join(output_dir, output_filename)
    
    # Use provided keypoints or extract from video
    if keypoints is not None:
        print(f"Using pre-extracted keypoints: {keypoints.shape}")
        keypoints_array = keypoints
    else:
        # Extract keypoints from video
        print(f"Extracting skeleton from {video_path}...")
        model = YOLO(model_name)
        
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv11-pose inference
            results = model(frame, conf=confidence, verbose=False)
            
            # Extract keypoints for the first (largest) person detected
            if (len(results) > 0 and 
                results[0].keypoints is not None and 
                len(results[0].keypoints.data) > 0):
                kpts = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
                keypoints_xy = kpts[:, :2]  # (17, 2)
            else:
                keypoints_xy = np.zeros((17, 2))
            
            keypoints_list.append(keypoints_xy)
        
        cap.release()
        
        if len(keypoints_list) == 0:
            print("Error: No frames extracted from video")
            return None
        
        keypoints_array = np.array(keypoints_list)  # (T, 17, 2)
        print(f"Extracted {len(keypoints_array)} frames with skeleton keypoints")
    
    # Create annotated video with per-window probabilities
    success = create_annotated_video(
        video_path=video_path,
        keypoints=keypoints_array,
        activities=activities,
        output_path=output_path,
        per_window_probs=per_window_probs,
        clip_length=clip_length,
        overlap=overlap,
        step=step
    )
    
    if success:
        return output_path
    else:
        return None

