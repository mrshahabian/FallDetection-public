"""
Video conversion utilities for browser-compatible playback
"""

import os
import cv2
import subprocess
import shutil
from typing import Optional, Tuple


def needs_conversion(filename: str) -> bool:
    """
    Check if video needs conversion for browser playback.
    
    Browser-compatible formats: MP4 (H.264), WebM (VP8/VP9), OGG
    Note: MP4 with mp4v codec may not work, so we convert AVI/MOV/MKV to WebM
    """
    browser_compatible = {'mp4', 'webm', 'ogg'}
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    # Always convert non-browser formats, and also convert MP4 if it might be mp4v
    return ext not in browser_compatible


def get_converted_filename(filename: str) -> str:
    """Get the converted filename (same name but .webm extension for better browser support)"""
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    return f"{base_name}_converted.webm"


def convert_video_opencv(input_path: str, output_path: str) -> bool:
    """
    Convert video to WebM using OpenCV (better browser compatibility than MP4 with mp4v).
    
    Args:
        input_path: Path to input video
        output_path: Path to output WebM video
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure valid FPS
        if fps <= 0:
            fps = 25.0
        
        # Determine output format from extension
        output_ext = output_path.rsplit('.', 1)[-1].lower() if '.' in output_path else 'webm'
        
        # Try different codecs based on output format
        if output_ext == 'webm':
            # WebM format - try VP8/VP9 codecs
            codecs = ['VP80', 'VP90', 'XVID', 'mp4v']
        else:
            # MP4 format - try H.264 variants first
            codecs = ['H264', 'avc1', 'X264', 'mp4v', 'XVID']
        
        writer = None
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    print(f"Using codec: {codec} for {output_ext}")
                    break
                if writer:
                    writer.release()
                writer = None
            except Exception as e:
                print(f"Codec {codec} failed: {e}")
                if writer:
                    writer.release()
                writer = None
                continue
        
        if writer is None or not writer.isOpened():
            print("Error: Could not create video writer with any codec")
            cap.release()
            return False
        
        # Convert frame by frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            writer.write(frame)
            frame_count += 1
            
            # Progress indicator for long videos
            if frame_count % 100 == 0:
                print(f"Converted {frame_count}/{total_frames} frames...")
        
        cap.release()
        writer.release()
        
        # Verify output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Conversion complete: {frame_count} frames written to {output_path}")
            return True
        else:
            print("Error: Output file is empty or doesn't exist")
            return False
            
    except Exception as e:
        print(f"Error converting video: {e}")
        return False


def convert_video_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Convert video to MP4 using ffmpeg (if available).
    
    This produces better quality and browser compatibility.
    
    Args:
        input_path: Path to input video
        output_path: Path to output MP4 video
        
    Returns:
        True if conversion successful, False otherwise
    """
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        return False
    
    try:
        # Convert to H.264 MP4 with AAC audio
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'fast',  # Encoding speed
            '-crf', '23',  # Quality (lower = better, 23 is default)
            '-c:a', 'aac',  # AAC audio codec
            '-movflags', '+faststart',  # Enable streaming
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"FFmpeg conversion complete: {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FFmpeg conversion timed out")
        return False
    except Exception as e:
        print(f"FFmpeg conversion error: {e}")
        return False


def convert_video(input_path: str, output_path: str) -> bool:
    """
    Convert video to browser-compatible format (WebM or MP4).
    
    Tries ffmpeg first (better quality), falls back to OpenCV.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video (WebM or MP4)
        
    Returns:
        True if conversion successful, False otherwise
    """
    # Try ffmpeg first (better quality and compatibility)
    if convert_video_ffmpeg(input_path, output_path):
        return True
    
    # Fall back to OpenCV
    print("FFmpeg not available, using OpenCV for conversion...")
    return convert_video_opencv(input_path, output_path)


def ensure_browser_compatible(upload_folder: str, filename: str) -> Tuple[str, bool]:
    """
    Ensure video is browser-compatible, converting if necessary.
    
    Args:
        upload_folder: Path to upload folder
        filename: Original filename
        
    Returns:
        Tuple of (playback_filename, was_converted)
        - playback_filename: Filename to use for browser playback
        - was_converted: True if video was converted
    """
    if not needs_conversion(filename):
        return filename, False
    
    input_path = os.path.join(upload_folder, filename)
    converted_filename = get_converted_filename(filename)
    output_path = os.path.join(upload_folder, converted_filename)
    
    # Check if already converted
    if os.path.exists(output_path):
        print(f"Using existing converted file: {converted_filename}")
        return converted_filename, True
    
    # Convert video
    print(f"Converting {filename} to browser-compatible format...")
    if convert_video(input_path, output_path):
        return converted_filename, True
    else:
        # Conversion failed, return original (fallback message will show)
        return filename, False

