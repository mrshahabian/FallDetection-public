"""
Rule-based fall detection using bounding box analysis
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque


class FallDetector:
    """
    Rule-based fall detector using bounding box aspect ratio analysis.
    
    The detector analyzes the height-to-width ratio of person bounding boxes.
    A fallen person typically has a lower aspect ratio (more horizontal) than
    a standing person (more vertical).
    """
    
    def __init__(self, aspect_ratio_threshold: float = 0.8,
                 consecutive_frames: int = 5,
                 min_confidence: float = 0.5):
        """
        Initialize fall detector.
        
        Args:
            aspect_ratio_threshold: Threshold for aspect ratio (h/w).
                                     If ratio < threshold, person is considered fallen.
            consecutive_frames: Number of consecutive frames below threshold
                                required to trigger fall detection.
            min_confidence: Minimum detection confidence to consider
        """
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.consecutive_frames = consecutive_frames
        self.min_confidence = min_confidence
        
        # History of aspect ratios for temporal smoothing
        self.aspect_ratio_history = deque(maxlen=consecutive_frames)
        self.fall_state = False
    
    def calculate_aspect_ratio(self, bbox: Tuple[float, float, float, float]) -> float:
        """
        Calculate aspect ratio (height/width) of bounding box.
        
        Args:
            bbox: Bounding box as (x, y, width, height)
            
        Returns:
            Aspect ratio (height/width)
        """
        x, y, w, h = bbox
        if w == 0:
            return 0.0
        return h / w
    
    def detect_fall_from_bbox(self, bbox: Tuple[float, float, float, float],
                              confidence: float = 1.0) -> Dict[str, any]:
        """
        Detect fall from a single bounding box.
        
        Args:
            bbox: Bounding box as (x, y, width, height)
            confidence: Detection confidence
            
        Returns:
            Dictionary with detection results:
            - 'fallen': bool, whether person is detected as fallen
            - 'aspect_ratio': float, calculated aspect ratio
            - 'confidence': float, detection confidence
        """
        if confidence < self.min_confidence:
            return {
                'fallen': False,
                'aspect_ratio': 0.0,
                'confidence': confidence,
                'reason': 'low_confidence'
            }
        
        aspect_ratio = self.calculate_aspect_ratio(bbox)
        self.aspect_ratio_history.append(aspect_ratio)
        
        # Check if enough consecutive frames are below threshold
        if len(self.aspect_ratio_history) >= self.consecutive_frames:
            avg_ratio = np.mean(self.aspect_ratio_history)
            fallen = avg_ratio < self.aspect_ratio_threshold
        else:
            # Not enough frames yet, use current ratio
            fallen = aspect_ratio < self.aspect_ratio_threshold
        
        self.fall_state = fallen
        
        return {
            'fallen': fallen,
            'aspect_ratio': aspect_ratio,
            'confidence': confidence,
            'avg_aspect_ratio': np.mean(self.aspect_ratio_history) if self.aspect_ratio_history else aspect_ratio,
            'frames_analyzed': len(self.aspect_ratio_history)
        }
    
    def detect_fall_from_yolo(self, yolo_results, frame_idx: int = 0) -> Dict[str, any]:
        """
        Detect fall from YOLO detection results.
        
        Args:
            yolo_results: YOLO detection results (from ultralytics)
            frame_idx: Frame index (for tracking)
            
        Returns:
            Dictionary with detection results
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Warning: ultralytics not available. Using placeholder.")
            return {'fallen': False, 'error': 'yolo_not_available'}
        
        if len(yolo_results) == 0 or yolo_results[0].boxes is None:
            return {
                'fallen': False,
                'aspect_ratio': 0.0,
                'confidence': 0.0,
                'reason': 'no_detection'
            }
        
        # Get the largest (most confident) person detection
        boxes = yolo_results[0].boxes
        if len(boxes) == 0:
            return {
                'fallen': False,
                'aspect_ratio': 0.0,
                'confidence': 0.0,
                'reason': 'no_boxes'
            }
        
        # Find person class (class 0 in COCO)
        person_boxes = []
        for i in range(len(boxes)):
            if boxes.cls[i] == 0:  # Person class
                box = boxes.xywh[i].cpu().numpy()  # (x_center, y_center, width, height)
                conf = boxes.conf[i].cpu().numpy()
                # Convert to (x, y, width, height) format
                x, y, w, h = box[0] - w/2, box[1] - h/2, box[2], box[3]
                person_boxes.append((x, y, w, h, conf))
        
        if not person_boxes:
            return {
                'fallen': False,
                'aspect_ratio': 0.0,
                'confidence': 0.0,
                'reason': 'no_person'
            }
        
        # Use the most confident detection
        person_boxes.sort(key=lambda x: x[4], reverse=True)
        x, y, w, h, conf = person_boxes[0]
        
        return self.detect_fall_from_bbox((x, y, w, h), confidence=conf)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     yolo_model: str = "yolo11n.pt") -> List[Dict[str, any]]:
        """
        Process entire video for fall detection.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            yolo_model: YOLO model name for person detection
            
        Returns:
            List of detection results for each frame
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Error: ultralytics not installed. Cannot process video.")
            return []
        
        # Load YOLO model
        model = YOLO(yolo_model)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer for output
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results_list = []
        frame_idx = 0
        
        print(f"Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            yolo_results = model(frame, verbose=False)
            
            # Detect fall
            detection = self.detect_fall_from_yolo(yolo_results, frame_idx)
            results_list.append(detection)
            
            # Draw on frame if output requested
            if out is not None:
                # Draw bounding box
                if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
                    boxes = yolo_results[0].boxes
                    for i in range(len(boxes)):
                        if boxes.cls[i] == 0:  # Person
                            box = boxes.xyxy[i].cpu().numpy()
                            x1, y1, x2, y2 = box.astype(int)
                            
                            # Color: red if fallen, green if not
                            color = (0, 0, 255) if detection['fallen'] else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add label
                            label = "FALLEN" if detection['fallen'] else "STANDING"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add aspect ratio info
                info_text = f"Aspect Ratio: {detection.get('aspect_ratio', 0):.2f}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        if out:
            out.release()
            print(f"Annotated video saved to {output_path}")
        
        # Summary
        fallen_frames = sum(1 for r in results_list if r.get('fallen', False))
        total_frames = len(results_list)
        fall_percentage = (fallen_frames / total_frames * 100) if total_frames > 0 else 0
        
        print(f"\nFall Detection Summary:")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames with fall detected: {fallen_frames} ({fall_percentage:.1f}%)")
        
        return results_list
    
    def reset(self):
        """Reset detector state (clear history)."""
        self.aspect_ratio_history.clear()
        self.fall_state = False


def detect_fall_simple(bbox: Tuple[float, float, float, float],
                      threshold: float = 0.8) -> bool:
    """
    Simple fall detection function (single frame, no temporal smoothing).
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        threshold: Aspect ratio threshold
        
    Returns:
        True if fallen, False otherwise
    """
    x, y, w, h = bbox
    if w == 0:
        return False
    aspect_ratio = h / w
    return aspect_ratio < threshold


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        detector = FallDetector(
            aspect_ratio_threshold=0.8,
            consecutive_frames=5
        )
        
        results = detector.process_video(video_path, output_path)
        
        # Save results
        if len(sys.argv) > 3:
            import json
            with open(sys.argv[3], 'w') as f:
                json.dump(results, f, indent=2)

