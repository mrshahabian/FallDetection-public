"""
Flask web application for activity recognition
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback

from config import (
    UPLOAD_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS,
    allowed_file, CLASS_NAMES, FEW_SHOT_VLM_CHECKPOINT
)
from model_inference import predict
from skeleton_visualizer import visualize_skeleton_with_activities
from simple_fall_detection import predict_simple_fall_detection

# Import VLM modules (may fail if VLM not available)
try:
    from vlm_video_description import predict_vlm_video_description
    from vlm_fall_detection_improved import predict_vlm_fall_detection_improved
    VLM_VIDEO_DESC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VLM modules not available: {e}")
    VLM_VIDEO_DESC_AVAILABLE = False
    predict_vlm_video_description = None
    predict_vlm_fall_detection_improved = None

try:
    from vlm_few_shot_fall_detection import predict_vlm_few_shot_fall_detection
    VLM_FEW_SHOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Few-shot VLM module not available: {e}")
    VLM_FEW_SHOT_AVAILABLE = False
    predict_vlm_few_shot_fall_detection = None

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Available models (VLM is separate, only for fall detection)
# KTH 6-activity models: walking, jogging, running, boxing, handwaving, handclapping
AVAILABLE_MODELS = [
    '3dcnn_simple',
    '3dcnn_deep',
    '2dcnn_resnet',
    '2dcnn_lenet',
    '2dcnn',  # Backward compatibility (defaults to ResNet)
    'vit',
    'stgcn',
    'tcnt',
    'simple_fall',  # Simple fall detection based on pose

    # Supervised fall/no-fall models: same architectures, trained directly on
    # binary Fall vs. No-Fall labels (evaluation/train_cnn_models.py), not KTH activities
    '2dcnn_resnet_fall',
    '3dcnn_simple_fall',
    'vit_fall',
]

# VLM modes (separate from activity recognition)
# 'zero_shot': CLIP + balanced fall/non-fall prompt bank (evaluation/prompts.py), no training
# 'few_shot': CLIP + trained linear classifier (evaluation/checkpoints/few_shot_vlm_classifier.pt)
VLM_MODES = ['zero_shot', 'few_shot', 'video_description']


@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html', models=AVAILABLE_MODELS, classes=CLASS_NAMES)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert video for browser playback if needed
        from video_converter import ensure_browser_compatible, needs_conversion
        
        playback_filename = filename
        was_converted = False
        
        if needs_conversion(filename):
            print(f"Video needs conversion for browser playback: {filename}")
            playback_filename, was_converted = ensure_browser_compatible(
                app.config['UPLOAD_FOLDER'], filename
            )
        
        return jsonify({
            'success': True,
            'filename': filename,  # Original filename for processing
            'playback_filename': playback_filename,  # For browser playback
            'was_converted': was_converted,
            'message': 'File uploaded successfully' + (' (converted for playback)' if was_converted else '')
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/video/<filename>')
def serve_video(filename):
    """Serve uploaded video file with proper MIME type"""
    # Determine MIME type based on file extension
    mime_types = {
        'mp4': 'video/mp4',
        'webm': 'video/webm',
        'ogg': 'video/ogg',
        'avi': 'video/x-msvideo',
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska'
    }
    
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    # Check if this is a converted file or annotated video
    if '_converted' in filename or '_annotated' in filename:
        # Converted/annotated files - determine by extension
        if filename.endswith('.webm'):
            mimetype = 'video/webm'
        elif filename.endswith('.mp4'):
            mimetype = 'video/mp4'
        else:
            mimetype = mime_types.get(file_ext, 'video/webm')
    else:
        mimetype = mime_types.get(file_ext, 'video/mp4')
    
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mimetype)
    # Add CORS headers if needed
    response.headers['Accept-Ranges'] = 'bytes'
    return response


@app.route('/predict/<model_type>', methods=['POST'])
def predict_activity(model_type):
    """Run inference with specified model"""
    try:
        # Handle simple fall detection separately
        if model_type == 'simple_fall':
            return predict_simple_fall()
        
        # Validate model type
        if model_type not in AVAILABLE_MODELS:
            return jsonify({
                'error': f'Invalid model type. Available: {", ".join(AVAILABLE_MODELS)}'
            }), 400
        
        # Get filename from request
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Check if anomaly detection is requested
        use_anomaly_detection = data.get('use_anomaly_detection', False)
        
        # Run prediction with per-window probabilities
        print(f"Running prediction with {model_type} on {filename}...")
        if use_anomaly_detection:
            print("  Anomaly detection mode enabled")
        from model_inference import predict_skeleton_model
        prediction_result = predict_skeleton_model(
            filepath, model_type, 
            return_per_window=True,
            use_anomaly_detection=use_anomaly_detection
        )
        
        # Get averaged probabilities for response
        probabilities = prediction_result['averaged']
        per_window_data = prediction_result.get('per_window', [])
        clip_length = prediction_result.get('clip_length', 32)
        overlap = prediction_result.get('overlap', 0.5)
        step = prediction_result.get('step', 16)
        keypoints = prediction_result.get('keypoints', None)  # Get pre-extracted keypoints
        
        # Get anomaly detection results if available
        anomaly_detection = prediction_result.get('anomaly_detection')
        
        # Find predicted class (highest probability)
        predicted_class = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = probabilities[predicted_class]
        
        # Create annotated video with skeleton and activity labels
        annotated_video_path = None
        annotated_video_filename = None
        try:
            print(f"Creating annotated video for {filename}...")
            annotated_video_path = visualize_skeleton_with_activities(
                video_path=filepath,
                activities=probabilities,
                output_dir=app.config['UPLOAD_FOLDER'],
                per_window_probs=per_window_data,
                clip_length=clip_length,
                overlap=overlap,
                step=step,
                keypoints=keypoints  # Pass pre-extracted keypoints to avoid re-extraction
            )
            
            if annotated_video_path:
                annotated_video_filename = os.path.basename(annotated_video_path)
                print(f"Annotated video ready: {annotated_video_filename}")
            else:
                print("Warning: Failed to create annotated video")
        except Exception as e:
            print(f"Warning: Could not create annotated video: {e}")
            traceback.print_exc()
        
        response_data = {
            'success': True,
            'model': model_type,
            'probabilities': probabilities,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        
        # Add anomaly detection results if available
        if anomaly_detection is not None:
            response_data['anomaly_detection'] = {
                'is_fall': bool(anomaly_detection['is_anomaly']),
                'anomaly_score': float(anomaly_detection['anomaly_score']),
                'threshold': float(anomaly_detection['threshold']),
                'confidence': float(anomaly_detection['confidence']),
                'entropy': float(anomaly_detection['entropy']),
                'per_window': anomaly_detection.get('per_window', [])
            }
        
        # Add annotated video filename if created
        if annotated_video_filename:
            response_data['annotated_video'] = annotated_video_filename
        
        return jsonify(response_data)
    
    except FileNotFoundError as e:
        return jsonify({
            'error': f'Model checkpoint not found: {str(e)}'
        }), 404
    
    except ValueError as e:
        return jsonify({
            'error': f'Processing error: {str(e)}'
        }), 400
    
    except RuntimeError as e:
        return jsonify({
            'error': f'Runtime error: {str(e)}'
        }), 500
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


def predict_simple_fall():
    """Handle simple fall detection prediction"""
    try:
        if 'filename' not in request.json:
            return jsonify({'error': 'No filename provided'}), 400
        
        data = request.json
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Run simple fall detection with per-window results
        print(f"Running simple fall detection on {filename}...")
        fall_result = predict_simple_fall_detection(
            filepath, 
            clip_length=32, 
            overlap=0.5, 
            return_per_window=True
        )
        
        # Get per-window data
        per_window_data = fall_result.get('per_window', [])
        clip_length = fall_result.get('clip_length', 32)
        overlap = fall_result.get('overlap', 0.5)
        step = fall_result.get('step', 16)
        keypoints = fall_result.get('keypoints', None)
        
        # Create annotated video with fall detection results
        annotated_video_path = None
        annotated_video_filename = None
        try:
            print(f"Creating annotated video with fall detection for {filename}...")
            
            # Convert fall detection to activity-like format for visualization
            activities = {
                'Fall': fall_result['fall_probability'],
                'No Fall': 1.0 - fall_result['fall_probability']
            }
            
            # Convert per-window fall detection to per-window probabilities format
            per_window_probs = []
            for window_data in per_window_data:
                window_probs = {
                    'Fall': window_data['fall_probability'],
                    'No Fall': 1.0 - window_data['fall_probability']
                }
                per_window_probs.append({
                    'window_num': window_data['window_num'],
                    'window_start': window_data['window_start'],
                    'window_end': window_data['window_end'],
                    'probabilities': window_probs
                })
            
            annotated_video_path = visualize_skeleton_with_activities(
                video_path=filepath,
                activities=activities,
                output_dir=app.config['UPLOAD_FOLDER'],
                per_window_probs=per_window_probs,
                clip_length=clip_length,
                overlap=overlap,
                step=step,
                keypoints=keypoints
            )
            
            if annotated_video_path:
                annotated_video_filename = os.path.basename(annotated_video_path)
                print(f"Annotated video ready: {annotated_video_filename}")
        except Exception as e:
            print(f"Warning: Could not create annotated video: {e}")
            traceback.print_exc()
        
        response_data = {
            'success': True,
            'model': 'simple_fall',
            'is_fall': fall_result['is_fall'],
            'fall_probability': fall_result['fall_probability'],
            'num_windows': fall_result['num_windows'],
            'windows_with_fall': fall_result['windows_with_fall'],
            'per_window': per_window_data
        }
        
        # Add annotated video filename if created
        if annotated_video_filename:
            response_data['annotated_video'] = annotated_video_filename
        
        return jsonify(response_data)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in simple fall detection: {error_trace}")
        return jsonify({
            'error': f'Fall detection failed: {str(e)}'
        }), 500


def _build_vlm_fall_detection_response(mode, filename, filepath, fall_result):
    """
    Shared post-processing for VLM fall-detection modes (zero_shot / few_shot):
    builds the annotated video and the JSON response from a predict_vlm_*_fall_detection result.
    """
    # Get per-window data
    per_window_data = fall_result.get('per_window', [])
    clip_length = fall_result.get('clip_length', 32)
    overlap = fall_result.get('overlap', 0.5)
    step = fall_result.get('step', 16)
    keypoints = fall_result.get('keypoints', None)

    # Create annotated video with fall detection results
    annotated_video_path = None
    annotated_video_filename = None
    try:
        print(f"Creating annotated video with VLM {mode} fall detection for {filename}...")

        # Convert fall detection to activity-like format for visualization
        activities = {
            'Fall': fall_result['fall_probability'],
            'No Fall': 1.0 - fall_result['fall_probability']
        }

        # Convert per-window fall detection to per-window probabilities format
        per_window_probs = []
        for window_data in per_window_data:
            window_probs = {
                'Fall': window_data['fall_probability'],
                'No Fall': 1.0 - window_data['fall_probability']
            }
            per_window_probs.append({
                'window_num': window_data['window_num'],
                'window_start': window_data['window_start'],
                'window_end': window_data['window_end'],
                'probabilities': window_probs
            })

        annotated_video_path = visualize_skeleton_with_activities(
            video_path=filepath,
            activities=activities,
            output_dir=app.config['UPLOAD_FOLDER'],
            per_window_probs=per_window_probs,
            clip_length=clip_length,
            overlap=overlap,
            step=step,
            keypoints=keypoints
        )

        if annotated_video_path:
            annotated_video_filename = os.path.basename(annotated_video_path)
            print(f"Annotated video ready: {annotated_video_filename}")
    except Exception as e:
        print(f"Warning: Could not create annotated video: {e}")
        traceback.print_exc()

    response_data = {
        'success': True,
        'mode': mode,
        'is_fall': fall_result['is_fall'],
        'fall_probability': fall_result['fall_probability'],
        'num_windows': fall_result['num_windows'],
        'windows_with_fall': fall_result['windows_with_fall'],
        'per_window': per_window_data,
        'annotated_video': annotated_video_filename if annotated_video_filename else None
    }
    if 'num_prompts' in fall_result:
        response_data['num_prompts'] = fall_result['num_prompts']

    return jsonify(response_data)


@app.route('/vlm/<mode>', methods=['POST'])
def vlm_prediction(mode):
    """Run VLM inference (zero-shot / few-shot fall detection, or video description)"""
    try:
        # Validate VLM mode
        if mode not in VLM_MODES:
            return jsonify({
                'error': f'Invalid VLM mode. Available: {", ".join(VLM_MODES)}'
            }), 400

        # Get filename from request
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400

        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        # Run prediction based on mode
        print(f"Running VLM {mode} on {filename}...")
        if mode == 'zero_shot':
            if not VLM_VIDEO_DESC_AVAILABLE or predict_vlm_fall_detection_improved is None:
                return jsonify({
                    'error': 'VLM zero-shot fall detection is not available. Please ensure transformers library is installed and CLIP model can be loaded.'
                }), 503

            fall_result = predict_vlm_fall_detection_improved(
                filepath,
                clip_length=32,
                overlap=0.5,
                return_per_window=True
            )
            return _build_vlm_fall_detection_response(mode, filename, filepath, fall_result)

        elif mode == 'few_shot':
            if not VLM_FEW_SHOT_AVAILABLE or predict_vlm_few_shot_fall_detection is None:
                return jsonify({
                    'error': 'VLM few-shot fall detection is not available. Please ensure transformers library is installed and CLIP model can be loaded.'
                }), 503

            fall_result = predict_vlm_few_shot_fall_detection(
                filepath,
                clip_length=32,
                overlap=0.5,
                classifier_path=FEW_SHOT_VLM_CHECKPOINT,
                return_per_window=True
            )
            return _build_vlm_fall_detection_response(mode, filename, filepath, fall_result)

        elif mode == 'video_description':
            # New window-based video description
            if not VLM_VIDEO_DESC_AVAILABLE or predict_vlm_video_description is None:
                return jsonify({
                    'error': 'VLM video description is not available. Please ensure transformers library is installed and CLIP model can be loaded.'
                }), 503
            
            description_result = predict_vlm_video_description(
                filepath,
                clip_length=32,
                overlap=0.5,
                return_per_window=True
            )
            return jsonify({
                'success': True,
                'mode': mode,
                'description': description_result['description'],
                'num_windows': description_result['num_windows'],
                'per_window': description_result.get('per_window', []),
                'fps': description_result.get('fps', 30.0),
                'total_frames': description_result.get('total_frames', 0)
            })
    
    except RuntimeError as e:
        error_msg = str(e)
        if 'not available' in error_msg.lower():
            return jsonify({
                'error': 'VLM model is not available. Please ensure transformers library is installed and CLIP model can be loaded.'
            }), 503
        return jsonify({
            'error': f'Runtime error: {error_msg}'
        }), 500
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in VLM prediction: {error_trace}")
        return jsonify({
            'error': f'VLM prediction failed: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
    app.run(debug=True, host='0.0.0.0', port=5000)

