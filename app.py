from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Strict detection parameters
HUMAN_CONFIDENCE = 0.85  # Only accept very confident human detections
EDGE_THRESHOLDS = (100, 200)  # Canny edge detection thresholds

def strict_human_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, []
    
    height, width = img.shape[:2]
    
    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = classes[class_id]
            
            # Only accept human detections with high confidence
            if label == 'person' and confidence >= HUMAN_CONFIDENCE:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                
                # Additional human verification - aspect ratio check
                aspect_ratio = h / w
                if 0.3 < aspect_ratio < 0.8:  # Human-like proportions
                    detections.append({
                        'label': label,
                        'confidence': float(confidence),
                        'box': [x, y, int(w), int(h)]
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, f"Human {confidence:.2f}", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img, detections

def edge_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Apply Canny edge detection
    edges = cv2.Canny(img, *EDGE_THRESHOLDS)
    
    # Convert to 3-channel BGR for consistency
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    operation = request.form.get('operation', 'detect_humans')
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    file.save(input_path)
    
    try:
        if operation == 'edge':
            processed = edge_detection(input_path)
            if processed is None:
                return jsonify({'error': 'Edge detection failed'}), 400
            cv2.imwrite(output_path, processed)
            return jsonify({
                'status': 'success',
                'processed': filename,
                'detections': [{'label': 'edges', 'confidence': 1.0}]
            })
        
        elif operation == 'detect_humans':
            processed_img, detections = strict_human_detection(input_path)
            if processed_img is None:
                return jsonify({'error': 'Human detection failed'}), 400
            cv2.imwrite(output_path, processed_img)
            return jsonify({
                'status': 'success',
                'processed': filename,
                'detections': detections
            })
        
        else:
            return jsonify({'error': 'Invalid operation'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def processed_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)