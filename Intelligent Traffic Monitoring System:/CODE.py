from flask import Flask, Response, send_file, stream_with_context
import cv2
import numpy as np

app = Flask(__name__)

# Define file paths
weights_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\intelligent traffic monitoring\yolov3.weights"
cfg_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\intelligent traffic monitoring\yolov3.cfg"
names_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\intelligent traffic monitoring\coco.names"
video_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\intelligent traffic monitoring\2099536-hd_1920_1080_30fps.mp4"
output_video_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\intelligent traffic monitoring\processed_video.mp4"

skip_frames = 10

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load classes
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define colors for different classes
colors = {
    "car": (0, 0, 255),        # Red
    "motorbike": (255, 0, 0),  # Blue
    "bus": (0, 255, 0),        # Green
    "truck": (0, 255, 255),    # Yellow
    "person": (255, 0, 255),   # Magenta
    "bicycle": (255, 165, 0)   # Orange for bicycles (added)
}

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

def draw_labels(boxes, confidences, class_ids, indexes, frame):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors.get(label, (255, 0, 0))  # Default to red if class not found
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video.")

    frame_count = 0
    boxes, confidences, class_ids, indexes = [], [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            boxes, confidences, class_ids, indexes = detect_objects(frame)
        frame = draw_labels(boxes, confidences, class_ids, indexes, frame)

        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1

    cap.release()

@app.route('/')
def index():
    return '''
        <!doctype html>
        <html>
        <head>
            <title>Traffic Monitoring</title>
        </head>
        <body>
            <h1>Traffic Monitoring</h1>
            <p>Choose an option:</p>
            <p><a href="/video_feed">View Live Video Feed</a></p>
            <p><a href="/download_video">Download Processed Video</a></p>
        </body>
        </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_video')
def download_video():
    # Create and save the processed video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    boxes, confidences, class_ids, indexes = [], [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            boxes, confidences, class_ids, indexes = detect_objects(frame)
        frame = draw_labels(boxes, confidences, class_ids, indexes, frame)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return send_file(output_video_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
