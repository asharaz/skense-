# Traffic Monitoring with YOLOv3
## Project Overview
This project is designed for real-time traffic monitoring using YOLOv3, a powerful object detection algorithm. It processes video streams to identify and classify various objects, such as cars, motorbikes, buses, trucks, bicycles, and pedestrians. The application uses Flask for the web interface, allowing users to view a live video feed and download processed videos with detected objects highlighted.


## Key Features:
* Real-Time Object Detection: The application processes video frames in real-time to detect objects using the pre-trained YOLOv3 model.
* Customizable Detection: The system can be easily extended to detect additional objects by modifying the configuration files and adding corresponding colors for visualization.

* Web-Based Interface: A user-friendly web interface is provided for viewing the live video feed or downloading the processed video.

* Efficient Processing: Frames are processed in intervals (e.g., every 10th frame) to optimize performance without sacrificing too much detection accuracy.

## Prerequisites
Before setting up the project in PyCharm, ensure you have the following prerequisites installed:

*   Python 3.6+
* Flask 
* OpenCV
* NumPy

#### Download YOLOv3 Files:

* Download the pre-trained YOLOv3 weights file (yolov3.weights) from the official YOLOv3 website.

* Ensure that the yolov3.cfg configuration file and coco.names file are in the same directory as the project.

* Update File Paths:  update the paths for the YOLOv3 weights, configuration, class names, and video file to match your local file paths.

## Code Explanation

### Importing Libraries

```python
from flask import Flask, Response, send_file, stream_with_context
import cv2
import numpy as np

```
* Flask: A micro web framework for Python used to create web applications. Flask is used here to create the web interface and handle HTTP requests.

* cv2: The OpenCV library for computer vision tasks, such as video processing and object detection.

* numpy: A library for numerical operations on arrays. It is used here to handle image data as arrays.

```python
app = Flask(__name__)
```
* app = Flask(__name__): Initializes a new Flask web application. __name__ is passed to Flask to determine the root path of the application.

#### File Paths Configuration
```python
weights_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\yolov3.weights"
cfg_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\yolov3.cfg"
names_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\coco.names"
video_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\2099536-hd_1920_1080_30fps.mp4"
output_video_path = r"C:\Users\saura\OneDrive\Desktop\sharaz\intelligent traffic monitoring\processed_video.mp4"
```
* weights_path: Path to the pre-trained YOLOv3 weights file, which contains the model's learned parameters.

* cfg_path: Path to the YOLOv3 configuration file that defines the network architecture.

* names_path: Path to a file containing class names (e.g., car, bus, person) that YOLOv3 can detect.

* video_path: Path to the input video file that will be processed for object detection.
* output_video_path = Path to the output video file that will be processed and downloaded.

#### YOLO Model Loading
```python
skip_frames = 10
```
*   skip_frames: Number of frames to skip between processing for object detection. This can help reduce computational load by not processing every frame.
```python
net = cv2.dnn.readNet(weights_path, cfg_path)
```
* cv2.dnn.readNet: Loads the YOLOv3 model using the specified weights and configuration files.
```python

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```
* layer_names: Retrieves the names of all layers in the network.
* output_layers: Identifies the layers responsible for producing the final output (detections) from the network.

#### Class Names and Colors
``` python
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
```
* classes: Reads the class names from the coco.names file into a list. Each line in the file represents a different object class.
```python
colors = {
    "car": (0, 0, 255),        # Red
    "motorbike": (255, 0, 0),  # Blue
    "bus": (0, 255, 0),        # Green
    "truck": (0, 255, 255),    # Yellow
    "person": (255, 0, 255),   # Magenta
    "bicycle": (255, 165, 0)   # Orange for bicycles
}
```
* colors: A dictionary mapping each class to a specific color. This is used to color-code the bounding boxes and labels drawn on the detected objects.
#### Object Detection Function
```python
def detect_objects(frame):
    height, width, channels = frame.shape
```
* frame.shape: Retrieves the dimensions of the video frame (height, width, and the number of color channels).
```python
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
```
* cv2.dnn.blobFromImage: Converts the image to a blob, which is a suitable format for input to the neural network. The blob is created by resizing the image to 416x416, normalizing pixel values, and performing other preprocessing steps.
* net.setInput: Sets the blob as the input to the YOLO model.
* net.forward: Passes the input through the network and gets the output from the specified output layers.
```python
class_ids = []
confidences = []
boxes = []
```
* class_ids: A list to store the IDs of the detected classes.
* confidences: A list to store the confidence scores for each detected object.
* boxes: A list to store the bounding boxes for each detected object.
```python
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
```
* detection[5:]: Extracts the scores for each class from the detection output.
* np.argmax(scores): Identifies the index of the highest score, which corresponds to the detected class.
* confidence > 0.5: Filters out detections with low confidence (less than 50%).
* center_x, center_y, w, h: Calculate the center coordinates and size of the bounding box.
* boxes.append([x, y, w, h]): Stores the bounding box coordinates.
* confidences.append(float(confidence)): Stores the confidence score.
* class_ids.append(class_id): Stores the class ID.
```python
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```
* cv2.dnn.NMSBoxes: Applies Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes with lower confidence scores, retaining only the most accurate ones.
```python
return boxes, confidences, class_ids, indexes
```
* return: Returns the detected bounding boxes, confidence scores, class IDs, and filtered indexes.
#### Label Drawing Function
```python
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
```
* draw_labels: This function draws the bounding boxes and labels on the detected objects in the video frame.
* x, y, w, h = boxes[i]: Extracts the coordinates and dimensions of the bounding box.
* label = str(classes[class_ids[i]]): Gets the class name corresponding to the detected object.
* color = colors.get(label, (255, 0, 0)): Retrieves the color for the class, defaults to red if the class is not found.
* cv2.rectangle: Draws the bounding box around the detected object.
* cv2.putText: Adds the label and confidence score above the bounding box.
* return frame: Returns the frame with drawn labels.
#### Frame Generator for Live Video Feed
```python
def generate_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video.")
```
* cv2.VideoCapture: Opens the video file for reading frames.
* cap.isOpened(): Checks if the video file was successfully opened.
* raise RuntimeError: Raises an error if the video file cannot be opened.
```python
frame_count = 0
boxes, confidences, class_ids, indexes = [], [], [], []
```
* frame_count: Counter for the frames processed.
* boxes, confidences, class_ids, indexes: Initialize empty lists to store detection data.
``` python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
```
* cap.isOpened(): Loops while the video is open.
* cap.read(): Reads a frame from the video.
* ret: A boolean that indicates whether the frame was successfully read.
* break: Exits the loop if no more frames can be read.
``` python
if frame_count % skip_frames == 0:
    boxes, confidences, class_ids, indexes = detect_objects(frame)
frame = draw_labels(boxes, confidences, class_ids, indexes, frame)
```
* frame_count % skip_frames == 0: Detects objects only every skip_frames frames.
* draw_labels: Draws the detected objects on the frame.
```python
Copy code
_, buffer = cv2.imencode('.jpg', frame)
frame = buffer.tobytes()
```
* cv2.imencode: Encodes the frame as a JPEG image.
* buffer.tobytes(): Converts the encoded image to bytes for streaming.
```python
yield (b'--frame\r\n'
       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
```       
* yield: Streams the frame in the multipart format for the HTTP response.
```python
frame_count += 1
cap.release()
frame_count += 1: Increments the frame count.
```
* cap.release(): Closes the video file. 
#### Web Routes and Handlers
```python
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
```
* @app.route('/'): Defines the route for the home page.
* index: Returns an HTML page with links to view the live video feed or download the processed video.
```python
@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
```
* @app.route('/video_feed'): Defines the route for streaming the live video feed.
* Response: Creates an HTTP response that streams the video frames using the generate_frames function.
* stream_with_context: Ensures the streaming context is properly managed.
```python
@app.route('/download_video')
def download_video():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video.")
```        
* @app.route('/download_video'): Defines the route for downloading the processed video.
* cv2.VideoCapture(video_path): Opens the video file.
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
```
* cv2.VideoWriter_fourcc(*'mp4v'): Specifies the codec for writing the video.
* cv2.VideoWriter: Creates a video writer object to save the processed video.
* output_video_path: Specifies the output file path for the processed video.
* int(cap.get(3)), int(cap.get(4)): Gets the width and height of the video frames.
```python
frame_count = 0
boxes, co`nfidences, class_ids, indexes = [], [], [], []
```
* Same as before, initializes frame counter and lists for detection data.
```python
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
```
* Processes and writes each frame to the output video file, similar to the live feed, but saves the processed frames to disk.
```python
return send_file(output_video_path, as_attachment=True)
```
* send_file: Sends the processed video file to the user as a downloadable attachment.
#### Running the Application
```python
if __name__ == '__main__':
    app.run(debug=True)
```    
* if __name__ == '__main__':: Ensures the Flask app runs only when this script is executed directly.
* app.run(debug=True): Starts the Flask server in debug mode, allowing for easier development and troubleshooting.

## Output

Run the code , make sure the path of the vedio is changed accordingly.
![1](https://github.com/user-attachments/assets/61414c92-f151-4b46-a9d5-ab73ab2ed081)


copy the link and paste it on the browser, a new page will open with live vedio feed and download vedio feed.
![Screenshot 2024-08-26 124656](https://github.com/user-attachments/assets/196f531b-163f-4072-9de3-9c150aac8fc2)


If you choose the live video feed it will show on the page without downloading it.
![Screenshot 2024-08-26 124711](https://github.com/user-attachments/assets/10133f3d-c66b-4770-a630-7a5a559727a8)


If you choose download process video , it will be downloaded on the output path that is mentioned in the code.
![Screenshot 2024-08-26 125207](https://github.com/user-attachments/assets/27fd2a99-6d88-482d-85db-52d510926513)
