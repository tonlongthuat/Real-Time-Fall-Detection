import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from flask import Flask, render_template, Response, request, jsonify
import queue
from esp32cam_streamer import ESP32CamStreamer
from video import VideoProcessor, FileVideoStreamer, VideoStreamer

app = Flask(__name__, template_folder='templates')

# Global variables
frame_queue = queue.Queue(maxsize=10)
ip_address = None
video_processor = VideoProcessor(frame_queue)
video_streamer_file = FileVideoStreamer(frame_queue)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_ip', methods=['POST'])
def set_ip():
    global ip_address
    data = request.get_json()
    ip_address = data.get('ip')
    print(f"Received IP address: {ip_address}")
    return jsonify({'message': 'IP address set successfully'}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the uploads directory exists
    uploads_dir = os.path.join(app.root_path, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Start processing the uploaded video
    video_processor.start_processing(file_path, camera_id=1)

    return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/video_feed')
def video_feed():
    if ip_address:
        esp32_cam = ESP32CamStreamer(f"http://{ip_address}/")
        streamer = VideoStreamer(esp32_cam, video_processor)
        return Response(streamer.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(video_streamer_file.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
