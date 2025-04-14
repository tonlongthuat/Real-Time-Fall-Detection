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
    global ip_address, video_active, video_processor

    data = request.get_json()
    ip_address = data.get('ip')

    # Reset lại bộ xử lý video để tránh bị kẹt ở video cũ
    video_processor.stop_processing()
    video_processor = VideoProcessor(frame_queue)

    video_active = True  # Bật luồng video mới
    print(f"Received IP address: {ip_address}")
    return jsonify({'message': 'IP address set successfully'}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    global video_active, ip_address, video_processor

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    uploads_dir = os.path.join(app.root_path, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Reset camera IP để tránh xung đột khi chuyển giữa video và camera
    ip_address = None

    # Reset lại bộ xử lý video
    video_processor.stop_processing()
    video_processor = VideoProcessor(frame_queue)
    video_processor.start_processing(file_path, camera_id=1)

    video_active = True  # Đánh dấu video đang chạy

    return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/video_feed')
def video_feed():
    if ip_address:
        esp32_cam = ESP32CamStreamer(f"http://{ip_address}/")
        streamer = VideoStreamer(esp32_cam, video_processor)
        return Response(streamer.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(video_streamer_file.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/close_cam', methods=['POST'])
def close_cam():
    global video_active
    video_active = False  # Tắt video
    return jsonify({'message': 'Camera closed successfully'}), 200
if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
