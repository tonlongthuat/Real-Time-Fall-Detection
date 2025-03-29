import cv2
import time
from threading import Thread
import mediapipe as mp
from fall_detector import FallDetector

class VideoProcessor:
    def __init__(self, frame_queue, confidence_threshold=0.5):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=confidence_threshold, min_tracking_confidence=confidence_threshold)
        self.frame_queue = frame_queue
        self.should_stop = False
        self.processing_thread = None
        self.fall_detector = FallDetector()  # Initialize FallDetector
        self.person_id_counter = 0  # Counter for assigning person IDs
        # Add variables for FPS calculation
        self.prev_frame_time = 0
        self.current_frame_time = 0
        self.fps = 0

    def process_frame(self, frame):
        # Calculate FPS
        self.current_frame_time = time.time()
        if self.prev_frame_time > 0:
            self.fps = 1 / (self.current_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.current_frame_time
        
        # Display FPS on the right corner
        cv2.putText(frame, f"FPS: {int(self.fps)}", (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        falling_detected = False

        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

            # Process pose landmarks for fall detection
            landmarks = results.pose_landmarks.landmark
            pose_result = self.fall_detector.determine_pose(landmarks)
            pose_label = pose_result[0]  # Get the pose label from the tuple
            person_id = self.person_id_counter
            self.person_id_counter += 1

            # Display posture on the frame
            cv2.putText(frame, f"Posture: {pose_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display angles for debugging
            if len(pose_result) > 1 and isinstance(pose_result[1], dict):
                pose_data = pose_result[1]
                if 'spine_angle' in pose_data:
                    cv2.putText(frame, f"Spine Angle: {pose_data['spine_angle']:.1f}°", 
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.fall_detector.detect_fall(person_id, pose_result):
                falling_detected = True
                # Display "FALL DETECTED" text on the frame with higher visibility
                cv2.putText(frame, "FALL DETECTED", (frame.shape[1]//2 - 150, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return frame

    def process_video(self, video_path, camera_id):
        self.should_stop = False
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened() and not self.should_stop:
            success, frame = cap.read()
            if not success:
                break

            processed_frame = self.process_frame(frame)

            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(processed_frame)

        cap.release()

    def start_processing(self, video_path, camera_id):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing()
        self.processing_thread = Thread(target=self.process_video, args=(video_path, camera_id))
        self.processing_thread.start()

    def stop_processing(self):
        self.should_stop = True
        if self.processing_thread:
            self.processing_thread.join()

class VideoStreamer:
    def __init__(self, esp32_cam, video_processor):
        self.esp32_cam = esp32_cam
        self.video_processor = video_processor

    def start(self):
        self.esp32_cam.start()

    def stop(self):
        self.esp32_cam.stop()

    def generate_frames(self):
        self.start()
        try:
            while True:
                frame = self.esp32_cam.get_frame()
                if frame is not None:
                    processed_frame = self.video_processor.process_frame(frame)
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    time.sleep(0.01)
        finally:
            self.stop()
            
class FileVideoStreamer:
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue

    def get_frame(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.01)
