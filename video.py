import cv2
import time
from threading import Thread
import mediapipe as mp
from fall_detector import FallDetector
import yagmail
class VideoProcessor:
    def __init__(self, frame_queue, confidence_threshold=0.7):  # Increased from 0.5 to 0.7
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
        # Thêm biến theo dõi thời gian nằm
        self.person_lying_times = {}  # Dictionary để lưu thời gian nằm của từng người
        self.tracked_persons = {}  # Dictionary lưu trạng thái người
        self.next_person_id = 0    # Counter cho ID mới
        self.EMAIL_DELAY = 2.5  # 5 giây delay trước khi gửi email
         # Email configuration
        self.yag = yagmail.SMTP('ngvietquang377@gmail.com', 'hjmx zlsf nrgr cwat')
        self.email_receivers = ["ngvietquang377@gmail.com", "ngvietquang37@gmail.com"]
    def _assign_person_id(self, landmarks):
        """Gán ID ổn định dựa trên vị trí trung bình các điểm mốc"""
        # Tính vị trí trung bình (có thể tối ưu thêm)
        avg_x = sum(lm.x for lm in landmarks.landmark) / len(landmarks.landmark)
        avg_y = sum(lm.y for lm in landmarks.landmark) / len(landmarks.landmark)
        
        # Tìm người gần nhất đang được theo dõi
        closest_id = None
        min_distance = float('inf')
        
        for pid, data in self.tracked_persons.items():
            last_pos = data['last_position']
            distance = (last_pos[0] - avg_x)**2 + (last_pos[1] - avg_y)**2
            
            # Ngưỡng khoảng cách để coi là cùng 1 người
            if distance < min_distance and distance < 0.01:  
                min_distance = distance
                closest_id = pid
        
        # Nếu không tìm thấy người phù hợp, tạo ID mới
        if closest_id is None:
            closest_id = self.next_person_id
            self.next_person_id += 1
            self.tracked_persons[closest_id] = {
                'last_position': (avg_x, avg_y),
                'lying_start': None,
                'email_sent': False
            }
        else:
            # Cập nhật vị trí mới nhất
            self.tracked_persons[closest_id]['last_position'] = (avg_x, avg_y)
        
        return closest_id
    def sending_email(self, person_id):
        """Gửi email sau chính xác 5 giây nằm liên tục"""
        if person_id not in self.tracked_persons:
            return
        
        person_data = self.tracked_persons[person_id]
        current_time = time.time()
        
        # Nếu mới bắt đầu nằm
        if person_data['lying_start'] is None:
            person_data['lying_start'] = current_time
            person_data['email_sent'] = False
            return
        
        # Tính thời gian đã nằm
        lying_duration = current_time - person_data['lying_start']
        
        # Gửi email nếu đủ thời gian
        if lying_duration >= self.EMAIL_DELAY and not person_data['email_sent']:
            try:
                subject = "⚠️ Cảnh báo ngã"
                contents = [
                    f"Phát hiện người {person_id} đã nằm quá {self.EMAIL_DELAY} giây",
                    f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "Vui lòng kiểm tra ngay!"
                ]
                self.yag.send(to=self.email_receivers, subject=subject, contents=contents)
                
                person_data['email_sent'] = True
                print(f"✅ Đã gửi email cho người {person_id}")
            except Exception as e:
                print(f"❌ Lỗi gửi email: {e}")
    def process_frame(self, frame):
        # Calculate FPS
        self.current_frame_time = time.time()
        if self.prev_frame_time > 0:
            self.fps = 1 / (self.current_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.current_frame_time
        
        # Smaller text for FPS
        cv2.putText(frame, f"FPS: {int(self.fps)}", (frame.shape[1] - 120, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
            person_id = self._assign_person_id(results.pose_landmarks)
            # Display posture on the frame (smaller text)
            cv2.putText(frame, f"Posture: {pose_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display angles for debugging (smaller text)
            if len(pose_result) > 1 and isinstance(pose_result[1], dict):
                pose_data = pose_result[1]
                if 'spine_angle' in pose_data:
                    cv2.putText(frame, f"Spine Angle: {pose_data['spine_angle']:.1f}°", 
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.fall_detector.detect_fall(person_id, pose_result):
                falling_detected = True
                # Display "FALL DETECTED" text (still visible but smaller)
                cv2.putText(frame, "FALL DETECTED", (frame.shape[1]//2 - 100, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if pose_label == "LYING":
                self.sending_email(person_id)
            else:
                # Reset nếu không còn nằm
                if person_id in self.tracked_persons:
                    self.tracked_persons[person_id]['lying_start'] = None
                    self.tracked_persons[person_id]['email_sent'] = False
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
