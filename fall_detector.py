import numpy as np
import time
import math

class FallDetector:
    def __init__(self, fall_angle_threshold=30, velocity_threshold=0.4, fall_duration=0.5):
        # Configuration parameters
        self.fall_angle_threshold = fall_angle_threshold
        self.velocity_threshold = velocity_threshold
        self.fall_duration = fall_duration
        
        # Tracking data
        self.person_trackers = {}
        self.prev_time = time.time()
        
    def calculate_vertical_angle(self, v1, v2):
        # Calculate the angle between a vector and the vertical (y-axis)
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
            
        cos_angle = dot / (norm_v1 * norm_v2)
        # Clamp to handle floating point errors
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def determine_pose(self, landmarks):
        """
        Determines the posture using key points from MediaPipe Pose landmarks.
        Returns a tuple of (posture_label, pose_data)
        """
        # Extract key points (normalized coordinates)
        nose = np.array([landmarks[0].x, landmarks[0].y])
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
        left_hip = np.array([landmarks[23].x, landmarks[23].y])
        right_hip = np.array([landmarks[24].x, landmarks[24].y])
        left_knee = np.array([landmarks[25].x, landmarks[25].y])
        right_knee = np.array([landmarks[26].x, landmarks[26].y])
        left_ankle = np.array([landmarks[27].x, landmarks[27].y])
        right_ankle = np.array([landmarks[28].x, landmarks[28].y])
        
        # Calculate midpoints
        shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)
        hip_mid = np.mean([left_hip, right_hip], axis=0)
        ankle_mid = np.mean([left_ankle, right_ankle], axis=0)
        knee_mid = np.mean([left_knee, right_knee], axis=0)
        
        # Vectors
        vertical = np.array([0, -1])  # Pointing upward
        spine_vector = shoulder_mid - hip_mid
        leg_vector = hip_mid - knee_mid
        
        # Calculate key angles
        spine_vertical_angle = self.calculate_vertical_angle(spine_vector, vertical)
        leg_vertical_angle = self.calculate_vertical_angle(leg_vector, vertical)
        
        # Height and orientation metrics
        height = np.linalg.norm(nose - ankle_mid)
        width = max(np.linalg.norm(left_shoulder - right_shoulder), 
                    np.linalg.norm(left_hip - right_hip))
        height_ratio = height / (width + 1e-6)  # Avoid division by zero
        
        # Position of body center relative to feet (used for stability assessment)
        hip_ankle_vector = hip_mid - ankle_mid
        stability_ratio = np.linalg.norm(hip_ankle_vector) / height if height > 0 else 0
        
        current_time = time.time()
        time_diff = current_time - self.prev_time
        self.prev_time = current_time
        
        # Combine features for posture classification
        pose_data = {
            "spine_angle": spine_vertical_angle,
            "leg_angle": leg_vertical_angle,
            "height_ratio": height_ratio,
            "stability_ratio": stability_ratio,
            "body_center": hip_mid,
            "timestamp": current_time
        }
        
        # Improved classification logic
        # Fall detection is more reliable when looking at the spine orientation
        if spine_vertical_angle > 45 and height_ratio < 1.0:
            return "LYING", pose_data
        elif spine_vertical_angle < 20 and stability_ratio > 0.4:
            return "STANDING", pose_data
        else:
            return "FALLING", pose_data
    
    def detect_fall(self, person_id, pose_result):
        """
        State machine to detect falls based on posture transitions and velocity.
        """
        pose_label, pose_data = pose_result
        
        if person_id not in self.person_trackers:
            self.person_trackers[person_id] = {
                'state': 'NORMAL',
                'state_start_time': time.time(),
                'prev_position': pose_data['body_center'],
                'prev_time': pose_data['timestamp'],
                'vertical_velocity': 0,
                'fall_detected': False,
                'fall_confirmed_time': None
            }
        
        tracker = self.person_trackers[person_id]
        current_time = pose_data['timestamp']
        
        # Calculate velocity (focus on vertical displacement)
        time_diff = current_time - tracker['prev_time']
        if time_diff > 0:
            prev_pos = tracker['prev_position']
            curr_pos = pose_data['body_center']
            # Vertical movement (y-coordinate) is more indicative of falls
            vertical_displacement = abs(curr_pos[1] - prev_pos[1])
            tracker['vertical_velocity'] = vertical_displacement / time_diff
        
        # Update tracking info
        tracker['prev_position'] = pose_data['body_center']
        tracker['prev_time'] = current_time
        
        # State transition logic
        if pose_label == "STANDING":
            # Reset fall detection if person is standing
            tracker['state'] = 'NORMAL'
            tracker['state_start_time'] = current_time
            tracker['fall_detected'] = False
            tracker['fall_confirmed_time'] = None
            return False
            
        elif pose_label == "FALLING":
            # Transition to falling state if not already falling
            if tracker['state'] != 'FALLING':
                tracker['state'] = 'FALLING'
                tracker['state_start_time'] = current_time
            
            # Check for fast motion while falling (indicative of actual falls)
            if tracker['vertical_velocity'] > self.velocity_threshold:
                # Mark potential fall, will be confirmed if followed by lying
                tracker['fall_detected'] = True
            
            return False
            
        elif pose_label == "LYING":
            # If we were falling and now lying down, it's likely a fall
            if tracker['fall_detected'] and tracker['state'] == 'FALLING':
                # Check if the transition from falling to lying happened quickly
                falling_duration = current_time - tracker['state_start_time']
                
                if falling_duration < 2.0:  # Falls typically happen quickly
                    tracker['state'] = 'FALLEN'
                    tracker['fall_confirmed_time'] = current_time
                    return True
            
            # If already in lying state, check how long
            if tracker['state'] == 'LYING':
                lying_duration = current_time - tracker['state_start_time']
                # Long period of lying may also indicate a fall
                if lying_duration > 3.0:
                    tracker['state'] = 'FALLEN'
                    return True
            else:
                # Transition to lying state
                tracker['state'] = 'LYING'
                tracker['state_start_time'] = current_time
        
        return tracker['state'] == 'FALLEN'