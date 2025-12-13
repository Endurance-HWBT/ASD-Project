from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time
from model import Model
import pyautogui

app = Flask(__name__)

# Global variables for camera states
mouse_camera_active = False
posture_camera_active = False
posture_warning = {"active": False, "message": "", "duration": 0}

class VirtualMouse:
    def __init__(self):
        model = Model()
        self.rcog_hands, self.screen_height, self.screen_width, self.mp_drawings = model.initialize_model(
            num_of_hands=1, confidence_score=0.8
        )
        self.prev_x, self.prev_y = 0, 0
        self.cap = None
        
    def move(self, target_x, target_y, smoothening, h, w):
        screen_x = np.interp(target_x, (100, w-100), (0, self.screen_width))
        screen_y = np.interp(target_y, (100, h-100), (0, self.screen_height))
        cur_x = self.prev_x + (screen_x - self.prev_x) / smoothening
        cur_y = self.prev_y + (screen_y - self.prev_y) / smoothening
        pyautogui.moveTo(cur_x, cur_y)
        self.prev_x, self.prev_y = cur_x, cur_y
        
    def detect_gestures(self, frame, h, w):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.rcog_hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                index_finger = lm[8]
                middle_finger = lm[12]
                thumb = lm[4]
                ring = lm[16]
                pinky = lm[20]
                
                # Get coordinates
                thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
                ring_x, ring_y = int(ring.x * w), int(ring.y * h)
                pinky_x, pinky_y = int(pinky.x * w), int(pinky.y * h)
                index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
                middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)
                
                # Draw visual feedback
                cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (ring_x, ring_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (pinky_x, pinky_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                cv2.line(frame, (thumb_x, thumb_y), (ring_x, ring_y), (255, 255, 0), 2)
                cv2.line(frame, (thumb_x, thumb_y), (pinky_x, pinky_y), (255, 255, 0), 2)
                cv2.line(frame, (thumb_x, thumb_y), (middle_x, middle_y), (255, 255, 0), 2)
                
                # Move cursor
                self.move(index_x, index_y, 4, h, w)
                
                # Left click detection
                distance1 = np.hypot(ring_x - thumb_x, ring_y - thumb_y)
                if distance1 < 20:
                    cv2.putText(frame, 'Left Click!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Right click detection
                distance2 = np.hypot(pinky_x - thumb_x, pinky_y - thumb_y)
                if distance2 < 20:
                    cv2.putText(frame, 'Right Click!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Tab switch detection
                distance3 = np.hypot(middle_x - thumb_x, middle_y - thumb_y)
                if distance3 < 20:
                    cv2.putText(frame, 'Switch Tab!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

class PostureMonitor:
    def __init__(self):
        self.net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
        self.bad_posture_start_time = None
        self.bad_posture_time = 0
        self.warning_given = False
        self.bad_posture_threshold = 10
        self.cap = None
        
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
        }
        
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
        ]
    
    def calculate_angle(self, a, b, c):
        if a is None or b is None or c is None:
            return None
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def detect_pose(self, frame):
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        self.net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368),
                                                (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = self.net.forward()
        out = out[:, :19, :, :]
        
        points = {}
        for i in range(len(self.BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points[i] = (int(x), int(y)) if conf > 0.2 else None
        
        return points
    
    def check_posture(self, points):
        try:
            nose = points.get(self.BODY_PARTS["Nose"])
            neck = points.get(self.BODY_PARTS["Neck"])
            l_shoulder = points.get(self.BODY_PARTS["LShoulder"])
            r_shoulder = points.get(self.BODY_PARTS["RShoulder"])
            l_hip = points.get(self.BODY_PARTS["LHip"])
            l_ear = points.get(self.BODY_PARTS["LEar"])
            
            if not all([nose, neck, l_shoulder, r_shoulder]):
                return None, 0, 0
            
            neck_angle = 180
            if l_ear and l_shoulder and l_hip:
                neck_angle = self.calculate_angle(l_ear, l_shoulder, l_hip)
                if neck_angle is None:
                    neck_angle = 180
            
            shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
            shoulder_alignment = shoulder_diff
            
            head_forward = False
            if nose and neck:
                head_forward = nose[0] < neck[0] - 30
            
            is_good_posture = (
                neck_angle > 150 and
                shoulder_alignment < 40 and
                not head_forward
            )
            
            return is_good_posture, neck_angle, shoulder_alignment
        except:
            return None, 0, 0
    
    def draw_skeleton(self, frame, points):
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]
            
            if points.get(idFrom) and points.get(idTo):
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.circle(frame, points[idFrom], 5, (0, 0, 255), thickness=-1)
                cv2.circle(frame, points[idTo], 5, (0, 0, 255), thickness=-1)
    
    def display_warning(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "WARNING: BAD POSTURE!", (50, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3)
        cv2.putText(frame, "Straighten your back!", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        global posture_warning
        current_time = time.time()
        
        points = self.detect_pose(frame)
        self.draw_skeleton(frame, points)
        
        is_good, neck_angle, shoulder_align = self.check_posture(points)
        
        if is_good is not None:
            if is_good:
                posture_status = "GOOD POSTURE"
                status_color = (0, 255, 0)
                if self.bad_posture_start_time is not None:
                    self.bad_posture_time += current_time - self.bad_posture_start_time
                self.bad_posture_start_time = None
                self.warning_given = False
                posture_warning = {"active": False, "message": "", "duration": 0}
            else:
                posture_status = "BAD POSTURE"
                status_color = (0, 0, 255)
                
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = current_time
                
                bad_posture_duration = current_time - self.bad_posture_start_time
                
                if bad_posture_duration >= self.bad_posture_threshold:
                    self.display_warning(frame)
                    if not self.warning_given:
                        self.warning_given = True
                    posture_warning = {
                        "active": True,
                        "message": "Bad posture detected! Please straighten your back.",
                        "duration": int(bad_posture_duration)
                    }
            
            cv2.putText(frame, posture_status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, status_color, 2)
            cv2.putText(frame, f"Neck: {int(neck_angle)}°", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Shoulder: {int(shoulder_align)}px", (10, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.bad_posture_start_time is not None:
                duration = int(current_time - self.bad_posture_start_time)
                cv2.putText(frame, f"Bad: {duration}s / 10s", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Detecting pose...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        
        return frame

# Initialize systems
virtual_mouse = VirtualMouse()
posture_monitor = PostureMonitor()

def generate_mouse_frames():
    virtual_mouse.cap = cv2.VideoCapture(0)
    while mouse_camera_active:
        ret, frame = virtual_mouse.cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame = virtual_mouse.detect_gestures(frame, h, w)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    virtual_mouse.cap.release()

def generate_posture_frames():
    posture_monitor.cap = cv2.VideoCapture(1)  # Camera 1 for posture
    while posture_camera_active:
        ret, frame = posture_monitor.cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = posture_monitor.process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    posture_monitor.cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_mouse')
def video_mouse():
    return Response(generate_mouse_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_posture')
def video_posture():
    return Response(generate_posture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_mouse', methods=['POST'])
def start_mouse():
    global mouse_camera_active
    mouse_camera_active = True
    return jsonify({"status": "started"})

@app.route('/stop_mouse', methods=['POST'])
def stop_mouse():
    global mouse_camera_active
    mouse_camera_active = False
    return jsonify({"status": "stopped"})

@app.route('/start_posture', methods=['POST'])
def start_posture():
    global posture_camera_active
    posture_camera_active = True
    return jsonify({"status": "started"})

@app.route('/stop_posture', methods=['POST'])
def stop_posture():
    global posture_camera_active
    posture_camera_active = False
    return jsonify({"status": "stopped"})

@app.route('/posture_status')
def posture_status():
    return jsonify(posture_warning)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)