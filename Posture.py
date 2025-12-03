import cv2
import numpy as np
import time
import urllib.request
import os


class PostureMonitor:
    def __init__(self):
        # Download model files if not present
        self.setup_model()

        # Load the model
        self.net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

        # Posture tracking variables
        self.bad_posture_start_time = None
        self.bad_posture_time = 0
        self.warning_given = False
        self.bad_posture_threshold = 10  # seconds (1 minute)

        # Body part indices
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

    def setup_model(self):
        """Download pose estimation model if not present"""
        model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
        graph_url = "https://github.com/quanhua92/human-pose-estimation-opencv/raw/master/graph_opt.pb"

        if not os.path.exists("graph_opt.pb"):
            print("Downloading pose estimation model (this may take a minute)...")
            try:
                urllib.request.urlretrieve(graph_url, "graph_opt.pb")
                print("Model downloaded successfully!")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("\nAlternative: Manual download instructions:")
                print(
                    "1. Download from: https://github.com/quanhua92/human-pose-estimation-opencv/raw/master/graph_opt.pb")
                print("2. Place 'graph_opt.pb' in the same folder as this script")
                exit(1)

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
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
        """Detect pose keypoints"""
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
        """Analyze posture based on detected keypoints"""
        try:
            nose = points.get(self.BODY_PARTS["Nose"])
            neck = points.get(self.BODY_PARTS["Neck"])
            l_shoulder = points.get(self.BODY_PARTS["LShoulder"])
            r_shoulder = points.get(self.BODY_PARTS["RShoulder"])
            l_hip = points.get(self.BODY_PARTS["LHip"])
            l_ear = points.get(self.BODY_PARTS["LEar"])

            if not all([nose, neck, l_shoulder, r_shoulder]):
                return None, 0, 0

            # Calculate neck angle
            neck_angle = 180  # default good angle
            if l_ear and l_shoulder and l_hip:
                neck_angle = self.calculate_angle(l_ear, l_shoulder, l_hip)
                if neck_angle is None:
                    neck_angle = 180

            # Check shoulder alignment
            shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
            shoulder_alignment = shoulder_diff

            # Check head forward position
            head_forward = False
            if nose and neck:
                head_forward = nose[0] < neck[0] - 30

            # Determine posture
            is_good_posture = (
                    neck_angle > 150 and  # Neck relatively straight
                    shoulder_alignment < 40 and  # Shoulders relatively level
                    not head_forward  # Head not too far forward
            )

            return is_good_posture, neck_angle, shoulder_alignment

        except Exception as e:
            print(f"Error in posture check: {e}")
            return None, 0, 0

    def draw_skeleton(self, frame, points):
        """Draw skeleton on frame"""
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
        """Display warning message"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, "WARNING: BAD POSTURE DETECTED!",
                    (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3)
        cv2.putText(frame, "Please straighten your back and neck",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Sit up straight, shoulders back!",
                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def run(self):
        """Main monitoring loop"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Posture Monitor Started!")
        print("Position yourself so your upper body is visible")
        print("Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape
            current_time = time.time()

            # Detect pose
            points = self.detect_pose(frame)

            # Draw skeleton
            self.draw_skeleton(frame, points)

            # Check posture
            is_good, neck_angle, shoulder_align = self.check_posture(points)

            if is_good is not None:
                if is_good:
                    posture_status = "GOOD POSTURE"
                    status_color = (0, 255, 0)

                    if self.bad_posture_start_time is not None:
                        self.bad_posture_time += current_time - self.bad_posture_start_time
                    self.bad_posture_start_time = None
                    self.warning_given = False

                else:
                    posture_status = "BAD POSTURE"
                    status_color = (0, 0, 255)

                    if self.bad_posture_start_time is None:
                        self.bad_posture_start_time = current_time

                    bad_posture_duration = current_time - self.bad_posture_start_time

                    if bad_posture_duration >= self.bad_posture_threshold and not self.warning_given:
                        self.display_warning(frame)
                        self.warning_given = True
                        print(f"\n⚠️  WARNING: Bad posture for {int(bad_posture_duration)} seconds!")
                        print("Advisory:")
                        print("  • Take a 5-minute break")
                        print("  • Stand up and stretch")
                        print("  • Adjust your chair height")
                        print("  • Position monitor at eye level")
                        print("  • Keep feet flat on floor")
                    elif bad_posture_duration >= self.bad_posture_threshold:
                        self.display_warning(frame)

                # Display status
                cv2.putText(frame, posture_status, (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, status_color, 2)

                cv2.putText(frame, f"Neck Angle: {int(neck_angle)}°", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Shoulder Diff: {int(shoulder_align)}px", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if self.bad_posture_start_time is not None:
                    duration = int(current_time - self.bad_posture_start_time)
                    cv2.putText(frame, f"Bad Posture: {duration}s / 10s", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Detecting pose...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

            cv2.putText(frame, "Press 'q' to quit", (10, image_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Posture Monitor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 50)
        print("Session Summary:")
        print(f"Total bad posture time: {int(self.bad_posture_time)} seconds")
        print("=" * 50)


if __name__ == "__main__":
    monitor = PostureMonitor()
    monitor.run()