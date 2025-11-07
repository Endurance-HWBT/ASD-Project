import cv2
from model import Model
class Idk():
    def __init__(self):
        model=Model()
        self.rcog_hands,self.screen_height,self.screen_width,self.mp_drawings = model.initialize_model(num_of_hands=1,confidence_score=0.7)
    def track(self,frame):
        results = self.rcog_hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                index_finger = lm[8]
                middle_finger = lm[12]
                thumb = lm[4]
                ring = lm[16]
                pinky = lm[20]
                thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
                ring_x, ring_y = int(ring.x * w), int(ring.y * h)
                pinky_x, pinky_y = int(pinky.x * w), int(pinky.y * h)
                index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
                middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)
                cv2.circle(frame, (middle_x, middle_y), 10, (255, 0, 0), -1)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (ring_x, ring_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (pinky_x, pinky_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    obj = Idk()
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        obj.track(frame=rgb)
        cv2.imshow('Virtual Mouse', rgb)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
                break
    cap.release()
    cv2.destroyAllWindows()