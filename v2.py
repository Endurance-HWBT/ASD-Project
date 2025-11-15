import cv2
import numpy as np
from model import Model
import pyautogui
import time

class Fuck():
    def __init__(self):
        model=Model()
        self.rcog_hands,self.screen_height,self.screen_width,self.mp_drawings = model.initialize_model(num_of_hands=1,confidence_score=0.8)
        print(self.screen_height,self.screen_width)

    def detect(self,h,w):
        results = self.rcog_hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                self.index_finger = lm[8]
                self.middle_finger = lm[12]
                self.thumb = lm[4]
                self.ring = lm[16]
                self.pinky = lm[20]
                self.wrist = lm [0]
                a=[self.index_finger,self.middle_finger,self.ring,self.pinky]
                avg = np.mean([np.hypot(int(self.wrist.x * w) - int(i.x*w),int(self.wrist.y * h) - int(i.y*h))for i in a])
                if avg < 30:
                     return False
                elif avg > 50:
                     return True
    
    def move(self,h,w,prev_x,prev_y):
        screen_x = np.interp((int(self.index_finger.x*w)+int(self.middle_finger.x*w))/2, (100, w-100), (0, self.screen_width))
        screen_y = np.interp((int(self.index_finger.y*h)+int(self.middle_finger.y*h))/2, (100, h-100), (0, self.screen_height))
        cur_x = prev_x + (screen_x - prev_x) / 3
        cur_y = prev_y + (screen_y - prev_y) / 3
        pyautogui.moveTo(cur_x, cur_y)
        return cur_x, cur_y

    def show(self,frame):
        cv2.circle(frame, (int((int(self.index_finger.x*w)+int(self.middle_finger.x*w))/2),int((int(self.index_finger.y*h)+int(self.middle_finger.y*h))/2)), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(self.thumb.x*h),int(self.thumb.y*w)), 10, (0, 255, 0), -1)
        # cv2.line(frame, (thumb_x, thumb_y), (ring_x, ring_y), (255, 255, 0), 2)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    obj = Fuck()
    value = False
    cor_x, cor_y =0,0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Virtual Mouse', rgb)
        value = obj.detect(h=h,w=w)
        if value:
            cor_x, cor_y = obj.move(h=h,w=w,prev_x=cor_x,prev_y=cor_y)
            obj.show(frame=rgb)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()