import cv2
import numpy as np
from model import Model
import pyautogui
import time
class Idk():
    def __init__(self):
        'Model setup'
        model=Model()
        self.rcog_hands,self.screen_height,self.screen_width,self.mp_drawings = model.initialize_model(num_of_hands=1,confidence_score=0.8)
    
    def move(self,target_x,target_y,smoothening,prev_x,prev_y,h,w):
        screen_x = np.interp(target_x, (100, w-100), (0, self.screen_width))
        screen_y = np.interp(target_y, (100, h-100), (0, self.screen_height))
        cur_x = prev_x + (screen_x - prev_x) / smoothening
        cur_y = prev_y + (screen_y - prev_y) / smoothening
        pyautogui.moveTo(cur_x, cur_y)
        return cur_x, cur_y

    def left_click(self,ring_x,ring_y,thumb_x,thumb_y):
        distance = np.hypot(ring_x - thumb_x, ring_y - thumb_y)    
        if distance < 20:
            pyautogui.click()
            cv2.putText(frame, 'Left Click!', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            cv2.line(frame, (thumb_x, thumb_y), (ring_x, ring_y), (255, 255, 0), 3)
            print('Left click')
    
    def right_click(self,middel2_x,middel2_y,thumb_x,thumb_y):
        distance = np.hypot(middel2_x - thumb_x, middel2_y - thumb_y)    
        if distance < 20:
            pyautogui.rightClick()
            cv2.putText(frame, 'Right Click!', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            cv2.line(frame, (thumb_x, thumb_y), (middel2_x, middel2_y), (255, 255, 0), 3)
            print('Right click')
    
    def switch_tab(self,thumb_x,thummb_y,middle2_x, middel2_y):
        distance = np.hypot(middle2_x - thumb_x, middel2_y - thummb_y)    
        if distance < 20:
            pyautogui.hotkey('ctrl', 'tab')
            print('switch tabs')

   
    # def Switch(self,frame,value):
    #     results = self.rcog_hands.process(frame)
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             lm = hand_landmarks.landmark            
    #             middle_tip = lm[12]
    #             middle_mcp = lm[9]
    #             middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
    #             middle2_x, middel2_y = int(middle_mcp.x * w), int(middle_mcp.y * h)
    #             if np.hypot(middle2_x - middle_x, middel2_y - middle_x) <10:
    #                 return True
    #     return False
    
    def track(self,frame,h,w):
        cor_x, cor_y =0,0
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
                cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (ring_x, ring_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (pinky_x, pinky_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
                cv2.line(frame, (thumb_x, thumb_y), (ring_x, ring_y), (255, 255, 0), 2)
                cv2.line(frame, (thumb_x, thumb_y), (pinky_y, pinky_y), (255, 255, 0), 2)
                cv2.line(frame, (thumb_x, thumb_y), (middle_x, middle_y), (255, 255, 0), 2)
                cor_x, cor_y =self.move(target_x=index_x,target_y=index_y,smoothening=4,prev_x=cor_x,prev_y=cor_y,h=h,w=w)
                self.left_click(ring_x=ring_x,ring_y=ring_y,thumb_x=thumb_x,thumb_y=thumb_y)
                self.right_click(middel2_x=pinky_x,middel2_y=pinky_y,thumb_x=thumb_x,thumb_y=thumb_y)
                self.switch_tab(middel2_y=middle_x,middle2_x=middle_y,thumb_x=thumb_x,thummb_y=thumb_y)



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    obj = Idk()
    value = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # value = obj.Switch(frame=rgb,value=value)
        # if value:
        obj.track(frame=rgb,h=h,w=w) #using the model
        
        cv2.imshow('Virtual Mouse', rgb)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()