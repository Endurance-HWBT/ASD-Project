# from hand_recog import Detect
from model import Model
import cv2
import pyautogui
import numpy as np

model=Model()
rcog,screen_height,screen_width = model.initialize_model(num_of_hands=1,confidence_score=0.7)
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
click_threshold = 20
smoothening = 3
scroll_threshold = 40
click_down = True
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # flip for natural movement
    h, w, _ = frame.shape

    # Convert frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = rcog.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract coordinates
            lm = hand_landmarks.landmark
            # x_coords = [int(p.x * w) for p in lm]
            # y_coords = [int(p.y * h) for p in lm]

            # z_values = [p.z for p in lm]
            # avg_depth = np.mean(z_values)
            # cv2.putText(frame, f"Dept:{avg_depth}",(0,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 0)

            # x_min, x_max = min(x_coords), max(x_coords)
            # y_min, y_max = min(y_coords), max(y_coords)

            # # Draw bounding box
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            # cv2.putText(frame, "Tracking Area", (x_min, y_min - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            index_finger = lm[8]
            middle_finger = lm[12]
            thumb = lm[4]
            ring = lm[16]
            pinky = lm[20] 

            x = int(((index_finger.x+middle_finger.x)/2) * w)
            y = int(((index_finger.y +middle_finger.y)/2)* h)

            # Convert to screen coordinates
            screen_x = np.interp(x, (100, w-100), (0, screen_width))
            screen_y = np.interp(y, (100, h-100), (0, screen_height))

            # screen_x = np.interp(x, (x_min, x_max), (0, screen_width))
            # screen_y = np.interp(y, (y_min, y_max), (0, screen_height))

            # Smooth movement
            cur_x = prev_x + (screen_x - prev_x) / smoothening
            cur_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y
            
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            ring_x, ring_y = int(ring.x * w), int(ring.y * h)
            pinky_x, pinky_y = int(pinky.x * w), int(pinky.y * h)
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
            middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)

            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (ring_x, ring_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (pinky_x, pinky_y), 10, (0, 255, 0), -1)
                
            cv2.line(frame, (thumb_x, thumb_y), (pinky_x, pinky_y), (255, 255, 0), 2)
            distance1 = np.hypot(ring_x - thumb_x, ring_y - thumb_y)
            distance2 = np.hypot(pinky_x - thumb_x, pinky_y - thumb_y)
            d1 = np.hypot(index_x - middle_x, index_y - middle_y)
            d2 = np.hypot(middle_x - ring_x, middle_y - ring_y)
            d3 = np.hypot(index_x - ring_x, index_y - ring_y)

            avg_dist = (d1 + d2 + d3) / 3 

            if distance1 < click_threshold:
                if not click_down:
                    pyautogui.click()
                    click_down = True
                    cv2.putText(frame, 'Left Click!', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                click_down = False

            if distance2 < click_threshold:
                if not click_down:
                    pyautogui.rightClick()
                    click_down = True
                    cv2.putText(frame, 'Right Click!', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                click_down = False

            if avg_dist < scroll_threshold:
                if index_y < ring_y:  # hand tilted up
                        pyautogui.scroll(50)   # scroll up
                        cv2.putText(frame, 'Scroll Up', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:  # hand tilted down
                        pyautogui.scroll(-50)  # scroll down
                        cv2.putText(frame, 'Scroll Down', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            



    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
