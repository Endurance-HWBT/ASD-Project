# from hand_recog import Detect
from model import Model
import cv2
import pyautogui
import numpy as np

model=Model()
rcog,screen_height,screen_width = model.initialize_model(num_of_hands=1,confidence_score=0.7)
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
click_threshold = 30
smoothening = 3
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
            index_finger = lm[8]  # index tip
            thumb = lm[4] 
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            # Convert to screen coordinates
            screen_x = np.interp(x, (100, w-100), (0, screen_width))
            screen_y = np.interp(y, (100, h-100), (0, screen_height))

            # Smooth movement
            cur_x = prev_x + (screen_x - prev_x) / smoothening
            cur_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y
            
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)

            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
            cv2.line(frame, (x, y), (thumb_x, thumb_y), (255, 255, 0), 2)
            distance = np.hypot(x - thumb_x, y - thumb_y)

            if distance < click_threshold:
                if not click_down:
                    pyautogui.click()
                    click_down = True
                    cv2.putText(frame, 'Click!', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                click_down = False

            # Draw landmarks
            # model.draw_landmarks(frame, hand_landmarks, model.HAND_CONNECTIONS)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
