from model import Model
import numpy as np

class Detect():
    def __init__(self):
        model = Model()
        self.hands,self.screen_height,self.screen_width = model.initialize_model(num_of_hands=1,confidence_score=0.7)
        self.smoothening = 5
        self.click_threshold = 30 
        self.click_down = False

    def predict(self,rgb,h,w):
        'predicts if fingers click or not inputs:frame,height and width'
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                index_finger = lm[8]
                thumb = lm[4]
                x = int(index_finger.x * w)
                y = int(index_finger.y * h)

                screen_x = np.interp(x, (100, w-100), (0, self.screen_width))
                screen_y = np.interp(y, (100, h-100), (0, self.screen_height))

                thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
                distance = np.hypot(x - thumb_x, y - thumb_y)

                if distance < self.click_threshold:
                    return True,x,y,
                else:
                    return False