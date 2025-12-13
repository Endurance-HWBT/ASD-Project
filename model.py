<<<<<<< HEAD
import mediapipe as mp
import pyautogui
class Model():
    def __init__(self):
        self
    
    def initialize_model(self,num_of_hands,confidence_score):
        'creates a model input the num of hands to detect and confidence threshold'
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(max_num_hands=num_of_hands, min_detection_confidence=confidence_score)
        screen_width, screen_height = pyautogui.size()
        return hands,screen_height,screen_width,mp_drawing
=======
import mediapipe as mp
import pyautogui
class Model():
    def __init__(self):
        self
    
    def initialize_model(self,num_of_hands,confidence_score):
        'creates a model input the num of hands to detect and confidence threshold'
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(max_num_hands=num_of_hands, min_detection_confidence=confidence_score)
        screen_width, screen_height = pyautogui.size()
        return hands,screen_height,screen_width,mp_drawing
>>>>>>> piyush-new
 