import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)
    # note to self: find out if there's a feature in the cv2 library that detects the amount of cameras.
    # Maybe everything from here down can be placed into a for loop that iterates with each camera input? 

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    # Assigns the hand detection AI model from mediapipe wiht given confidence paramters to keyword hands. 
    while capture.isOpened():
        ret, frame = capture.read()
        # assigns ret to the boolean of whether or not cam is recoding and frame to the actual image data
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_image = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # orients the camera input, changes color scheme, runs the model on the image, reverse color scheme change.
        # Reasoning for this color switch is that mediapipe uses RGB while cv2 operates on BGR, so you need to change it temporarily. 

        hands_list =  detected_image.multi_hand_landmarks
        if hands_list:                                    
        # if there is a hand on screen, then...
            for lm_list in hands_list:
            # for each each set of landmark lists corresponding to each detected hand...
                mp_drawing.draw_landmarks(image, lm_list, 
                                            mp_hands.HAND_CONNECTIONS,
                                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                color=(255, 0, 255), thickness=4, circle_radius=2),
                                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                color=(20, 180, 90), thickness=2, circle_radius=2)
                                        )
                                            # draw the purple dots at each knuckle/wrist detected and the interconnectin green lines. \

                # any of the dimentional components (x,y or z) can be accessed using lm_list.landmark[num].dimensional_comp
                # Ex: lm_list.landmark[0].x returns the x compnent of the first landmark, which happens to be the wrist. 
                # The z component also exists, potentially no need for a second camera? 

                print(lm_list.landmark[0].z * 1e07)
                # insert algorithm to be used on each hand here. Interconnect with wireless tranmission protocool. 
                

        cv2.imshow('Webcam', image)
  
        if cv2.waitKey(1) & 0xFF == ord('x'):
           break

capture.release()
cv2.destroyAllWindows()