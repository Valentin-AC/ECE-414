import cv2
import mediapipe as mp
import time
import math as math
import os


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):     
            # custom constructor made for objects of the HandTrackingDynamic class. The four parameters are attributes which are set to defaults as shown within the parantheses. 
        self.__mode__   =  mode                                                     
        self.__maxHands__   =  maxHands                                             
        self.__detectionCon__   =   detectionCon
        self.__trackCon__   =   trackCon
            # these four attributes are created either by the defaults assigned to the parameters above or by manual assignment.
            # the fact that the self.(attribute) isn't exactly the same as the atrribute name (given the underscores) tells us that these are just ways to access the attributes. 
       
        self.tipIds = [4, 8, 12, 16, 20]
            # this attribute serves to tell us which landmarks (out of the 21) are the finger tips. 
                                                        
        self.handsMp = mp.solutions.hands
                # Assigns the hand detection AI model from mediapipe wiht given confidence paramters to attribute handsMp                                         
        self.hands = self.handsMp.Hands()
        self.mpDraw= mp.solutions.drawing_utils
            # these three come from the mediapipe library mostly. As a reminder, the mediapipe library is a pre-trained computer vision AI model. 

       

    def drawHandLandmarks(self, frame, draw=True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #Changes color scheme, runs the model on the image.
        # Reasoning for this color switch is that mediapipe uses RGB while cv2 operates on BGR, so you need to change it.   
        
        if self.results.multi_hand_landmarks: 
            # if there is a hand on screen, then...
            for handLms in self.results.multi_hand_landmarks:
                # for each each set of landmark lists corresponding to each detected hand...
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)
                    # draw the red dots at each knuckle/wrist detected and interconnecting white lines.

        return frame


    def findPosition( self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        zList = []
        bbox =  []
        self.lmsList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
                # because the handNo parameter is set to 0 by default, this variable refers to ONLY the first hand detected
            for id, lm in enumerate(myHand.landmark):
                # for each landmark on the detected hand...
                h, w, c = frame.shape
                    # the .shape method from the mediapipe library assigns the height and width of the screen, in pixels. c is color channel. 
                cx, cy = int(lm.x * w), int(lm.y * h)
                    # the range of the coordinate values of each landmark gets converted into the possible range of pixel values instead of the arbitrary 0-1 range. 
                z = int(lm.z)
                    # There is no corresponding pixels for z motion, so range stays the default. 
                    # This may change once second camera is implemented. 

                xList.append(cx)
                yList.append(cy)
                zList.append(z)

                self.lmsList.append([id, cx, cy, z])

                if draw:
                    # As shown in paramater declaration above, by default, draw = true. 
                    cv2.circle(frame,  (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                        # Draw purple circles on each landmark, overtop the normal red ones.. 

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
                #print( "Hands Keypoint")
                #print(bbox)
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255 , 0), 2)
                # draw a green rectangle that is 20 pixels larger than the hand in all four directions. 

        return self.lmsList, bbox
    

    def findDistanceXY(self, p1, p2, frame, draw= True, r=5, t=3):
         
        x1 , y1 = self.lmsList[p1][1:3]
        #Assigns the x and y coords of the first target landmart to x1 and y1. 
        x2, y2 = self.lmsList[p2][1:3]
        #Assigns the x and y coords of the second  target landmart to x2 and y4. 
        xMid , yMid = (x1+x2)//2 , (y1 + y2)//2
        #Finds midpoint components. 

        if draw:
              cv2.line(frame,(x1, y1),(x2,y2) ,(50,255,50), t)
                #Draws line between target landmarks.
              cv2.circle(frame,(x1,y1),r,(255,0,135),cv2.FILLED)
              cv2.circle(frame,(x2,y2),r, (255,0,135),cv2.FILLED)
              cv2.circle(frame,(xMid,yMid), int(r*0.65) ,(255,0,135),cv2.FILLED)
                #Draws circles on each target landmark + midpoint. Made midpoint circle smaller. 
                    #74,26,181 is a rose red color in case you want to use that. 
        len= math.hypot(x2-x1,y2-y1)
            #finds distance between target landmarks. 

        return len, [x1, y1, x2, y2, xMid, yMid]
    
    
    def findFingerUp(self):
        fingers=[]

        if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0]-2][1]:
                # Checks whether the thumb tip is to the right (for left-hand tracking) or left (for right-hand tracking) compared to the preceding joint.
                # This is necessary because the thumb bends sideways (unlike the other fingers, which bend vertically).
            fingers.append(1)
                #Append 1 to the list of fingers up. 
        else:
            fingers.append(0)

        for id in range(1, 5):            
            if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id]-3][2]:
                #The other four fingers (index, middle, ring, pinky) bend vertically, so their conditions compare y-coordinates instead of x-coordinates.
                #Here, 2 represents the y-coordinate. If the fingertip’s y-coordinate is smaller (higher up), it means the finger is extended.
                   fingers.append(1)
            else:
                   fingers.append(0)

            # As of right now, this only works well when the hand is upright.
            # When hand is downward, it registers fingers as up (instead of down) when hand is closed. 
            # Will be fixed by "uprightness coeffecient" (to be made)

            #Note to self: Include a larger implementation here that alters whether a finger is either 0 or 1 based on both it being up/down AND and "uprightness coeffecient".

        if sum(fingers[1:5]) == 0:
            handClosed = True
            handMsg = "closed"
            #Regardless of thumb, if the four fingers of a hand are down, hand is closed. 
            #Remember, count starts from 0. 
            #Also, remember that index ranges are exclusive on the end index. 
        else: 
            if 0 < sum(fingers[0:5]) < 5:
                handMsg = "partially open"
            else: 
                handMsg = "open"
            handClosed = False
            #regardless of how many fingers are up, if all four aren't down, the hand will be considered closed. 
        

        return fingers, handMsg, handClosed

    #Will need to add method for rotation coeffecient here. 


def main():
        
        ctime=0
        ptime=0
        cap = cv2.VideoCapture(0)
        #Takes video input from the first deteted camera. 
        detector = HandTrackingDynamic()
            # This declares detector to be an object of the HandTrackingDyanmic class, which gives it access to all the functions (methods) above.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, frame = cap.read()

            frame = detector.drawHandLandmarks(frame)
            lmsList = detector.findPosition(frame)
            if len(lmsList[0]) != 0:
                # This if statement is necessary because without it, the program kills itself when your hand isn't on screen lol. 
                fingers, handMsg, handClosed = detector.findFingerUp()
                distance, info = detector.findDistanceXY(4, 8, frame)
                print("Fingers Up: ", fingers, (sum(fingers[0:5])), "  ", distance)
                    # Output finger states to console
                
                cv2.putText(frame, ("Hand is " + handMsg), (5,80), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                #On-screen hand status. 

            #frame = detector.findDistanceXY(1,2,frame)

            #note that the findDistanceXY method are not used in the main function
            
            #if len(lmsList)!=0:
                #print(lmsList[0])
                    #This is a print used for debugging, commenented out for now. 

            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime = ctime
                #ctime is the time at which the loop was last ran. ptime stores the previous ctime. 
                #Hence, the FPS actually refers to how often landmark (knuckle) locations are calculated per second. 

            cv2.putText(frame, ("FPS: " + str(int(fps))), (5,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                #On screen FPS counter. Second argument is text to be displayed, third is location and the rest is font/color/formatting. 

            if cv2.waitKey(1) == ord('x'):
                break
                    #break condition: if x is pressed, stops loop
 
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #Creates a greyscale version of the camera output. It doesn't get used. 
            cv2.imshow('Hand Movement Interpreter', frame)
                #Opens a window with the name Hand Movement Interpreter and displays the result of running the above code on the camera input. 

if __name__ == "__main__":
            main()

        #These two lines just make sure that main() doesnt run unless this script is run directly. Prevents it from running unintentionally if this script is imported into another program. 