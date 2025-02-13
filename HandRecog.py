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

    # A quick note about the coordinate system: (0,0) is intially located at the TOP RIGHT of the screen and the x and y values respectively get higher as go to the left and down. 
    # Not sure why the coordinate system is like this, it just is. 
    # The flip function applied to the frame in the drawHandLandmarks method below flips the x axis. 
    # For consistenty and convenience, we will also flip the y axis below to make (0,0 the bottom left)

    def processAndCorrectView(self, frame): 

        frame = cv2.flip(frame, 1)
            #flips frame to match user's hands.

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
            # Changes color scheme, runs the model on the image.
            # Reasoning for this color switch is that mediapipe uses RGB while cv2 operates on BGR, so you need to change it.
        
        return frame
    
    def drawHandLandmarks(self, frame, draw=True):  
        
        if self.results.multi_hand_landmarks: 
            # if there is a hand on screen, then...
            for handLms in self.results.multi_hand_landmarks:
                # for each each set of landmark lists corresponding to each detected hand...
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)
                    # draw the red dots at each knuckle/wrist detected and interconnecting white lines.

        return frame

    def findAndMark_Positions( self, frame, handNo=0, draw=True):
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
                
                z = lm.z
                    # There is no corresponding pixels for z motion, so range stays the default. 
                    # This may change once second camera is implemented. 

                xList.append(cx)
                yList.append(cy)
                zList.append(z)

                cyDraw = cy
                #Defines seperate y coordinates specifically for drawing because the flip applied below flips the orientation of drawings.

                cy = int((1 - lm.y) * h) 
                #Flips the y axis to go from bottom to top as described earlier, causing (0,0) to be at the bottom right.
                self.lmsList.append([id, cx, cy, z, cyDraw])

                
                if draw:
                    # As shown in paramater declaration above, by default, draw = true. 
                    cv2.circle(frame,  (cx, cyDraw), 5, (255, 0, 255), cv2.FILLED)
                        # Draw purple circles on each landmark, overtop the normal red ones.
                
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
                #No need to flip values in xList and yList for rectangle, works as is. 

                #print( "Hands Keypoint")
                #print(bbox)
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255 , 0), 2)
                # draw a green rectangle that is 20 pixels larger than the hand in all four directions. 
                

        return self.lmsList, bbox
    

    def drawMarkers(self, p1, p2, color, frame, r=5, t=3):

        if color == "red":
            lineColor = (74,26,200)
            dotColor = (37,13,100)
        elif color == "green": 
            lineColor = (50,255,50)
            dotColor = (25,120,25)

        x1, x2 = self.lmsList[p1][1], self.lmsList[p2][1]
            #Assigns the x coords of the first and second target landmart to x1 and x2, respectively.
        xMid = (x1+x2)//2  

        y1Draw, y2Draw = self.lmsList[p1][4], self.lmsList[p2][4]
        yMidDraw = (y1Draw + y2Draw)//2

        cv2.line(frame,(x1, y1Draw),(x2, y2Draw) ,(lineColor), t)
            # Draws line between target landmark in bright color.
        cv2.circle(frame,(x1, y1Draw),r,(dotColor),cv2.FILLED)
        cv2.circle(frame,(x2, y2Draw),r, (dotColor),cv2.FILLED)
        cv2.circle(frame,(xMid,yMidDraw ), int(r*0.65) ,(dotColor),cv2.FILLED)
            # Draws dark colored circles on each target landmark + midpoint. Made midpoint circle smaller. 
            # 74,26,200 is a bright rose red color in case you want to use that.

        return frame


    def findAndMark_XYDistanceAndOrientation(self, p1, p2, frame):
         
        x1 , y1 = self.lmsList[p1][1:3]
            #Assigns the x and y coords of the first target landmart to x1 and y1. 
        x2, y2 = self.lmsList[p2][1:3]
            #Assigns the x and y coords of the second  target landmart to x2 and y2. 
        xDist, yDist = (x2-x1), (y2-y1)
            #assigns the difference between x and y coords to xDist and yDist, respectively. 
        xMid , yMid = (x1+x2)//2 , (y1 + y2)//2
            #Finds midpoint components. 
 
        absoluteDist = math.hypot(xDist,yDist)
            #finds absolute value of distance between target landmarks.

        angle = math.atan2(yDist, xDist)
            # Finds angle (in radians) between yDist and xDist vectors.
        uprightness = math.sin(angle)
            #Takes the sine of this angle to determine how upright the chosen section is. 
        horizontalness = math.cos(angle)
            #Takes the cosine of this angle to determine an horizontal orientation coeffecient for the chosen section. 
        
        #print(xDist, yDist, angle, uprightness) 
        
        return absoluteDist, horizontalness, uprightness, [x1, y1, x2, y2, xMid, yMid, xDist, yDist, angle]

    
    def findFingersOpen(self,frame):
        fingers=[]

        _, handhorizontalness, _, _ = self.findAndMark_XYDistanceAndOrientation(1, 0 , frame)
            # Measures the horizontalness from the wrist to the first landmark along the thumb. 
        frame = self.drawMarkers(1, 0, "green", frame)

        if handhorizontalness > 0: 
            thumbOnLeft = True
            thumbDefault = 1
        else: 
            thumbOnLeft = False
            thumbDefault = 0
            #This if statement solves the issue of not being able to distinguish between left and right hand being up as well as if a hand is flipped. 

        if self.lmsList[self.tipIds[0]][1] < self.lmsList[self.tipIds[0]-2][1]:
                # Checks whether the thumb tip is to the right (for left-hand tracking) compared to the preceding joint.
                # This is necessary because the thumb bends sideways (unlike the other fingers, which bend vertically).
            fingers.append(thumbDefault)
                #Append 1 to the list of fingers up. 
        else:
            fingers.append(1-thumbDefault)

        _, _, handUprightness, _ = self.findAndMark_XYDistanceAndOrientation(0, 10, frame)
            # Measures the verticality from the wrist to the first knuckle of the middle finger. 
        frame = self.drawMarkers(0, 10, "green", frame)

        if handUprightness > 0: 
            handIsUpright = True
            fingerDefault = 1
        else: 
            handIsUpright = False
            fingerDefault = 0
                # This if statement solves the issue of fingers being counted as down when the hand is upside down by making status dependent on the hand's vertical orientation. 
                # Not a perfect solution, there is a range of hand movement where the hand is parallel to the table/ground where the program thinks it's closed by it's not. 
        
        for id in range(1, 5):            
            if self.lmsList[self.tipIds[id]][2] > self.lmsList[self.tipIds[id]-2][2]:
                #The other four fingers (index, middle, ring, pinky) bend vertically, so their conditions compare y-coordinates instead of x-coordinates.
                #Here, 2 represents the y-coordinate. If the fingertip is higher or lower (depeding on handIsUpright) than the landmark two knuckles below, it signfies whether or not the finger is up or down. 
                   fingers.append(fingerDefault)
            else:
                   fingers.append(1-fingerDefault)

        if sum(fingers[1:5]) == 0:
            handisClosed = True
            handMsg = "closed"
                #Regardless of thumb, if the four fingers of a hand are down, hand is closed. 
                #Remember, count starts from 0. 
                #Also, remember that index ranges are exclusive on the end index. 
        else: 
            if 0 < sum(fingers[0:5]) < 5:
                handMsg = "partially open"
            else: 
                handMsg = "open"
            
            handisClosed = False
                #regardless of how many fingers are up, if all four aren't down, the hand will be considered closed. 
        
        return fingers, handMsg, handisClosed, handIsUpright, thumbOnLeft


    def findRotation(self, frame): 
        
        zPointerBaseKnuckle, zPinkieBaseKnuckle = self.lmsList[5][3], self.lmsList[17][3]
        dist, _, _, _ = self.findAndMark_XYDistanceAndOrientation(5, 17, frame)
            # Retrieves information about z coords of target knuckles as well as distance in between.
        _, _, _, _, thumbOnLeft = self.findFingersOpen(frame)
        frame = self.drawMarkers(5, 17, "red", frame)

        bufferAndScalingFactor = 0.5
            #A value from 0-1 which determines how much the hand needs to be rotated from the starting position to activate rotation tracking and also how senstive the rotation is past this buffer point.
        
        global maxDistance
        maxDistance = 150
        if dist > maxDistance: 
              maxDistance = dist
                # If absolute distance ever grows larger than any point in the past during the current instance, maxDistance is updated.
            # A starting value of 150 pixels is used to prevent from very small movements before the max distance value becomes accurate from heavily influence rotation. It's a buffer value. 
        #Even with the global variable, the max Dist values is always the intial assignment until dist grows larger. It doesn't stick once pushed higher, not sure why. 

        unbufferedRotation = dist/maxDistance
            # This statement inherently makes it so that the neutral position is when both the palm and back of the hand are facing perpendicaular to the camerage.
            # In this position, the pinkie is in front of all the other fingers (from the POV of the camera) and Dist ~= 0. 
            # By dividing by maxDistance, unsighnedRotion will never be over 1 

        if unbufferedRotation > bufferAndScalingFactor:
            unsignedRotation = ((unbufferedRotation/bufferAndScalingFactor) - 1)
                # Unsimplified, this value is ((unbufferedRotation- bufferAndScalingFactor)/bufferAndScalingFactor) since this adjusts when rotation actually starts to get counted so its still in the 0-1 range. 
                # Seperating the terms and factoring out maxDistance on the second term results in unbufferedRotation/bufferAndScalingFactor - 1.
        else: 
            unsignedRotation= 0
                # If the handrotation is higher than the buffer value, then the temporary value of unsignedRotation kicks in and is scaled by the buffer factor to make up for not kicking in until reaching buffer point.  

        if thumbOnLeft:
            rotation =  unsignedRotation 
        else: 
            rotation = unsignedRotation * -1
                #If the thumb is on the left, the rotation value will cause counterclockwise (positive) rotaiton. If not, counter-clockwise (negative) rotation will occur. 

        return rotation, maxDistance



def main():
        
        ctime = 0
        ptime = 0
        #Setting values for important variables for later. 

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
                #take camera input

            frame = detector.processAndCorrectView(frame)
                #process camera input and flip view
            frame = detector.drawHandLandmarks(frame)
                #draw intial landmark drawings
            lmsList = detector.findAndMark_Positions(frame)
                #Determine pixel positions and do secondary landmark drawing. 
            if len(lmsList[0]) != 0:
                # This if statement is necessary because without it, the program kills itself when your hand isn't on screen lol. 
                fingers, handMsg, _, _, _ = detector.findFingersOpen(frame)
                verticalDistance, handHorizontalness, handUprightness, info = detector.findAndMark_XYDistanceAndOrientation(0,10, frame)
                horizontalDistance, _, _, _ = detector.findAndMark_XYDistanceAndOrientation(5, 17, frame)
                rotation, maxDistance = detector.findRotation(frame)

                print(detector.lmsList[0][3]*3.5e5)
                #print("Fingers Open: ", fingers, (sum(fingers[0:5])), "  ", verticalDistance, handHorizontalness, handUprightness, horizontalDistance, maxDistance, rotation)
                    # Output finger states and other info to console
                
                cv2.putText(frame, ("Hand is " + handMsg), (5,80), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                    #On-screen hand status. 
                cv2.putText(frame, ("Rotation coeffecient is " + str(rotation)), (5,120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
            
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