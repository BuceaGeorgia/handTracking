import cv2

import mediapipe as mp
import time

# the parameter is the device index
# normally one camera will be connected
# so the parameter in this case is 0
cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
mpDraw=mp.solutions.drawing_utils
hands=mpHands.Hands()

pTime=0
cTime=0
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                # id and x and y coordinates of a landmark
                print(id,lm)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),5)
    # for el in
    cv2.imshow(" Image",img)
    cv2.waitKey(1)