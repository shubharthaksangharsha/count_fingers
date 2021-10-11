import cv2
import mediapipe as mp 
import time 
import os
import HandTrackingModule as htm
wCam, hCam = 640, 480 
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folderpath = "/home/shubharthak/Desktop/shubhi_handmodule/hand_detector_shubh/countfingers_shubh/fingers"
myList = os.listdir(folderpath)
myList.sort()
print(myList)
overlayList = []
for impath in myList:
    image = cv2.imread(f'{folderpath}/{impath}')
    # print((f'{folderpath}/{impath}'))
    overlayList.append(image)
# print(len(overlayList))
detector = htm.handDetector(detectionConfidence=0.75)
tipID= [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lm = detector.findPosition(img, draw= False)
    # print(lm)
    if len(lm) != 0:
        fingers = []
        #Thumb
        if lm[tipID[0]][1] > lm[tipID[0] - 1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        #4 Fingers
        for id in range(1,5):
            if lm[tipID[id]][2] < lm[tipID[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        h,w,c = overlayList[0].shape
        # print(h, w, c)
        # img[0:h,0:w] = overlayList[0]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers),(45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)
    cv2.imshow("Finger-Counter-Shubharthak", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
