import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('resources/push_up1_faster.mp4')
pTime = 0
# Adjust frames to be processed for speedup
frame_time = 10

# Set up custom window for viewing
cv2.namedWindow('Track Upper Body', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('Track Upper Body', [500, 800])

# Pushup Logic
pushup_position = False
num_pushups = 0

detector = pm.poseDetector()

while True:
    # img is in bgr
    success, img = cap.read()
    # Convert bgr to rgb for mediapipe models
    img = detector.getPose(img)
    lmList = detector.getPosition(img)
    if len(lmList) != 0:
        pass

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Frames per second
    #cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Pushup Counter
    #cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Track Upper Body', img)
    cv2.waitKey(1)