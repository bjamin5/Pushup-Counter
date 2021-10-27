import pdb
import cv2
import mediapipe as mp
import time
import numpy as np
import math

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

class poseDetector():
    def __init__(self, mode=False,
                 up_body=False,
                 smooth=True,
                 detection_con = 0.5,
                 track_con = 0.5):
        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, #true - will always be trying to detect from inputs, false will try to detect and when the confidence is high it will keep tracking
                   upper_body_only=self.up_body,
                   smooth_landmarks=self.smooth,
                   min_detection_confidence=self.detection_con, #if confidence is more than 0.5 it will have detected the person and will proceed to go to tracking
                   min_tracking_confidence=self.track_con) #if the confidence is more than 0.5 it will keep tracking else it will return to detecting
        # Bool for pushup position
        self.ready_to_start = False
        self.timer_started = False
        self.start_time = None
        self.end_time = None
        self.going_up = False
        self.going_down = True
        self.last_state = (0,0,0)

        self.state_color = YELLOW # Yellow default color

        self.counter = 0

        self.angle = 0

    def getPose(self, img, draw=True): # Ask the user do you want to display on image or not
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Will give us the detection of our pose
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                pass
                # Draw landmarks on screen
                # self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.UPPER_BODY_POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lm_list = []
        body_lines = []
        left_side = []
        right_side = []
        # check if results are available
        if self.results.pose_landmarks:
            # See website for information about different pose landmarks (id)
            # https://www.analyticsvidhya.com/blog/2021/05/pose-estimation-using-opencv/
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # We need this because the landmark is just a ratio location for the image
                # print(id, lm)
                # We need the exact pixel value by multiplying the by the width and height
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id, cx, cy])

                if id in [11, 12, 13, 14, 15, 16]:
                    if draw:
                        if id % 2 == 0:
                            right_side.append((cx, cy))
                        else:
                            left_side.append((cx, cy))
                        # Overlay on the previous points if we are detecting it properly
                        # cv2.circle(img, (cx, cy), 10, (255,0,0), cv2.FILLED)
            if len(lm_list) != 0:
                body_lines = right_side[::-1] + left_side
                body_lines = np.int32(body_lines)
                # pts = body_lines.reshape((-1, 1, 2))
                # pdb.set_trace()
                cv2.polylines(img, [body_lines], False, color=self.state_color, thickness=10)
        return lm_list

    def checkPushupPosition(self, lm_list):
        left_shoulder = lm_list[0]
        right_shoulder = lm_list[1]
        left_elbow = lm_list[2]
        right_elbow = lm_list[3]
        left_wrist = lm_list[4]
        right_wrist = lm_list[5]
        # landmark[1] = x , landmark[2] = y
        # vector between 2 points P and Q
        # PQ-> = (X_q - X_p, Y_q - Y_p)

        # Angle between 2 vectors
        # cos(theta) = (a dot v) / (||a|| * ||v||)

        # vector between elbow and wrist [x,y]
        # right elbow to wrist
        r_e_w = np.array([right_wrist[1] - right_elbow[1],
                         right_wrist[2] - right_elbow[2]])
        r_e_s = np.array([right_shoulder[1] - right_elbow[1],
                         right_shoulder[2] - right_elbow[2]])

        unit_vector_1 = r_e_w / np.linalg.norm(r_e_w)
        unit_vector_2 = r_e_s / np.linalg.norm(r_e_s)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) * (180/np.pi)
        # pdb.set_trace()
        self.angle = angle
        return self.angle
        # vector between elbow and shoulder

    def checkIfReady(self):
        if self.angle > 170 and not self.ready_to_start:
            if self.start_time == None:
                # If timer not started, start it
                self.start_time = time.perf_counter()

            curr_time = time.perf_counter()
            if curr_time - self.start_time > 4.0:
                self.ready_to_start = True
                self.state_color = BLUE
        elif not self.ready_to_start:
            self.start_time = None
            # Reset start time to None if doesn't stay in pushup position for 3 seconds

    def countPushups(self):
        if self.ready_to_start:
            if self.going_down:
                if self.angle < 130: # If pass threshold
                    self.going_up = True
                    self.going_down = False
                    self.state_color = RED
            elif self.going_up:
                if self.angle > 165: # peak threshold
                    self.counter += 1
                    self.going_down = True
                    self.going_up = False
                    self.state_color = BLUE

    def print_status(self):
        if self.going_down and self.last_state[0] != self.going_down:
            print("Going down")
        if self.going_up and self.last_state[1] != self.going_up:
            print("Going up")
        if not self.ready_to_start and self.last_state[2] != self.ready_to_start:
            print("Not ready to start")
        if self.ready_to_start and self.last_state[2] != self.ready_to_start:
            print("Started")
        self.last_state = (self.going_down, self.going_up, self.ready_to_start)


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('resources/push_up3_test.mp4')
    # cap = cv2.VideoCapture('resources/push_up_2.mp4')
    pTime = 0
    # Adjust frames to be processed for speedup
    frame_time = 10

    # Set up custom window for viewing
    cv2.namedWindow('Track Upper Body', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Track Upper Body', [500, 800])

    # Pushup Logic
    pushup_position = False

    detector = poseDetector()
    theta = 0
    while True:
        # img is in bgr
        success, img = cap.read()
        # Convert bgr to rgb for mediapipe models
        img = detector.getPose(img)

        # Start timer to activate pushup mode
        detector.checkIfReady()
        # Get list of landmarks
        lmList = detector.getPosition(img)
        # Check pushup position
        if len(lmList) != 0:
            angle = detector.checkPushupPosition(lmList[11:17]) # (11th -> 16th)
            # pdb.set_trace()
            cv2.putText(img, str(int(angle)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3)
            cv2.putText(img, str(int(detector.counter)), (900, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3)

        # Count Pushups
        detector.countPushups()
        detector.print_status()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # print(theta)
        # cv2.putText(img, str(int(theta)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('Track Upper Body', img)
        cv2.waitKey(1)

# If we just run this file it will run main instead. If we call a fn in this file it won't run main.
if __name__ == "__main__":
    main()