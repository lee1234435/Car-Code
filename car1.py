import cv2
import time 
import os 
from jd_opencv_lane_detect import JdOpencvLaneDetect

 
# OpenCV line detector object
cv_detector = JdOpencvLaneDetect()


# Camera object: reading image from camera 
cap = cv2.VideoCapture("Lesson_robotSensor\car_video.avi")
# Setting camera resolution as 320x240
cap.set(3, 320)
cap.set(4, 240)

 
# real driving routine 
while True:
    time.sleep(0.1)
    ret, img_org = cap.read()
    if ret:
        # Find lane angle
        lanes, img_lane = cv_detector.get_lane(img_org)
        angle, img_angle = cv_detector.get_steering_angle(img_lane, lanes)
        if img_angle is None:
            print("can't find lane...")
            pass
        else:
            cv2.imshow('lane', img_angle)
            print(angle)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("cap error")

cap.release()
cv2.destroyAllWindows()

