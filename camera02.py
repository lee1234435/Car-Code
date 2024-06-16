import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(args=None): 
    pass

cap1 = cv2.VideoCapture(0) # 노트북 카메라 on
cap1.set(3,720) # width
cap1.set(4,480) # height
cap1.set(10,125) # 범위 0 ~ 255 , 10 -> light
cv2.namedWindow("HSV") #
cv2.namedWindow("CANNY") #

#HSV 전용 트랙바 만들기 
cv2.createTrackbar("h_low", "HSV", 0, 180, nothing)
cv2.createTrackbar("h_high", "HSV", 0, 180, nothing)
cv2.createTrackbar("s_low", "HSV", 0, 200, nothing)
cv2.createTrackbar("s_high", "HSV", 50, 255, nothing)
cv2.createTrackbar("v_low", "HSV", 0, 200, nothing)
cv2.createTrackbar("v_high", "HSV", 50, 255, nothing)

#CANNY 전용 트랙바 만들기
cv2.createTrackbar("t_low", "CANNY", 0, 255, nothing)
cv2.createTrackbar("t_high", "CANNY", 0, 255, nothing)

# 카메라 작동
while True:
    ret1, img1 = cap1.read()
    
    img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    
    t_low = cv2.getTrackbarPos("t_low", "CANNY")
    t_high = cv2.getTrackbarPos("t_high", "CANNY")
    
    h_low = cv2.getTrackbarPos("h_low","HSV")
    h_high = cv2.getTrackbarPos("h_high","HSV")
    s_low = cv2.getTrackbarPos("s_low","HSV")
    s_high = cv2.getTrackbarPos("s_high","HSV")
    v_low = cv2.getTrackbarPos("v_low","HSV")
    v_high = cv2.getTrackbarPos("v_high","HSV")
    
    low_t = np.array([h_low,s_low,v_low])
    high_t = np.array([h_high,s_high,v_high])
    
    mask = cv2.inRange(img_hsv, low_t, high_t)
    
    result = cv2.bitwise_and(img1, img1, mask=mask)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    cv2.imshow("original",img1)
    cv2.imshow("result",mask)
    cv2.imshow("result",result)

    
cap1.release()
cv2.destroyAllWindows()
