import cv2
import sys

cap1 = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(0)

if not cap1.isOpened():
    print("Could not open a Camera!")
    sys.exit()
    
# 캡처 화면 이기 때문에 while 문에 넣어두지 않고 한번만 실행    
ret, background = cap1.read() #return , background

if not ret:
    print("Cann't load background")
    sys.exit()

background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) #
blur = cv2.GaussianBlur(background_gray, (5,5), 0) # (5,5) 블러처리 되는 정도   
blur_finish = cv2.imshow("blur_finish",blur)

while True:
    ret1, img1 = cap1.read()
    #ret2, img2 = cap2.read()
    camera = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    camera = cv2.GaussianBlur(camera, (5,5), 0)
    
    difference = cv2.absdiff(blur, camera)
    #img_gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img_color2 = cv2.cvtColor(img_gray2,cv2.COLOR_GRAY2BGR)
    # cv2.imshow("Camera",img1)
    #cv2.imshow("Camera",img_color2)
    #cv2.imshow("Camera",img_color)
    _, difference = cv2.threshold(difference, 100, 255, cv2.THRESH_BINARY) # _ -> 쓸때없는 변수는 사용안하겠다는 의미 (return 값을 받기는 받는다는 의미)
    
    # 
    
    
    cv2.imshow("difference",difference)
    
    lc, _, stats, _ = cv2.connectedComponentsWithStats(difference) # 
    
    for i in range(1, lc): # stats 
        x, y, w, h, s = stats[i]
        print(stats[i][:])
        
        if (s < 200):
            continue
            
        cv2.rectangle(img1, (x,y,w,h), (0,0,255), 2, 2)
    
    cv2.imshow("Camera_live",img1)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap1.release()
#cap2.release()
cv2.destroyAllWindows()
