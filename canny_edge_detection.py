import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 흑백 및 선 추출
# img = cv.imread('chunsik.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# edges = cv.Canny(img,100,200)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

width = 600
height = 600
img = cv.imread('chunsik.jpg')
img = cv.resize(img, (height, width))

img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edge = cv.Canny(img2, 100, 200)

blur = cv.GaussianBlur(img, (9,9))

while True:
    #ret2, img2 = cap2.read()
    
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_color2 = cv2.cvtColor(img_gray2,cv2.COLOR_GRAY2BGR)
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    
    cv.imshow("chunsik",img2)
    cv.imshow("edge",edge)
    #cv2.imshow("Camera",img_color2)
    #cv2.imshow("Camera",img_color)
    
#cap2.release()
cv.destroyAllWindows()


