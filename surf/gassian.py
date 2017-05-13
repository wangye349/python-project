import cv2

img = cv2.imread("e:/test6.bmp", 0)
result = cv2.blur(img, (5,5))

cv2.imwrite('test3.bmp',result)
