from sys import argv
script,filename = argv
import cv2
import numpy as np
import time

img = cv2.imread(filename)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# surf.hessianThreshold=3000
surf = cv2.SURF(3000)
print "**********88"
kp,res = surf.detectAndCompute(gray,None)
print res.shape
img=cv2.drawKeypoints(gray,kp)
#img = cv2.drawKeypoints(img,kp,None,(255,0,255),4)
print(len(kp))

cv2.namedWindow("SURF")
cv2.imshow("SURF", img)
cv2.imwrite("surf.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
