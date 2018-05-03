import numpy as np
import cv2


img = np.zeros((512,512,3), np.uint8)

cv2.imshow('image', img)
cv2.waitKey()
