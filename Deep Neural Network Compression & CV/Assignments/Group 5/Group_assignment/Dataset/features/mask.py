import cv2
import numpy as np

def get_fruit_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_not(white_mask)
