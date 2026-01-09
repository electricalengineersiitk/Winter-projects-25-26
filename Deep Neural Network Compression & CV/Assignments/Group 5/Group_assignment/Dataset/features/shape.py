import cv2
import numpy as np

def get_shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0]*6

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = w/h if h else 0
    circularity = (4*np.pi*area)/(perimeter**2) if perimeter else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area else 0

    extent = area/(w*h) if w*h else 0

    return [area, perimeter, aspect_ratio, circularity, solidity, extent]
