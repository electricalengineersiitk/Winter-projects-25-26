import cv2
import numpy as np

def get_lbp_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    lbp = np.zeros((h,w), dtype=np.uint8)

    for i in range(1,h-1):
        for j in range(1,w-1):
            center = gray[i,j]
            binary = ""
            for dx,dy in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
                binary += '1' if gray[i+dx,j+dy] >= center else '0'
            lbp[i,j] = int(binary, 2)
    return lbp
