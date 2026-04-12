import cv2

def get_canny_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.Canny(blur, 100, 200)
