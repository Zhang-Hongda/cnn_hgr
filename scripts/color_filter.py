import numpy as np
import cv2

# filter image based on hsv color range
def color_filter(area, lower_range, upper_range):
    hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    result = cv2.bitwise_and(hsv, hsv, mask = mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result

