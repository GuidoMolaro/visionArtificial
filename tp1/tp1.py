import cv2 as cv

from gitFabri.tp_deteccion.contour import get_contours, get_biggest_contour, compare_contours
from gitFabri.tp_deteccion.frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours
from gitFabri.tp_deteccion.trackbar import create_trackbar, get_trackbar_value

def main():

    window_name = 'Window'
    trackbar_name = 'Trackbar'
    slider_max = 151
    cv.namedWindow(window_name)
    cap = cv.VideoCapture(0)
    biggest_contour = None
    color_white = (255, 255, 255)
    create_trackbar(trackbar_name, window_name, slider_max)
    # saved_hu_moments = load_hu_moments(file_name="hu_moments.txt")
    saved_contours = []

    while True:
        ret, frame = cap.read()
        grayF = cv.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        thresh = cv.adaptiveThreshold(grayF, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        cv.imshow('thresh', thresh)

main()