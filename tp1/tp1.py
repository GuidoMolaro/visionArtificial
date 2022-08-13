import cv2 as cv
import numpy as np

from gitFabri.tp_deteccion.contour import get_contours, get_biggest_contour, compare_contours
from gitFabri.tp_deteccion.frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours
from gitFabri.tp_deteccion.trackbar import create_trackbar, get_trackbar_value


def main():
    # cap = cv.VideoCapture(0)
    im = cv.imread('shapes.jpg')
    biggest_contour = None
    color_white = (255, 255, 255)
    # saved_hu_moments = load_hu_moments(file_name="hu_moments.txt")
    saved_contours = []

    while True:
        # ret, frame = cap.read()
        grayF = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        thresh = cv.adaptiveThreshold(grayF, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        cv.imshow('thresh', thresh)
        contoursShapes, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        im2 = cv.imread('shapes2.jpg')
        gray2 = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)
        ret2, thresh2 = cv.threshold(gray2, 127, 255, cv.THRESH_BINARY)
        contours2, hierarchy = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        for contour in contours2:
            moments_alphabet = cv.moments(contour)
            huMoments_alphabet = cv.HuMoments(moments_alphabet)
            for cnt1 in contoursShapes:
                if cv.matchShapes(contour, cnt1, cv.CONTOURS_MATCH_I2, 0) < 0.4:
                    cv.drawContours(thresh2, contour, -1, (0, 0, 255), 2)
                    cv.imshow("contornos", thresh2)
                    cv.waitKey(0)
        cv.waitKey(0)

main()
