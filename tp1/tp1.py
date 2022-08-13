import cv2

from gitFabri.tp_deteccion.contour import get_contours, get_biggest_contour, compare_contours
from gitFabri.tp_deteccion.frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours
from gitFabri.tp_deteccion.trackbar import create_trackbar, get_trackbar_value

def main():
    win