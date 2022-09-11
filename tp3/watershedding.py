import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def lam(x):
    return x


def open(img, val1):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (val1 + 1, val1 + 1))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=2)
    return img


def dilate(img, val1):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (val1 + 1, val1 + 1))
    return cv.dilate(img, kernel, iterations=3)

def erode(img, val):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (val+1, val+1))
    return cv.erode(img, kernel, iterations=1)

def setBinary(image, val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255, cv.THRESH_BINARY)
    # aplica funcion thresh / ret1 si es true --> significa q no tenemos error
    return thresh1


cv.namedWindow('Tracks')
cv.createTrackbar('ThreshFg', 'Tracks', 100, 255, lam)
cv.createTrackbar('ThreshBg', 'Tracks', 20, 255, lam)
cv.createTrackbar('Open', 'Tracks', 3, 10, lam)
cv.createTrackbar('Dilate', 'Tracks', 20, 100, lam)
cv.createTrackbar('Erode', 'Tracks', 3, 100, lam)


def main():
    # Get and show Cells
    img = cv.imread("levadura.png")

    # Get binaryImg and find all the Nuclei
    threshold = cv.getTrackbarPos('ThreshFg', 'Tracks')
    thresholdBg = cv.getTrackbarPos('ThreshBg', 'Tracks')
    binary = setBinary(img, threshold)
    binaryBg = setBinary(img,thresholdBg)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    quant = str(len(contours)-1)

    # Noise Removal
    open1 = cv.getTrackbarPos('Open', 'Tracks')
    opened = open(binary, open1)
    opened2 = open(binaryBg, open1)

    # Sure Background Area
    dil1 = cv.getTrackbarPos('Dilate', 'Tracks')
    background = dilate(opened2, dil1)
    cv.imshow('Background', background)

    # Sure Foreground (Nuclei)
    dist_transform = cv.distanceTransform(opened, cv.DIST_L2, 5)
    cv.imshow('DT', dist_transform)
    ret, sure_fg = cv.threshold(dist_transform, 0.1* dist_transform.max(), 255, 0)
    foreground = np.uint8(sure_fg)

    # Unknown Region
    unknown = cv.subtract(background, foreground)

    #Show Images
    cv.putText(binary, quant, (0, 670), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv.imshow('Binary', binary)
    cv.imshow('BinaryBg', binaryBg)
    cv.imshow('SureNuclei', foreground)
    cv.imshow('Unknown', unknown)

    # Markers
    ret, markers = cv.connectedComponents(foreground)
    markers = markers + 10
    markers[unknown == 255] = 0
    plt.imshow(markers, cmap='jet')
    plt.show()
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv.imshow('Watershed', img)

    if cv.waitKey(0) == ord("r"):
        main()


main()
