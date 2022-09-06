import cv2 as cv
import numpy as np


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
cv.createTrackbar('Thresh', 'Tracks', 100, 255, lam)
cv.createTrackbar('Open', 'Tracks', 3, 10, lam)
cv.createTrackbar('Dilate', 'Tracks', 3, 100, lam)
cv.createTrackbar('Erode', 'Tracks', 3, 100, lam)


def main():
    # Get and show Cells
    img = cv.imread("levadura.png")
    cv.imshow('cells', img)

    # Get binaryImg and find all the Nuclei
    threshold = cv.getTrackbarPos('Thresh', 'Tracks')
    binary = setBinary(img, threshold)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    quant = str(len(contours))

    # Noise Removal
    open1 = cv.getTrackbarPos('Open', 'Tracks')
    opened = open(binary, open1)

    # Sure Background Area
    dil1 = cv.getTrackbarPos('Dilate', 'Tracks')
    background = dilate(opened, dil1)
    cv.imshow('Background', background)

    # Sure Foreground (Nuclei)
    ero = cv.getTrackbarPos('Erode', 'Tracks')
    foreground = erode(opened, ero)
    foreground = np.uint8(foreground)

    # Unknown Region
    unknown = cv.subtract(background, foreground)

    #Show Images
    cv.putText(binary, quant, (0, 670), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv.imshow('Binary', binary)
    cv.imshow('SureNuclei', foreground)
    cv.imshow('Unknown', unknown)

    # Markers
    ret, markers = cv.connectedComponents(foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    #color = cv.applyColorMap(unknown, cv.COLORMAP_JET)
    #cv.imshow('ColorMap', color)
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv.imshow('Watershed', img)

    if cv.waitKey(0) == ord("r"):
        main()


main()
