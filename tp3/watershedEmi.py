import cv2 as cv
import numpy as np
def setBinary(image, val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255,cv.THRESH_BINARY)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return thresh1

def lam(x):
    return x


def main():
    cv.namedWindow('Tracks')
    cv.createTrackbar('Thresh', 'Tracks', 100, 255, lam)
    cv.createTrackbar('Open', 'Tracks', 3, 10, lam)
    cv.createTrackbar('Dilate', 'Tracks', 20, 100, lam)
    cv.createTrackbar('Erode', 'Tracks', 3, 100, lam)

    while True:
        key = cv.waitKey(30)
        threshVal = cv.getTrackbarPos('Thresh', 'Tracks')
        img = cv.imread("levadura.png")
        imgBinary = setBinary(img, threshVal)
        cv.imshow("bin", imgBinary)

        # noise removal
        openVal = cv.getTrackbarPos('Open', 'Tracks')
        kernelOpen = np.ones((openVal, openVal), np.uint8)
        opening = cv.morphologyEx(imgBinary, cv.MORPH_OPEN, kernelOpen, iterations=2)

        # sure background area
        dilateVal = cv.getTrackbarPos('Dilate', 'Tracks')
        kernel = np.ones((dilateVal, dilateVal), np.uint8)
        sure_bg = cv.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        cv.imshow("sureFg", sure_fg)
        cv.imshow("subreBg", sure_bg)

        if key == 27:
            break

main()

