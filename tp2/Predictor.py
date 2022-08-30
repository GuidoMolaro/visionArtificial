from math import copysign, log10

import cv2 as cv
from joblib import load

webcam = cv.VideoCapture(1)
classifier = load('classifier.joblib')

def setBinary(image, val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255, cv.THRESH_BINARY_INV)
    return thresh1

def dilatation(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return cv.dilate(img, kernel, iterations=1)

def erosion(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return cv.erode(img, kernel, iterations=1)

def denoise(img, val1, val2):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (val1 + 1, val2 + 1))
    tempImg = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(tempImg, cv.MORPH_CLOSE, kernel)

def getContours(binary, img):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    for i in contours:
        area = cv.contourArea(i)
        if area > 0.5:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            cv.drawContours(img, contours, -1, (255, 0, 0), 3)
    return contours

def lam(x):
    pass
def main():
    cv.namedWindow('binary')
    cv.createTrackbar('Thresh', 'binary', 100, 255, lam)
    cv.namedWindow('denoised')
    cv.createTrackbar('KSize', 'denoised', 1, 5, lam)
    while True:
        tecla = cv.waitKey(30)
        ret, img = webcam.read()

        valBinary = cv.getTrackbarPos("Thresh", "binary")

        binaryImg = setBinary(img, valBinary)
        cv.imshow('binary', binaryImg)

        valKS = cv.getTrackbarPos('KSize', 'denoised')
        denoisedImg = denoise(binaryImg, valKS, valKS)
        cv.imshow('denoised', denoisedImg)

        contours = getContours(denoisedImg, img)

        for i in contours:
            if cv.contourArea(i) > 1000:
                moments = cv.moments(i)
                huMoments = cv.HuMoments(moments)
                for j in huMoments:
                    huMoments[j] = -1 * copysign(1.0, huMoments[j]) * log10(abs(huMoments[j]))
                # analyze = huMoments.reshape(1,-1)
                result = classifier.predict([huMoments])
                x, y, w, h = cv.boundingRect(i)
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(img, str(result), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.imshow('webcam', img)
        if tecla == 27:
            break

main()