# Ejercicios con la cámara, en tiempo real, mostrando siempre en una ventana la imagen de la cámara.  Requiere imágenes binarias,
# que se pueden obtener por thresholding.
# Dilatar una imagen binaria, controlar el tamaño del elemento estructural con un trackbar, de 1 a 50
# Erosionar
# Aplicar opening y closing consecutivamente, para filtrar ruidos
import math

from PIL import Image
import cv2 as cv
import numpy as np

webcam = cv.VideoCapture(1)


def setBinary(image, val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255,cv.THRESH_BINARY_INV)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return thresh1

def setBinaryAutom(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
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

    # cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv.imshow("Rayo", img)
    for i in contours:
        area = cv.contourArea(i)
        if area > 1000:
            cv.drawContours(img, i, -1, (255, 255, 255), 7)
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
    cv.imshow("Contours", img)
    return contours

def getBiggestContour(contours):
    try:
        max_cnt = contours[0]
        for cnt in contours:
            if cv.contourArea(cnt) > cv.contourArea(max_cnt):
                max_cnt = cnt
        return max_cnt
    except:
        return contours
def imagesContours(): #devuelve un array con todos los contornos de las img
    circulo = setBinaryAutom(np.array(Image.open('circulo.png')))
    #circulo = setBinaryAutom('tp1/circulo.png')
    triangulo = setBinaryAutom(np.array(Image.open('triangulo.png')))
    rectangulo = setBinaryAutom(np.array(Image.open('rectangulo.png')))

    circuloContour, hierarchy = cv.findContours(circulo, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    trianguloContour, hierarchy = cv.findContours(triangulo, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    rectanguloContour, hierarchy = cv.findContours(rectangulo, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    contours = {
        "circulo": getBiggestContour(circuloContour),
        "rectangulo": getBiggestContour(rectanguloContour),
        "triangulo": getBiggestContour(trianguloContour)
    }
    return contours

def match(contour):
    moments = cv.moments(contour)
    huMoments = cv.HuMoments(moments)

    contours = imagesContours()
    for i in contours.keys():
        distance = cv.matchShapes(contour,contours[i],cv.CONTOURS_MATCH_I2,0)
        if distance < 0.3:
            return i
    return "null"


def main():
    cv.namedWindow('binary')
    cv.createTrackbar('Thresh', 'binary', 0, 255, setBinary)
    cv.namedWindow('denoised')
    cv.createTrackbar('KSize', 'denoised', 0, 5, denoise)
    while True:
        tecla = cv.waitKey(30)
        ret, img = webcam.read()
        cv.imshow('webcam', img)

        val = cv.getTrackbarPos("Thresh", "binary")

        binaryImg = setBinary(img, val)
        cv.imshow('binary', binaryImg)

        val1 = cv.getTrackbarPos('KSize', 'denoised')
        denoisedImg = denoise(binaryImg, val1, val1)
        cv.imshow('denoised', denoisedImg)

        contours = getContours(denoisedImg, img)

        if match(getBiggestContour(contours)) == "null":
            for i in contours:
                cv.drawContours(img, i, -1, (0, 0, 255), 7)
        else:
            for i in contours:
                cv.drawContours(img, i, -1, (0, 255, 0), 7)
        if tecla == 27:
            break


main()