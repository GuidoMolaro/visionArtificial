from PIL import Image
import cv2 as cv
import numpy as np
from sklearn import tree
from joblib import dump, load
from tabulate import tabulate

webcam = cv.VideoCapture(0)

def setBinary(image, val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255,cv.THRESH_BINARY_INV)
    return thresh1

def setBinaryAutom(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
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
    #cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    for i in contours:
        area = cv.contourArea(i)
        if area > 1000:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            cv.drawContours(img, contours, -1, (255, 0, 0), 3)
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

def match(contour, val):
    contours = imagesContours()
    for i in contours.keys():
        distance = cv.matchShapes(contour, contours[i], cv.CONTOURS_MATCH_I2, 0)
        if distance < val:  # el error ponerlo con la barra al tope.
            return i
    return "False"

def descriptionGenerator():
    imCircle = cv.imread('circulo.png', cv.IMREAD_GRAYSCALE)
    _, imCircle = cv.threshold(imCircle, 128, 255, cv.THRESH_BINARY)
    momentCircle = cv.moments(imCircle)

    imRectangle = cv.imread('rectangulo.png', cv.IMREAD_GRAYSCALE)
    _, imRectangle = cv.threshold(imRectangle, 128, 255, cv.THRESH_BINARY)
    momentRectangle = cv.moments(imRectangle)

    imTriangle = cv.imread('triangulo.png', cv.IMREAD_GRAYSCALE)
    _, imTriangle = cv.threshold(imTriangle, 128, 255, cv.THRESH_BINARY)
    momentTriangle = cv.moments(imTriangle)

    huMomentCircle = cv.HuMoments(momentCircle)  # 1
    huMomentRectangle = cv.HuMoments(momentRectangle)  # 2
    huMomentTriangle = cv.HuMoments(momentTriangle)  # 3

    tagDictionary = [["1", "circle"],
                     ["2", "rectangle"],
                     ["3", "triangle"]]
    print(tabulate(tagDictionary))
    dataset = [huMomentCircle,
               huMomentRectangle,
               huMomentTriangle]
def trainer():


    # dataset
    X = [
        [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28,
         -1.12639633e-37],
        [6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37,
         6.53608067e-04],
        [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28,
         -1.12639633e-37],
        [1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16,
         9.67218398e-18],
        [8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19,
         -1.18450102e-37],
        [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28,
         -1.12639633e-37],
        [9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04,
         6.07480284e-16],
    ]

    # etiquetas, correspondientes a las muestras
    Y = [1, 1, 1, 2, 2, 3, 3]

    # entrenamiento
    clasificador = tree.DecisionTreeClassifier().fit(X, Y)

    # visualización del árbol de decisión resultante
    tree.plot_tree(clasificador)

    # guarda el modelo en un archivo
    dump(clasificador, 'filename.joblib')

    # en otro programa, se puede cargar el modelo guardado
    clasificadorRecuperado = load('filename.joblib')



