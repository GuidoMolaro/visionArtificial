#Ejercicios con la cámara, en tiempo real, mostrando siempre en una ventana la imagen de la cámara.  Requiere imágenes binarias,
#que se pueden obtener por thresholding.
#Dilatar una imagen binaria, controlar el tamaño del elemento estructural con un trackbar, de 1 a 50
#Erosionar
#Aplicar opening y closing consecutivamente, para filtrar ruidos
import cv2 as cv
webcam = cv.VideoCapture(0)

def setBinary(image,val):
    imWebcam = image
    gray = cv.cvtColor(imWebcam, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return thresh1


def dilatation(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    return cv.dilate(img, kernel, iterations=1)

def erosion(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    return cv.erode(img,kernel,iterations=1)

def denoise(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    tempImg = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(tempImg,cv.MORPH_CLOSE, kernel)

def contours(binary, img):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # contours = [cnt1, cnt2, cnt3]
    cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv.imshow("Rayo", img)

while True:
    tecla = cv.waitKey(30)
    ret, img = webcam.read()
    cv.imshow('webcam',img)

    binaryImg = setBinary(img,85)
    cv.imshow('bianry',binaryImg)

    denoisedImg = denoise(binaryImg)
    cv.imshow('denoised',denoisedImg)

    contours(denoisedImg,img)
    if tecla == 27:
        break