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
    ret1, thresh1 = cv.threshold(gray, val, 255, cv.THRESH_BINARY_INV)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return thresh1


def dilatation(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    return cv.dilate(img, kernel, iterations=1)

def erosion(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    return cv.erode(img,kernel,iterations=1)

def denoise(img, val1, val2): #TODO: el diam de el kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(val1+1, val2+1))
    tempImg = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(tempImg,cv.MORPH_CLOSE, kernel)

def contours(binary, img):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    #cv.imshow("Rayo", img)
    for i in contours:
        area = cv.contourArea(i)
        if area > 1000:
            cv.drawContours(img, i, -1, (255,0,255),7)
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i,0.02 * peri, True)
    cv.imshow("Contours", img)

def main():
    cv.namedWindow('binary')
    cv.createTrackbar('Thresh', 'binary', 0, 255, setBinary)
    cv.namedWindow('denoised')
    cv.createTrackbar('KSize', 'denoised', 0, 5, denoise)
    while True:
        tecla = cv.waitKey(30)
        ret, img = webcam.read()
        cv.imshow('webcam',img)

        val = cv.getTrackbarPos("Thresh", "binary")

        binaryImg = setBinary(img, val)
        cv.imshow('binary', binaryImg)

        val1 = cv.getTrackbarPos('KSize', 'denoised')
        denoisedImg = denoise(binaryImg, val1, val1)
        cv.imshow('denoised',denoisedImg)

        contours(denoisedImg,img)
        if tecla == 27:
            break

main()