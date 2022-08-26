import cv2 as cv
import pandas as pd
import openpyxl as openpyxl
def descriptionGenerator(filename):
    im = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    _, im = cv.threshold(im, 128, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    moment = cv.moments(im)
    huMoment = cv.HuMoments(moment).flatten()
    return huMoment

def dataset(info):
    #circulo 1, rectangulo 2, triangulo 3
    dataset = [
        descriptionGenerator('10triangles/t8.jpeg'),
        descriptionGenerator('10triangles/t9.jpeg'),
        descriptionGenerator('10triangles/t10.jpeg'),
        descriptionGenerator('10circles/c1.png'),
        descriptionGenerator('10circles/c2.png'),
        descriptionGenerator('10rectangles/r1.png'),
        descriptionGenerator('10rectangles/r2.png'),
        descriptionGenerator('10triangles/t5.jpeg'),
        descriptionGenerator('10rectangles/r3.png'),
        descriptionGenerator('10circles/c3.png'),
        descriptionGenerator('10rectangles/r5.png'),
        descriptionGenerator('10rectangles/r6.png'),
        descriptionGenerator('10circles/c4.png'),
        descriptionGenerator('10circles/c5.png'),
        descriptionGenerator('10circles/c6.png'),
        descriptionGenerator('10rectangles/r7.png'),
        descriptionGenerator('10rectangles/r8.png'),
        descriptionGenerator('10triangles/t1.jpeg'),
        descriptionGenerator('10triangles/t2.jpeg'),
        descriptionGenerator('10triangles/t3.jpeg'),
        descriptionGenerator('10triangles/t4.jpeg'),
        descriptionGenerator('10rectangles/r9.jpg'),
        descriptionGenerator('10rectangles/r10.png'),
        descriptionGenerator('10circles/c7.png'),
        descriptionGenerator('10circles/c8.png'),
        descriptionGenerator('10rectangles/r4.png'),
        descriptionGenerator('10circles/c9.png'),
        descriptionGenerator('10triangles/t6.jpeg'),
        descriptionGenerator('10triangles/t7.jpeg'),
        descriptionGenerator('10circles/c10.png'),
    ]
    tags = [3, 3, 3, 1, 1, 2, 2, 3, 2, 1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 3, 3, 1]
    if info ==1:
        return dataset
    if info==0:
        return tags
def main():
    data = dataset(1)
    tags = dataset(0)
    col1 = "tags"
    col2 = "huInvariants"
    data = pd.DataFrame({col1: tags, col2: data})
    data.to_excel('data.xlsx', sheet_name='sheet1', index=False)

main()