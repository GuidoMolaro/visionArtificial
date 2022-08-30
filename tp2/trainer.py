from math import copysign, log10
import numpy as np
from sklearn import tree
from joblib import dump, load
from sklearn.datasets import load_iris
from DescriptorGenerator import dataset
import matplotlib.pyplot as plt
# Python3 code to select
# data from excel
import xlwings as xw

# Specifying a sheet
ws = xw.Book("data.xlsx").sheets['sheet1']
y = ws.range("A2:A31").value
hu1 = ws.range("B2:B31").value
hu2 = ws.range("C2:C31").value
hu3 = ws.range("D2:D31").value
hu4 = ws.range("E2:E31").value
hu5 = ws.range("F2:F31").value
hu6 = ws.range("G2:G31").value
hu7 = ws.range("H2:H31").value

x = [
    [float(hu1[i].strip("[]")),
     float(hu2[i].strip("[]")),
     float(hu3[i].strip("[]")),
     float(hu4[i].strip("[]")),
     float(hu5[i].strip("[]")),
     float(hu6[i].strip("[]")),
     float(hu7[i].strip("[]"))]
    for i in range(30)
]

classifier = tree.DecisionTreeClassifier().fit(x, y)
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(classifier)
fig.savefig("decistion_tree.png")
# guarda el modelo en un archivo
dump(classifier, 'classifier.joblib')
