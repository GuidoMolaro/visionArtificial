from sklearn import *
from joblib import dump
import matplotlib.pyplot as plt
import xlwings as xw

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

dump(classifier, 'classifier.joblib')
