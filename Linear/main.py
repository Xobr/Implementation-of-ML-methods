import Linear.LinearRegression as linearR
import Linear.LogisticRegression as logisticR
import dataSetManager as dm
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
import math as mt


def tranform_x(x):
    res = list()
    for i in range(0,len(x[0])):
        ls = [elem[i] for elem in x]
        res.append(ls)
    return res

def evaluate_linear_regression():
    x_train, y_train, x_test, y_test = dm.get_LR_datset()

    lrsk = LinearRegression()
    lrsk.fit(tranform_x(x_train),y_train)
    prdC = lrsk.predict(tranform_x(x_test))
    forecast_errors1 = [mt.fabs(y_test[i]-prdC[i]) for i in range(len(prdC))]
    bias = sum(forecast_errors1) * 1.0/len(prdC)
    print('SK Bias: %f' % bias)

    lr = linearR.LinearRegression()
    lr.fit(x_train, y_train)
    pred = lr.predict_for_many(x_test)

    forecast_errors = [mt.fabs(y_test[i]-pred[i]) for i in range(len(pred))]
    bias = sum(forecast_errors) * 1.0/len(pred)
    print('Bias: %f' % bias)

def evaluate_logistic_regression():
    x_train, y_train, x_test, y_test = dm.getZeroOneDataSet()

    lrsk = LogisticRegression()
    lrsk.fit(tranform_x([x_train]), y_train)
    prdC = lrsk.predict(tranform_x([x_test]))
    print('Accuracy', metrics.accuracy_score(y_test, prdC))

    lr = logisticR.LogisticRegression()
    lr.fit(x_train, y_train)
    pred = lr.predict_for_many(x_test)
    print('Accuracy', metrics.accuracy_score(y_test, pred))


evaluate_logistic_regression()