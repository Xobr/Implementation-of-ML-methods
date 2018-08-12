import random as rn
import numpy as np
import math

exp = math.exp(1.0)
class LogisticRegression:

    def __init__(self, alpha = 0.1, accuracy = 0.0001 , max_iteration = 10000):
        self.alpha = alpha
        self.accuracy = accuracy
        self.max_iteration = max_iteration
        self.weight = None

    def main_fnk(self, w, x):
        res = 1.0 / float(1.0 + math.exp(-w * x))
        if res == 1:
            res -= self.alpha
        return res

    def cost_function(self, w , x , y , count_features = 0):
        if count_features == 0:
            count_features = len(y)
        sum = 0.0
        for i in range(0, count_features):
            f = self.main_fnk(w,  x[i])
            sum += y[i] * math.log(f,exp) + ((1.0 - y[i]) * math.log(1.0 - f,exp))
        return -sum/count_features

    def gradient(self, x , y, w = None):
        count_features = len(y)
        if w is None: # move it to fit method
            w = rn.randrange(-100, 100)
        last_cost_fnk = self.cost_function(w, x, y, count_features)

        for i in range(0, self.max_iteration):
            if w is None:
                print("Trig")
            w_new = w - self.alpha * (1 / count_features) * self.calculate_sum(x, y, w, count_features)
            cost_fnk = self.cost_function(w_new, x, y, count_features)
            w = w_new

            if math.fabs(last_cost_fnk - cost_fnk) < self.accuracy:
                print('LR iterations: ', i)
                # break
            last_cost_fnk = cost_fnk
        return w

    def calculate_sum(self, x, y, w ,count_features = 0):
        sum = 0
        if count_features == 0:
            count_features = len(x[0])
        for i in range(0, count_features):
            sum += (self.main_fnk(w, x[i]) - y[i]) * x[i]
        return sum

    def fit(self, x, y):
        if self.weight is None:
            self.weight = self.gradient(x, y)
        else:
            self.weight = self.gradient(x, y, self.weight[:])
        return self.weight

        try:
            if self.weight is None:
                self.weight = self.gradient(x, y)
            else:
                self.weight = self.gradient(x, y, self.weight[:])
        except OverflowError:
            print('Something want wrong with gradient. Please change alpha(speed of learning) to lowest value.')
            raise OverflowError()

    def predict(self , x):
        return self.main_fnk(self.weight, x)

    def predict_for_many(self,x):
        y = list()
        for i in range(0, len(x)):
            y.append(self.predict(x[i]))
        return y
