import random as rn
import numpy as np
import math

class LinearRegression:

    weights = []

    def __init__(self, alpha = 0.000001, accuracy = 0.01 , max_iteration = 10000):
        self.aplpha = alpha
        self.accuracy = accuracy
        self.max_iteration = max_iteration

    def main_fnk(self, w, x):
        len_v = len(w)
        len_x = len(x)
        if(len_v != len_x):
            raise Exception('Lenghth of arrays must be the same')
        sum = 0
        for i in range(0,len_v):
            sum += w[i] * x[i]
        return sum

    def initialize_weights(self,count):
        for i in range(0,count):
           yield rn.randrange(0,10)

    def cost_function(self, w , x , y , count_features = 0):
        if(count_features == 0):
            count_features = len(y)
        sum = 0.0
        for i in range(0, count_features):
            sum += math.pow(self.main_fnk(w, [elem[i] for elem in x]) - y[i] , 2)
        return sum/(2 * count_features)

    def gradient(self, x , y, w = None):
        count = len(x)
        count_features = len(y)
        if w is None:
            w = list(self.initialize_weights(count))
        last_cost_fnk = self.cost_function(w, x,y,count_features)

        for i in range(0, self.max_iteration):
            w_new = w[:]
            for j in range(0,count):
                w_new[j] = w[j] - self.aplpha * (1/count_features) * self.calculate_sum(x, y, w, j, count_features)
            cost_fnk = self.cost_function(w_new, x, y, count_features)
            w = w_new[:]
            if math.fabs(last_cost_fnk - cost_fnk) < self.accuracy:
                print('LR iterations: ', i)
                break
            last_cost_fnk = cost_fnk

        return w

    def calculate_sum(self, x, y, w ,j ,count_features = 0):
        sum = 0
        if(count_features == 0):
            count_features = len(x[0])
        for i in range(0, count_features):
            sum += (self.main_fnk(w, [elem[i] for elem in x]) - y[i]) * x[j][i]
        return sum

    def fit(self, x, y):
        try:
            x_trans = list()
            x_trans.append(list(np.full(len(x[0]),1)))
            x_trans.extend([elem[:] for elem in x])
            if len(self.weights) == 0:
                self.weights = self.gradient(x_trans, y)
            else:
                self.weights = self.gradient(x_trans, y, self.weights[:])
        except OverflowError:
            print('Something want wrong with gradient. Please change alpha(speed of learning) to low value.')

    def predict(self , x):
        x_trans = [1]
        x_trans.extend(x)
        return self.main_fnk(self.weights,x_trans)

    def predict_for_many(self,x):
        y = list()
        for i in range(0, len(x[0])):
            param = [elem[i] for elem in x]
            y.append(self.predict(param))
        return y
