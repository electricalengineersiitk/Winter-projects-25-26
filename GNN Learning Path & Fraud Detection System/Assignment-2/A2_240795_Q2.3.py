import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self,x,y):
        
        x_rows ,x_column = x.shape

        self.w = np.zeros(x_column)
        self.b = 0

        y_final = np.where(y <=0 ,-1 ,1)

        for _ in range(self.n_iters):
            for idx , x_i in enumerate(x):
                condition = y_final[idx] *(self.w@x-b)

                if(condition >=1):
                    gradient = 2*self.lambda_param*self.w
                    self.w = self.w -(self.learning_rate*gradient)
                
                else:
                    gradient = 2*self.lambda_param*self.w - y_final[idx]*x_i
                    self.w = self.w -(self.learning_rate*gradient)

                    db = y_final[idx]
                    self.b = b-self.learning_rate*db
    def predict(self,x):
        output = self.w@x -self.b

        return np.sign(output)