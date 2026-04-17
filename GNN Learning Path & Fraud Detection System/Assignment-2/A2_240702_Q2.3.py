import numpy as np
class LinearSVM:
    def __init__(self,learning_rate,lambda_param,n_iters):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx]*(np.dot(self.w,x_i)-self.b) >=1
                if condition:
                    dw = 2*self.lambda_param*self.w
                    self.w -= self.lr*dw
                else:
                    dw = 2*self.lambda_param*self.w - y_[idx]*x_i
                    db = y_[idx]
                    self.w -= self.lr*dw
                    self.b -= self.lr*db
    def predict(self, X):
        linear_output = np.dot(X,self.w)-self.b
        return np.sign(linear_output)
