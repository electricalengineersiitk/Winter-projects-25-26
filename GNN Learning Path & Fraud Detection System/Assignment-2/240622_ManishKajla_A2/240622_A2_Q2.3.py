import numpy as np

class LinearSVM:
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Weight vector
        self.b = None  # Bias
        
    def _convert_labels(self, y):
        
        y_new = np.copy(y)
        y_new[y_new == 0] = -1
        return y_new

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        y_ = self._convert_labels(y)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_i = y_[idx]
                margin = y_i * (np.dot(self.w, x_i) - self.b)
                
                dw = np.zeros_like(self.w)
                db = 0
                
                if margin >= 1:
                    dw = 2 * self.lambda_param * self.w
                    db = 0 
                else:
                    dw = 2 * self.lambda_param * self.w 
                    dw -= y_i * x_i 
                    
                    db = y_i

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
        
    def predict(self, X):
        
        linear_output = np.dot(X, self.w) - self.b
        
        return np.sign(linear_output)
