import numpy as np
class LinearSVM:
    def __init__(self,learning_rate,lambda_param,n_iters):
        self.lr=learning_rate
        self.lmb=lambda_param
        self.iters=n_iters
    def fit(self,X,y):
        for i in range(len(y)):
            if y[i]<=0:
                y[i]=-1
            else:
                y[i]=1
        n_rows,n_cols=X.shape
        self.w=np.zeros(n_cols)
        self.b=0
        for i in range(self.iters):
            for j in range(n_rows):
                hinge=y[j]*(np.dot(self.w,X[j])-self.b)
                if hinge>=1:
                    self.w-=self.lr*(2*self.lmb*self.w)
                else:
                    self.w-=self.lr*(2*self.lmb*self.w-y[j]*X[j])
                    self.b-=self.lr*y[j]
    def predict(self,X):
        output=np.dot(X,self.w)-self.b
        if output>0:
            print(1)
        else:
            print(-1)