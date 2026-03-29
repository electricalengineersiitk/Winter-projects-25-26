import numpy as np

class MLP_XOR_Classifier:

    def __init__(self, hidden_neurons=2, learning_rate=0.1, n_epochs=10000):
        self.lr = learning_rate
        self.epochs = n_epochs
        self.H = hidden_neurons 
        
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        np.random.seed(42) 
        
        
        self.W1 = np.random.randn(2, self.H) * 0.01
        self.b1 = np.zeros((1, self.H))
        
        
        self.W2 = np.random.randn(self.H, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self):
        
        N = self.X.shape[0] 
        
        for epoch in range(self.epochs):

            # Forward
            Z1 = np.dot(self.X, self.W1) + self.b1 
            A1 = self._sigmoid(Z1)                
            
            # Hidden to Output Layer
            Z2 = np.dot(A1, self.W2) + self.b2   
            A2 = self._sigmoid(Z2)  
            
            # Loss 
            E = A2 - self.y 
            loss = np.mean(E**2)
            
            # Backpropagation 
            Delta2 = (2 * E / N) * self._sigmoid_derivative(A2) 
            
            dW2 = np.dot(A1.T, Delta2) 
            db2 = np.sum(Delta2, axis=0, keepdims=True)
            
            dLdA1 = np.dot(Delta2, self.W2.T) 
            
            Delta1 = dLdA1 * self._sigmoid_derivative(A1) 
            
        
            dW1 = np.dot(self.X.T, Delta1) 
            db1 = np.sum(Delta1, axis=0, keepdims=True) 
            
            # Update 
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        print(f"\nTraining Complete. Final Loss: {loss:.6f}")
        
    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._sigmoid(Z1)
        
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)
        
        return (A2 >= 0.5).astype(int)
