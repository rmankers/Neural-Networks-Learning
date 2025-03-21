from sklearn.preprocessing import StandardScaler

# Create an instance of StandardScaler
scaler = StandardScaler()

# Normalize the matrix X column-wise
normalized_X = scaler.fit_transform(X)

# Print the normalized matrix
print(normalized_X)

class neural_network:



    def __init__(self, hidden_layers: int, node_counts: list):

        # Number of hidden layers 
        self.hidden_layers = hidden_layers
        if type(self.hidden_layers) != int:
            raise ValueError('hidden_layers must be an integer')
        
        # Number of nodes in each hidden layer
        self.node_counts = node_counts
        if type(self.node_counts) != list:
            raise ValueError('node_counts must be a list with length = hidden_layers')
        

        # Check if length of nodes_array matches the number of hidden_layers
        if hidden_layers != len(node_counts):
            raise ValueError('Number of hidden layers must match number of node counts')
        

    # Activation Functions
    def relu(self,x):
        return np.maximum(0,x)
    
    
    def relu_deriv(self,Z):

        return Z > 0

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x),axis = 0)
    

    def ohe(self,Y):
    
        one_hot_Y = np.zeros(shape = (y.size, y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    

    def loss(self,pred, test):

        return ((pred - ohe(test))**2).mean()

    def get_predictions(self,A2):
        return np.argmax(A2, 0)

    def get_accuracy(self,predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    # Intializing Network Parameters
    def init_params(self,X,h1_n, h2_n, input_size):

        #  Weight Matrix centered at zero
        W1 = np.random.rand(h1_n,n) - 0.5
        W2 = np.random.rand(h2_n,h1_n) - 0.5

        # Initializing Biases Vectors
        b1 = np.zeros(shape = (h1_n,1))
        b2 = np.zeros(shape = (h2_n,1))

        return W1, b1, W2,b2 


    # Forward Prop
    def forward_prop (self,X, W1, b1, W2, b2):

        Z1 = W1.dot(X) + b1
        A1 = self.relu(Z1)

        Z2 = W2.dot(A1) +b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2
    
    # Backward Propogation
    def back_prop (self,X, y, W1, Z1, A1, W2, A2):
        dZ2 = A2 - self.ohe(y)
        dW2 = (1/m) * np.dot(dZ2,A1.T)
        db2 = (1/m) * np.sum(dZ2)

        dZ1 = np.dot(W2.T,dZ2) * self.relu_deriv(Z1)
        dW1 = (1/m) * np.dot(dZ1,X.T)
        db1 = (1/m) * np.sum(dZ1)


        return dW1, db1, dW2, db2
    
    # Param Update
    def updated_params(self,W1,dW1, W2,dW2, b1, db1, b2, db2, learning_rate = 1 ):
        W2 = W2 - (learning_rate * dW2)
        b2 = b2 - (learning_rate * db2)

        W1 = W1 - (learning_rate * dW1)
        b1 = b1 - (learning_rate * db1)
        return W1, b1, W2 , b2
    
    # Gradient Descent

    def fit(self,X,y, learning_rate = 0.1, epochs = 1000):
        W1, b1, W2, b2 = self.init_params(X,self.node_counts[0],self.node_counts[0],X.shape[0])

        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward_prop(X, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = self.back_prop (X, y, W1, Z1, A1, W2, A2)
            W1, b1, W2, b2 = self.updated_params(W1,dW1, W2,dW2, b1, db1, b2, db2, learning_rate = 0.1 )

            if i % 10 == 0:
                    print("Iteration: ", i)
                    predictions = self.get_predictions(A2)
                    print(self.get_accuracy(predictions, y))

nn = neural_network(hidden_layers = 2,node_counts= [10,10] )
nn.fit(X = normalized_X, y = y , learning_rate = 0.1, epochs = 1000)
