# Creating Activation Functions
def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis = 0)

def ohe(Y):
    
    one_hot_Y = np.zeros(shape = (y.size, y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def loss(pred, test):

    return ((pred - ohe(test))**2).mean()

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def relu_deriv(Z):

    return Z > 0



# Converting data to numpy array and Transposing
X0 = training_data_df.drop(columns = ['label']).to_numpy().T
X0 = X0/ 100
y = training_data_df['label'].to_numpy().T

# Getting Size of Features and Samples
n,m = X0.shape

# Initializing Size of Hidden Layer Nodes
h1_n = 10
h2_n = 10

def init_params(X,h1_n, h2_n):
    #  Intializing Weight Matrix 
    W1 = np.random.rand(h1_n,n) - 0.5
    W2 = np.random.rand(h2_n,h1_n) -0.5

    # Initializing Biases Vectors
    b1 = np.zeros(shape = (h1_n,1))
    b2 = np.zeros(shape = (h2_n,1))

    return W1, b1, W2,b2 
def forward_prop (X, W1, b1, W2, b2):

    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)

    Z2 = W2.dot(A1) +b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2



def back_prop (X, y, W1, Z1, A1, W2, A2):
    dZ2 = A2 - ohe(y)
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) * np.sum(dZ2)

    dZ1 = np.dot(W2.T,dZ2) * relu_deriv(Z1)
    dW1 = (1/m) * np.dot(dZ1,X.T)
    db1 = (1/m) * np.sum(dZ1)


    return dW1, db1, dW2, db2

def updated_params(W1,dW1, W2,dW2, b1, db1, b2, db2, learning_rate = 1 ):
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)

    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    return W1, b1, W2 , b2

W1, b1, W2, b2 = init_params(X0,10,10)

for i in range(500):
    Z1, A1, Z2, A2 = forward_prop(X0, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = back_prop (X0, y, W1, Z1, A1, W2, A2)
    W1, b1, W2, b2 = updated_params(W1,dW1, W2,dW2, b1, db1, b2, db2, learning_rate = 0.1 )

    if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, y))

