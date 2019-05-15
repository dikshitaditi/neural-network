
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
class NeuralNet:
    scaler = MinMaxScaler()
    activation = " "
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        train_dataset = pd.read_csv(train)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols-1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)     
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y)
        #print(self.X_test.shape)
        LE = LabelEncoder()
        self.y_train[:,0] = LE.fit_transform(self.y_train[:,0])
        #print(self.y_train)
        self.y_test[:,0] = LE.fit_transform(self.y_test[:,0])

        self.X_train = self.preprocess(self.X_train)      
        #
        # Find number of input and output layers from the dataset
        #

        input_layer_size = len(self.X_train[0])
        if not isinstance(self.y_train[:,0], np.ndarray):
            output_layer_size = 3
            print("op layer size", output_layer_size)
        else:
            output_layer_size = len(self.y_train[0])
           # print(self.y_train[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X_train
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X_train), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X_train), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    #definition of activation functions
    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self,x)
        if activation == "relu":
            self.__relu(self,x)

    def __sigmoid(self, x):
        x = x.astype('float64')
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
         x = x.astype('float64')
         return ((np.exp(x) - np.exp(-x))/ (np.exp(x) + np.exp(-x)))

    def __relu(self, x):
         x = x.astype('float64')
         return np.maximum(0,x)

    # derivative of activation function, indicates confidence about existing weight
    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh": 
            self.__tanh_derivative(self, x)
        if activation == "relu": 
            self.__relu_derivative(self, x)

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return (1 - (x**2))

    def __relu_derivative(self,x):
        #print("x",x.shape)
        for i in range(111):
            for j in range (x[i].size):
                 x[i,j] = 1 if x[i,j]>0 else 0
        return x

    #pre-process function
    def preprocess(self, X):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X = normalize(X, norm='l2')
        return X

    # Below is the training function 
    def train(self, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(self.X_train)
            error = 0.5 * np.power((out - self.y_train), 2)
            update_layer2 = 0.00
            update_layer1 = 0.00
            update_input  = 0.00
            self.backward_pass(out, self.activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 =self.w23 + update_layer2
            self.w12 = self.w12 + update_layer1
            self.w01 = self.w01 + update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, X):
        # pass our inputs through our neural network
        if(self.activation == "sigmoid"):
            in1 = np.dot(X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
            return out
        if(self.activation == "tanh"):
            in1 = np.dot(X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
            return out
        if(self.activation == "relu"):
            in1 = np.dot(X, self.w01 )
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
            return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)


    def compute_output_delta(self, out, activation):
        self.delta_output = 0*self.deltaOut;
        if activation == "sigmoid":
            delta_output = (self.y_train - out) * (self.__sigmoid_derivative(out))

        if activation == "tanh":
            delta_output = (self.y_train - out) * (self.__tanh_derivative(out))

        if activation == "relu":
            delta_output = (self.y_train - out) * (self.__relu_derivative(out))
        self.deltaOut = delta_output


    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))

        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        if activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2


    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    #predict function 
    def predict(self, X_test, header = True):
        X_test = self.scaler.transform(self.X_test)
        out = self.forward_pass(X_test)
        error = 0.5*np.power((out - self.y_test), 2)
        return error


if __name__ == "__main__":
    neural_network = NeuralNet("train.csv")
    neural_network.activation = "relu"
    neural_network.train()
    testError = neural_network.predict("X_test")
    print("The testerror is : ",np.sum(testError))