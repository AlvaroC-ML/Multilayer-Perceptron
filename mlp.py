import numpy as np
from numpy import random

import functions as func

class myNeuralNetwork:
    # The first and last elements are the inputs and outputs
    # For now, all activation functions are sigmoid.
    # For now, the cost function is the mean squared error.
    def __init__(self, layer_dimensions):
        self.dimensions = layer_dimensions
        print("Starting neural network.")
        
        print("Number of layers:", len(layer_dimensions))
        self.weights = []
        self.biases = []
        # Initializing weights and biases. Indices are off by one.
        
        self.weights = [random.rand(i, j) 
                        for i, j 
                        in zip(self.dimensions[:-1], self.dimensions[1:])]
        self.biases = [random.rand(1, i) 
                        for i
                        in self.dimensions[1:]]
    
    # Input needs to be a 2D numpy array, row.
    def evaluate(self, data):
        for w, b in zip(self.weights, self.biases):
            data = func.sigmoid(np.matmul(data, w) + b)
        return data
    
    # Returns the values calculated during forwardprop.
    # Input need to be a 2D numpy array, row.
    def forwardprop(self, data):
        z = data
        a = [z] # Results pre-activation
        counter = 0
        for w, b in zip(self.weights, self.biases):
            a.append(np.matmul(z,w) + b)
            z = func.sigmoid(a[counter+1])
            counter = counter+1
        return a        
            
    # Applies stochatic gradient descent using the data given
    # Input needs to be in a list where each element is data. Alpha is the learning rate.
    def StochasticGradientDescent(self, data_input, data_result, batch_size, alpha):
        num_data = len(data_input)
        num_full_batches = int(num_data / batch_size)
        leftover = num_data - (batch_size*num_full_batches)
        
        # Shuffle data
        shuffle = random.permutation(len(data_input))
        data_input = [data_input[i] for i in shuffle]
        data_result = [data_result[i] for i in shuffle]
        
        # Go over each batch
        for i in range(num_full_batches):
            # Create batches
            batch_in = data_input[i:i+batch_size]
            batch_res = data_result[i:i+batch_size]
            
            # Calculate derivatives
            update_w, update_b = self.batchProcessing(batch_in, batch_res)
            
            # Update weights and biases
            self.weights = [original - ((alpha/batch_size)*update)
                           for original, update
                            in zip(self.weights, update_w)]
            self.biases = [original - ((alpha/batch_size)*update)
                           for original, update
                            in zip(self.biases, update_b)]
        
        # Leftovers are treated separately
        if leftover>0:
            batch_in = data_input[num_data-leftover:num_data]
            batch_res = data_res[num_data-leftover:num_data]

            update_w, update_b = self.batchProcessing(batch_in, batch_res)
            
            self.weights = [original - ((alpha/leftover)*update)
                           for original, update
                            in zip(self.weights, update_w)]
            self.biases = [original - ((alpha/left_over)*update)
                           for original, update
                            in zip(self.biases, update_b)]       

    # Calculates the aggregate of a batch
    # Inputs need to be in a list where each element is an input
    def batchProcessing(self, data_input, data_result):    
        # For each data part, we calculate the derivatives, and we add them to the holder.
        counter = 0
        holder_w = [np.zeros(weight.shape) for weight in self.weights]
        holder_b = [np.zeros(bias.shape) for bias in self.biases]
        
        for d_in, d_res in zip(data_input, data_result):
            h_w, h_b = self.backprop(d_in, d_res)
            holder_w = [original + new for original, new in zip(holder_w, h_w)]
            holder_b = [original + new for original, new in zip(holder_b, h_b)]
        
        return holder_w, holder_b

    
    # Calculates the derivative of all weights and biases in network.
    # Inputs need to be a 2D numpy array, row.
    def backprop(self, data_input, data_result):
        L = len(self.biases) # Number of layers excluding input
        
        a = self.forwardprop(data_input)
        z = [func.sigmoid(a_i) for a_i in a]
        # L is the index of the last layer
        
        # Initialize outputs
        deltaC_weights = list(range(L))
        deltaC_biases = list(range(L))
        # L-1 is the index of the last layer.
        
        # Derivatives of last layer
        deltaC_aL = np.multiply(z[L]-data_result,
                                 func.sigmoid_p(a[L])) 
        deltaC_biases[L-1] = np.multiply(deltaC_aL, func.sigmoid_p(z[L]))
        
        # Derivatives of weights to last layer
        deltaC_weights[L-1] = np.multiply(deltaC_biases[L-1], 
                                          np.transpose(a[L-1]))
        
        # Other derivatives
        for i in reversed(range(L-1)):
            deltaC_biases[i] = np.multiply(np.matmul(deltaC_biases[i+1], np.transpose(self.weights[i+1])),
                                           func.sigmoid_p(z[i+1]))
            deltaC_weights[i] = np.multiply(deltaC_biases[i],
                                          np.transpose(a[i]))

        return deltaC_weights, deltaC_biases
            
    # Input needs to be a list where each item is data in rows.
    def costFunction(self, data_input, data_result):
        inputs = int(len(data_input))
        current_cost = 0
        for data_in, data_res in zip(data_input, data_result):
            current_cost = current_cost + func.MSE(self.evaluate(data_in),
                                              data_res)
        return current_cost/(2*inputs)
            
    def train(self, data_input, data_result, epochs, batch_size, alpha):
        print("Number of epochs:", epochs)
        for i in range(epochs):
            print("Cost before epoch", i+1, "is:",
                 self.costFunction(data_input, data_result))
            self.StochasticGradientDescent(data_input, data_result, batch_size, alpha)
            print("Epoch", i+1, "completed")
        print("Learning done. Final cost:", self.costFunction(data_input, data_result))
    
    # Input needs to be a 2D array. Outputs true if prediction is correct.
    def prediction_works(self, data_input, data_result):
        result = self.evaluate(data_input)
        if np.array_equal(result >= np.amax(result), data_result == 1): # Correct prediction
            return True 
        else:
            return False
                    
    # Receives many data points. Returns how many are right.
    def performance(self, data_input, data_result):
        inputs = int(len(data_input))
        counter = 0
        for data_in, data_res in zip(data_input, data_result):
            if self.prediction_works(data_in, data_res):
                counter = counter + 1
        print("Number of inputs correctly classified:", counter)
        print("Number of inputs:", inputs)
        print("Performance of model:", counter/inputs * 100)