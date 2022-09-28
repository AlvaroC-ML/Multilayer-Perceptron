import numpy as np

#################
# TRAINNING
#################

def sigmoid(z):
    return np.divide(1, 1+np.exp(-z))

def sigmoid_p(z):
    return np.divide(np.exp(-z), np.power(1+np.exp(-z), 2))

# Only works for row vectors
def MSE(v1, v2):
    return np.vdot(v1-v2, v1-v2) / (2*v1.shape[1])


#################
# DATA PROCESSING
#################


# Input is a 2 dimensional grayscale picture.
# Output is a row vector with the grayscale values.
def image_flat(image):
    x_dim = image.shape[0]
    flat = np.array([])
    for i in range(x_dim):
        flat = np.append(flat, image[i], axis = 0)
    return np.array(flat, ndmin = 2) # Convert it into a row vector

# It is assumed all pictures have the same dimensions
def input_list(images):
    inputs = []
    for image in images:
        if image.shape == ():
            continue
        inputs.append(image_flat(image))
    return inputs

# The minimum value is assumed to be 0, the max value is max_value-1
# Output is a row vector
def one_hot_encoding(value, max_value):
    vector = np.zeros((1, max_value))
    vector[0, value] = 1
    return vector

# Transforms a list of outputs into a matrix with outputs hot-encoded
def output_list(outputs, max_value):
    output = []
    for value in outputs:
        output.append(one_hot_encoding(value, max_value))
    return output