import functions as func
import mlp
from tensorflow import keras

# Data loading
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Normalizing
x_train, x_test = x_train/255, x_test/255
# Preparing for network
x_train, x_test = func.input_list(x_train), func.input_list(x_test)
y_train, y_test = func.output_list(y_train, 10), func.output_list(y_test, 10)

#Hyperparameters
epochs = 10
batch_size = 50
learning_rate = 0.5

nn = mlp.myNeuralNetwork([28*28, 10, 10])

print("\nHere's how the model does without any training.\n")

nn.performance(x_test, y_test)

print("\nNow let's train.\n")

nn.train(x_train, y_train, epochs, batch_size, learning_rate)

print("\nLet's check performance.\n")

nn.performance(x_test, y_test)