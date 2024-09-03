# -*- coding:utf8 -*-
# @Author  :Dodo

import numpy as np

class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            # print(error)

            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            # print(error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output

        #converting values to floats
        inputs = inputs.astype(float)

        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output

    def predict(self, inputs):
        #predicting the inputs and get the binary output zero or one

        inputs = inputs.astype(float)

        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        for index, item in enumerate(output):
            if item > 0.5:
                output[index] = 1
            else :
                output[index] = 0

        return output

if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T
    # training_outputs1 = np.array([[1,0,0,1]]).T
    # print(training_outputs * training_outputs1)

    #training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    # test = [[1, 1,0],
    #         [0,1,0]]
    # test = np.array(test)
    # output = neural_network.predict(test)
    # print(output)