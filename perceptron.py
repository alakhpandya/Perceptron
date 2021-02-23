import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=10, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, lables):
        print("\nweights\t\tbias\tError\n")
        for _ in range(self.threshold):
            for inputs, lable in zip(training_inputs, lables):
                prediction = self.predict(inputs)
                print( self.weights[1:], "\t", self.weights[0], "\t", lable-prediction)
                self.weights[1:] += self.learning_rate * (lable - prediction) * inputs
                self.weights[0] += self.learning_rate * (lable - prediction)