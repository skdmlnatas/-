import numpy as np

class Perceptron():
    def __init__(self, input_size, threshold = 100) -> None:
        self.threshold = threshold
        self.weights = np.zeros(input_size + 1)
        self.count = 0
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) * self.weights[0]

        if summation >0:
            activation = 1
        else:
            activation = -1
        return activation

    def train(self, training_inputs, labels):
        print("training...")
        for _ in range(self.threshold):
            m = 0
            for input, label in zip(training_inputs, labels):
                summation = np.dot(input, self.weights[1:]) + self.weights[0]
                if summation * label <= 0:
                    self.weights[1:] += label * input
                    self.weights[0] += label
                    m += 1
                    self.count += 1
                if m == 0:
                    break
    
    def print_result(self, inputs):
        print("trained weights: ", self.weights)
        print("total counts: ", self.count)
        for x in inputs:
            print(x, self.predict(x))

training_inputs = np.array([[1,1], [1,0],[0,1],[0,0]])
labels = np.array([1,-1,-1,-1])

perc = Perceptron(2, threshold=10)
perc.train(training_inputs, labels)
perc.print_result(training_inputs)