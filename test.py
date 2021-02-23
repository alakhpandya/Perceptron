import numpy as np
from perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

lables = np.array([1,0,0,0])

perceptron = Perceptron(2)
perceptron.train(training_inputs, lables)

a = [1, 1, 0, 0]
b = [1, 0, 1, 0]
dataset = zip(a, b)
print("\nInputs\tPerceptron Output\tActual Output")
i = 0
for x, y in dataset:
    test_inputs1 = np.array([x, y])
    print(test_inputs1, "\t\t", perceptron.predict(test_inputs1),"\t\t", lables[i])
    i += 1
