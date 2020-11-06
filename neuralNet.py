import numpy as np

'''
all 16 possible examples of a 2x2 grid of white or black pixels
   black=0, white=1
with a bias neuron included (last neuron all 1s)

the first 5 examples are the darks, the rest are brights
'''
x = np.array(([0, 0, 0, 0, 1],
              [1, 0, 0, 0, 1],
              [0, 1, 0, 0, 1],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1],
              [1, 1, 0, 0, 1],
              [1, 0, 1, 0, 1],
              [1, 0, 0, 1, 1],
              [0, 1, 1, 0, 1],
              [0, 1, 0, 1, 1],
              [0, 0, 1, 1, 1],
              [1, 1, 1, 0, 1],
              [1, 1, 0, 1, 1],
              [1, 0, 1, 1, 1],
              [0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]), dtype=float)

# the corresponding "correct" answer to the above examples
y = np.array(([0],
              [0],
              [0],
              [0],
              [0],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1]), dtype=float)


class NeuralNet:
    def __init__(self):
        # layer sizes
        self.inputLayerSize = 5  # x1, x2, x3, x4, Bias
        self.hiddenLayerSize = 2
        self.outputLayerSize = 1  # y1 (bright or dark)
        # weight arrays (initialized as random)
        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def feedforward(self, x):
        # sigmoid applied to the dot of x with the 1st layer weights
        self.i = self.sigmoid(np.dot(x, self.w1))
        # sigmoid of the dot of the previous sigmoid with the 2nd layer weights
        out = self.sigmoid(np.dot(self.i, self.w2))
        return out

    def backProp(self, x, y, out):
        # calculate errors and deltas
        self.outError = y - out
        self.outDelta = self.outError * self.sigmoid(out, firstDerivative=True)
        # note that "T" refers to the transpose of a numpy array
        self.iError = self.outDelta.dot(self.w2.T)
        self.iDelta = self.iError * self.sigmoid(self.i, firstDerivative=True)
        # adjust weights with backprop technique
        self.w1 += x.T.dot(self.iDelta)
        self.w2 += self.i.T.dot(self.outDelta)

    def train(self, x, y):
        # "train" is just shorthand for feedforward and backprop
        out = self.feedforward(x)
        self.backProp(x, y, out)

    def sigmoid(self, s, firstDerivative=False):
        if firstDerivative:
            return s * (1 - s)
        else:
            return 1 / (1 + np.exp(-s))


def main():
    nn = NeuralNet()
    xPredict = np.array(([1, 0, 1, 0, 1]), dtype=int)

    # clear the loss file
    with open("lossPerEpoch.csv", "w") as lossFile:
        lossFile.seek(0)
        lossFile.truncate()

    if trace:
        print("Input: \n" + str(x))
        print("Expected Output: \n" + str(y) + '\n\n')

    # each epoch runs inside this loop
    for i in range(trainingEpochs):
        print("Epoch # " + str(i))
        if trace:
            print("\nActual Output: \n" + str(nn.feedforward(x)))
        # calculate loss
        loss = np.mean(np.square(y - nn.feedforward(x)))
        # save loss into file
        with open("lossPerEpoch.csv", "a") as lossFile:
            lossFile.write(str(i)+","+str(loss.tolist())+'\n')

        if trace:
            print("Loss: " + str(loss) + '\n')
        # train with updated weights
        nn.train(x, y)
        if trace:
            print("Predicted output data based on trained weights: ")
            print("Expected (x1, x2, x3, x4, and Bias): \n" + str(xPredict))
        print("Output (y1): \n" + str(nn.feedforward(xPredict)) + '\n\n')


# choose number of epochs
trainingEpochs = 100
# optionally print more information about each epoch
trace = False
main()
