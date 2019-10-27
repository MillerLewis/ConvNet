import numpy as np
from HelperMethods.layer_help import softmax

# Define a ConvNet as a class
# Within the class there are a list of layers and a depth
# We'll also define a forward pass and backprop methods
class NeuralNet:
    def __init__(self):
        # The only attributes we'll have is depth and layers
        # We're including the depth in case it's helpful
        self.depth = 0
        self.layers = []
        self.cache = range(self.depth)

    def appendLayer(self, layer):
        ###Should throw an exception if we try to append a non-layer
        self.depth += 1
        self.layers.append(layer)

    def insertLayer(self, index, layer):
        ###Should throw an exception if we try to insert a non-layer
        self.depth += 1
        self.layers = self.layers[:index] + [layer] + self.layers[index + 1:]

    def forwardPass(self, inp):
        # Take the ConvNet itself and an input and compute the forward pass
        # We compute the forward pass simply by computing the forward pass
        # on each seperate layer on the ConvNet
        # Should output a (1,1,K) np array. K := #classes if FINAL scores
        returnVal = np.copy(inp)
        for layer in self.layers:
            returnVal = layer.forwardPass(returnVal)
        # print returnVal.shape
        return returnVal

    def backProp(self, inp, label, regularization):
        # Take the ConvNet itself and an input and a cache of variables we need and compute the backwards pass
        # Assume that the cache is a list of dictionaries and the keys are
        # dWeights, output
        # Assume input is of side [1,1,K]
        outputs = [inp]
        for layer in self.layers:
            outputs.append(layer.forwardPass(outputs[-1]))
        # print outputs[-1]
        dScores = softmax(outputs[-1])
        dScores[0, 0, label] -= 1
        derivs = range(self.depth)
        derivs = [{'dInputs': dScores}]

        for i in range(self.depth - 1, -1, -1):
            x = self.layers[i].backProp(outputs[i], derivs[-1]['dInputs'], regularization=regularization)
            derivs.append(x)

        return derivs[::-1]

    def optimize(self, inp, label, regularization, learningRate):
        derivs = self.backProp(inp, label, regularization=regularization)
        for i in range(self.depth):
            if 'dWeights' in derivs[i]:
                self.layers[i].weights += -learningRate * derivs[i]['dWeights']
            if 'dBiases' in derivs[i]:
                self.layers[i].biases += -learningRate * derivs[i]['dBiases']

    def predictLabel(self, inp):
        # This will be predicting the label of just ONE image
        # the output scores hould be of the form (1,1,K) K := #class labels
        scores = softmax(self.forwardPass(inp))
        # print scores
        if scores.shape[0:2] != (1, 1):
            print("CLASS SCORES ARE IN INCORRECT FORMAT. RAISING EXCEPTION")
            raise Exception("PREDICT LABEL HAS FAILED")
        else:
            scores = np.reshape(scores, scores.shape[2])
            return np.argmax(scores)