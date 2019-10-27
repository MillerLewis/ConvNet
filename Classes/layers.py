from HelperMethods.layer_help import isclose, convolution, flipMatrix, initializeRandomMatrix
import numpy as np

#Define a general class of layers just so other layers are subclasses
#We'll also define some methods that so we can have a guideline for all other
#layers
class Layer:
    def __init__(self):
        self.cache = None
    def forwardPass(self, inp):
        pass
    def backProp(self, cache):
        #Would like to define the cache as a dictionary
        #The keys would be:
        # dWeights, dInputs, output
        pass


class ConvLayer(Layer):
    # Assume that the weights is a numpy array, ###THROW EXCEPTION IF UNTRUE
    # Assume that the weights are a square matrix but with a depth. i.e. a 3d array
    # Would like to assume that the weights array is 4-D with the 4th dimension representing
    # which filter.

    ###METHOD TO INITIALIZE WEIGHTS RANDOMLY??
    def __init__(self, numFilters, weights=None, filterSize=2, stride=1, padding=0, depth=1):
        super().__init__()
        self.filterSize = filterSize
        self.numFilters = numFilters
        self.stride = stride
        self.padding = padding
        self.depth = depth

        if weights is not None:
            self.weights = weights  ###WEIGHTS SHOULD BE [F x F x depth x numFilters]
        elif weights is None:
            shape = (self.filterSize, self.filterSize, self.depth, self.numFilters)
            self.weights = initializeRandomMatrix(shape)
        self.biases = np.zeros(self.numFilters)
        self.shape = self.weights.shape

    def __str__(self):
        return 'ConvLayer \n' + str(self.weights)



    def forwardPass(self, inp):
        outputSize = (inp.shape[0] - self.filterSize - 2 * self.padding) / self.stride + 1
        if not isclose(outputSize, int(round(outputSize))):
            raise Exception("Output size will not be valid in ConvLayer", str(self))
        outputSize = int(round(outputSize))

        outputArray = np.zeros((outputSize, outputSize, self.numFilters))
        # print(convolution(inp,self.weights[:,:,:,0]))
        for k in range(self.numFilters):
            convd = convolution(inp, self.weights[:, :, :, k], padding=self.padding, stride=self.stride)
            convd = convd[:, :, 0]
            outputArray[:, :, k] = convd + self.biases[k]
        return outputArray

    def backProp(self, inp, deriv, regularization):
        # We want to compute: the derivative wrt weights, biases, inputs
        cache = {}
        dWeights = np.zeros(shape=self.weights.shape)
        for l in range(self.numFilters):
            for k in range(self.weights.shape[2]):
                dWeights[:, :, k, l] = convolution(deriv[:, :, l], inp[:, :, k], stride=self.stride)

        dWeights += regularization * self.weights

        newWeights = flipMatrix(self.weights)
        dInputs = np.zeros(shape=inp.shape)

        for k in range(dInputs.shape[2]):
            dInputs[:, :, k] = convolution(deriv, newWeights[:, :, k, :], mode='full', stride=self.stride)[:, :, 0]

        dBiases = inp.shape[0] * inp.shape[1] * np.sum(deriv, axis=0).sum(axis=0)  ###CHECK THIS

        cache = {'dWeights': dWeights, 'dInputs': dInputs, 'dBiases': dBiases}
        return cache

    def optimize(self, inp, deriv, stepSize):
        self.backprop(inp, deriv)
        dWeights = self.derivatives['dWeights']
        dBiases = self.derivatives['dBiases']

        self.weights += -stepSize * dWeights
        self.biases += -stepSize * dBiases


class ReLULayer(Layer):
    # This will be just a simple ReLU layer that thresholds at zero. I'm sure we could implement
    # more complex non-linearity layers, but for now, let's keep it simple

    def __str__(self):
        return 'ReLU layer'

    def forwardPass(self, inp):
        inp[inp <= 0] = 0
        return inp

    def backProp(self, inp, deriv, regularization):
        deriv[inp <= 0] = 0
        return {'dInputs': deriv}
