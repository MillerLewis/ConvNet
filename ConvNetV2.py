#ConvNetV2. Assume we'll only train using online training
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import klepto
import multiprocessing as mp
import scipy.signal

def unwrap_self_f(arg, **kwarg):
    return ConvNet.forwardPass(*arg, **kwarg)

def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def softmax(scores):
    #Assuming scores is 1D array
    x = np.exp(scores)
    return x/np.sum(x)

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


#Define a ConvNet as a class
# Within the class there are a list of layers and a depth
# We'll also define a forward pass and backprop methods
class ConvNet:
    def __init__(self):
        #The only attributes we'll have is depth and layers
        #We're including the depth in case it's helpful
        self.depth = 0
        self.layers = []
        self.cache = range(self.depth)

    def appendLayer(self,layer):
        ###Should throw an exception if we try to append a non-layer
        self.depth += 1
        self.layers.append(layer)

    def insertLayer(self,index,layer):
        ###Should throw an exception if we try to insert a non-layer
        self.depth += 1
        self.layers = self.layers[:index] + [layer] + self.layers[index+1:]

        

    def forwardPass(self,inp):
        #Take the ConvNet itself and an input and compute the forward pass
        #We compute the forward pass simply by computing the forward pass
        # on each seperate layer on the ConvNet
        #Should output a (1,1,K) np array. K := #classes if FINAL scores
        returnVal = np.copy(inp)
        for layer in self.layers:
            returnVal = layer.forwardPass(returnVal)
        #print returnVal.shape
        return returnVal

                
                
    def backProp(self, inp, label, regularization):
        #Take the ConvNet itself and an input and a cache of variables we need and compute the backwards pass
        #Assume that the cache is a list of dictionaries and the keys are
        #dWeights, output
        #Assume input is of side [1,1,K]
        outputs = [inp]
        for layer in self.layers:
            outputs.append(layer.forwardPass(outputs[-1]))
        #print outputs[-1]
        dScores = softmax(outputs[-1])
        dScores[0,0,label] -= 1
        derivs = range(self.depth)
        derivs = [{'dInputs':dScores}]

        
        for i in range(self.depth-1,-1,-1):
            x = self.layers[i].backProp(outputs[i], derivs[-1]['dInputs'], regularization = regularization)
            derivs.append(x)

        return derivs[::-1]

    def optimize(self,inp,label,regularization, learningRate):
        derivs = self.backProp(inp,label, regularization = regularization)
        for i in range(self.depth):
            if 'dWeights' in derivs[i]:
                self.layers[i].weights += -learningRate * derivs[i]['dWeights']
            if 'dBiases' in derivs[i]:
                self.layers[i].biases += -learningRate * derivs[i]['dBiases']

        
        

        
        


    def predictLabel(self,inp):
        #This will be predicting the label of just ONE image
        #the output scores hould be of the form (1,1,K) K := #class labels
        scores = softmax(self.forwardPass(inp))
        #print scores
        if scores.shape[0:2] != (1,1):
            print "CLASS SCORES ARE IN INCORRECT FORMAT. RAISING EXCEPTION"
            raise Exception("PREDICT LABEL HAS FAILED")
        else:
            scores = np.reshape(scores,scores.shape[2])
            return np.argmax(scores)

#Define a general class of layers just so other layers are subclasses
#We'll also define some methods that so we can have a guideline for all other
#layers
class Layer:
    def __init__(self):
        self.cache = None
    def forwardPass(self):
        pass
    def backProp(self, cache):
        #Would like to define the cache as a dictionary
        #The keys would be:
        # dWeights, dInputs, output
        pass


#Define a function that pads a matrix M with zeros in the xy dimensions
def pad(M, padding):
    try:
        #One output if it's 3D
        if len(M.shape) >= 3:
            paddedM = np.zeros(shape = (M.shape[0] + 2*padding, M.shape[1] + 2*padding)+M.shape[2:])
            paddedM[padding:M.shape[0] + padding, padding:M.shape[1] + padding] = M

        #Different output if 2D
        elif len(M.shape) == 2:
            paddedM = np.zeros(shape = (M.shape[0] + 2*padding, M.shape[1] + 2*padding))
            paddedM[padding:M.shape[0] + padding, padding:M.shape[1] + padding] = M[:,:]

        return paddedM
    except Exception as e:
        print "Possibly tried a bad padding!"
        print e


def convolution(inp, filt, padding = 0, stride = 1,mode = 'valid'):
    newInp = pad(inp,padding)
    output = scipy.signal.convolve(newInp,filt[::-1,::-1], mode = mode)[::stride][::stride]
    return output

#For the derivatives we also need to rotate the matrix 180 degrees in the widthxheight
def flipMatrix(X):
    return X[::-1,::-1]



class ConvLayer(Layer):
    #Assume that the weights is a numpy array, ###THROW EXCEPTION IF UNTRUE
    #Assume that the weights are a square matrix but with a depth. i.e. a 3d array
    #Would like to assume that the weights array is 4-D with the 4th dimension representing
    # which filter.

    ###METHOD TO INITIALIZE WEIGHTS RANDOMLY??
    def __init__(self, numFilters, weights = None, filterSize = 2, stride = 1, padding = 0, depth = 1):
        self.filterSize = filterSize
        self.numFilters = numFilters
        self.stride = stride
        self.padding = padding
        self.depth = depth
        if weights is not None:
            self.weights = weights ###WEIGHTS SHOULD BE [F x F x depth x numFilters]
        elif weights is None:
            self.initializeWeightsGauss()
        self.biases = np.zeros(self.numFilters)
        self.shape = self.weights.shape

        #self.cache = {}

        
    def __str__(self):
        return 'ConvLayer \n' + str(self.weights)


    def initializeWeightsGauss(self, mean = 0, std = 1):
        #Essentially, we want to randomly initialize the convolutional layers weights with a
        # random gaussian distn
        shape = (self.filterSize,self.filterSize,self.depth,self.numFilters)
        outputArray = (np.random.randn(*shape)*std + mean)
        #print outputArray
        self.weights = outputArray


    def forwardPass(self, inp):
        outputSize = (inp.shape[0]-self.filterSize-2*self.padding)/self.stride + 1
        if not isclose(outputSize, int(round(outputSize))):
            raise Exception("Output size will not be valid in ConvLayer", str(self))
        outputSize = int(round(outputSize))
        
        outputArray = np.zeros((outputSize,outputSize,self.numFilters))
        #print(convolution(inp,self.weights[:,:,:,0]))
        for k in xrange(self.numFilters):
            convd = convolution(inp,self.weights[:,:,:,k],padding = self.padding, stride = self.stride)
            convd = convd[:,:,0]
            outputArray[:,:,k] =  convd + self.biases[k]
        return outputArray
        

    def backProp(self, inp, deriv, regularization):
        #We want to compute: the derivative wrt weights, biases, inputs
        cache = {}
        dWeights = np.zeros(shape = self.weights.shape)
        for l in range(self.numFilters):
            for k in range(self.weights.shape[2]):
                dWeights[:,:,k,l] = convolution(deriv[:,:,l],inp[:,:,k],stride = self.stride)

        dWeights += regularization * self.weights
        
        newWeights = flipMatrix(self.weights)
        dInputs = np.zeros(shape = inp.shape)

        
        for k in range(dInputs.shape[2]):
            dInputs[:,:,k] = convolution(deriv,newWeights[:,:,k,:], mode = 'full', stride = self.stride)[:,:,0]
        
            
            
        dBiases = inp.shape[0]*inp.shape[1]*np.sum(deriv,axis = 0).sum(axis=0)   ###CHECK THIS
        
        cache = {'dWeights':dWeights, 'dInputs':dInputs, 'dBiases':dBiases}
        return cache


    def optimize(self, inp, deriv, stepSize):
        self.backprop(inp,deriv)
        dWeights = self.derivatives['dWeights']
        dBiases = self.derivatives['dBiases']
                                   
        self.weights += -stepSize*dWeights
        self.biases += -stepSize*dBiases

class ReLULayer(Layer):
    #This will be just a simple ReLU layer that thresholds at zero. I'm sure we could implement
    # more complex non-linearity layers, but for now, let's keep it simple

    def __str__(self):
        return 'ReLU layer'
    def forwardPass(self, inp):
        inp[inp<=0] = 0
        return inp

    def backProp(self, inp, deriv, regularization):
        deriv[inp <= 0] = 0
        return {'dInputs':deriv}



