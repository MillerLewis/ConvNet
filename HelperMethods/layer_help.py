import scipy.signal
import numpy as np

def convolution(inp, filt, padding = 0, stride = 1,mode = 'valid'):
    newInp = pad(inp,padding)
    output = scipy.signal.convolve(newInp,filt[::-1,::-1], mode = mode)[::stride][::stride]
    return output


#Define a function that pads a matrix M with zeros in the xy dimensions
def pad(M, padding):
    try:
        paddedM = None
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
        print("Possibly tried a bad padding!")
        print(e)

def initializeRandomMatrix(shape, mean=0, std=1):
    # Essentially, we want to randomly initialize the convolutional layers weights with a
    #  random gaussian distn
    outputArray = (np.random.randn(*shape) * std + mean)
    # print outputArray
    return outputArray



def softmax(scores):
    #Assuming scores is 1D array
    x = np.exp(scores)
    return x/np.sum(x)

#For the derivatives we also need to rotate the matrix 180 degrees in the widthxheight
def flipMatrix(X):
    return X[::-1,::-1]

def is_close(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
