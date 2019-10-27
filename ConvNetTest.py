import Classes.NeuralNet
import numpy as np
import klepto
import time
import multiprocessing as mp
import gzip
import os
import pickle

path = 'fixedmnist'
dataDir = klepto.archives.dir_archive(path, cached = True, serialized = True)
#print dataDir
dataDir.load('training_data')
dataDir.load('validation_data')
dataDir.load('test_data')

trainingData = dataDir['training_data']
validationData = dataDir['validation_data']
testData = dataDir['test_data']


testIm = trainingData[0][0]
cLayer = NeuralNet.ConvLayer(numFilters = 5, filterSize = 1, depth = 1)
cLayer2 = NeuralNet.ConvLayer(numFilters = 10, filterSize = 28, depth = 5)
N = 5*1*1 + 10*28*5
cLayer.initializeWeightsGauss(std = 1.0/(2*N))
cLayer2.initializeWeightsGauss(std = 1.0/(2*N))
reluLayer = NeuralNet.ReLULayer()
cNet = NeuralNet.NeuralNet()
cNet.appendLayer(cLayer)
cNet.appendLayer(reluLayer)
cNet.appendLayer(cLayer2)
n = 15
testIms = trainingData[0][:2**n]
testImLabels = trainingData[1][:2**n]


def checkAccuracy(inputs, labels):
    S = 0
    for i, j in zip(inputs, labels):
        print(cNet.predictLabel(i),j)
        if cNet.predictLabel(i) == j:
            S += 1
    
    return S/len(inputs)*100



testIm = testIms[1]
testImLabel = testImLabels[1]

LOAD = False
if LOAD:
    cNet = pickle.load(path)

TRAIN = True
a = np.random.randint(0,10000)
#checkAccuracy(testIms,testImLabels)
if TRAIN:
    t0 = time.time()
    learningRate = 0.0001
    regularization = 0
    print(testImLabel)
    print(2**n)
    randomIms = []
    
    for i in range(2**15):
        randomIms.append(np.random.randint(0,2**2))
        if i%100 == 0:
            print(i)#, randomIms[-1]
        #cNet.optimize(testIms[randomIms[-1]],testImLabels[randomIms[-1]],learningRate = learningRate)
        cNet.optimize(testIms[i%4], testImLabels[i%4], regularization = regularization, learningRate = learningRate)
        if time.time() - t0 >= 10:
            break
    

#print cNet.predictLabel(testIms[a]), testImLabels[a]
checkAccuracy(testIms[:4],testImLabels[:4])
#checkAccuracy(testIms[randomIms], testImLabels[randomIms])

#print NeuralNet.softmax(cNet.forwardPass(testIm))
#print cNet.predictLabel(testIm)
#print time.time() - t0
#print checkAccuracy(testIms[randomIms], testImLabels[randomIms])

path = 'weights.pkl'
OVERWRITE = True
"""
if not os.path.exists(path) or OVERWRITE:
    pickle.dump(cNet, path)

cNet = pickle.load(path)
print cNet.predictLabel(testIm), testImLabel"""




"""def multiForward(NeuralNet, inputs):
    pool = mp.Pool(processes = 4)
    scores = np.array([pool.apply(NeuralNet.forwardPass, args = (inputs[i],)) for i in range(inputs.shape[0])])
    return scores    """

"""if __name__ == '__main__':
    
    
    for j in [True, False]:
        print "BEGINNING"
        t0 = time.time()
        #for i in range(N):
        t1 = time.time()
        cNet.forwardPassMultiple(testIms,j)
        #print multiForward(cNet,testIms)
        print time.time() - t1
        print
        print j,"   FINAL TIME: ", (time.time() - t0)

    t0 = time.time()
    #x = cNet.predictLabelsMultiple(testIms)
    #print x.shape
    #print sum(x == testImsLabels)/2**n * 100
    #print time.time() - t0
    print trainingData[0].shape
    for i in trainingData[0][0:2**n]:
        #print i.shape
        cLayer2.forwardPass(i)
        
    print(time.time() - t0)
    t0 = time.time()
    #for i in testIms:
    #cNet.forwardPass(testIm[0])
    #print(time.time() - t0)"""

