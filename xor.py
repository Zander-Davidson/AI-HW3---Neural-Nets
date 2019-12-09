import math
import random
import datetime


def f(x) = 1 / (1 + math.e^(-x))
def fp(x) = math.e^(-x) / (1 + math.e^(-x))^2
def fp(y) = y(1 - y)  # = fp(f(x))


X = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]] # training set, an XOR truth table
N = 3 # number of layers in neural net
neurons = [2, 2, 1] # number of neurons in the hidden and output layers, respectively
learnRate = 0.1
weight = [] # [layer][neuron][weight], stores the weights of all neural connections
delta = [] # [layer][neuron][weight], stores the gradients from neuron i to j, with indices corresponding to the weight list
bias = [] # [layer][neuron], stores biases for each nueron


# weight = [[[20, 20], [-20, -20]], [[20, 20]]]
# bias = [[-10, 30], [-30]]


def sigmoid(a):
    return 1 / (1 + math.exp(-a))

# derivative of sigmoid function w.r.t. y
def sigDeriv(y):
    return y * (1 - y)

# yj: output of a neuron in the final layer.
# tj: expected output for this neuron
def deltaOutput(yj, tj, numConnections):
    return [sigDeriv(yj) * (yj - tj) for j in range(numConnections)]

# yi: output from neuron i (hidden layer). 
# dj: list of deltas for every input to neuron j (hidden layer + 1).
# wj: list of weights for every input to neuron j (hidden layer + 1).
# xih: input from neuron h (hidden layer - 1) to neuron i (hidden layer).
# returns a list of deltas for each connection from each h (hidden layer - 1) to the single i (hidden layer)
def deltaHidden(yi, dj, wj, xi):
    # output of n
    derivOutput = sigDeriv(yi)
    weightSum = 0
    for n in range(dj):
        weightSum += dj[n] * wj[n]
    return [derivOutput * weightSum * xih for xih in xi]


def backpropagate():
    for j in range(neurons[-1])
        deltaWeight = learnRate * deltaOutput(neurons[-2]) * 







# assign random initial weights

random.seed(datetime.datetime.now())
# output layer has no connection to another
for l in range(1, N):
    # print('\tLayer ' + str(l))
    layerBias = []
    layerWeight = []

    # 1 axon per neuron -> "neurons" gives number of axons per layer
    for n in range(neurons[l]):
        # print('\t\tNeuron ' + str(n))
        layerBias.append(0)
        neuronWeight = []

        # if there are h neurons in layer n+1, we need h connections from the current axon to it
        for w in range(neurons[l-1]):
            # print('\t\t\tWeight ' + str(w))
            neuronWeight.append(random.random())
        
        layerWeight.append(neuronWeight)

    bias.append(layerBias)
    weight.append(layerWeight)

# samples in training set
for x in X:
    print('Training set ')

    xIn = x
    yOut = []
    
    # loop through layers, excluding input layer
    for l in range(1, N):
        print('\tLayer ' + str(l))

        # loop through neurons in each layer
        for n in range(neurons[l]):
            print('\t\tNeuron ' + str(n) + ': ' + str(bias[l-1][n]))

            sum = 0

            # loop through weights for a given neuron in layer j
            for w in range(neurons[l-1]):
                print('\t\t\tWeight ' + str(w) + ': ' + str(weight[l-1][n][w]))

                sum += weight[l-1][n][w] * xIn[w]

            yOut.append(sigmoid(sum + bias[l-1][n]))

        print('\txIn = yOut: ' + str(yOut[-1]))
        xIn = yOut

    backpropagate()




    