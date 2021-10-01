import pandas as pd
import random
import numpy as np
from colors import *
from mapVals import *
import pygame

#redesigned for numpy arrays
#new nomenclature
#more freedom in dimensions


''''
This includes the following functions:
getWeightedArray()
seed_weights()
normalizeState()
getDecision()
compressOutput()
returnUpdatedArrays()
displayNetwork()
add_names()
'''
def getWeightedArrays(dim_array,std=0.8):
    pd_weights = []
    np_biases = []
    for i in range(len(dim_array) - 1):
        weights = pd.DataFrame()
        biases = np.array([])
        for j in range(dim_array[i]):
            seeds = seed_weights(dim_array[i+1],std)
            weights[j] = seeds
        for j in range(dim_array[i+1]):
            seed = np.random.normal(scale=std)
            biases = np.append(biases,seed)
        pd_weights.append(weights)
        np_biases.append(biases)
    np_weights = []
    for i in pd_weights:
        np_weights.append(pd.DataFrame(i).to_numpy())
    pd_biases = []
    for i in np_biases:
        pd_biases.append(pd.DataFrame(i))
    return(pd_weights, np_weights, pd_biases, np_biases)

def seed_weights(rows,std):
    '''
    Returns random weights ranging from +/- 1
    :param rows: The number of rows in the array
    :return: One column for the final array with the proper amount of rows
    '''
    #np.random.seed(40)
    seeds = []
    for i in range(rows):
        seeds.append(np.random.normal(scale=std))
    return (seeds)

def linNormalizeState(state, d=1):
    for i in range(state.size):
        state[i] = state[i]/d
    return (state)

def normalizeState(state, f=1, s=1):
    '''
    Normalizes the state series to have values only between +/- 1
    :param state: The original array, composed of readings from program
    :param f: Controls the translating of the original state to its new value
    A higher value will lead to smaller values of the state converging to +/- 1
    :param s: The max boundary value to which all values will converge. A
    value of 1 will lead to all outputs converging to +/- 1
    :return:
    '''
    for i in range(state.size):
        state[i] = s*((1 / (1 + np.exp(-f * state[i]))) - 0.5)
    return (state)

def normalize(series, f=2.5, s=1):
    '''
    Normalizes the state series to have values only between +/- 1
    :param state: The original array, composed of readings from program
    :param f: Controls the translating of the original state to its new value
    A higher value will lead to smaller values of the state converging to +/- 1
    :param s: The max boundary value to which all values will converge. A
    value of 1 will lead to all outputs converging to +/- 1
    :return:
    '''
    for i in range(series.size):
        series[i] = s*((1 / (1 + np.exp(-f * series[i]))))
    return (series)

def getDecision(state, w_arrays, biases, f=1, s=1):
    '''
    Calculates the decisions from the neural network
    :param state: The readings from the program, often normalized
    :return:
    '''
    #change to numpy arrays, no more "python" arrays
    layer_z = []
    layer_a = []
    a = state
    layer_a.append(a)
    for i in range(len(w_arrays)):
        z = (np.dot(w_arrays[i], a)) + biases[i]
        #a = sigmoid(z)
        a = s * 2 * ((1 / (1 + np.exp(-f * z))) - 0.5)
        layer_z.append(z)
        layer_a.append(a)
    return(layer_a,layer_z)

def sigmoid(z):
    return (1/(1 + np.exp(-z)))

def sigmoid_derv(a):
    sigmoid_d = np.multiply(a,np.add(a,1))
    return(sigmoid_d)

def ComputeCost(desired,layer_a):
    cost = 0.5*np.power(np.subtract(desired,layer_a[-1]),2)
    cost_derv = np.subtract(layer_a[-1],desired)
    return(cost,cost_derv)

def delta_L(cost_derv,layer_z):
    s = sigmoid(layer_z[-1])
    delta_L = np.multiply(cost_derv,s)
    return(delta_L,s)

def delta_ls(cost_derv,layer_a,np_weights):
    delta_ls = []
    layers = len(np_weights)
    #print(layers)
    s = sigmoid_derv(layer_a[-1])
    delta_l = np.multiply(cost_derv, s)
    delta_ls.append(delta_l)
    for i in range(layers-1):
        j = layers - i - 1
        s = sigmoid_derv(layer_a[j])
        '''print()
        print(np_weights[j].transpose())
        print(np.array([np_weights[j]]).transpose())
        print()
        print(delta_ls[i])
        print(np.array([delta_ls[i]]))
        print()
        print(s)
        print(np.array([s]))'''
        delta_l = np.multiply(np.dot(np_weights[j].transpose(),delta_ls[i]),s)
        delta_ls.append(delta_l)
    delta_ls.reverse()
    return(delta_ls)

def delta_ws(delta_ls,layer_a):
    delta_ws = []
    for i in range(len(delta_ls)):
        a = np.array([layer_a[i]])
        d = np.array([delta_ls[i]]).transpose()
        delta_w = np.multiply(a, d)
        delta_ws.append(delta_w)
    return delta_ws

def updateWeights(np_weights,delta_ws,T):
    for i in range(len(np_weights)):
        r, c = np_weights[i].shape
        for j in range(r):
            for k in range(c):
                np_weights[i][j][k] -= T * delta_ws[i][j][k]
    return np_weights

def updateBiases(np_biases,delta_ls,T):
    for i in range(len(np_biases)):
        r = np_biases[i].size
        for j in range(r):
            np_biases[i][j] -= T * delta_ls[i][j]
    return np_biases

#This needs to be reevaluated
def returnUpdatedArrays(dim_array, weights, biases, sigma=1):
    '''
    This function is designed to produce new random arrays that are based
    off of a previous trial
    :param sigma: The standard deviation of the change added to the arrays.
    A standard deviation of 1 will cause 68% of values to fall within +/- 1.
    This value is added to the passed in array. A std of 0.5 will cause 68%
    to fall within +/- 0.5
    :return: Returns the same arrays passed in after having the change added
    '''
    pd_weights, weightsAdjustments, pd_biases, biasesAdjustments = getWeightedArrays(dim_array,sigma)
    #print(weightsAdjustments)
    #print(biasesAdjustments)
    #for i in range(len(weights)):
        #weights[i] = np.add(weightsAdjustments[i],weights[i])
        #biases[i] = np.add(biasesAdjustments[i],biases[i])
    weights2 = []
    biases2 = []
    #Change this to go item by item?
    for i in range(len(weights)):
        weights2.append(np.add(weights[i],weightsAdjustments[i]))
    for i in range(len(biases)):
        biases2.append(np.add(biases[i],biasesAdjustments[i]))
    return(weights2,biases2)





    '''
    # Finds the shape of the passed in array
    h1_shape = h1_weights_master.shape
    # Creates an empty array of the same size
    h1_weights = pd.DataFrame(np.nan, index=range(h1_shape[0]), columns=range(h1_shape[1]))
    # Adds in values from original array plus a change to every element
    for i in range(h1_shape[0]):
        for j in range(h1_shape[1]):
            change = np.random.normal(loc=0.0, scale=sigma)
            h1_weights.iloc[i, j] = h1_weights_master.iloc[i, j] + change
    # Changes any values outside of +/- 1 to +/- 1
    h1_weights = h1_weights.mask(h1_weights > 1, 1)
    h1_weights = h1_weights.mask(h1_weights < -1, -1)

    h2_shape = h2_weights_master.shape
    h2_weights = pd.DataFrame(np.nan, index=range(h2_shape[0]), columns=range(h2_shape[1]))
    for i in range(h2_shape[0]):
        for j in range(h2_shape[1]):
            change = np.random.normal(loc=0.0, scale=sigma)
            h2_weights.iloc[i, j] = h2_weights_master.iloc[i, j] + change
    h2_weights = h2_weights.mask(h2_weights > 1, 1)
    h2_weights = h2_weights.mask(h2_weights < -1, -1)

    z_shape = z_weights_master.shape
    z_weights = pd.DataFrame(np.nan, index=range(z_shape[0]), columns=range(z_shape[1]))
    for i in range(z_shape[0]):
        for j in range(z_shape[1]):
            change = np.random.normal(loc=0.0, scale=sigma)
            z_weights.iloc[i, j] = z_weights_master.iloc[i, j] + change
    z_weights = z_weights.mask(z_weights > 1, 1)
    z_weights = z_weights.mask(z_weights < -1, -1)
    return (h1_weights, h2_weights, z_weights)
    '''

def displayNetwork(dim_array, np_weights, width=400,height=250,state=None,layer_a=None):
    # Set up the parameters for drawing
    pygame.init()
    radius = 8
    line_width = 2
    # Set up the stuff for pygame
    screen = pygame.Surface((width, height))
    yspacing = []
    for i in dim_array:
        yspacing.append(int(height/(i+1)))
    xspacing = []
    xspace = 0
    for i in range(len(dim_array)):
        xspace += int(width/(1+len(dim_array)))
        xspacing.append(xspace)
    # Fills the screen with black
    screen.fill(black)
    # Find all of the positions of the circles
    positions = []
    for i in range(len(dim_array)):
        y_current = 0
        columnPos = []
        for j in range(dim_array[i]):
            y_current += yspacing[i]
            columnPos.append((xspacing[i],y_current))
            #add to stuff two D array
        positions.append(columnPos)
    #for i in positions:
        #print(i)
    for i in range(len(positions)-1):  #for all columns but the last
        for j in range(len(positions[i])):  #for all items in current column
            for k in range(len(positions[i+1])):
                lineColor = get_color(np_weights[i][k][j])
                pygame.draw.line(screen, lineColor, positions[i][j], positions[i+1][k], line_width)
    for i in range(len(positions)):
        for j in range(len(positions[i])):
            pygame.draw.circle(screen, white, positions[i][j], radius)
    return screen

def get_color(w):
    redRange = (160, 255)
    greenRange = (85, 255)
    if w > 1:
        w = 1
    if w < -1:
        w = -1
    if (w < 0):
        color = (mapVals(w, redRange), 0, 0)
    else:
        color = (0, mapVals(w, greenRange), 0)
    return color

def add_names(h1_weights=None, h2_weights=None, z_weights=None):
    '''
    Indices and columns names are added to the arrays
    :param h1_weights: Array of weights detailing the relationship between
    inputs and hidden layer 1 nodes
    :param h2_weights:Array of weights detailing the relationship between
    hidden layer 1 and hidden layer 2 nodes
    :param z_weights:Array of weights detailing the relationship between
    hidden layer 2 and z layer nodes or hidden layer 1 and z layer nodes
    :return: Returns the arrays with added indices and columns
    '''
    if (isinstance(h1_weights, pd.DataFrame)):
        h1_columns = []
        h1_indices = []
        h1_shape = h1_weights.shape  # rows,columns
        for i in range(h1_shape[0]):
            h1_indices.append("a_" + str(i))
        for i in range(h1_shape[1]):
            h1_columns.append("h1_" + str(i))
        h1_weights.columns = [h1_columns]
        h1_weights.index = [h1_indices]
    if (isinstance(h2_weights, pd.DataFrame)):
        h2_columns = []
        h2_indices = []
        h2_shape = h2_weights.shape  # rows,columns
        for i in range(h2_shape[0]):
            h2_indices.append("h1_" + str(i))
        for i in range(h2_shape[1]):
            h2_columns.append("h2_" + str(i))
        h2_weights.columns = [h2_columns]
        h2_weights.index = [h2_indices]
    if (isinstance(z_weights, pd.DataFrame) and isinstance(h2_weights, pd.DataFrame)):
        z_columns = []
        z_indices = []
        z_shape = z_weights.shape  # rows,columns
        for i in range(z_shape[0]):
            z_indices.append("h2_" + str(i))
        for i in range(z_shape[1]):
            z_columns.append("z_" + str(i))
        z_weights.columns = [z_columns]
        z_weights.index = [z_indices]
    if (isinstance(z_weights, pd.DataFrame) and isinstance(h2_weights, pd.DataFrame) == False):
        z_columns = []
        z_indices = []
        z_shape = z_weights.shape  # rows,columns
        for i in range(z_shape[0]):
            z_indices.append("h1_" + str(i))
        for i in range(z_shape[1]):
            z_columns.append("z_" + str(i))
        z_weights.columns = [z_columns]
        z_weights.index = [z_indices]
    return (h1_weights, h2_weights, z_weights)

def gradientCloseLayer(h1_weights,output,desired_output):
    #only works when there is only one hidden layer and one output
    error = output-desired_output
    derv_output = output*(1-output)
    gradients = []
    shape = h1_weights.shape
    for i in range(shape[0]):
        gradient = error*derv_output*h1_weights.iloc[i]
        gradients.append(gradient)
    gradients = np.array(gradients)
    return(gradients)
