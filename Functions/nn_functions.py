#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:03:54 2018

@author: Young
"""


# Import modules
import os
import sys
import numpy as np
import pandas as pd

# Import custom functions
utilities_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Functions'
sys.path.append(utilities_dir)
from eval_util import mse


# =============================================================================
# nn_functions
# =============================================================================
# Logistic function
def logistic(x, derivative=False):
    if derivative:
        return np.exp(x)/(1+np.exp(x))**2
    else:
        return 1/(1+np.exp(-x))

# Softmax function
def softmax(vector, derivative=False):
    s = np.exp(vector) / np.sum(np.exp(vector))
    if derivative:
        return s * (1-s)
    else:
        return s
    
# Activation function
def neuron_activation(w, b, a, activation_function = logistic):
    '''
    w   := weights of previous neurons
    b   := bias
    a   := previous neuron outputs
    
    Example:
        >>> w = np.random.random((4,4))
        >>> b = [0.1, 0.1, 0.1, 0.1]
        >>> a = [0.3, 0.1, 0.2, 0.5]
        >>> neuron_activation(w,b,a)
    '''
    w = np.array(w)
    b = np.array(b)
    a = np.array(a)
    
    # Check dimensions
    new_num_neurons = np.shape(w)[0]
    old_num_neurons = np.shape(w)[1]
    if not (len(a) == old_num_neurons) & (len(b) == new_num_neurons):
        raise Exception('Incorrect dimensions.')
    
    # Calculate new a
    z = w.dot(a) + b
    if activation_function:
        new_a = activation_function(z)
        return new_a
    else:
        return z

# Forward propagation
def forward_propagate(x_value, w, b, layers, sigmoids):
    # Fire neurons
    a = []
    z = []
    for layer_idx in range(len(layers)-1):
        w_layer = w[layer_idx]
        b_layer = b[layer_idx]
        # First layer
        if layer_idx == 0:
            z_layer = neuron_activation(w_layer, b_layer, np.array(x_value), activation_function = None)
            a_layer = neuron_activation(w_layer, b_layer, np.array(x_value), activation_function = sigmoids[layer_idx])
        # Other layers
        else:
            z_layer = neuron_activation(w_layer, b_layer, a[layer_idx-1], activation_function = None)
            a_layer = neuron_activation(w_layer, b_layer, a[layer_idx-1], activation_function = sigmoids[layer_idx])
        z.append(z_layer)
        a.append(a_layer)
        
    return a,z

# Backward propagation
def back_propagate(x_value, y_value, w, b, layers, sigmoids, error_func = mse):
    # Forward propagation
    a, z = forward_propagate(x_value, w, b, layers, sigmoids)
    
    # Error
    error = error_func(a[-1], np.array(y_value))
    
    # Backpropagation
    grad_w = []
    grad_b = []
    y_temp = np.array(y_value)
    for layer_idx in reversed(range(len(layers)-1)):
        # Check first layer input neurons
        if layer_idx == 0:
            deriv_z_by_w = np.array(x_value)
        else:
            deriv_z_by_w = a[layer_idx-1]
        
        # Final layer
        if layer_idx == len(layers)-2:
            deriv_a_by_z = sigmoids[layer_idx](z[layer_idx], derivative=True)#logistic(z[layer_idx], derivative=True)
            deriv_c_by_a = error_func(y_temp, a[layer_idx], derivative=True)#2 * (a[layer_idx] - y_temp)
            
            grad_final_layer_w = np.outer(deriv_z_by_w, deriv_a_by_z * deriv_c_by_a).transpose()
            grad_final_layer_b = deriv_a_by_z * deriv_c_by_a
    
            grad_w.append(grad_final_layer_w)
            grad_b.append(grad_final_layer_b)
        
        # Iterate through lower layers
        else:
            deriv_a_by_z = sigmoids[layer_idx](z[layer_idx], derivative=True)
            deriv_c_by_a = np.matmul(deriv_c_by_a * sigmoids[layer_idx](z[layer_idx + 1], derivative=True), w[layer_idx + 1])
            
            grad_layer_w = np.outer(deriv_z_by_w, deriv_a_by_z * deriv_c_by_a).transpose()
            grad_layer_b = deriv_a_by_z * deriv_c_by_a
    
            grad_w.append(grad_layer_w)
            grad_b.append(grad_layer_b)
            
    # Reverse list
    grad_w = list(reversed(grad_w))
    grad_b = list(reversed(grad_b))
    
    return grad_w, grad_b, error

# Chunked back propagation
def chunked_back_propagate(minibatch, x, y, w, b, layers, sigmoids, error_func = mse):
    return list(map(lambda train_idx: back_propagate(x_value = x.iloc[train_idx],
                                                   y_value = y.iloc[train_idx],
                                                   w = w,
                                                   b = b,
                                                   layers = layers,
                                                   sigmoids=sigmoids,
                                                   error_func=error_func), minibatch))

def predict(x_value, w, b, layers, sigmoids, error_func = None, y_value = None):    
    # Forward propagation
    a, _ = forward_propagate(x_value, w, b, layers, sigmoids)
    
    # Error
    prediction = a[-1]
    try:
        error = error_func(prediction, np.array(y_value))
        return prediction, error
    except TypeError:
        return prediction