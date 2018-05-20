#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:15:13 2018

@author: Young
"""


# Import modules
import os
import sys
import random
import numpy as np
import pandas as pd
import pickle as cPickle
import matplotlib.pyplot as plt
from sklearn import datasets
from functools import reduce
from datetime import datetime
from sklearn.model_selection import train_test_split

# Import custom functions
utilities_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Functions'
sys.path.append(utilities_dir)
from gen_util import set_title
from eval_util import mse
from eval_util import plot_conf_mat
from eval_util import plot_error_changes
from munge_util import binarize_labels
from nn_functions import logistic
from nn_functions import softmax
from nn_functions import neuron_activation
from nn_functions import predict
from nn_functions import back_propagate

# Import parallelisation modules
import multiprocessing
from joblib import Parallel, delayed


# =============================================================================
# Define NN Class
# =============================================================================
class nn():
    def __init__(self, layers = [4,3,3], 
                 sigmoids = [logistic, softmax],
                 random_seed = None, 
                 w = None, 
                 b = None,
                 previous_epochs = 0):
        self.layers = layers
        self.epochs = previous_epochs
        self.sigmoids = sigmoids
        
        # Check size of sigmoids and layers
        if len(layers) != (len(sigmoids) + 1):
            raise Exception('\nLength of sigmoids needs to be length of layers - 1.\nThis is because the layers also count the first layer as the feature inputs.')
        
        if random_seed:
            np.random.seed(random_seed)
            
        # Set up random initialiser - weights and biases
        self.w = []
        self.b = []
        self.inertia_w = []
        self.inertia_b = []
        for layer_idx in range(len(layers)-1):
            num_neurons_previous_layer = layers[layer_idx]
            num_neurons_layer = layers[layer_idx+1]
            
            self.w.append(np.random.random((num_neurons_layer, num_neurons_previous_layer)))
            self.b.append(np.random.random((num_neurons_layer)))
            self.inertia_w.append(np.zeros((num_neurons_layer, num_neurons_previous_layer)))
            self.inertia_b.append(np.zeros((num_neurons_layer)))
        if w:
            self.w = w
        if b:
            self.b = b
            
        # Compute number of weights
        count_weights = list(map(lambda x: np.ma.size(x), self.w))
        count_bias = list(map(lambda x: np.ma.size(x), self.b))
        total_params = sum(count_weights) + sum(count_bias)       
        count_string = []
        for num_weight, num_bias in zip(count_weights, count_bias):
            count_string.append('(' + str(num_weight) + '+' + str(num_bias) + ')')
        set_title ('Neural network initialised')
        print ('Number of weights = ', ' + '.join(count_string))
        print ('                  = ', str(total_params))

    # Load nn
    @classmethod
    def load(cls, load_path):
        layers, sigmoids, w, b, epochs = cPickle.load(open(load_path, 'rb'))
        print ('Neural network loaded')
        return cls(layers = layers, sigmoids=sigmoids, w=w, b=b, previous_epochs=epochs)
    
    # Save nn
    def save(self, save_path):
        save_data = self.layers, self.sigmoids, self.w, self.b, self.epochs
        cPickle.dump(save_data, open(save_path, 'wb')) 
        print ('Neural network saved')
        
    # Fit nn
    def fit(self, x, y, 
            learning_rate,
            influence_of_inertia,
            size_minibatch=None, 
            epochs=1000, error_func=mse,
            verbose=False,
            n_jobs_data_parallelisation=8):
        set_title ('Training')
        st = datetime.now()
        self.learning_rate = learning_rate
        self.size_minibatch = size_minibatch
        self.epochs = self.epochs+epochs
        
        avg_errors = []
        if not size_minibatch:
            size_minibatch = len(x)
        for i in range(epochs):
            # Backpropagation
            grad_ws = []
            grad_bs = []
            errors = []
            size_minibatch / n_jobs_data_parallelisation
            minibatch = random.sample(range(len(x)), size_minibatch)
            n_jobs_data_parallelisation
            for train_idx in minibatch:
                grad_w, grad_b, error = back_propagate(x_value = x.iloc[train_idx],
                                               y_value = y.iloc[train_idx],
                                               w = self.w,
                                               b = self.b,
                                               layers = self.layers,
                                               sigmoids=self.sigmoids,
                                               error_func=error_func)
            
                grad_ws.append(grad_w)
                grad_bs.append(grad_b)
                errors.append(error)
        
            # Average grad w
            grad_avg_w = []
            for layer in range(len(self.layers)-1):
                grad_layer_ws = list(map(lambda x: x[layer], grad_ws))
                sum_w = sum(grad_layer_ws)
                grad_avg_w.append(sum_w / float(len(grad_ws)))
            
            # Average grad b
            grad_avg_b = []
            for layer in range(len(self.layers)-1):
                grad_layer_bs = list(map(lambda x: x[layer], grad_bs))
                sum_b = sum(grad_layer_bs)
                grad_avg_b.append(sum_b / float(len(grad_bs)))
            
            # Average error
            avg_error = np.mean(errors)
            avg_errors.append(avg_error)
            
            # Update weights
            for layer in range(len(self.layers)-1):
                self.w[layer] = self.w[layer] - ( (1 - influence_of_inertia) * (learning_rate * grad_avg_w[layer]) + influence_of_inertia * (self.inertia_w[layer]) )
                self.b[layer] = self.b[layer] - ( (1 - influence_of_inertia) * (learning_rate * grad_avg_b[layer]) + influence_of_inertia * (self.inertia_b[layer]) )
                self.inertia_w[layer] = ( (1 - influence_of_inertia) * (learning_rate * grad_avg_w[layer]) + influence_of_inertia * (self.inertia_w[layer]) )
                self.inertia_b[layer] = ( (1 - influence_of_inertia) * (learning_rate * grad_avg_b[layer]) + influence_of_inertia * (self.inertia_b[layer]) )
        
        # Print error
        et = datetime.now()
        print('------------')
        print('Training Error\t', avg_error, '\t[ Done in', et-st, ']')
        print('------------')
        
        # Plot error
        if verbose:
            plot_error_changes(avg_errors)

    # Predict
    def predict(self, x):
        errors = []
        preds = []
        for test_idx in range(len(x)):
            x_value = x.iloc[test_idx]
            pred = predict(x_value=x_value, w=self.w, b=self.b, 
                           layers=self.layers, 
                           sigmoids=self.sigmoids)
            preds.append(pred)
            
        print('Prediction complete')
        return preds
        
    # Validate
    def validate(self, x, y, error_func = mse, verbose=True):
        st = datetime.now()
        errors = []
        preds = []
        for test_idx in range(len(x)):
            x_value = x.iloc[test_idx]
            y_value = y.iloc[test_idx]
            
            pred, error = predict(x_value = x_value, 
                                  w=self.w, b=self.b, 
                                  layers = self.layers, 
                                  sigmoids=self.sigmoids,
                                  error_func = error_func,
                                  y_value = y_value)
            preds.append(pred)
            errors.append(error)
        # Print error
        et = datetime.now()
        print('\n-----------')
        print('Testing Error\t', np.average(errors), '\t[ Done in', et-st, ']')
        print('-----------\n')        
        
        # Confusion matrix
        if verbose:
            if len(preds[0]) == 1:
                pred_classes = (np.array(preds) > 0.5) * 1
                true_classes = np.array(y)

            elif len(preds[0]) > 1:
                pred_classes = np.argmax(preds,1)
                true_classes = np.argmax(np.array(y),1)
            plot_conf_mat(true_classes, pred_classes, figsize=(5,5))
        
        

## =============================================================================
## Manual Training
## =============================================================================
## Import modules
#import os
#import sys
#import random
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import datasets
#from sklearn.model_selection import train_test_split
#
## Import custom functions
#utilities_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Functions'
#nn_class_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Class'
#sys.path.append(utilities_dir)
#sys.path.append(nn_class_dir)
#from eval_util import mse
#from munge_util import binarize_labels
#from nn_functions import logistic
#from nn_functions import softmax
#
## Import parallelisation modules
#import multiprocessing
#from joblib import Parallel, delayed
#
## Import some data to play with
#iris = datasets.load_iris()
#data = pd.DataFrame(iris.data)
#data.columns = iris.feature_names
#labels = pd.DataFrame(iris.target)
#labels.columns = ['labels']
#
## Normalise data
#normalised_data = (data - np.mean(data))/np.std(np.array(data),0)
#
## Split data
#train_x,test_x,train_y,test_y = train_test_split(normalised_data, labels, train_size = 0.6, test_size = 0.4, random_state=1)
#bin_train_y = binarize_labels(train_y)
#bin_test_y = binarize_labels(test_y)
#
## Allocate variables
#nn_layers = [4,50,3]
#w = []
#b = []
#
## Create random inialised nn
#for layer_idx in range(len(nn_layers)-1):
#    num_neurons_previous_layer = nn_layers[layer_idx]
#    num_neurons_layer = nn_layers[layer_idx+1]
#    
#    w.append(np.random.random((num_neurons_layer, num_neurons_previous_layer)))
#    b.append(np.random.random((num_neurons_layer)))
#
## Train
#learning_rate = 0.10
#size_training_data = len(train_x)
#size_minibatch = 30
#for i in range(1000):
#    grad_ws = []
#    grad_bs = []
#    errors = []
#    for train_idx in random.sample(range(size_training_data), size_minibatch):
#        grad_w, grad_b, error = back_propagate(x_value = train_x.iloc[train_idx],
#                                       y_value = bin_train_y.iloc[train_idx],
#                                       w = w,
#                                       b = b,
#                                       layers = nn_layers)
#    
#        grad_ws.append(grad_w)
#        grad_bs.append(grad_b)
#        errors.append(error)
#
#    # Average grad w
#    grad_avg_w = []
#    for layer in range(len(nn_layers)-1):
#        grad_layer_ws = list(map(lambda x: x[layer], grad_ws))
#        sum_w = sum(grad_layer_ws)
#        grad_avg_w.append(sum_w / float(len(grad_ws)))
#    
#    # Average grad b
#    grad_avg_b = []
#    for layer in range(len(nn_layers)-1):
#        grad_layer_bs = list(map(lambda x: x[layer], grad_bs))
#        sum_b = sum(grad_layer_bs)
#        grad_avg_b.append(sum_b / float(len(grad_bs)))
#    
#    # Average error
#    avg_error = np.mean(errors)
#    
#    # Update weights
#    for layer in range(len(nn_layers)-1):
#        w[layer] = w[layer] - learning_rate * grad_avg_w[layer]
#        b[layer] = b[layer] - learning_rate * grad_avg_b[layer]
#
#print('Training MSE:\t', avg_error)
#
#size_testing_data = len(test_x)
#errors = []
#preds = []
#for test_idx in range(size_testing_data):
#    x_value = test_x.iloc[test_idx]
#    y_value = bin_test_y.iloc[test_idx]
#    
#    pred, error = predict(x_value = x_value, w=w, b=b, layers = nn_layers, y_value = y_value)
#    preds.append(pred)
#    errors.append(error)
#print('Testing MSE:\t', np.average(errors))
#preds
#pred_classes = np.argmax(preds,1)
#plot_conf_mat(np.array(test_y), pred_classes, figsize=(5,5), cmap=plt.cm.Blues)
