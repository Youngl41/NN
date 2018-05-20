#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:28:50 2018

@author: Young
"""


# Import modules
import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# =============================================================================
# Error Handling
# =============================================================================
# Error caluclation
def mse(y_vector, p_vector, derivative=False):
    y_vector = np.array(y_vector)
    p_vector = np.array(p_vector)
    
    # Derivative wrt p_vector
    if derivative:
        derivative = (p_vector - y_vector)*2
        return derivative
    # MSE
    else:
        mse_value = np.sum((y_vector - p_vector)**2) / float(p_vector.size)
        return mse_value

# Plot errors
def plot_error_changes(errors):
    df = pd.DataFrame(errors)
    df.columns = ['errors']
    group_size = int(np.ceil(len(errors)/20.0))
    df.loc[:,'grouped'] = (df.index / group_size).astype(int)+1
    df.groupby('grouped')['errors'].agg('mean').plot(figsize = (9,5))
    plt.title('Avg error after each epoch')
    plt.xlabel('Epoch count (in ' + str(group_size) + "'s)")
    plt.ylabel('Average error')
    plt.show()
    
# Confusion matrix    
figsize_ = (13,13)
def plot_conf_mat(y_test, p_test, figsize = figsize_, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    # Display results
    classes_            = []
    classes_.append(list(p_test))
    classes_.append(list(y_test))
    classes             = np.unique(classes_)
    cfmt                = confusion_matrix(y_test, p_test)
    
    plt.figure(figsize=figsize)
    plt.imshow(cfmt, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cfmt.max() / 2.
    for i, j in itertools.product(range(cfmt.shape[0]), range(cfmt.shape[1])):
        plt.text(j, i, cfmt[i, j],
                 horizontalalignment="center",
                 color="white" if cfmt[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print (classification_report(y_test, p_test))

    return cfmt