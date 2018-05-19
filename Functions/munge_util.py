#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:26:15 2018

@author: Young
"""

# Import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


# =============================================================================
# Data Pre-Processing Functions
# =============================================================================
def normalise(pandas_df):
    mean_col_wise = np.mean(pandas_df)
    standard_dev_col_wise = np.std(np.array(pandas_df),0)
    standardised_df = (pandas_df - mean_col_wise)/standard_dev_col_wise
    return standardised_df

# Convert labels
def binarize_labels(labels):
    lb = LabelBinarizer()
    lb.fit(labels)
    binary_labels = lb.transform(labels)
    binary_names = lb.classes_
    binary_labels_df = pd.DataFrame(binary_labels)
    binary_labels_df.columns = binary_names
    return binary_labels_df
