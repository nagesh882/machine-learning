"""
Multiple Variable Linear Regression

Or

Multivariate Linear Regression
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



dataFrame = pd.read_csv("/home/ubuntu/Documents/machine-learning/homeprices.csv")

dataFrame.columns = dataFrame.columns.str.strip().str.lower()
print(dataFrame)