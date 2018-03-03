#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:35:23 2018

@author: Diogo Rodrigues
"""
# Required Python Packages
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd #import and manager datasets
from sklearn.preprocessing import Imputer # Take care of missing data
from sklearn.cross_validation import train_test_split # Split the dataset into the Training set and Test set
from sklearn.preprocessing import LabelEncoder # Encoding categorial data
from sklearn.preprocessing import OneHotEncoder # Dummy encoding
from sklearn.preprocessing import StandardScaler # Features Scaling

# File Paths
INPUT_PATH = '../dataset/Data.csv'

def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return:
    """
    data = pd.read_csv(path)

    return data
    
def handle_missing_data(dataset, first_column, limit_column):
    """
    Taking care of the missing data. Obs.:strategy can be 'mean', 'median','most_frequent'
    :param first_column
    :param last_column
    :return:
    """
    
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(dataset[:, first_column:limit_column])
    dataset[:, first_column:limit_column] = imputer.transform(dataset[:, first_column:limit_column])
    
    return dataset

def enconding_categorical_data(dataset):
    """
    Encoding string to number
    param dataset:
    :return:
    """
    
    label_encoder = LabelEncoder()
    dataset = label_encoder.fit_transform(dataset)
    
    return dataset

def dummy_encoding(dataset, column):
    """
    :param dataset:
    :return:
    """
    one_hot_encoding = OneHotEncoder(categorical_features = [column])
    dataset = one_hot_encoding.fit_transform(dataset).toarray()
    
    return dataset   

def split_dataset(features, target, train_percentage):
    """
    Splitting the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param features:
    :param target:    
    :return: training set and Test set
    """
    # Splitting the dataset into the Training set and Test set    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = train_percentage, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def scaling_features(train, test):
    """
    :param dataset:
    :return:
    """
    standard_scaler = StandardScaler()
    train = standard_scaler.fit_transform(train)
    test = standard_scaler.transform(test)
    
    return train, test
 
def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    
    print (dataset.describe())

def main():
    """
    Main function
    :return:
    """

    # Importing the dataset
    dataset = read_data(INPUT_PATH)
    
    # Matrix of features X and dependent variables vector y
    X = dataset.iloc[:, :-1].values #all rows and columns less the last column  
    y = dataset.iloc[:, 3].values #all rows and columns until the third column
    
    # Taking care of missing data
    X = handle_missing_data(X, 1, 3)
    
    # Encoding categorical data
    X[:, 0] = enconding_categorical_data(X[:, 0])
    y = enconding_categorical_data(y)
    
    # Dummy encoding
    X = dummy_encoding(X, 0)
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)
    
    #Features Scaling
    X_train, X_test = scaling_features(X_train, X_test)
    print(dataset)
    
if __name__ == "__main__":
    main()
