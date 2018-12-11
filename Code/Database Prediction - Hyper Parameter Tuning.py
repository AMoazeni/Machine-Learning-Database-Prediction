# Artificial Neural Network
# Part 5 - Improve and Tune Hyper-Parameters for the ANN (takes a long time to compute)

# This method is the most robust architecture, allows you to find the best hyper-parameters and accuracies

# Dropout Regularization to reduce overfitting if needed
# GridSearch tries several Tuning Hyper Parameters to find the best ones

# K-Fold Cross Validation breaks up the data into 'K' chunks
# It then trains 'K' times, choosing a different chunk every time, this improves accuracy


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Import data and extract input x and output y
data_url = 'https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Database-Prediction/master/Data/Bank_Customer_Data.csv'
dataset = pd.read_csv(data_url)
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Data pre-processing
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# Prepare training and test sets, also apply feature scaling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# This function has an input (Optimizer) so we can try different ones
# 'Adam' and 'rmsprop' (also good for RNN) are good optimizers for stochastic gradient descent
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# Build Neural Network Classifier with K-Fold Cross Validation training, tune Hyper-Parameters here
# Try 'epochs': [100, 500] for major improvements to accuracy
# Try 'cv = 10' for increased K-Validation segmentation
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [5, 10],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 2)


# Fit Model to data using grid_search to try various Hyper Parameter
grid_search = grid_search.fit(x_train, y_train)
# Output best parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


print('\n\n\nThis model was trained with an accuracy of %.2f%%\n' % (best_accuracy*100))

print('\nThe best hyper-parameters for this model:\n', best_parameters, '\n')

print('\nUse the above Hyper-Parameters to retrain your model and make improved predictions.')