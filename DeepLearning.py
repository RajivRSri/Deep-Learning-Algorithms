"""
Customer churn analysis
"""

#Load libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("C:/R")

#Read Dataset
data = pd.read_csv("telecom.csv")

#Check Loaded Dataset
data.head()
data.shape
data.info()

#Check for null values if any
data.isnull().values.sum()
sns.heatmap(data.isnull(),
            yticklabels = False,
            cbar = False,
            cmap = "viridis")

#Check for imbalanced dataset
data.churn.value_counts()
sns.set_style("whitegrid")
sns.countplot(x = "churn", data = data, palette = "RdBu_r")

#Create dummy variables
from sklearn.preprocessing import LabelEncoder
data["international plan"] = pd.get_dummies(data["international plan"], drop_first = True)
data["voice mail plan"] = LabelEncoder().fit_transform(data["voice mail plan"])

#Drop variable "Phone Number"
data = data.drop("phone number", axis = 1)

#Mean Encoding for categorical variable "state"
states = data.state.sort_values().unique()

meanVal = data.groupby(["state"])["churn"].mean()

myDict = {}
for index in range(len(states)):
    myDict.update({states[index] : meanVal[index]})

def replaceWithMean(X):
    return (myDict[X])

data.state = data.state.apply(lambda X : replaceWithMean(X))

#Divide dataset into dependent and independent variables
x = data.drop("churn", axis = "columns")
y = data["churn"]

#Handle imbalanced dataset
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 42)
x_res, y_res = smk.fit_sample(x, y)
print(x_res.shape, y_res.shape)

#Divide dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, train_size = 0.7, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
stdScaler = StandardScaler()
x_train = stdScaler.fit_transform(x_train)
x_test = stdScaler.transform(x_test)

#Build ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

model = Sequential()

#Add input layer and first hidden layer
model.add(Dense(units = 10, kernel_initializer = "he_normal", activation = "relu", input_dim = 19))
model.add(Dropout(0.2))

#Second Hidden Layer
model.add(Dense(units = 15, kernel_initializer = "he_normal", activation = "relu"))
model.add(Dropout(0.3))

#Third Hidden Layer
model.add(Dense(units = 20, kernel_initializer = "he_normal", activation = "relu"))
model.add(Dropout(0.4))

#Output Layer
model.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))

#Compile
model.compile(optimizer = "Adamax", loss = "binary_crossentropy", metrics = ["accuracy"])

#Fit model
result = model.fit(x_train, y_train, validation_split = 0.3, batch_size = 10, nb_epoch = 100)

print(result.history.keys())

# summarize history for accuracy
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Model Prediction
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

#Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)

#Hyperparameter Otimization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Activation
from keras.activations import relu, sigmoid

layers = [[20], [40, 20], [45, 35, 10]]
activations = ['sigmoid', 'relu']

def my_model (layers, activation):
    model = Sequential()
    for index, nodes in enumerate(layers):
        if (0 == index):
            model.add(Dense(nodes, input_dim = x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    model.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model

model = KerasClassifier(build_fn = my_model, verbose = 0)

param_val = dict(layers = layers, activation = activations, batch_size = [128, 256], epochs = [30])

grid = GridSearchCV(estimator = model, param_grid = param_val, cv = 5)

grid_result = grid.fit(x_train, y_train)

grid_result.best_estimator_

grid_result.best_params_