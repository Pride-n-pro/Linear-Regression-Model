import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import necessary modules
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.plotting import plot_learning_curves


# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense


#importing and seeing the data
df = pd.read_csv("C:\\Users\\Prajnan Karmakar\\OneDrive\\Desktop\\python programming\\pats_wags.csv")
print(df)
#print(df.describe())

#normalising and selecting the target variables
target_columns=['Y']
predictors = ['X']
#df[predictors] = df[predictors]/df[predictors].max()
#print(df)
#print(df.describe())

#train test split
X = df[predictors].values
y = df[target_columns].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#print(X_train.shape); print(X_test.shape)


# Define model
model = Sequential()
model.add(Dense(100, input_dim=1, activation= "relu"))
model.add(Dense(20, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))
model.summary() #Print model Summary

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)


pred_train= model.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train)))

pred= model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred)))

#printing train test data
print(X_test)
print(model.predict(X_test))

#plt.scatter(X_test, pred ,color='g') 
#plt.plot(X_test,y_test)


#learning curve


# save model and architecture to single file
#model.save("model.h5")
#print("Saved model to disk")


# load and evaluate a saved model
#from numpy import loadtxt
#from tensorflow.keras.models import load_model
 
# load model
#model = load_model('model.h5')
# summarize model.
#model.summary()
#model.predict(df)

#Learning Curve
#history = model.fit(X_train, y_train,validation_split = 0.5, epochs=100, batch_size=4)
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

