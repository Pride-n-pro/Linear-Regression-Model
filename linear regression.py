import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt

#prepare data
datapoints_length=1000
#generate random x values in the range -10 to 1-
x= np.random.uniform(low=-10,high=10, size=(datapoints_length,1))
y= np.random.uniform(low=-10,high=10, size=(datapoints_length,1))
z=5*x + 6*y + 5
noise=np.random.uniform(low=-1, high=1, size=(datapoints_length,1))
z=5*x + 6*y + 5 + noise
input= np.column_stack((x,y))

#define the model achitechture
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])

#compile the data
model.compile(
    loss='mean_squared_error',
    optimizer='sgd',
    metrics=['mse'])

from tensorflow.keras.callbacks import History
history= History()

model.fit(input, z , epochs=15, verbose=1, validation_split=0.2, callbacks=[history])


print("Predicted z for x=2, y=3 ----> ", model.predict([[2,3]]))

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')