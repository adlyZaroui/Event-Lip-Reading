''' This file contains code for training an LSTM model.'''

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare your data
X = np.column_stack((t_sampled, x_sampled, y_sampled))
X = X.reshape((X.shape[0], 1, X.shape[1])) # LSTM expects input to be in [samples, timesteps, features] format

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_sampled, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 3)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Fit the model to your data
model.fit(X_train, y_train, epochs=200, verbose=0)