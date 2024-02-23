import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

mse_list = []
rmse_list = []

for layers in range(1, 21):  # Iteration from 1 - 20 layers
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(3,)))  # Input layer number
    for _ in range(layers):  # Add NN layers
        model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)  # Model training

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    mse_list.append(mse)
    rmse_list.append(rmse)

# Plot MSE and RMSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 21), mse_list, marker='o', label='MSE')
plt.title('Model MSE vs. Number of Layers')
plt.xlabel('Number of Layers')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 21), rmse_list, marker='o', color='red', label='RMSE')
plt.title('Model RMSE vs. Number of Layers')
plt.xlabel('Number of Layers')
plt.ylabel('RMSE')
plt.legend()

plt.tight_layout()
plt.show()
