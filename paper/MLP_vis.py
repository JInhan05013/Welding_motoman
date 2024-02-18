import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

def visualize_predictions(y_true, y_pred, title='Predicted vs Actual Values'):
    plt.figure(figsize=(14, 6))

    # Assuming first column is the mean and second column is the std deviation
    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], [y_true[:, 0].min(), y_true[:, 0].max()], 'k--', lw=2)
    plt.xlabel('Actual Mean (mm)')
    plt.ylabel('Predicted Mean (mm)')
    plt.title('Mean Comparison')

    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], [y_true[:, 1].min(), y_true[:, 1].max()], 'k--', lw=2)
    plt.xlabel('Actual Std Deviation')
    plt.ylabel('Predicted Std Deviation')
    plt.title('Std Deviation Comparison')

    plt.suptitle(title)
    plt.show()

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

model = Sequential([
    Dense(32, activation='relu', input_shape=(3,)),  # Adjusted to match your feature vector size
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')  # Outputs two features: mean and std deviation of height differences
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test dataset
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print evaluation results
print("MSE: ", mse)
print("RMSE: ", rmse)



visualize_predictions(y_test, predictions)