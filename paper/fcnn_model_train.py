import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle


with open('Test_data/ER4043_model_150ipm_2023_10_08_08_23_41_with_centroids.pkl', 'rb') as file:
    layers_data = pickle.load(file)

X = []  
y = []  

for layer_name, layer_data in layers_data.items():
    for segment_data in layer_data['segments']:
        # print(segment_data['spectral_centroid_mean'])
        X.append([segment_data['spectral_centroid_mean']])
        y.append(np.mean(segment_data['height_segment']))  

X = np.array(X)
y = np.array(y)

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data normilization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = MLPRegressor(hidden_layer_sizes=(50,100,100,100,100,100,100,), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R^2: {r2}')

# Visualization
plt.figure(figsize=(10, 6))  

# Plot observated data
plt.plot(y_test, 'o-', color='black', label='Actual Height Average')

# Plot predicted data
plt.plot(y_pred, 'o-', color='red', label='Predicted Height Average')

plt.xlabel('Sample Index')
plt.ylabel('Height Average')
plt.title('Comparison of Actual and Predicted Height Averages')
plt.legend()
plt.show()