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

for layers in range(1, 21):  # 迭代1到20层
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(3,)))  # 输入层
    for _ in range(layers):  # 添加变化的中间层
        model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))  # 输出层

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)  # 训练模型，verbose=0表示不输出训练日志

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    mse_list.append(mse)
    rmse_list.append(rmse)

# 绘制MSE和RMSE的变化曲线
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
