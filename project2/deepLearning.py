import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 加载数据
train_data = sio.loadmat('./dataset/task1/PA_data_train.mat')
test_data = sio.loadmat('./dataset/task1/PA_data_test.mat')

# 提取输入和输出
pa_input_train = train_data['paInput']
pa_output_train = train_data['paOutput']
pa_input_test = test_data['paInput']
pa_output_test = test_data['paOutput']

# 将复杂数据分为实部和虚部
pa_input_train_real = np.real(pa_input_train)
pa_input_train_imag = np.imag(pa_input_train)
pa_output_train_real = np.real(pa_output_train)
pa_output_train_imag = np.imag(pa_output_train)

pa_input_test_real = np.real(pa_input_test)
pa_input_test_imag = np.imag(pa_input_test)
pa_output_test_real = np.real(pa_output_test)
pa_output_test_imag = np.imag(pa_output_test)

# 确保数据维度匹配
pa_input_train_real = pa_input_train_real.reshape(-1, 1)
pa_input_train_imag = pa_input_train_imag.reshape(-1, 1)
pa_output_train_real = pa_output_train_real.reshape(-1, 1)
pa_output_train_imag = pa_output_train_imag.reshape(-1, 1)

pa_input_test_real = pa_input_test_real.reshape(-1, 1)
pa_input_test_imag = pa_input_test_imag.reshape(-1, 1)
pa_output_test_real = pa_output_test_real.reshape(-1, 1)
pa_output_test_imag = pa_output_test_imag.reshape(-1, 1)

# 组合实部和虚部
X_train = np.hstack((pa_input_train_real, pa_input_train_imag))
y_train = np.hstack((pa_output_train_real, pa_output_train_imag))
X_test = np.hstack((pa_input_test_real, pa_input_test_imag))
y_test = np.hstack((pa_output_test_real, pa_output_test_imag))

# 标准化数据
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train = scaler_X.transform(X_train)
y_train = scaler_y.transform(y_train)
X_test = scaler_X.transform(X_test)
y_test = scaler_y.transform(y_test)

# 构建模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 预测
y_pred = model.predict(X_test)

# 反标准化
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

# 计算NMSE
Iout_test = y_test_inv[:, :pa_output_test_real.shape[1]]
Qout_test = y_test_inv[:, pa_output_test_real.shape[1]:]
Iout_pred = y_pred_inv[:, :pa_output_test_real.shape[1]]
Qout_pred = y_pred_inv[:, pa_output_test_real.shape[1]:]

nmse = 10 * np.log10(
    np.sum((Iout_test - Iout_pred) ** 2 + (Qout_test - Qout_pred) ** 2) /
    np.sum(Iout_test ** 2 + Qout_test ** 2)
)

print(f'NMSE (Test): {nmse}')

# 绘图结果
plt.figure(figsize=(14, 14))

# 绘制实际值和预测值的对比图 (实部)
plt.subplot(3, 2, 1)
plt.plot(Iout_test, label='Actual Real')
plt.plot(Iout_pred, label='Predicted Real')
plt.title('Test Data (Real Part)')
plt.legend()

# 绘制实际值和预测值的对比图 (虚部)
plt.subplot(3, 2, 2)
plt.plot(Qout_test, label='Actual Imaginary')
plt.plot(Qout_pred, label='Predicted Imaginary')
plt.title('Test Data (Imaginary Part)')
plt.legend()

# 仅绘制实际值 (实部)
plt.subplot(3, 2, 3)
plt.plot(Iout_test, label='Actual Real', color='blue')
plt.title('Test Data Actual (Real Part)')
plt.legend()

# 仅绘制实际值 (虚部)
plt.subplot(3, 2, 4)
plt.plot(Qout_test, label='Actual Imaginary', color='green')
plt.title('Test Data Actual (Imaginary Part)')
plt.legend()

# 仅绘制预测值 (实部)
plt.subplot(3, 2, 5)
plt.plot(Iout_pred, label='Predicted Real', color='orange')
plt.title('Test Data Predicted (Real Part)')
plt.legend()

# 仅绘制预测值 (虚部)
plt.subplot(3, 2, 6)
plt.plot(Qout_pred, label='Predicted Imaginary', color='red')
plt.title('Test Data Predicted (Imaginary Part)')
plt.legend()

plt.tight_layout()
plt.show()
