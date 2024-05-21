import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
train_data = sio.loadmat('./dataset/task1/PA_data_train.mat')
test_data = sio.loadmat('./dataset/task1/PA_data_test.mat')

x_train = train_data['paInput'].flatten()
y_train = train_data['paOutput'].flatten()
x_test = test_data['paInput'].flatten()
y_test = test_data['paOutput'].flatten()

# Separate real and imaginary parts
x_train_real = x_train.real
x_train_imag = x_train.imag
y_train_real = y_train.real
y_train_imag = y_train.imag

x_test_real = x_test.real
x_test_imag = x_test.imag
y_test_real = y_test.real
y_test_imag = y_test.imag

# Data preprocessing
scaler_x_real = StandardScaler()
scaler_x_imag = StandardScaler()
scaler_y_real = StandardScaler()
scaler_y_imag = StandardScaler()

x_train_real_scaled = scaler_x_real.fit_transform(x_train_real.reshape(-1, 1)).flatten()
x_train_imag_scaled = scaler_x_imag.fit_transform(x_train_imag.reshape(-1, 1)).flatten()
y_train_real_scaled = scaler_y_real.fit_transform(y_train_real.reshape(-1, 1)).flatten()
y_train_imag_scaled = scaler_y_imag.fit_transform(y_train_imag.reshape(-1, 1)).flatten()

x_test_real_scaled = scaler_x_real.transform(x_test_real.reshape(-1, 1)).flatten()
x_test_imag_scaled = scaler_x_imag.transform(x_test_imag.reshape(-1, 1)).flatten()
y_test_real_scaled = scaler_y_real.transform(y_test_real.reshape(-1, 1)).flatten()
y_test_imag_scaled = scaler_y_imag.transform(y_test_imag.reshape(-1, 1)).flatten()

# Feature engineering (for non-linear regression)
def create_features(x, memory_depth):
    X = np.zeros((len(x), memory_depth))
    for i in range(memory_depth):
        X[i:, i] = x[:len(x) - i]
    return X[memory_depth - 1:]

memory_depth = 10
X_train_real = create_features(x_train_real_scaled, memory_depth)
X_train_imag = create_features(x_train_imag_scaled, memory_depth)
Y_train_real = y_train_real_scaled[memory_depth - 1:]
Y_train_imag = y_train_imag_scaled[memory_depth - 1:]

X_test_real = create_features(x_test_real_scaled, memory_depth)
X_test_imag = create_features(x_test_imag_scaled, memory_depth)
Y_test_real = y_test_real_scaled[memory_depth - 1:]
Y_test_imag = y_test_imag_scaled[memory_depth - 1:]

# Model construction
model_real = LinearRegression()
model_imag = LinearRegression()

model_real.fit(X_train_real, Y_train_real)
model_imag.fit(X_train_imag, Y_train_imag)

# Performance evaluation
y_pred_train_real = model_real.predict(X_train_real)
y_pred_train_imag = model_imag.predict(X_train_imag)
y_pred_test_real = model_real.predict(X_test_real)
y_pred_test_imag = model_imag.predict(X_test_imag)

nmse_train_real = mean_squared_error(Y_train_real, y_pred_train_real)
nmse_train_imag = mean_squared_error(Y_train_imag, y_pred_train_imag)
nmse_test_real = mean_squared_error(Y_test_real, y_pred_test_real)
nmse_test_imag = mean_squared_error(Y_test_imag, y_pred_test_imag)

nmse_train = 10 * np.log10(nmse_train_real + nmse_train_imag)
nmse_test = 10 * np.log10(nmse_test_real + nmse_test_imag)

print(f'NMSE (Train): {nmse_train}')
print(f'NMSE (Test): {nmse_test}')

# Plotting results
plt.figure(figsize=(14, 14))

# 实际值和预测值的对比 (训练数据 实部)
plt.subplot(3, 2, 1)
plt.plot(Y_train_real, label='Actual Real')
plt.plot(y_pred_train_real, label='Predicted Real')
plt.title('Train Data (Real Part)')
plt.legend()

# 实际值和预测值的对比 (训练数据 虚部)
plt.subplot(3, 2, 2)
plt.plot(Y_train_imag, label='Actual Imaginary')
plt.plot(y_pred_train_imag, label='Predicted Imaginary')
plt.title('Train Data (Imaginary Part)')
plt.legend()

# 实际值和预测值的对比 (测试数据 实部)
plt.subplot(3, 2, 3)
plt.plot(Y_test_real, label='Actual Real')
plt.plot(y_pred_test_real, label='Predicted Real')
plt.title('Test Data (Real Part)')
plt.legend()

# 实际值和预测值的对比 (测试数据 虚部)
plt.subplot(3, 2, 4)
plt.plot(Y_test_imag, label='Actual Imaginary')
plt.plot(y_pred_test_imag, label='Predicted Imaginary')
plt.title('Test Data (Imaginary Part)')
plt.legend()

# 仅展示实际值 (测试数据 实部)
plt.subplot(3, 2, 5)
plt.plot(Y_test_real, label='Actual Real', color='blue')
plt.title('Test Data Actual (Real Part)')
plt.legend()

# 仅展示实际值 (测试数据 虚部)
plt.subplot(3, 2, 6)
plt.plot(Y_test_imag, label='Actual Imaginary', color='green')
plt.title('Test Data Actual (Imaginary Part)')
plt.legend()

plt.tight_layout()
plt.show()
