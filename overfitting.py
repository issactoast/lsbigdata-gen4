import numpy as np
import pandas as pd

# np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=10, replace=False)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
# y = 2*x + 3 + np.random.normal(0, 0.2, len(x))
# y = 2*x + 3
# y = np.sin(2 * np.pi * x)

mydata = pd.DataFrame({'x': x, 'y': y})
mydata = mydata.sort_values('x').reset_index(drop=True)
print(mydata)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, color='blue',
#             label='Data Points')  # 데이터 점

import matplotlib.pyplot as plt

x2 = np.linspace(0, 1, 100)
y2 = np.sin(2 * np.pi * x2)
# y2 = 2*x2 + 3

plt.figure(figsize=(6, 4))
plt.scatter(mydata['x'], mydata['y'],
            color='black', label='Observed')
plt.plot(x2, y2, color='red', label='True Curve')
plt.axhline(y=np.mean(mydata['y']),
            color='blue',
            label='Mean Model')
plt.title('Data and True Curve')
plt.legend()
plt.grid(True)
plt.show()


x = np.random.choice(np.arange(0, 1.05, 0.05), size=10, replace=False)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
mydata = pd.DataFrame({'x': x, 'y': y})
mydata = mydata.sort_values('x').reset_index(drop=True)
print(mydata)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly1 = PolynomialFeatures(degree=9, include_bias=True)
X1 = poly1.fit_transform(mydata[['x']])
model1 = LinearRegression().fit(X1, mydata['y'])
y1_pred = model1.predict(poly1.transform(x2.reshape(-1, 1)))

plt.figure(figsize=(6, 4))
# plt.scatter(mydata['x'], mydata['y'], color='black', label='Observed')
# 여러 번 반복하면서 파란 선 겹치기
for _ in range(200):  # 원하는 반복 횟수 (ex: 50번)
    x = np.random.choice(np.arange(0, 1.05, 0.05), size=10, replace=False)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
    mydata = pd.DataFrame({'x': x, 'y': y}).sort_values('x').reset_index(drop=True)

    poly1 = PolynomialFeatures(degree=9, include_bias=True)
    X1 = poly1.fit_transform(mydata[['x']])
    model1 = LinearRegression().fit(X1, mydata['y'])
    y1_pred = model1.predict(poly1.transform(x2.reshape(-1, 1)))

    plt.plot(x2, y1_pred, color='blue', alpha=0.2)  # alpha 낮게 설정하여 투명하게

plt.plot(x2, y2, color='red', label='True Curve')
plt.plot(x2, y1_pred, color='blue', label='Degree 1 Fit')
plt.title('1-degree Polynomial Regression')
plt.ylim((-2.0, 2.0))
plt.legend()
plt.grid(True)
# plt.show()


# train 셋
np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=40, replace=True)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
data_for_learning = pd.DataFrame({'x': x, 'y': y})

# test 셋
x_test = np.random.choice(np.arange(0, 1.05, 0.05), size=5, replace=True)
true_test_y = np.sin(2 * np.pi * x_test) + np.random.normal(0, 0.2, size=len(x_test))
data_for_testing = pd.DataFrame({'x': x_test, 'y': [np.nan]*len(x_test)})
data_for_testing.head()

# train 셋 나누기 -> train, valid
from sklearn.model_selection import train_test_split
train, valid = train_test_split(data_for_learning, test_size=0.3, random_state=1234)
print(train.shape)
print(valid.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

i=19
poly1 = PolynomialFeatures(degree=i, include_bias=True)
train_X = poly1.fit_transform(train[['x']])
model1 = LinearRegression().fit(train_X, train['y'])
model_line_blue = model1.predict(poly1.transform(x2.reshape(-1, 1)))

# 예측값 계산
train_y_pred = model1.predict(poly1.transform(train[['x']]))
valid_y_pred = model1.predict(poly1.transform(valid[['x']]))

# MSE 계산
mse_train = mean_squared_error(train['y'], train_y_pred)
mse_valid = mean_squared_error(valid['y'], valid_y_pred)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1행 2열 서브플롯

# 왼쪽: 학습 데이터와 모델 피팅 결과
axes[0].scatter(train['x'], train['y'], color='black', label='Train Observed')
axes[0].plot(x2, y2, color='red', alpha=0.3, label='True Curve')
axes[0].plot(x2, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[0].text(0.05, -1.8, f'MSE: {mse_train:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[0].set_title(f'{i}-degree Polynomial Regression (Train)')
axes[0].set_ylim((-2.0, 2.0))
axes[0].legend()
axes[0].grid(True)

# 오른쪽: 검증 데이터
axes[1].scatter(valid['x'], valid['y'], color='green', label='Valid Observed')
axes[1].plot(x2, y2, color='red', alpha=0.3, label='True Curve')
axes[1].plot(x2, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[1].text(0.05, -1.8, f'MSE: {mse_valid:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[1].set_title(f'{i}-degree Polynomial Regression (Valid)')
axes[1].set_ylim((-2.0, 2.0))
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

perform_train = []
perform_valid = []

i=1
for i in range(1, 21):
    poly = PolynomialFeatures(degree=i, include_bias=True)
    X_train = poly.fit_transform(train[['x']])
    X_valid = poly.transform(valid[['x']])
    model = LinearRegression().fit(X_train, train['y'])
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    mse_train = mean_squared_error(train['y'], y_train_pred)
    mse_valid = mean_squared_error(valid['y'], y_valid_pred)
    perform_train.append(mse_train)
    perform_valid.append(mse_valid)

best_degree = np.argmin(perform_valid) + 1
print("Best polynomial degree:", best_degree)






