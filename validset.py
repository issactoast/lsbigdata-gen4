import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train 셋
np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=40, replace=True)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
data_for_learning = pd.DataFrame({'x': x, 'y': y})

# train 셋 나누기 -> train, valid
from sklearn.model_selection import train_test_split
train, valid = train_test_split(data_for_learning, test_size=0.3, random_state=1234)
print(train.shape)
print(valid.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

i=15   # i = 1에서 변동시키면서 MSE 체크 할 것
k = np.linspace(0, 1, 100)
sin_k = np.sin(2 * np.pi * k)

poly1 = PolynomialFeatures(degree=i, include_bias=True)
train_X = poly1.fit_transform(train[['x']])
model1 = LinearRegression().fit(train_X, train['y'])
model_line_blue = model1.predict(poly1.transform(k.reshape(-1, 1)))

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
axes[0].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')
axes[0].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[0].text(0.05, -1.8, f'MSE: {mse_train:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[0].set_title(f'{i}-degree Polynomial Regression (Train)')
axes[0].set_ylim((-2.0, 2.0))
axes[0].legend()
axes[0].grid(True)

# 오른쪽: 검증 데이터
axes[1].scatter(valid['x'], valid['y'], color='green', label='Valid Observed')
axes[1].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')
axes[1].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[1].text(0.05, -1.8, f'MSE: {mse_valid:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[1].set_title(f'{i}-degree Polynomial Regression (Valid)')
axes[1].set_ylim((-2.0, 2.0))
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# tr / val
0.2460 / 0.5427
0.2457 / 0.5443
0.0448 / 0.1667
0.0273 / 0.2645
0.0243 / 2.3836
0.0094 / 1530607.15


# tr / val 2
0.2460 / 0.5327
0.2457 / 0.5243
0.0448 / 0.1667
0.0273 / 0.2645
0.0243 / 2.3836
0.0094 / 1530607.15


# tr / val 3
0.2460 / 0.5827
0.2457 / 0.4943
0.0448 / 0.1667
0.0273 / 0.2645
0.0243 / 2.3836
0.0094 / 1530607.15

최종 tr / val
   / 0.55    # (0.5427 + 0.5327 + 0.5927)
   / 

b=np.array([2, -3, 7])
np.sqrt(sum(b**2))