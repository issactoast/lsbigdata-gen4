import numpy as np
import matplotlib.pyplot as plt

# 입력 x, 출력 y 데이터
x = np.array([4, 7, 13, 2, 1, 5, 9])
y = np.array([1, 3, 5, 7, 2, 3, 2])

# 손실 함수 (MSE)
def loss(beta1, beta2):
    y_pred = beta1 * x + beta2
    return np.sum((y - y_pred) ** 2)

# 기울기 계산
def gradients(beta1, beta2):
    y_pred = beta1 * x + beta2
    error = y - y_pred
    d_beta1 = -2 * np.sum(x * error)
    d_beta2 = -2 * np.sum(error)
    return d_beta1, d_beta2

# 초기값 및 설정
beta1, beta2 = 0.0, 0.0        # 시작점
lr = 0.001                    # 학습률
steps = 10000                    # 반복 횟수
history = []                   # 학습 경로 저장

# 경사하강법 수행
for _ in range(steps):
    d_b1, d_b2 = gradients(beta1, beta2)
    beta1 -= lr * d_b1
    beta2 -= lr * d_b2
    history.append((beta1, beta2, loss(beta1, beta2)))

# 최종 파라미터
final_beta1, final_beta2 = beta1, beta2

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='orange', label='Data Points')  # 데이터 점
x_line = np.linspace(min(x), max(x), 100)
y_line = final_beta1 * x_line + final_beta2
plt.plot(x_line, y_line, color='red', label='Fitted Line')  # 학습된 직선
plt.title('Linear Regression via Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 결과 출력
print(f"최종 beta1 (기울기): {final_beta1}")
print(f"최종 beta2 (절편): {final_beta2}")