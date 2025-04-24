import numpy as np

# f(x)=x^2
def my_f(x):
    return np.sin(x)

x=np.linspace(-10, 10, 100)
x

f_x=my_f(x)
f_x

import matplotlib.pyplot as plt
plt.scatter(x, f_x, color="blue")

# f_x의 도함수
# (f(x+h) - f(x))/h
x=2
h=0.000001 # h 고정
(my_f(x+h) - my_f(x)) / h

def my_fprime(x):
    h=0.000001
    return (my_f(x+h) - my_f(x)) / h

my_fprime(1)
my_fprime(1.5)
my_fprime(-3)
my_fprime(3)
my_fprime(2)


import matplotlib.pyplot as plt
x=np.linspace(-10, 10, 100)
plt.scatter(x, my_fprime(x), color="red")



# 입력값 2개인 함수
def my_f2(beta1, beta2):
    return (beta1-1)**2 + (beta2-2)**2

my_f2(4, 2)

# GPT를 사용해서 3차원 평면에 해당 함수를 
# 그려보세요

def my_f2(beta1, beta2):
    return (beta1-1)**2 + (beta2-2)**2

my_f2(4, 2)

from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
# β1, β2 범위 설정
beta1 = np.linspace(-1, 3, 200)
beta2 = np.linspace(-1, 5, 200)
B1, B2 = np.meshgrid(beta1, beta2)

# 함수값 계산
Z = my_f2(B1, B2)

# 3D 플롯
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 표면 그리기
surf = ax.plot_surface(B1, B2, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# 등고선 추가 (바닥면)
ax.contour(B1, B2, Z, zdir='z', offset=Z.min(), cmap='viridis')

# 축 레이블
ax.set_xlabel('beta1')
ax.set_ylabel('beta2')
ax.set_zlabel('my_f2(beta1, beta2)')
ax.set_title('3D Surface Plot of my_f2')

# 컬러바
fig.colorbar(surf, shrink=0.5, aspect=10, label='f value')

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 함수 정의
def my_f2(beta1, beta2):
    return (beta1 - 1)**2 + (beta2 - 2)**2

# 그리드 정의
beta1_vals = np.linspace(-1, 5, 100)
beta2_vals = np.linspace(-1, 5, 100)
B1, B2 = np.meshgrid(beta1_vals, beta2_vals)

# 함수 값 계산
Z = my_f2(B1, B2)

# 등고선 그래프 그리기
plt.figure(figsize=(8, 6))
contour = plt.contour(B1, B2, Z, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.title('Contour plot of my_f2(beta1, beta2)')
plt.xlabel('beta1')
plt.ylabel('beta2')
plt.grid(True)
plt.axis('equal')
plt.show()
