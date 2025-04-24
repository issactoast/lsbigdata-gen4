import numpy as np

# 고객 데이터 만들기
np.random.seed(1234)
age = np.random.randint(20, 81, 10000)
gender=np.random.randint(0, 2, 10000)
gender=np.where(gender == 1, "여자", "남자")
price=np.random.randint(0, 200, 10000)

age[gender == "남자"].sum()-age[gender == "여자"].sum()

price[(gender == "남자") & (age >= 40) & (age < 50)].mean()


# X는 독립 변수 행렬 (예시로 3x3 행렬)
X = np.array([[2, 4],
              [1, 7],
              [7, 8]])

# y는 종속 변수 벡터 (예시로 3x1 벡터)
y = np.array([10, 5, 15])

# Hat 행렬 H 계산
H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)

# H @ y
np.matmul(H, y)


np.random.seed(2025)
array_2d = np.random.randint(1, 13, 12).reshape((3, 4))
array_2d

result=np.apply_along_axis(min_max_numbers, axis=1, arr=array_2d)
result.shape


import numpy as np

x=np.array([21, 12, 24, 18, 25, 28, 22, 22, 29, 14, 20, 45, 16, 18, 15, 17, 23, 55, 19, 26])

x.sort()
len(x)
q2=(x[9] + x[10])/2

np.sort(x[x < q2])
q1=17.5
np.sort(x[x > q2])
q3=25.5

iqr=q3-q1

q1-1.5*iqr
q3+1.5*iqr

x[(x < 5.5) | (x > 37.5)]


import numpy as np
x = np.array([21, 12, 24, 18, 25, 28, 22, 22, 29, 14, 20, 45, 16, 18, 15, 17, 23, 55, 19, 26])

# Q1, Q3, IQR 계산
q2=np.median(x)
q1 = np.median(x[x < q2])  # 1사분위수
q3 = np.median(x[x > q2])  # 3사분위수
iqr = q3 - q1

# 이상치 경계 계산
l_bound = q1 - 1.5 * iqr
u_bound = q3 + 1.5 * iqr

# 이상치 탐지
outliers=x[(x < l_bound) | (x > u_bound)]

# 이상치의 합
outlier_sum = sum(outliers)
print(f"Q1 = {q1}, Q3 = {q3}, IQR = {iqr}")
print(f"Lower bound = {l_bound}, Upper bound = {u_bound}")
print(f"Outliers = {outliers}")
print(f"Outlier sum = {outlier_sum}")


from scipy.stats import norm
평균 3, 표준편차 2
q1=norm(loc=3, scale=2).ppf(0.25)
q3=norm(loc=3, scale=2).ppf(0.75)

q3-q1


from scipy.stats import binom

# 어느 한 공장에서 제품을 하나 만들때 불량률 2%
# 해당 제품은 20개씩 박스에 포장되어 출고된다.
# 박스에 하나라도 불량품이 있을 확률은?
# Y: 박스에 들어있는 불량품 개수
# Y ~ B(n=20, p=0.02)
# P(Y=1)+P(Y=2)+...+P(Y=30)
# 1-(P(Y=0)+P(Y=1))
p_box=1-binom.cdf(1, n=20, p=0.02)
p_box

binom.pmf(1, n=3, p=p_box)

15*14*13

import math

result = math.factorial(15)
result/(2*2*2)

# 어느 한 공장에서 제품을 하나 만들때 불량률 1%
# 해당 제품은 30개씩 박스에 포장되어 출고된다.
# 박스가 하자가 있다고 판단기준: 3개 이상 불량
# 박스가 불량일 확률은?
# Y: 박스에 들어있는 불량품 개수
# Y ~ B(n=30, p=0.01)
# P(Y=3)+P(Y=4)+...+P(Y=30)
# 1-P(Y<=2)
p_box=1-binom.cdf(2, n=30, p=0.01)

# 회사 A 350박스를 판매함. 고객사 판매한 것의 3개
# 넘어가는 박스가 불량인 경우, 전화가 옴.
# A회사에서 항의전화 올 확률은?
# Z: 350개의 판매된 박스 중 불량박스 수
# Z ~ B(n=350, p=p_box)
# P(Z > 3)=1-P(Z <= 3)
1-binom.cdf(3, n=350, p=p_box)
