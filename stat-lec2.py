import numpy as np

# X 기대값
x=np.array([1, 2, 3])
p_x=np.array([0.2, 0.5, 0.3])
e_x=sum(x * p_x)
e_x

# X 분산
var_x=sum((x-e_x)**2 * p_x)
var_x

np.sqrt(var_x) # 표준편차

# Y = 2X + 3
# 기대값의 선형성
# E[Y] = E[2X + 3] = E[2X] + E[3] = 2E[X] + 3
2 * e_x + 3

# 분산의 성질
# Var(Y) = Var(2X + 3) = 2^2 Var(X) + 0
2**2 * var_x

# 문제 5. X 기대값, 분산산
x=np.array([0, 1, 2, 3])
p_x=np.array([0.1, 0.3, 0.4, 0.2])
e_x=sum(x * p_x)
e_x

var_x=sum((x-e_x)**2 * p_x)
var_x

# 동전 3번 던졌을때, 나온 앞면의 수: Y
# Y=y: 0, 1, 2, 3
# P(Y=y): 3C0 * 0.5^3, 3C1 * 0.5^3, 3C2 * 0.5^3, 3C3 * 0.5^3
import math as m
y=np.array([0, 1, 2, 3])
p_y=np.array([m.comb(3, 0), m.comb(3, 1), m.comb(3, 2), m.comb(3, 3)])
p_y=p_y * 0.5**3
p_y
e_y=sum(y * p_y)
e_y

var_y=sum((y-e_y)**2 * p_y)
var_y

# Y = X1 + X2 + X3
# Xi: i번째 동전에서 앞이 나온 횟수
# Xi = 0, 1 (1이 나올 확률 0.5)
# E[Y] = E[X1 + X2 + X3] = E[X1] + E[X2] + E[X3]
# E[X]=0.5, E[Y]=1.5
# Var[X] = E[X^2] - E[X]^2


y=np.array([1, 2, 3, 4, 5, 6])
p_y=np.repeat(1/6, 6)
e_y=sum(y * p_y)
e_y

var_y=sum((y-e_y)**2 * p_y)
var_y
0.075 / 0.25


# 심화 7번
y=np.array([1, 2, 3, 4, 5, 6, 7]) * 5
p_y=np.array([0.1, 0.15, 0.2, 0.3, 0.15, 0.05, 0.05])
e_y=sum(y * p_y)
e_y

var_y=sum((y-e_y)**2 * p_y)
sigma=np.sqrt(var_y)

e_y - sigma
e_y + sigma

import math as m

x1=m.comb(20, 19) * 0.02**19 * 0.98**1
x2=m.comb(20, 20) * 0.02**20 * 0.98**0
x3=m.comb(20, 18) * 0.02**18 * 0.98**2
x4=m.comb(20, 17) * 0.02**17 * 0.98**3

x1 + x2 + x3 + x4

p_y=np.array([m.comb(20, x) * 0.02**(x) * 0.98**(20-x) for x in range(0, 21)])
p_y[0]
P(Y <= 1)
p_y[0] + p_y[1]
P(Y <= 1.5)
p_y[0] + p_y[1]
P(Y <= 2)
p_y[0] + p_y[1] + p_y[2]


from scipy.stats import binom

# Y ~ B(20, 0.02),
# P(Y=0) = 20C0 * 0.98**20 * 0.02**0
binom.pmf(k=0, n=20, p=0.02)
binom.pmf(k=1, n=20, p=0.02)
binom.pmf(k=2, n=20, p=0.02)