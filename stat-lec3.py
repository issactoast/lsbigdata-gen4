import numpy as np

rng=np.random.default_rng(seed=2025)
rng.uniform(0, 1)

from scipy.stats import uniform

uniform.pdf(3, loc=2, scale=3)
uniform.pdf(0, loc=2, scale=3)

# 0에서 5까지 균일분포(2, 5)의 pdf를 그려보세요!
k=np.linspace(0, 7, 100)
density_k=uniform.pdf(k, loc=2, scale=3)

import matplotlib.pyplot as plt

plt.plot(k, density_k)
plt.xlabel('x')
plt.ylabel('f_(x)')
plt.show()

# F(x)=P(X <= x)
# 0에서 7까지 균일분포(2, 5)의 cdf를 그려보세요!
k=np.linspace(0, 7, 100)
cdf_k=uniform.cdf(k, loc=2, scale=3)

plt.plot(k, cdf_k)
plt.xlabel('x')
plt.ylabel('f_(x)')
plt.show()

# 균일분포(2, 5) 표본 5개 추출
uniform.rvs(loc=2, scale=3, size=5)

# ppf 함수()
# 균일분포(2, 5)에서 하위 50% 해당하는 값?
uniform.ppf(0.5, loc=2, scale=3)

# X~U(2, 5) 일때, P(1 < X <=4)=?
P(1 < X <=4)

uniform.cdf(4, loc=2, scale=3) - uniform.cdf(1, loc=2, scale=3)

# 기대값 균일분포(2, 5)
uniform.expect(loc=2, scale=3)
uniform.var(loc=2, scale=3)

# X ~ 이항분포 (5, 0.3)
from scipy.stats import binom

# 문제: 이 확률변수의 기대값, 분산은?
# E[X]=np, Var[X]=np(1-p)
binom.expect(args=(5, 0.3))
binom.var(n=5, p=0.3)

# 문제: 동전을 7번 던져서 나온 앞면의 수가
# 3회보다 작은 확률을 구하세요. 
# 동전이 앞면이 나올 확률은 0.47
# X~B(7, 0.47)
# P(X < 3) = P(X=0)+P(X=1)+P(X=2)
binom.pmf(0, n=7, p=0.47)
binom.pmf(1, n=7, p=0.47)
binom.pmf(2, n=7, p=0.47)
sum(binom.pmf([0, 1, 2], n=7, p=0.47))

binom.cdf(2, n=7, p=0.47)

# 어느 한 공장에서 제품을 하나 만들때 불량률 3%
# 해당 제품은 30개씩 박스에 포장되어 출고된다.
# 박스에 하나라도 불량품이 있을 확률은?
# Y: 박스에 들어있는 불량품 개수
# Y ~ B(n=30, p=0.03)
# P(Y=1)+P(Y=2)+...+P(Y=30)
# 1-P(Y=0)
1-binom.pmf(0, n=30, p=0.03)

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

# 문제 포아송분포 확률질량함수 그래프 그려보세요!
# lambda가 3인 포아송분포의 -1에서 20까지
from scipy.stats import poisson
# X~Poi(3)
# P(X=0)
x=np.arange(-1, 21)
p_x=poisson.pmf(x, mu=3)

plt.bar(x, p_x) # 막대 그래프
plt.title('X~Poisson(3)')
plt.show()

# X~Poi(?) 표본을 5개 뽑는법은?
poisson.rvs(mu=3, size=30)

lambda_hat=3.88
np.mean([1, 6, 3, 2, 5, 8, 1, 4, 5])

# 오늘 항의전화 안올 확률은?
poisson.pmf(0, mu=3.88)