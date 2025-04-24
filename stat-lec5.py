import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
# U(3, 7)
# 실제 모분산: 1.33333
uniform(loc=3, scale=4).var()

n=20
x=uniform.rvs(size=n, loc=3, scale=4)
x_bar=np.mean(x)

# 95% 신뢰구간은 계산해보세요!
# (4.034, 5.046)
# (4.471, 5.483)
# X_bar ~ N(x_bar, 1.3333/20)
from scipy.stats import norm
norm.ppf(0.025, loc=x_bar, scale=np.sqrt(1.3333/n))
norm.ppf(0.975, loc=x_bar, scale=np.sqrt(1.3333/n))

# 과연 위 신뢰구간은 언제나 모평균을 포함하고 있을까?
# 1000번의 신뢰구간을 계산하고, 그 신뢰구간이 모평균 5를
# 포함하는 비율을 계산하세요.
norm.ppf(0.025, loc=np.array([1, 2]), scale=np.sqrt(1.3333/n))

n=20
x=uniform.rvs(size=n*1000, loc=3, scale=4)
x_mat=np.reshape(x, (1000, 20))
x_bar=x_mat.mean(axis=1)

a=norm.ppf(0.025, loc=x_bar, scale=np.sqrt(1.3333/n))
b=norm.ppf(0.975, loc=x_bar, scale=np.sqrt(1.3333/n))
sum((a < 5) & (5 < b)) / 1000

norm.ppf(0.975, loc=0, scale=1)
norm.ppf(0.01, loc=0, scale=1)


n=20
x=uniform.rvs(size=n, loc=3, scale=4)
x_bar=np.mean(x)

# x_bar=4.952
# n=20
# 모분산 1.33333
# 모평균에 대한 86% 신뢰구간을 구하세요.
z_007=norm.ppf(0.93, loc=0, scale=1)
x_bar - np.sqrt(1.33333/n) * z_007 
x_bar + np.sqrt(1.33333/n) * z_007



n=9
x=uniform.rvs(size=n, loc=3, scale=4)

x=np.array([6.663, 5.104, 3.026, 6.917, 5.645,
            4.138, 4.058, 6.298, 6.506])
# 검정통계량 값
z=(x.mean() - 4) / (np.sqrt(2)/3)
z
# p-value
2*(1-norm.cdf(z, loc=0, scale=1))


# 문제 연습
# 어느 커피숍에서 판매하는 커피 한잔의 평균 온도
# 가 75도씨라고 주장하고 있습니다. 이 주장에 
# 의문을 가진 고객이 10잔의 커피 온도를 측정한
# 결과 다음과 같은 값을 얻었습니다.
# 단, 모표준편차는 1.2라고 알려져 있습니다.

72.4, 74.1, 73.7, 76.5, 75.3,
74.8, 75.9, 73.4, 74.6, 75.1

# 1) 귀무가설 대립가설을 설정하세요.
# H0: mu=75, vs. HA: mu!=75

# 2) z 검정통계량 값을 구하세요.
x=np.array([72.4, 74.1, 73.7, 76.5, 75.3,
            74.8, 75.9, 73.4, 74.6, 75.1])
z=(x.mean() - 75) / (1.2/np.sqrt(10))

# 3) 유의확률 값을 계산하세요.
# norm.cdf(z, loc=0, scale=1)
p_value=2*norm.cdf(z)
p_value

# 4) 유의수준 5%에서 통계적 판단은?
# p_value가 유의수준인 0.05보다 크므로, 귀무가설을
# 기각 할 수 없다.


# 새로운 분포의 특성 t 분포
from scipy.stats import t
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt

# 모수가 1개: 자유도(df)
# 분포의 퍼짐을 관장하는 모수
# 자유도가 작으면 작을 수록 => 넓게 퍼짐
# 자유도가 커질수록 => 표준정규분포와 비슷
# t(df=inf) = Z
# t 분포는 일반적으로 표준정규분포에 비해
# 꼬리가 두껍다
# t 분포는 항상 중심이 0
k=np.linspace(-3, 3, 100)
pdf_t=t.pdf(k, df=3)
pdf_z=norm.pdf(k, loc=0, scale=1)
plt.plot(k, pdf_t) # 막대 그래프
plt.plot(k, pdf_z, color="r") # 막대 그래프
plt.title('t df=3 vs N(0, 1)')
plt.show()

