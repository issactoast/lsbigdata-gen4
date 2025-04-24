import matplotlib.pyplot as plt
from scipy.stats import bernoulli

x = [0, 1]
p = 0.7
pmf = bernoulli.pmf(x, p)
plt.bar(x, pmf)
plt.title("Bernoulli PMF (p=0.7)")
plt.xlabel("x")
plt.ylabel("P(X=x)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

visits = [0, 1, 2, 0, 3, 1, 4, 2, 2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 4, 2]
np.mean(visits)

# 관측값
values, counts = np.unique(visits, return_counts=True)
prob_obs = counts / len(visits)

# 추정된 파라미터
lambda_hat = np.mean(visits)
x = np.arange(0, max(values)+1)
pmf_theory = poisson.pmf(x, mu=lambda_hat)

# 시각화
plt.bar(x -   0.2, prob_obs, width=0.4, label="Observed", color="skyblue")
plt.bar(x + 0.2, pmf_theory, width=0.4, label="Poisson Fit", color="orange")
plt.xlabel("Visits")
plt.ylabel("Probability")
plt.title("Observed vs. Fitted Poisson PMF")
plt.legend()
plt.grid(True)
plt.show()


# 문제 포아송분포 확률질량함수 그래프 그려보세요!
# lambda가 3인 포아송분포의 -1에서 20까지
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
# X~Poi(3), E[X]=lambda=3
# P(X=0)
x=np.arange(-1, 16)
p_x=poisson.pmf(x, mu=3)

# 아래 그래프에 분포의 기대값 자리에 빨간 세로선
# 추가해주세요!
mu=poisson.expect(args=(3,))
plt.bar(x, p_x) # 막대 그래프
plt.axvline(3, color="r")
plt.title('X~Poisson(3)')
plt.show()

# X~Poi(?) 표본을 5개 뽑는법은?
np.mean(poisson.rvs(mu=3, size=30))
np.mean(poisson.rvs(mu=3, size=9))

lambda_hat=3.88
np.mean([1, 6, 3, 2, 5, 8, 1, 4, 5])

# 오늘 항의전화 안올 확률은?
poisson.pmf(0, mu=3.88)

# 정규분포
from scipy.stats import norm

# X~ N(5, 4^2) pdf 그려보세요!
# Y~ N(5, 1^2) pdf 겹쳐그려보자!
k=np.linspace(-10, 20, 300)
p_x=norm.pdf(k, loc=5, scale=4)
p_y=norm.pdf(k, loc=7, scale=4)

plt.plot(x, p_x) # 막대 그래프
plt.plot(x, p_y, color="r") # 막대 그래프
plt.title('N(5, 4^2) vs. N(5, 3^2)')
plt.show()

# X~ N(5, 2^2) 일때
# P(X <= 5)를 계산하는 코드는?
norm.cdf(5, loc=5, scale=4)
# P(X <= 7)를 계산하는 코드는?
norm.cdf(7, loc=5, scale=2)-norm.cdf(3, loc=5, scale=2)

# 표준정규분포 Q1
norm.ppf(0.25, loc=0, scale=1)

# -2, 12까지 N(5, 2^2) pdf 그려보자
k=np.linspace(-3, 3, 300)
p_z=norm.pdf(k, loc=0, scale=1)
plt.scatter(k, p_z)
plt.xlim(-3, 3)
# 표준정규분포 표본 3000개 히스토그램
x=norm.rvs(loc=5, scale=2, size=3000)
plt.hist((x-5)/2, color="red", density=True, 
         alpha=0.3, edgecolor="black")

np.mean((x-5)/2)
np.std((x-5)/2, ddof=1)

P(X_bar > 32)=1-P(X_bar <= 32)
1-norm.cdf(32, loc=30, scale=np.sqrt(5**2/20))

# 균일분포(2, 14)
from scipy.stats import uniform
from scipy.stats import norm

# uniform.expect(loc=2, scale=12)
# uniform.expect(loc=2, scale=12)
uniform(2, 12).mean()
uniform(2, 12).var()
norm.cdf(6, loc=8, scale=np.sqrt(12/30))

from scipy.stats import expon

expon(scale=3).mean()
expon(scale=3).var()