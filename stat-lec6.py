# 데이터에서 모수(기대값) 추정하기
# X 의 기대값(평균) => mu=E[X]
# x1, x2, ..., xn 
# sum(x)/n 

# X 의 기대값 E[X^3]
# x1, x2, ..., xn 
# sum(x^3)/n

# X 의 기대값 E[X^3 - X]
# x1, x2, ..., xn 
# sum(x^3 - x)/n

# X 의 기대값 E[X^4 + X^2 - 3]
# sum(x^4 + x^2 - 3) / n

# X 의 기대값(분산) => sigma^2=Var(X)=E[(X-E[X])^2]
# x1, x2, ..., xn 
# sum( (x - x_bar)^2 )/n

from scipy.stats import uniform
import numpy as np
# U(3, 7)
# 실제 모분산: 1.33333
uniform(loc=3, scale=4).var()
# Var(X)=E[(X-E[X])^2]

n=100000
x=uniform.rvs(size=n, loc=3, scale=4)
x_bar=np.mean(x)

# 표본 분산
sum((x - x_bar)**2)/(n-1)

# 스튜던트 t정리 3번 확인
from scipy.stats import norm
from scipy.stats import chi2
import matplotlib.pyplot as plt

x=norm.rvs(size=17 * 1000, loc=10, scale=2)
x=np.reshape(x, (1000, 17))
x.shape
# (17-1)*(sum((x-x.mean())**2)/(n-1)) / 2**2
def myf(x):
    return (17-1)*x.var(ddof=1) / 2**2

n_1s=np.apply_along_axis(myf, axis=1, arr=x)

k=np.linspace(0, 40, 100)
pdf_chi2=chi2.pdf(k, df=(17-1))
plt.plot(k, pdf_chi2) 
plt.hist(n_1s, edgecolor="black",
         density=True, alpha=0.5)
plt.title('t df=3 vs N(0, 1)')
plt.show()

# t 검정통계량 분포 확인
x=norm.rvs(size=17 * 1000, loc=10, scale=2)
x=np.reshape(x, (1000, 17))
def myf2(x):
    return (x.mean() - 10) / (x.std(ddof=1)/np.sqrt(17))

t_sample=np.apply_along_axis(myf2, axis=1, arr=x)

from scipy.stats import t

k=np.linspace(-3.5, 3.5, 100)
pdf_t=t.pdf(k, df=(17-1))
plt.plot(k, pdf_t) 
plt.hist(t_sample, edgecolor="black",
         density=True, alpha=0.5)
plt.title('t df=16 vs t_sample hist')
plt.show()

# t 검정 예제
from scipy.stats import t

x=np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11, 11.97,
            2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9])
x
mu0=7
t_value=(x.mean() - mu0) / (x.std(ddof=1)/np.sqrt(15))
# p-value
2*t.cdf(t_value, df=14)

# 유의수준 0.05 하에서
# p-value가 0.05보다 크므로, 귀무가설을 기각하지
# 못한다. 해당 데이터는 모평균이 7인 분포에서 뽑혀져
# 나왔다고 판단할 수 있다.


# 커피온도 t 검정
x=np.array([72.4, 74.1, 73.7, 76.5, 75.3,
            74.8, 75.9, 73.4, 74.6, 75.1])
t_value=(x.mean() - 75) / (x.std(ddof=1)/np.sqrt(10))
t_value

2*t.cdf(t_value, df=9)

# p-value가 0.05보다 크므로, 귀무가설을 기각하지 못한다.
# 즉, 해당 커피숍의 커피 한잔의 평균 온도가 75도씨라고 판단한다.

# 기각역을 통해서 판단
t.ppf(0.025, df=9) # -2.26
# 기각역 -2.26보다 작거나, 2.26 보다 큰 영역
# t_value -1.08이라는 값이 기각역에 속하지 않으므로
# 귀무가설을 기각할 수 없다.
