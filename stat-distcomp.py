import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# 가상 데이터 만들기
x=norm.rvs(size=10, loc=3, scale=2)
x.sort()
x
0.2 * x[1] + 0.8 * x[2]


# 백분위수 예제 (25 백분위수)
x=np.array([155, 126, 27, 82, 115,
            140, 73, 92, 110, 134])
x.sort()
x

## j, h 구하기
n=len(x); p=25
(n-1)*(p/100)+1
j=int((n-1)*(p/100)+1) 
h=((n-1)*(p/100)+1) % 1
q_25=(1-h)*x[j-1] + h*x[j]
q_25

np.percentile(x, 25)
np.percentile(x, 50)
np.percentile(x, 75)

# 표본분산 n-1 vs. n
# 1. n이 작은경우 n-1로 나눈 것이 좀 더 정확한 추정값
# 2. n-1로 나눈 값이 이루는 분포 중심이 실제 모분산과 동일
# (기대값이 같다, 불편추정량)
# 균일분포 (3, 7)
# 분산: 1.3333
# 표본분산 n=10 의 히스토그램을 그려보면 

# 나의 데이터가 정규분포를 따를까?
np.percentile(x, 25) # 데이터 25백분위수
norm.ppf(0.25, loc=x.mean(), scale=x.std(ddof=1)) # 40


data_x=np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
                 11.97, 2.16, 3.24, 10.91, 11.36, 0.87])

import scipy.stats as sp
import matplotlib.pyplot as plt
sp.probplot(data_x, dist="norm", plot=plt)


# ECDF
from statsmodels.distributions.empirical_distribution import ECDF
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
                   11.97, 2.16, 3.24, 10.91, 11.36, 0.87])
# data_x=norm.rvs(size=2000, loc=3, scale=2)
ecdf = ECDF(data_x)
x = np.linspace(min(data_x), max(data_x))
y = ecdf(x)
k=np.linspace(-2, 12, 100)
cdf_k=norm.cdf(k,
               loc=data_x.mean(),
               scale=data_x.std(ddof=1))
plt.plot(x,y,marker='_', linestyle='none')
plt.plot(k, cdf_k, color="red")
plt.title("Estimated CDF")
plt.xlabel("X-axis")
plt.ylabel("ECDF")
plt.show()


k=np.linspace(-5, 15, 100)
f_x=norm.cdf(k,
             loc=data_x.mean(),
             scale=data_x.std(ddof=1))
plt.plot(k, f_x * (1-f_x), color="red")
plt.xlabel("X")
plt.ylabel("F(x)(1-F(x))")
plt.show()


from scipy.stats import anderson, norm
sample_data = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
                        11.97, 2.16, 3.24, 10.91, 11.36, 0.87])
result = anderson(sample_data, dist='norm')
result[0] # 검정통계량
result[1] # 임계값
result[2] # 유의수준

