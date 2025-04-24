# 카이제곱 분포 알아보기
from scipy.stats import chi2

# 1. 표준정규분포 확률변수를 사용해서 만들수 있다.
# Z: 표준정규분포 (평균:0, 분산:1)
# X = Z^2 ~ 카이제곱(1)
# X = Z1^2 + Z2^2 + Z3^2 ~ 카이제곱(3)
# 2. 나오는 값의 범위가 0~무한대


# 표준정규분포에서 3개 샘플 → 제곱합
z = norm.rvs(size=3 * 10000)
z = np.reshape(z, (10000, 3))
def sum_of_squares(x):
    return np.sum(x ** 2)
chi_samples = np.apply_along_axis(sum_of_squares, axis=1, arr=z)

# 카이제곱분포 (df=3)
k = np.linspace(0, 20, 200)
pdf_chi = chi2.pdf(k, df=3)
plt.hist(chi_samples, bins=30, density=True, edgecolor="black", alpha=0.5)
plt.plot(k, pdf_chi, label='Chi-squared(df=3)', color='green')
plt.title('Sum of squares of 3 N(0,1) vs Chi-squared(df=3)')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# 이항분포 근사 시각화
n, p = 1000, 0.5
x = binom.rvs(n=n, p=p, size=10000)

# 정규 근사값 (mean=np, std=sqrt(np(1-p)))
k = np.linspace(min(x), max(x), 100)
pdf_norm = norm.pdf(k, loc=n*p, scale=np.sqrt(n*p*(1-p)))
plt.hist(x, bins=30, density=True, edgecolor="black", alpha=0.5)
plt.plot(k, pdf_norm, label='Normal Approx', color='red')
plt.title('Binomial(n=1000, p=0.5) vs Normal Approximation')
plt.legend()
plt.show()

binom.cdf(460, n=1000, p=0.45)
norm.cdf(460, loc=450, scale=np.sqrt(1000*0.45*0.55))

# 1 표본 분산 검정
x=np.array([10.67, 9.92, 9.62, 9.53, 9.14,
            9.74, 8.45, 12.65, 11.47, 8.62])
n=len(x)
s_2=x.var(ddof=1)
v_value=(n-1)*s_2 / 1.3
v_value

p_value=chi2.sf(v_value, df=n-1)
p_value

1/(2.7/(n-1)*s_2)
1/(19/(n-1)*s_2)

1-chi2.cdf(15.55, df=1)