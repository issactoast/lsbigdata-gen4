import numpy as np

# 1표본 t검정 - 모듈사용
x=np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52,
            14.83, 13.03, 16.46, 10.84, 12.45])

from scipy.stats import ttest_1samp

statistic, p_value = ttest_1samp(x, popmean=10, alternative="two-sided")
statistic
p_value

# 독립 2표본 t검정
gender=np.array(["F"]*7 + ["M"]*5)
gender

import pandas as pd
my_tab=pd.DataFrame({
    "score": x,
    "gender": gender
})
my_tab

## 귀무가설
# mu_male==mu_female

## 대립가설
# mu_male > mu_female

from scipy.stats import ttest_ind

male_score = my_tab[my_tab["gender"] == "M"]["score"]
female_score = my_tab[my_tab["gender"] == "F"]["score"]
t_value, p_value=ttest_ind(male_score, female_score,
                           equal_var=True, 
                           alternative="greater")

t_value

# 1-t.cdf(t_value, df=(5+7-2))
p_value

# 유의수준 0.05 보다 p_value가 작으므로, 귀무가설 기각!
# 따라서, 남학생 그룹의 평균이 여학생 그룹의 평균보다 높다고
# 판단할 통계적인 근거가 충분하다.

# 대응표본 t검정
before_score=np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15])
after_score=np.array([10.52,14.83, 13.03, 16.46, 10.84, 12.45])

from scipy.stats import ttest_rel
t_value, p_value=ttest_rel(after_score, before_score, 
                           alternative="greater")

# 유의수준 0.05보다 p-value값이 작으므로,
# 귀무가설 기각 -> 교육프로그램 효과가 있다라고 판단.
after_score - before_score

t_value, p_value=ttest_1samp(after_score - before_score, 
                             popmean=0, alternative="greater")

# F 검정
oj_len=np.array([17.6, 9.7, 16.5, 12.0, 21.5, 23.3, 23.6,
                 26.4, 20.0, 25.2, 25.8, 21.2, 14.5, 27.3, 23.8])
vc_len=np.array([7.6, 4.2, 10.0, 11.5, 7.3, 5.8, 14.5, 10.6,
                 8.2, 9.4, 16.5, 9.7, 8.3, 13.6, 8.2])

f_value= oj_len.var(ddof=1) / vc_len.var(ddof=1)
f_value

# F(15-1, 15-1)
import matplotlib.pyplot as plt
from scipy.stats import f

k=np.linspace(0, 6, 100)
pdf_f=f.pdf(k, 14, 14)
plt.plot(k, pdf_f) 
plt.title('f 14,14')
plt.show()

# 왼쪽 p-value: 0.03555
f.cdf(f_value, 14, 14)

# 오른쪽 p-value: 0.03555
1-f.cdf(f_value, 14, 14)

p_value=2*(1-f.cdf(f_value, 14, 14))
p_value

# 유의수준 0.05 보다 p-value값이 크므로, 
# 귀무가설 기각 하지 못함 -> 분산 같다고 판단.

# 2표본 검정시 equl_var=True로 놓고 분석
# oj_len
# vc_len


# F 검정 함수 정의
from scipy.stats import f

def f_test(x, y, alternative="two_sided"):
    x = np.array(x)
    y = np.array(y)
    df1 = len(x) - 1
    df2 = len(y) - 1
    f = np.var(x, ddof=1) / np.var(y, ddof=1) # 검정통계량
    if alternative == "greater":
        p = 1.0 - f.cdf(f, df1, df2)
    elif alternative == "less":
        p = f.cdf(f, df1, df2)
    else:
        # two-sided by default
        p = 1-f.cdf(f, df1, df2)
        p = 2.0 * min(p, 1-p)
    return f, p

0.8