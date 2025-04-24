import numpy as np
from scipy.stats import rankdata

x=np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15,
            10.52, 14.83, 13.03, 16.46, 10.84, 12.45])

eta_0=10
len(x)

# r_i = eta_0에서 떨어진 거리의 순위
r_i=rankdata(abs(x-eta_0))
psi_i=np.where(x-eta_0 >= 0, 1, 0)

w_plus= sum(psi_i * r_i)
w_plus


from scipy.stats import wilcoxon

stat, pvalue=wilcoxon(x-eta_0, alternative="two-sided")
78.0 - stat
pvalue