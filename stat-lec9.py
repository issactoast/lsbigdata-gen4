import numpy as np

x=np.array([13, 23, 24, 20, 27, 18, 15])
exp_x=np.repeat(20, 7)


from scipy.stats import chisquare
from scipy.stats import chi2
statistic, p_value=chisquare(x, f_exp=exp_x)

# 통계량: sum((x-exp_x)**2 / exp_x)
statistic
# p-value: chi2.sf(statistic, df=6)
p_value

import pandas as pd
import numpy as np
odors = ['Lavender', 'Rosemary', 'Peppermint']
minutes_lavender = [10, 12, 11, 9, 8, 12, 11, 10, 10]
minutes_rosemary = [14, 15, 13, 16, 14, 15, 14, 13, 14, 16]
minutes_peppermint = [18, 17, 18, 16, 17, 19, 18, 17]
anova_data = pd.DataFrame({
    'Odor': np.array(["Lavender"]*9 + ["Rosemary"]*10 + ["Peppermint"]*8) ,
    'Minutes': minutes_lavender + minutes_rosemary + minutes_peppermint
})

anova_data.groupby(['Odor']).describe()

# SST = SSG + SSE

# 전체 평균
grand_mean = anova_data['Minutes'].mean()

# SSG: 집단 간 제곱합 = Σ nj * (meanj - grand_mean)^2
group_means = anova_data.groupby('Odor')['Minutes'].mean()
group_counts = anova_data.groupby('Odor')['Minutes'].count()
ssg = ((group_means - grand_mean)**2 * group_counts).sum()

# SSE: 집단 내 제곱합 = Σ (x_ij - meanj)^2
anova_data = anova_data.join(group_means, 
                             on='Odor',
                             rsuffix='_groupmean')
sse = ((anova_data['Minutes'] - anova_data['Minutes_groupmean'])**2).sum()

# SST: 총 제곱합 = Σ (x_ij - grand_mean)^2
sst = ((anova_data['Minutes'] - grand_mean)**2).sum()

sst
ssg + sse

from scipy.stats import f_oneway

f_stat, p_value=f_oneway(minutes_lavender,
                         minutes_rosemary,
                         minutes_peppermint)
f_stat
p_value


res=anova_data['Minutes'] - anova_data['Minutes_groupmean']

import matplotlib.pyplot as plt
x = anova_data['Odor']
y = res 
plt.scatter(x, y, color="red")
