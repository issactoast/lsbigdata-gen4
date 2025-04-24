import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']


# 회귀분석 기본코드
from statsmodels.formula.api import ols

model1 = ols("Petal_Length ~ Petal_Width", # 종속 ~ 독립 
            data=iris).fit()
print(model1.summary())

# 회귀분석 ANOVA
import statsmodels.api as sm
sm.stats.anova_lm(model1)

# 변수 3개 다중회귀분석
model2 = ols("Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width", # 종속 ~ 독립 
            data=iris).fit()
print(model2.summary())

# 모델 선택 검정
# H0: Reduced Model(변수 개수 작은 모델)
# HA: Full Model(변수 개수 많은 모델)
table = sm.stats.anova_lm(model1, model2) #anova
print(table)
# F 값이 크므로, p-value 작음. 유의수준 0.05하에서
# 귀무가설 기각 -> Full 모델 선택


# 잔차 뽑아오기
import scipy.stats as stats
import matplotlib.pyplot as plt

residuals = model2.resid
fitted_values = model2.fittedvalues
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals)
plt.subplot(1,2,2)
stats.probplot(residuals, plot=plt)
plt.show()

from statsmodels.stats.diagnostic import het_breuschpagan
bptest = het_breuschpagan(model2.resid, model2.model.exog)
print('BP-test statistics: ', bptest[0])
print('p-value: ', bptest[1])

y=iris["Petal_Length"]
y_hat=fitted_values
sigma2_hat=sum((y-y_hat)**2) / (150 - 3 - 1)

residuals**2 / sigma2_hat

# 더빈왓슨 0~4 사이 값
# 0 ~ 1.0	강한 양의 자기상관 있음 (문제 심각)
# 1.0 ~ 1.5	양의 자기상관 존재 가능성 있음
# 1.5 ~ 2.5	자기상관 없음으로 간주 가능 (안전한 범위)
# 2.5 ~ 3.0	음의 자기상관 존재 가능성 있음
# 3.0 ~ 4.0	강한 음의 자기상관 있음 (문제 심각)
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(model2.resid)
print(dw_stat)


# 팔머펭귄
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(penguins.head())


np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)

train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
train_data.head()

import matplotlib.pyplot as plt
import seaborn as sns
# Scatter plot using seaborn
plt.figure(figsize=(8,4))
sns.scatterplot(data=train_data,
                x='bill_length_mm',
                y='bill_depth_mm',
                edgecolor='w', s=50)
plt.title('Bill Length vs Bill Depth by Species')
plt.grid(True)
plt.show()


from statsmodels.formula.api import ols
model1 = ols("bill_length_mm ~ bill_depth_mm", data=train_data).fit()
model1.params

sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                edgecolor='w', s=50)
x_values = train_data['bill_depth_mm']
y_values = 55.4110 - 0.7062 * x_values
plt.plot(x_values, y_values, 
         color='red', label='Regression Line')
plt.grid(True)
plt.legend()
plt.show()

print(model1.summary())

model2 = ols("bill_length_mm ~ bill_depth_mm + species",
             data=train_data).fit()
model2.params
print(model2.summary())

sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep', edgecolor='w', s=50)

train_data['fitted'] = model2.fittedvalues
# 산점도 (실제 데이터)
sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep', edgecolor='w', s=50)
# 그룹별(facet별)로 fitted 선 그리기
for species, df in train_data.groupby('species'):
    df_sorted = df.sort_values('bill_depth_mm')  # X축 기준 정렬
    sns.lineplot(data=df_sorted,
                 x='bill_depth_mm', y='fitted')
plt.title("Regression Lines(fitted)")
plt.legend()
plt.show()

14.57 + 1.32*18 + 9.88 * 1

import scipy.stats as stats
import pingouin as pg

residuals = model2.resid
fitted_values = model2.fittedvalues
plt.figure(figsize=(7.5,3))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals)
plt.subplot(1,2,2)
pg.qqplot(residuals, dist='norm', confidence=0.95)
plt.tight_layout()
plt.show()

robust_model = model2.get_robustcov_results()
print(robust_model.summary())