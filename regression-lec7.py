from palmerpenguins import load_penguins
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df = load_penguins()
# 입력값: 펭귄 종, 부리 길이
# 결과값: 부리 깊이
# 선형회귀 모델 적합하기 문제
model=LinearRegression()
penguins=df.dropna()
penguins_dummies = pd.get_dummies(penguins, 
                            columns=['species'],
                            drop_first=True)
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies['bill_depth_mm']
x["bill_Chinstrap"] = x["bill_length_mm"] * x["species_Chinstrap"]
x["bill_Gentoo"] = x["bill_length_mm"] * x["species_Gentoo"]
model.fit(x, y)

model.coef_
model.intercept_


# statmodels 사용한 분석과 시각화
# 팔머펭귄
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(penguins.head())


np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)

train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
train_data.head()

model = ols("bill_length_mm ~ bill_depth_mm*species",
             data=train_data).fit()
model.params
print(model.summary())

sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep', edgecolor='w', s=50)

train_data['fitted'] = model.fittedvalues

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