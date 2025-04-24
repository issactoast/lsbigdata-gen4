import pandas as pd
import numpy as np

# 집가격 데이터 불러오세요!

df = pd.read_csv('./data/houseprice/train.csv')
df.head()
df.info()

# 미국 Ames 집 값
df["MSSubClass"].unique()

model2 = ols("house_price ~ MSSubClass",
             data=train_data).fit()
model2.params

### 스텝와이즈

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
from sklearn.datasets import load_iris

# Load iris data
iris = load_iris()
X = iris.data[:, [0, 1, 3]]
y = iris.data[:, 2]
names = np.array(iris.feature_names)[[0, 1, 3]]

# Define model
lr = LinearRegression()

# Define custom feature selector
def aic_score(estimator, X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("Model AIC:", model.aic)
    return -model.aic

# Perform SFS
sfs = SFS(lr,
          k_features=(1,3),
          forward=True,
          scoring=aic_score,
          cv=0,
          verbose = 0)
sfs.fit(X, y)

print('Selected features:', np.array(names)[list(sfs.k_feature_idx_)])

import numpy as np
import pandas as pd

# 종속 변수 y
y = np.array([
    78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7,
    72.5, 93.1, 115.9, 83.8, 113.3, 109.4
])

# 독립 변수 X
X = np.array([
    [7, 26, 6, 60],
    [1, 29, 15, 52],
    [11, 56, 8, 20],
    [11, 31, 8, 47],
    [7, 52, 6, 33],
    [11, 55, 9, 22],
    [3, 71, 17, 6],
    [1, 31, 22, 44],
    [2, 54, 18, 22],
    [21, 47, 4, 26],
    [1, 40, 23, 34],
    [11, 66, 9, 12],
    [10, 68, 8, 12]
])

from sklearn.metrics import r2_score

def adjusted_r2_score(estimator, X, y):
    y_pred = estimator.predict(X)
    n = X.shape[0]
    p = X.shape[1]
    r2 = r2_score(y, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

# Perform SFS - Adj R^2
sfs = SFS(lr,
          k_features=(1,4),
          forward=True,
          scoring=adjusted_r2_score,
          cv=0,
          verbose = 1)
sfs.fit(X, y)

selected_indices = list(sfs.k_feature_idx_)
print('Selected features:', selected_indices)


# Perform SFS - AIC
sfs = SFS(lr,
          k_features=(1,4),
          forward=True,
          scoring=aic_score,
          cv=0,
          verbose = 1)
sfs.fit(X, y)

selected_indices = list(sfs.k_feature_idx_)
print('Selected features:', selected_indices)