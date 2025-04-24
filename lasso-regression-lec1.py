import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']


# 회귀분석 기본코드
from statsmodels.formula.api import ols

# 변수 3개 다중회귀분석
model = ols("Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width", # 종속 ~ 독립 
            data=iris).fit()
print(model.summary())

model.params

import numpy as np
b_ols=np.array([-0.26, 1.44, 0.72, -0.64])
sum(abs(b_ols)) # L1
sum(b_ols**2) # L2

b_lasso=np.array([-0.2, 1.3, 0.7, -0.6])
sum(abs(b_lasso))
sum(b_lasso**2)


import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

X = iris[['Petal_Width', 'Sepal_Length', 'Sepal_Width']]
# X = np.column_stack((np.ones(len(X)), X))  # 상수항 추가
y = iris['Petal_Length'].values

# 회귀 계수 추정: β = (X^T X)^(-1) X^T y
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
beta_hat = XtX_inv @ Xty


# 람다가 10인 경우 릿지 회귀계수
my_l=10
XtX = X.T @ X + my_l*np.eye(4)
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
beta_hat_rid = XtX_inv @ Xty

# 라쏘 회귀계수 구하기
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=1.1) # alpha가 람다를 의미
lasso_model.fit(X, y)
lasso_model.intercept_
lasso_model.coef_

# 람다를 크게 설정하면 죽는 변수 개수 많아짐.
# 즉, 람다를 얼마로 설정함에 따라 
# 람다 작으면: 변수를 많이 선택하는 셈
# 람다 면: 변수를 적게 선택하는 셈

# 릿지 회귀
from sklearn.linear_model import Ridge

lasso_model = Ridge(alpha=1.1) # alpha가 람다를 의미
lasso_model.fit(X, y)
lasso_model.intercept_
lasso_model.coef_

# 람다가 커도 계수 0이 되지는 않음
# 단, 베타 계수 벡터의 L2놈 값이 작아짐
# 즉, 베타 계수 값 자체가 큰 값이 없어짐

# house price 데이터 최적 람다를 찾아보기!

