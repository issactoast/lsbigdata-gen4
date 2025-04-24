import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

X = iris[['Petal_Width', 'Sepal_Length', 'Sepal_Width']]
# X = np.column_stack((np.ones(len(X)), X))  # 상수항 추가
y = iris['Petal_Length'].values


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

from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet(alpha=1.0, l1_ratio=0, max_iter=10000)
elastic_model.fit(X, y)
elastic_model.intercept_
elastic_model.coef_

# CV를 통한 람다 찾기 - 라쏘
import pandas as pd
import numpy as np

data_X = pd.read_csv('./data/QuickStartExample_x.csv')
y = pd.read_csv('./data/QuickStartExample_y.csv')
data_X.head()

# 라쏘 회귀직선 구하기 - 람다 0.01 ~ 0.5인 경우 확인
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01) # alpha가 람다를 의미
lasso_model.fit(data_X, y)
lasso_model.intercept_
lasso_model.coef_

# data_X -> train, valid
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(data_X, y, 
                                                      test_size=0.3,
                                                      random_state=2025)

# 예: 람다 0.1 라쏘 적합 후 예측성능 평가
lasso_model = Lasso(alpha=0.2) # alpha가 람다를 의미
lasso_model.fit(X_train, y_train)

## validation 셋에서의 성능?
y_valid_hat=lasso_model.predict(X_valid)
# sum((y_valid_hat - y_valid)**2)
sum((y_valid_hat - y_valid["V1"])**2)


# for loop를 사용해서 최적 람다 찾기
alphas = np.linspace(0, 0.5, 1000)
valid_mse = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_valid_hat = model.predict(X_valid)
    valid_mse.append( sum((y_valid_hat - y_valid["V1"])**2) )

valid_mse

best_alpha = alphas[np.argmin(valid_mse)]
print("Best alpha (lambda):", best_alpha)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.scatter(alphas, valid_mse, color='blue')


# 최적 람다값을 통한 라쏘 회귀
lasso_model = Lasso(alpha=best_alpha)
lasso_model.fit(X_train, y_train)

lasso_model.intercept_
lasso_model.coef_

# Cross Validation
from sklearn.linear_model import LassoCV


alphas = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alphas,
                   cv=5,
                   max_iter=10000)
lasso_cv.fit(data_X, y)
lasso_cv.alpha_
lasso_cv.mse_path_.shape
lasso_cv.mse_path_