import numpy as np

# 설정: 표본 크기
n = 1000

# 평균 벡터
mean = [3, 4]

# 공분산 행렬
cov = [[2**2, 3.6],
       [3.6, 3**2]]

# 다변량 정규분포로부터 난수 추출
np.random.seed(42)
data = np.random.multivariate_normal(mean,
                                     cov,
                                     size=n)

# 각 변수 추출
x1 = data[:, 0]
x2 = data[:, 1]

import matplotlib.pyplot as plt

# 히스토그램 그리기
plt.hist(x1, bins=30, edgecolor='black')
plt.hist(x2, bins=30, edgecolor='black')

# 문제
# 각 X1, X2의 표본 평균과 표본 분산
# 표본 공분산, 표본 상관계수
x1_bar=x1.mean()
x2_bar=x2.mean()
x1_s2=x1.var(ddof=1)
x2_s2=x2.var(ddof=1)
x12_cov=sum((x1-x1_bar) * (x2-x2_bar)) / (n-1)
x12_cov
3.6/6

upper_r=sum((x1-x1_bar) * (x2-x2_bar)) 
lower_r=np.sqrt(sum((x1-x1_bar)**2)) * np.sqrt(sum((x2-x2_bar)**2))
r=upper_r/lower_r
r

# 표본 공부산 함수 / 상관계수 함수
np.cov(x1, x2, ddof=1)[0, 1]
np.corrcoef(x1, x2)[0, 1]

# 산점도
plt.scatter(x1, x2, alpha=0.3)

# 상관계수 검정
import scipy.stats as stats

corr_coef, p_value=stats.pearsonr(x1, x2)
corr_coef
p_value


# iris 데이터
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. Iris 데이터 로드
df_iris = load_iris()

# 2. pandas DataFrame으로 변환
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] #컬럼명 변경시

# 3. 타겟(클래스) 추가
iris["species"] = df_iris.target

# 4. 클래스 라벨을 실제 이름으로 변환 (0: setosa, 1: versicolor, 2: virginica)
iris["species"] = iris["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
iris

iris['species'].unique()

# 회귀분석 (단순회귀)
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols("Petal_Length ~ Petal_Width",
                data = iris).fit()
print(model.summary())

x=iris["Petal_Width"]
y=iris["Petal_Length"]

(y - 2*x + 3)

data_X=np.column_stack([np.repeat(1, 150), x])
data_X

# 정규방정식으로 회귀계수 추정
XtX = data_X.T @ data_X         # X^T X
XtX_inv = np.linalg.inv(XtX)    # (X^T X)^-1
Xty = data_X.T @ y              # X^T y

beta_hat = XtX_inv @ Xty        # 회귀계수 벡터
beta_hat

b=model.params["Intercept"]
a=model.params["Petal_Width"]

x_line = np.linspace(min(x), max(x), 100)
y_line = a * x_line + b
plt.scatter(x, y)
plt.plot(x_line, y_line, color='red')
plt.axhline(y.mean())

2.2299*0.7+1.0836


# 다중회귀분석
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Width + Sepal_Length",
                data = iris).fit()
print(model.summary())

model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length",
                data = iris).fit()
print(model.summary())


x1=iris["Petal_Width"]
x2=iris["Sepal_Width"]
x3=iris["Sepal_Length"]

data_X=np.column_stack([np.repeat(1, 150), x1, x2, x3])
data_X

# 정규방정식으로 회귀계수 추정
XtX = data_X.T @ data_X         # X^T X
XtX_inv = np.linalg.inv(XtX)    # (X^T X)^-1
Xty = data_X.T @ y              # X^T y

beta_hat = XtX_inv @ Xty        # 회귀계수 벡터
beta_hat


x=iris["Sepal_Width"]
y=iris["Petal_Length"]

beta0=model.params["Intercept"]
beta1=model.params["Petal_Width"]
beta2=model.params["Sepal_Width"]

x_line = np.linspace(min(x), max(x), 100)
y_line = beta2 * x_line + beta0
plt.scatter(x, y)
plt.plot(x_line, y_line, color='red')

# y_hat=2.2582+2.155x1+(-0.355)*x2

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 3D 플롯 준비
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 산점도 그리기
x = iris["Petal_Width"]
y = iris["Sepal_Width"]
z = iris["Petal_Length"]
ax.scatter(x, y, z, color='blue', 
           alpha=0.6, label='Observed data')


# 회귀 평면 생성
x_surf, y_surf = np.meshgrid(
    np.linspace(x.min(), x.max(), 30),
    np.linspace(y.min(), y.max(), 30)
)
z_surf = beta0 + beta1 * x_surf + beta2 * y_surf

# 회귀 평면 그리기
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.4, label='Regression plane')

# 축 라벨
ax.set_xlabel('Petal Width')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title('Multiple Linear Regression: 3D Visualization')

plt.legend()
plt.tight_layout()
plt.show()