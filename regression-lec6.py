import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
from palmerpenguins import load_penguins

# 1. 데이터 불러오기
df = load_penguins()
df = df.dropna()

numeric_df = df.select_dtypes(include=['number'])
categorical_df = df.select_dtypes(include=['object', 'category'])
categorical_df = pd.get_dummies(categorical_df, drop_first=True).astype(int)
df= pd.concat([numeric_df, categorical_df], axis=1)

# 2. 종속변수와 독립변수 분리
y = df['bill_length_mm']
X = df.drop(columns=['bill_length_mm'])

# 3. 범주형 변수 더미코딩

# 4. 선형회귀 모델 정의
model = LinearRegression()

# Define custom feature selector
def aic_score(estimator, X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("Model AIC:", model.aic)
    return -model.aic

# 5. SFS 실행
sfs = SFS(model,
          k_features='best',  # 또는 (1, X.shape[1])로 범위 지정 가능
          forward=True,
          scoring=aic_score,
          cv=0)

sfs = sfs.fit(X, y)

# 6. 결과 출력
print("선택된 피처 인덱스:", sfs.k_feature_idx_)
print("선택된 피처 이름:", sfs.k_feature_names_)
print("AIC score:", sfs.k_score_)

# 7. 중간 결과 보기
pd.DataFrame.from_dict(sfs.get_metric_dict()).T


