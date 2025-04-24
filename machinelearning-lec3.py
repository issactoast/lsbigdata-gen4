import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 데이터 불러오기
train_df = pd.read_csv('./data/houseprice/train.csv')
test_df = pd.read_csv('./data/houseprice/test.csv')

# 파생 변수 생성
# train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
# test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']

# train_df['TotalBath'] = (train_df['FullBath'] + 0.5 * train_df['HalfBath'] +
#                          train_df['BsmtFullBath'] + 0.5 * train_df['BsmtHalfBath'])
# test_df['TotalBath'] = (test_df['FullBath'] + 0.5 * test_df['HalfBath'] +
#                         test_df['BsmtFullBath'] + 0.5 * test_df['BsmtHalfBath'])

# train_df['HasPool'] = (train_df['PoolArea'] > 0).astype(int)
# test_df['HasPool'] = (test_df['PoolArea'] > 0).astype(int)

# train_df['HouseAge'] = train_df['YrSold'] - train_df['YearBuilt']
# test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']

# 타겟 로그변환
y_train = np.log(train_df['SalePrice'] + 1)
X_train = train_df.drop(columns='SalePrice')
X_test = test_df

# 이상치 제거
# X_train = X_train.drop([523, 1298, 332])
# y_train = y_train.drop([523, 1298, 332])

# ID 제거
X_train = X_train.drop(columns=['Id'])
X_test = X_test.drop(columns=['Id'])

# 숫자형/범주형 변수 분리
num_columns = X_train.select_dtypes(include=['number']).columns
cat_columns = X_train.select_dtypes(include=['object']).columns

# 결측값 처리
freq_impute = SimpleImputer(strategy='constant', fill_value='unknown')
mean_impute = SimpleImputer(strategy='constant', fill_value=-1)

X_train[cat_columns] = freq_impute.fit_transform(X_train[cat_columns])
X_test[cat_columns] = freq_impute.transform(X_test[cat_columns])

X_train[num_columns] = mean_impute.fit_transform(X_train[num_columns])
X_test[num_columns] = mean_impute.transform(X_test[num_columns])

# 인코딩 및 스케일링
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = onehot.fit_transform(X_train[cat_columns])
X_test_cat = onehot.transform(X_test[cat_columns])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_columns])
X_test_num = scaler.transform(X_test[num_columns])

# 최종 입력 데이터 결합
X_train_all = np.concatenate([X_train_num, X_train_cat], axis=1)
X_test_all = np.concatenate([X_test_num, X_test_cat], axis=1)

# 모델 정의
elasticnet = ElasticNet(max_iter=10000, l1_ratio=0)
tree = DecisionTreeRegressor()

# 파라미터 그리드
elastic_params = {'alpha': np.arange(0, 1, 0.1)}
tree_params = {'max_depth': [5, 10, 20, None], 
               'min_samples_split': [2, 5, 10]}

# 교차검증 설정
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 그리드서치
elastic_search = GridSearchCV(estimator=elasticnet, param_grid=elastic_params, cv=cv, scoring='neg_mean_squared_error')
tree_search = GridSearchCV(estimator=tree, param_grid=tree_params, cv=cv, scoring='neg_mean_squared_error')

# 학습
elastic_search.fit(X_train_all, y_train)
tree_search.fit(X_train_all, y_train)

# # 예측
# y_pred1 = elastic_search.predict(X_test_all)
# y_pred2 = tree_search.predict(X_test_all)

# # # 결과 저장
# submit = pd.read_csv('./data/houseprice/sample_submission.csv')
# submit['SalePrice'] = np.expm1((y_pred1 + y_pred2) / 2)
# submit.to_csv('./data/houseprice/baseline_eln_tree.csv', index=False)


# 기본 모델 예측 결과 생성 (train, test)
y_pred1_test = elastic_search.predict(X_test_all)
y_pred2_test = tree_search.predict(X_test_all)

y_pred1_train = elastic_search.predict(X_train_all)
y_pred2_train = tree_search.predict(X_train_all)

# 메타모델 학습을 위한 새로운 train, test 데이터 생성
meta_X_train = np.vstack([y_pred1_train, y_pred2_train]).T
meta_X_test = np.vstack([y_pred1_test, y_pred2_test]).T

# 메타 회귀모델 정의 및 학습
meta_model = LinearRegression()
meta_model.fit(meta_X_train, y_train)

# 최종 예측
meta_pred = meta_model.predict(meta_X_test)

# 예측 복원 및 저장
submit = pd.read_csv('./data/houseprice/sample_submission.csv')
submit['SalePrice'] = np.expm1(meta_pred)
submit.to_csv('./data/houseprice/stacked_eln_tree.csv', index=False)

