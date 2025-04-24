import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/houseprice/train.csv')
test_df = pd.read_csv('./data/houseprice/test.csv')

# pd.set_option('display.max_columns', None)
# train_df.info()

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df.drop(columns='SalePrice')
y_train = np.log(train_df['SalePrice'] + 1)
# y_train = train_df['SalePrice']


# 총 면적 변수: 지하층 + 1층 + 2층
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']

# 총 욕실 수 변수 (지하 포함, 반 욕실은 0.5로 계산)
train_df['TotalBath'] = (train_df['FullBath'] + 0.5 * train_df['HalfBath'] +
                         train_df['BsmtFullBath'] + 0.5 * train_df['BsmtHalfBath'])
test_df['TotalBath'] = (test_df['FullBath'] + 0.5 * test_df['HalfBath'] +
                        test_df['BsmtFullBath'] + 0.5 * test_df['BsmtHalfBath'])

# 수영장 유무 (0 또는 1)
train_df['HasPool'] = (train_df['PoolArea'] > 0).astype(int)
test_df['HasPool'] = (test_df['PoolArea'] > 0).astype(int)

# 주택 연식 (판매 시점 기준)
train_df['HouseAge'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']


import matplotlib.pyplot as plt

# SalePrice 히스토그램 그리기
plt.figure(figsize=(8, 5))
plt.hist(y_train, bins=50, edgecolor='black')
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 몇번째 행 (이상치)을 제거해야 할까?
plt.figure(figsize=(8, 5))
plt.scatter(train_df['LotArea'], train_df['SalePrice'], alpha=0.5, edgecolor='k')
plt.title('LotArea vs SalePrice')
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.grid(True)
plt.tight_layout()
plt.show()

# X_train = X_train.drop([523, 1298, 332, 440, 496, 581, 1061, 1190])
# y_train = y_train.drop([523, 1298, 332, 440, 496, 581, 1061, 1190])

X_train = X_train.drop([523, 1298, 332])
y_train = y_train.drop([523, 1298, 332])

X_test = test_df

X_train = X_train.drop(columns=['Id'])
X_test = X_test.drop(columns=['Id'])

# 칼럼 선택
num_columns = X_train.select_dtypes(include=['number']).columns
cat_columns = X_train.select_dtypes(include=['object']).columns

from sklearn.impute import SimpleImputer

# freq_impute = SimpleImputer(strategy='most_frequent')
# mean_impute = SimpleImputer(strategy='mean')
freq_impute = SimpleImputer(strategy='constant', fill_value='unknown')
mean_impute = SimpleImputer(strategy='constant', fill_value=-1)

X_train[cat_columns] = freq_impute.fit_transform(X_train[cat_columns])
X_test[cat_columns] = freq_impute.transform(X_test[cat_columns])

X_train[num_columns] = mean_impute.fit_transform(X_train[num_columns])
X_test[num_columns] = mean_impute.transform(X_test[num_columns])

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False)

X_train_cat = onehot.fit_transform(X_train[cat_columns])
X_test_cat = onehot.transform(X_test[cat_columns])

std_scaler = StandardScaler()
X_train_num = std_scaler.fit_transform(X_train[num_columns])
X_test_num = std_scaler.transform(X_test[num_columns])

X_train_all = np.concatenate([X_train_num, X_train_cat], axis = 1)
X_test_all = np.concatenate([X_test_num, X_test_cat], axis = 1)

from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

# 파라미터 확인 
ElasticNet().get_params()
DecisionTreeRegressor().get_params()

elasticnet = ElasticNet(max_iter=1000)
elastic_params = {'alpha' : np.arange(0, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}

dct_reg = DecisionTreeRegressor(min_samples_leaf=5)
dct_params = {'max_depth' : np.array([2, 4, 6, 8]),
              'ccp_alpha': np.arange(0.01, 0.3, 0.05)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 그리드서치
elastic_search = GridSearchCV(estimator=elasticnet, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

dct_search = GridSearchCV(estimator=dct_reg, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

elastic_search.fit(X_train_all, y_train)
dct_search.fit(X_train_all, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(elastic_search.cv_results_))
print(pd.DataFrame(dct_search.cv_results_))

# best prameter
print(elastic_search.best_params_)
print(dct_search.best_params_)

# 교차검증 best score 
print(-elastic_search.best_score_)
print(-dct_search.best_score_)

# 최종 예측 
y_pred1 = elastic_search.predict(X_test_all)
y_pred2 = dct_search.predict(X_test_all)

submit = pd.read_csv('./data/houseprice/sample_submission.csv')
submit["SalePrice"]=np.exp((y_pred1+y_pred2)/2) - 1
# submit["SalePrice"]=y_pred

# CSV로 저장
submit.to_csv('./data/houseprice/baseline_elnlog_outlier_varadd.csv', index=False)

