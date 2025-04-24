import pandas as pd
import numpy as np

train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_train.csv') 
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_test.csv')

train_X = train.drop(['grade'], axis = 1)
train_y = train['grade']

test_X = test.drop(['grade'], axis = 1)
test_y = test['grade']

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer, make_column_transformer 
from sklearn.pipeline import Pipeline, make_pipeline

num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(
     SimpleImputer(strategy="mean"),
     StandardScaler() 
     )

cat_preprocess = make_pipeline( 
    # SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False) 
    )
 
 
preprocess = ColumnTransformer( 
    [("num", num_preprocess, num_columns),
     ("cat", cat_preprocess, cat_columns)] 
     )

from sklearn.tree import DecisionTreeRegressor

full_pipe = Pipeline( 
       [ ("preprocess", preprocess), 
         ("regressor", DecisionTreeRegressor()) ]
        )


DecisionTreeRegressor().get_params()

decisiontree_param = {'regressor__ccp_alpha': np.arange(0.01, 0.3, 0.05)}

from sklearn.model_selection import GridSearchCV 

decisiontree_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = decisiontree_param,
    cv = 5,
    scoring = 'neg_mean_squared_error'
    )

decisiontree_search.fit(train_X, train_y)

decisiontree_search.best_params_

y_pred=decisiontree_search.predict(test_X)
y_pred
test_y