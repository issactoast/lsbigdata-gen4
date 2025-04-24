import numpy as np
import pandas as pd

url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)

mydata

mydata[mydata['gender'] == "F", :]      # 에러발생
mydata.loc[mydata['gender'] == "F", :]  # 작동
mydata[mydata['gender'] == "F"]         # 작동
check_f=np.array(mydata['gender'] == "F") # 작동
mydata.iloc[check_f, :]

mydata[mydata['midterm'] <= 15]

# 중간고사 점수 45 ~ 60점 사이 학생은 몇명인가요?
# 17명
check_score=(mydata['midterm'] >= 45) & (mydata['midterm'] <= 60)
mydata[check_score].shape[0]

# 라벨을 이용한 인덱싱 loc()
mydata.loc[mydata['midterm'] <= 15]
mydata.loc[mydata['midterm'] <= 15, "student_id"]
mydata.loc[mydata['midterm'] <= 15, ["student_id"]]
mydata.loc[mydata['midterm'] <= 15, ["student_id", "gender"]]

mydata['midterm'].iloc[0]
mydata['midterm'].isin([28, 38, 52])
# not mydata['midterm'].isin([28, 38, 52])
~mydata['midterm'].isin([28, 38, 52]) # not의 역할

mydata.loc[mydata['midterm'].isin([28, 38, 52])]
mydata.loc[~mydata['midterm'].isin([28, 38, 52])]

# 데이터에 빈칸이 뚫려있는 경우
mydata.iloc[3, 2] = np.nan
mydata.iloc[10, 3] = np.nan
mydata.iloc[13, 1] = np.nan

mydata["gender"].isna()

mydata.loc[mydata["gender"].isna()]
mydata.loc[~mydata["gender"].isna()]

mydata.dropna()

# Q. mydata에서 중간고사와 기말고사가
# 다 채워진 행들을 가져오세요.
# 중간고사 채워진 애들
cond1=~mydata["midterm"].isna()
# 기말고사 채워진 애들
cond2=~mydata["final"].isna()
mydata.loc[cond1 & cond2]

# 새로운 열 만들기
mydata["midterm"].isna()
mydata.loc[mydata["midterm"].isna(), "midterm"] = 50

mydata["midterm"].isna().sum() # NA 체크

mydata.loc[mydata["final"].isna(), "final"] = 30
mydata["final"].isna().sum() # NA 체크

mydata['total'] = mydata['midterm'] + mydata['final']
mydata

avg_col=(mydata['total'] / 2).rename('average')
mydata=pd.concat([mydata, avg_col], axis=1)

# 열 삭제
del mydata["gender"]

mydata.columns


# 데이터 프레임 합치기
import pandas as pd

df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})
df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5']
})
result = pd.concat([df1, df2])

df3 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
})
result = pd.concat([df1, df3], axis=1)

pd.concat([df1, df2], ignore_index=True)

df4 = pd.DataFrame({
    'A': ['A2', 'A3', 'A4'],
    'B': ['B2', 'B3', 'B4'],
    'C': ['C2', 'C3', 'C4']
})
df4

pd.concat([df1, df4], join='inner')
pd.concat([df1, df4], join='outer') # 기본설정

df_wkey=pd.concat([df1, df2], keys=['df1', 'df2'])
df_wkey.loc['df1']

mydata.head()
mydata.tail()
mydata.info()
mydata.describe()
mydata.sort_values("midterm")
mydata.mean()

# apply() 함수
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
df.apply(max, axis=0)
df.apply(max, axis=1)

def my_func(x, const=3):
    return max(x)**2 + const

df.apply(my_func, axis=0)
df.apply(my_func, axis=0, const=5)

# 팔머펭귄 데이터 불러오기
# pip install palmerpenguins
from palmerpenguins import load_penguins
penguins = load_penguins()

penguins.info()

# 각 펭귄 종별 특징 알아내서 발표
# 이제까지 배웠던 코드를 사용해서 분석 할 것