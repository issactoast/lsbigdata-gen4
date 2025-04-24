# conda install pandas
import pandas as pd

# 데이터 프레임 생성
df = pd.DataFrame({
    'col-str': ['one', 'two', 'three', 'four', 'five'],
    'col-num': [6, 7, 8, 9, 10]
})
df
print(df)

df['col-str']
df['col-num']
df

print(df['col-str'].dtype)
print(df['col-num'].dtype)

# df['col1'].dtype

# 판다스 시리즈
data = [10, 20, 30]
df_s = pd.Series(data, name = "aa")
print(df_s)

df.columns
df_s.name


# 데이터 프레임 생성
my_df = pd.DataFrame({
    'student_id': [1, 2, 3],
    'gender': ['F', 'M', 'F'],
    'midterm': [38, 42, 53]
}, index=["first", "second", "third"])
print(my_df)

my_df['gender']

my_s = pd.Series(['F', 'M', 'F'], 
                 name = "gender",
                 index=["first", "second", "third"])
my_s
pd.DataFrame(my_s)

# 외부 데이터 가져오기
import pandas as pd

url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)

mydata.head()

mydata['gender'].head()
mydata[['gender', 'student_id']].head()

# 대괄호 안에 숫자를 써서 인덱스 바로는 불가
# .iloc 함수를 사용하면 가능
mydata[1:4, 0]     # 에러 발생
mydata.iloc[0, 0]
mydata.iloc[1:4, 1:3]
mydata.head()

# .iloc 함수는 인덱스가 문자여도 잘 작동
mydata2=mydata.iloc[0:4, 0:3]
mydata2.index
mydata2.index = ["first", "second", "third", "fourth"]
mydata2
mydata2.iloc[0:2, 0]

# .iloc 함수는 :도 잘 작동함!
mydata2.iloc[:, 0]

# .iloc 함수는 결과값의 타입이 변동함
mydata2.iloc[0, 1] # 결과: 값 하나
mydata2.iloc[0:2, 0] # 결과: 시리즈
mydata2.iloc[2, 0:2] # 결과: 시리즈
mydata2.iloc[0:3, 0:2] # 결과: 데이터프레임
result1=mydata2.iloc[:, 0] # 결과: 시리즈
result2=mydata2.iloc[:, [0]] # 결과: 데이터프레임
# result1.iloc[1]
# result2.iloc[1, 0]
mydata2.iloc[:, [0]].squeeze() # 결과: 시리즈

# .iloc 함수는 연속되지 않은 열들, 행들을
# 리스트 형태로 받아서 인덱스
mydata2.iloc[:, [0, 2]]
mydata2.iloc[:, [0, 2, 1]]    # 열 순서 바뀜
mydata2.iloc[:, [0, 2, 1, 0]] # 중복 선택가능

mydata.shape

# 짝수번째 행들만 선택하려면?
import numpy as np

a = np.array([1, 2, 3])
mydata.iloc[a, :]

n=mydata.shape[0]
my_index=np.arange(1, n, 2)
mydata.iloc[my_index, :]

# 필터링이 될까?
check_f=np.array(mydata['gender']) == "F"
mydata.iloc[check_f, :]

# 시리즈 바로 넣을 경우 loc 함수 사용
mydata.loc[mydata['gender'] == "F", :]
