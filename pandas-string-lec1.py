import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

data = {
    'date': ['2024-01-01 12:34:56', '2024-02-01 23:45:01', '2024-03-01 06:07:08'],
    'value': [100, 201, 302]
}
df = pd.DataFrame(data)
df.info()

# 데이터 형식 변환: 날짜 전처리하기 쉽게!
df['date'] = pd.to_datetime(df['date'])
df.info()

pd.to_datetime('03-11-2024')
pd.to_datetime('2024-03-11')
pd.to_datetime('2024/03/11')
pd.to_datetime('03/11/2024')
# pd.to_datetime('11/2024/03') # 입력형식이 맞지 않음
pd.to_datetime('11-2024-03', format='%d-%Y-%m')
pd.to_datetime('11-24-03', format='%d-%y-%m')

dt_obj=pd.to_datetime('2025년 03월 11일',
                      format='%Y년 %m월 %d일')
dt_obj.year
dt_obj.month
dt_obj.day
dt_obj.hour
dt_obj.minute
dt_obj.second
dt_obj.weekday()
dt_obj.day_name()

df['year'] =df['date'].dt.year
df['month']=df['date'].dt.month
df['day']  =df['date'].dt.day
df['hour']=df['date'].dt.hour
df['minute']=df['date'].dt.minute
df['second']=df['date'].dt.second

df

# 날짜 계산
current_date = pd.to_datetime('2025-03-11')
current_date - df['date']

# 날짜 벡터 만들기
pd.date_range(start='2021-01-01',
              end='2022-01-10', freq='D')

pd.date_range(start='2016-01-01',
              end='2022-01-10', freq='ME')

# 날짜 합치기
df.year
df.month
df.day
pd.to_datetime(dict(year=df.year, 
                    month=df.month, 
                    day=df.day))

# 문자열 다루기
import pandas as pd

data = {
    '가전제품': ['냉장고', '세탁기', '전자레인지', '에어컨', '청소기'],
    '브랜드': ['LG LG', 'Samsung', 'Panasonic', 'Daikin', 'Dyson']
}
df = pd.DataFrame(data)

df['가전제품'].str.len()
df['브랜드'].str.lower()
df['브랜드'].str.upper()
df['브랜드'].str.title()
df['브랜드'].str.contains("a")

df.columns.str.contains("가")
df['브랜드변형']=df['브랜드'].str.replace("a", "aaaaB")
df['브랜드'].str.replace("a", "")
df['브랜드변형'].str.replace("a", "")

df[['브랜드_첫부분', '브랜드_두번째', '브랜드_세번째']]=df['브랜드'].str.split('a', expand=True)
df


df['가전제품']=df['가전제품'].str.replace('전자레인지', ' 전자 레인지 ')
df['가전제품'].str.strip()

df['가전제품'].iloc[2]
df['가전제품'].str.strip().iloc[2]