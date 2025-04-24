import pandas as pd

# pip install nycflights13
from nycflights13 import flights, planes

flights.info()

# 비행기 출/도착 데이터분석

# 분석 주제 정할 것.

# * 인사이트 도출 2가지 이상
# * 해당하는 인사이트를 보여주는 근거 데이터 생성
# * 분석 시 날짜 데이터로 변환 후 요일 정보 같은 직접적으로 
#   뽑을 수 없는 정보도 생각해보세요!


# =======================================
import pandas as pd

data = {
'주소': ['서울특별시 강남구! 테헤란로 123', 
       '부산광역시 해운대@구 센텀중앙로? 45',
       '경기도 안성시 서운면 바우덕이로 248']
}
df = pd.DataFrame(data)
df

df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)')
df['도시'] = df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)', expand=False)
df


df['주소'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
df['주소_특수문자제거']=df['주소'].str.replace(r'[^a-zA-Z0-9가-힣\s]', '', regex=True)

# 테헤란로, 센텀중앙로, 바우덕이로 같은 도로명 칼럼 만들기!
data = {
'주소': ['서울특별시 강남구! 테헤란로 123', 
       '부산광역시 해운대@구 센텀중앙로? 45',
       '경기도 안성시 서운면 바우덕이로 248']
}
df = pd.DataFrame(data)
df

df['주소']=df['주소'].str.replace(r'[^a-zA-Z0-9가-힣\s]', '', regex=True)
df['도로명']=df['주소'].str.extract(r'([가-힣]+로)')

# 숫자만 꺼내오려면?
df['주소'].str.extract(r'([0-9]+)')