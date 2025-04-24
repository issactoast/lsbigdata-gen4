import numpy as np

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(42)
a = np.random.randint(1, 21, 10)
print(a)

a[1]
a[1:4]
a[1:4:1]
a[1:8:2]
a[::2]

a=np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
b=np.array([1, 2, 3])

a.shape
b.shape
a + b

import numpy as np
np.random.seed(20250221)
np.random.randint(1, 12, 1)
# np.random.choice(range(1,12), 1, replace=False)



import numpy as np
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec

str_vec[0]
str_vec[[0, 2, 1, 0]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked

import numpy as np

x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
choices = ["양수", "0", "음수"]
result = np.select(conditions, choices, default="기타") # 기본값을 문자열로 설정

print(result)

# 고객 데이터 만들기
np.random.seed(2025)
age = np.random.randint(20, 81, 3000)
gender=np.random.randint(0, 2, 3000)
gender=np.where(gender == 1, "여자", "남자")
price=np.random.normal(50000, 3000, 3000)

# 고객연령층: 2030, 4050, 6070 벡터를 만들어보세요!
age
conditions = [(age >= 20) & (age < 40), #2030
              (age >= 40) & (age < 60), #2030
              (age >= 60) & (age < 80)]
choices = ["20-30대", "40-50대", "60-70대"]
result = np.select(conditions, choices, default="80대이상!") # 기본값을 문자열로 설정
