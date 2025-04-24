import numpy as np

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생
b = np.array(["apple", "banana", "orange"])
c = np.array([True, False, True, True]) #
print("Numeric Vector:", a)

x = np.empty(3)
x[0] = 3
x[1] = 3
x[2] = 3


np.arange(1, 3, 0.5)
arr1 = np.arange(10)
arr2 = np.arange(10, step=0.5)
arr2

np.linspace(1, 3, 10)

linear_space1 = np.linspace(0, 1, 5)
print("0부터 1까지 5개 원소:", linear_space1)


arr1 = np.arange(10)
np.repeat(arr1, 2)


np.repeat([8, 1, 2], repeats=[1, 2, 3])
arr3=np.tile([8, 1, 2], 3)
len(arr3)
arr3.shape
arr3.size

# 2차원 배열
len(np.array([1, 2, 3, 4]))
b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b
a - b
a * b
a / b
a ** b

2 * b


# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])

matrix.shape
vector.shape

matrix + vector


matrix = np.array([[ 0.0, 0.0, 0.0],
[10.0, 10.0, 10.0],
[20.0, 20.0, 20.0],
[30.0, 30.0, 30.0]])
# 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector = vector.reshape(4, 1)
vector.shape
print(vector.shape, matrix.shape)

matrix + vector


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)
dot_product


# 길이가 다른 두 벡터
a = np.array([1, 2, 3, 4])
b = np.array([1, 2])


# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2025)
a = np.random.randint(1, 101, 20)
a

a[a > 50]

# Q1
# 시드는 2025 고정
# 고객 데이터를 만들기 나이 20~80
# 3000명 발생
# 이 중 40대 이상 고객은 몇명?
np.random.seed(2025)
customer_age = np.random.randint(20, 81, 3000)
customer_age[customer_age >= 40].size

# 40대 고객은 몇명?
customer_age[(customer_age >= 40) & (customer_age < 50)].size
a = np.array([5, 3, 1, 10, 24, 3])
a

a[(a > 5) & (a < 15)]

a[a!=10]

a = np.array([True, True, False])
b = np.array([False, True, False])
a & b

# 1에서 300 사이 숫자들 중에서
# 7의 배수의 합을 구하세요!
vec1 = np.arange(1, 301)
sum(vec1[vec1 % 7 == 0])

# 시드 2025
# 각 숫자가 나올 가능성이 같은가요?
# np.random.seed(2025)
vec2=np.random.randint(1, 4, 3000)

vec2[vec2 == 1].size, vec2[vec2 == 2].size, vec2[vec2 == 3].size

vec3=np.random.rand(10)
vec3[vec3 > 0.5].size / 10 # 0: 뒷면, 1: 앞면

# 고객 데이터 만들기
np.random.seed(2025)
customer_age = np.random.randint(20, 81, 3000)

# 3000명에 대하여 성별 벡터를 만들어보세요!
# 0: 남자, 1: 여자
# 50%, 50% 비율
# gender 벡터를 만들것
gender=np.random.randint(0, 2, 3000)
gender

# 0: 남자, 1: 여자로 바꾸기
gender=np.where(gender == 1, "여자", "남자")

# 총 남자 여자는 몇명 있나요?
gender[gender=="남자"].size
gender[gender=="여자"].size

customer_age[0]
gender[0]

# 나이 벡터에서 여자에 해당하는 나이들은 어떻게
# 걸러낼까?
sum(customer_age[gender == "여자"])/1503
sum(customer_age[gender == "남자"])/1497

# a = np.array([5, 3, 1, 10, 24, 3])
# a[a > 3] = 10
# a
# np.where(a == 10, "남자", "여자")

# 고객 데이터 만들기
np.random.seed(2025)
age = np.random.randint(20, 81, 3000)
gender=np.random.randint(0, 2, 3000)
gender=np.where(gender == 1, "여자", "남자")
price=np.random.normal(50000, 3000, 3000)

# Q1. 각 연령대별 평균 나이 계산해주세요!
# 80은 70대로 설정
# 2030, 4050, 6070 그룹으로 설정
age[(age >= 20) & (age < 40)].mean()
age[(age >= 40) & (age < 60)].mean()
age[(age >= 60) & (age < 81)].mean()

# Q2. 성별, 연령대별 구매액을 구하고
# 평균 구매액이 가장 큰 그룹은?
# 구매액이 가장 큰 그룹은?
mean_price = np.zeros(6)
mean_price[0]=price[(age >= 20) & (age < 40) & (gender == "남자")].mean()
mean_price[1]=price[(age >= 20) & (age < 40) & (gender == "여자")].mean()
mean_price[2]=price[(age >= 40) & (age < 60) & (gender == "남자")].mean()
mean_price[3]=price[(age >= 40) & (age < 60) & (gender == "여자")].mean()
mean_price[4]=price[(age >= 60) & (age < 81) & (gender == "남자")].mean()
mean_price[5]=price[(age >= 60) & (age < 81) & (gender == "여자")].mean()

np.where(mean_price == mean_price.max())

total_price = np.zeros(6)
total_price[0]=price[(age >= 20) & (age < 40) & (gender == "남자")].sum()
total_price[1]=price[(age >= 20) & (age < 40) & (gender == "여자")].sum()
total_price[2]=price[(age >= 40) & (age < 60) & (gender == "남자")].sum()
total_price[3]=price[(age >= 40) & (age < 60) & (gender == "여자")].sum()
total_price[4]=price[(age >= 60) & (age < 81) & (gender == "남자")].sum()
total_price[5]=price[(age >= 60) & (age < 81) & (gender == "여자")].sum()

total_price.max()
np.where(total_price == total_price.max())


a=np.array([1, 2, 3], dtype=np.int64)
a
a=np.array([1.5, 2, 3], dtype=np.float64)
a

a=np.array([1.5, 0, 0.5], dtype=np.bool_)
a

a=np.array([1.5, 0, 0.5], dtype=np.str_)
a

gender=np.random.randint(0, 2, 3000)
gender=np.array(gender, dtype=np.str_)
gender[gender == '0'] = "남자"
gender[gender == '1'] = "여자"
gender

np.nan + 1
None + 1

a = np.array([20, np.nan, 13, 24, 309, np.nan])
a[np.isnan(a)] = np.nanmean(a)
a

~np.isnan(a)
a[~np.isnan(a)]

a_filtered = a[~np.isnan(a)]
a_filtered




