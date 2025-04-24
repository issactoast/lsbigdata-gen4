def add_two_numbers(num1, num2):
    result = num1 + num2
    return result

add_two_numbers(1, 2)

# Quiz: 하나의 숫자를 입력 받아서 제곱 한 후 + 10을
# 한 결과를 반환하는 함수, my_f를 만들어 주세요.
def my_f(num1):
    result = num1**2 + 10
    return result

# Quit 2: 하나의 숫자를 입력 받아서 제곱 한 후, 
# 내가 원하는 숫자 (입력변수 이름 my_add)를 더한
# 결과를 반환하는 함수, my_f를 만들어 주세요.
def my_f(num1, my_add):
    result = num1**2 + my_add
    return result

my_f(10, 3)

# 초기값 설정
def my_f(num1, my_add=10):
    result = num1**2 + my_add
    return result

my_f(5)
my_f(5, 10)
my_f(5, 3)

# 초기값을 두 변수에 모두 설정
# 첫번째 입력값 base: 3
# 두번째 입력값 base: 10
def my_f(num1=3, my_add=10):
    result = num1**2 + my_add
    return result

my_f()

# 출력값이 여러개인 함수
import numpy as np
a=np.arange(10)

def min_max_numbers(arr1):
    max_num = max(arr1)
    min_num = min(arr1)
    return max_num, min_num

min_max_numbers(a)

# 첫번째 값 꺼내오려면?
result=min_max_numbers(a)
result[0]
result[1]

# 시험 범위
# 진도 나간 PPT + 넘파이(행렬) + 점프투파이썬 p120까지
# 객관식 20문제
# 내일 오후 4시 (1시간 ~ 1시간 30분)

# 결과값 두개 함수 apply 적용하기
np.random.seed(2025)
array_2d = np.random.randint(1, 13, 12).reshape((3, 4))
array_2d

result=np.apply_along_axis(min_max_numbers, axis=1, arr=array_2d)
result.shape

# 성적데이터 만들기
np.random.seed(2025)
z=np.random.randint(1, 21, 20).reshape(4, 5)
z

# 1~5월 모의고사 성적
# 4명의 학생이 존재
# 각 학생의 점수 최고점, 최저점, 평균을 구해보세요!
# arr1=z[0]
def my_f(arr1):
    return max(arr1), min(arr1), np.mean(arr1)

np.apply_along_axis(my_f, axis=1, arr=z)


x = 5

if x > 4:
    y = 1
else:
    y = 2

y = 1 if x > 4 else 2

x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)


x = 0

if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"
print(result)

def num_checker(num):
    # 숫자 체크
    if num > 0:
        result = "양수"
    elif num == 0:
        result = "0"
    else:
        result = "음수"
    # 결과 반환
    return result

num_checker(3)
num_checker(-3)
num_checker(0)

# 숫자가 하나 들어왔을때,
# 홀수인지 짝수인 판단하는 함수를 만들어보세요!
num = 3
def even_odd(num):
    if num % 2 == 0:
        result = "짝수"
    else:
        result = "홀수"

    return result

even_odd(5)
# even_odd 함수의 입력값으로 문자열을 입력한 경우
# 어떻게 처리할 것인가? 생각해보세요.
# 각 조의 최종 코드를 확정해주세요.
even_odd("5")

# np.random.seed(114226)
# np.random.randint(1, 13) # 발표조
# np.random.randint(1, 4)  # 발표자

for i in range(1, 4):
    print(f"Here is {i}")


for i in [1, 3, 2, 5]:
    print(f"Here is {i}")

x=15
print(f"Here is {x}")

# 1에서 100까지 위 코드를 실행시키려면?
# 1에서 100까지 짝수만 찍어보려면?
# 1에서 100까지 7의 배수만 찍어보면?
for i in np.arange(7, 101, 7):
    print(f"Here is {i}")

result=np.repeat(0, 100)
for i in np.arange(100):
    result[i]=i**2
result

# 3의 배수를 쭉 채워넣어보세요!
result=np.repeat(0, 100)
for i in np.arange(100):
    result[i]=3*i + 3 # 3 * (i + 1)

result

# result의 4번째 칸마다 0을 입력하는 코드
for i in np.arange(100):
    if (i+1) % 4 == 0:
        result[i]=0

result

result[result % 4 == 0] = 0

for i in np.arange(3, 100, 4):
    result[i]=0


names = ["John", "Alice"]
ages = [25, 30]

greetings = [f"Hello, my name is {name} and I am {age} years old." 
                                    for name, age in zip(names, ages)]


# while loop
i = 0
while i <= 10:
    i += 3 # i = i + 3
    print(i)

# while 루프를 사용해서 1에서 100까지의 간격이 1인 수열을
# 찍어보세요!
i = 1
while i <= 100:
    print(i)
    i += 1 # i = i + 1

i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)


# 1에서 100 사이의 7의 배수를 찍어보는 루프 작성
# while & break 사용
i = 7
while True:
    print(i)
    i += 7
    if i > 100:
        break

i = 0
while i < 10:
    i += 1
    if i % 2 == 0:
        continue # 짝수일 때 다음 반복으로 이동
    print(i)

# 고객 데이터 만들기
np.random.seed(2025)
age = np.random.randint(20, 81, 3000)
gender=np.random.randint(0, 2, 3000)
gender=np.where(gender == 1, "여자", "남자")
price=np.random.normal(50000, 3000, 3000)

# 고객연령층: 2030, 4050, 6070 벡터를 만들어보세요!
age
conditions = [(age >= 20) & (age < 40), #2030
              (age >= 40) & (age < 60), #4060
              (age >= 60) & (age < 80)]
choices = ["20-30대", "40-50대", "60-70대"]
result = np.select(conditions, choices, default="80대이상!") # 기본값을 문자열로 설정

# price 변경 구매액 여자인 경우 + 1000, 남자인 경우 변경없음
# gender 벡터를 쭉 훑어보다가 여자인 경우 변경하고,
# 남자인 경우에는 건너뛰는 while & continue 루프
# gender.shape
i=0
while i < len(gender):
    i += 1
    if gender[i-1] == "남자":
        continue
    price[i-1] += 1000 # 여자
    
price



y = 2
def my_func(x):
    y = 1
    result = x + y
    return result

my_func(3)
print(y)


def outer_func(input):
    def inner_func(input):
        result = input + 2
        return result

    result = input + inner_func(input)
    return result

outer_func(5)



y = 2
def my_func():
    global y
    y += 1

print(y)
my_func()
print(y)

# global 키워드는 한단계 위 환경의 y를 불러내는가?
# 전체 최상위 환경의 y를 불러낼까?
del y

def outer_func():
    # global y
    # y = 10
    def inner_func():
        return y + 2
    result = inner_func()
    return result

outer_func()


