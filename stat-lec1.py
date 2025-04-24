import numpy as np

rng=np.random.default_rng(seed=2025)
rng.uniform(0, 1)

# 함수: Bernulli()
# 입력값은 없고, 0, 1 둘 중 하나를 가짐
# rng.uniform(0, 1) 값 0.5 작거나 같으면
# 1, 그렇지 않으면 0

def Bernulli():
    rng=np.random.default_rng()
    u=rng.uniform(0, 1)
    if u <= 0.5:
        return 1
    else:
        return 0
    
Bernulli()


# 확률변수 X는 0, 1, 2
# 대응하는 확률이 2/10, 5/10, 3/10이다.
# 위 코드를 응용해서 확률변수 X를 만들어보세요!
def X():
    rng=np.random.default_rng()
    u=rng.uniform(0, 1)
    if u <= 0.2:
        return 0
    elif u <= 0.7:
        return 1
    else:
        return 2

X()

# X(size=10) 입력값을 만들어서
# 결과값 넘파이 벡터로 나올수있도록 개조
rng.uniform(0, 1, 10)
s=10
def X(s=1):
    rng=np.random.default_rng()
    u=rng.uniform(0, 1, s)
    conditions = [u <= 0.2, u <= 0.7, u <= 1]
    choices = [0, 1, 2]
    result = np.select(conditions, choices, default=np.nan)
    return result

X(10)
X()

# 확률변수 X의 확률 질량함수 시각화
import matplotlib.pyplot as plt

x=np.array([0, 1, 2])
p_x=np.array([0.2, 0.5, 0.3])
plt.bar(x, p_x) # 막대 그래프
plt.title('R.V. X pmf')
plt.show()

# 200명을 뽑자!
X(s=200).mean()

# Y 기대값
y=np.array([0, 1, 2])
p_y=np.array([0.2, 0.4, 0.4])
sum(y * p_y)
# Y 분산
sum((y-1.2)**2 * p_y)
np.sqrt(sum((y-1.2)**2 * p_y)) # 표준편차


# 문제: 확률변수 X의 확률 분포표는 다음과 같습니다.
# X=x    1, 2, 3, 4
# P(X=x) 0.1, 0.3, 0.2, 0.4

# 1) 평균을 구하세요.
# 2) 분산을 구하세요.
# 3) X에서 평균보다 큰 값이 나올 확률은 얼마인가요?
# 4) X의 확률질량 함수를 Bar 그래프로 그려보세요.
# 5) 5개의 표본을 무작위로 추출한 값의 산술평균을 계산해보세요. (표본평균)
# 6) 4번에서 그린 그래프에, 확률변수의 평균값을 빨간 세로선
# 으로 표시하고, 5번에서 계산한 표본평균을 파란 세로선으로 표시
# 하세요.(파란선은 코드를 돌릴때마다 위치가 바뀌어야 함.)
# *) 생각해볼거리 - 
# 5개의 표본으로 계산한 표본평균이 코드를 돌릴때 움직이는 정도 vs.
# 20개의 표본으로 계산한 표본평균이 코드를 돌릴때 움직이는 정도


# 숙제 1: 확률변수 X의 확률 분포표는 다음과 같습니다.
# X=x    1에서 20까지 정수
# P(X=x) 각 1/20

# 1) 평균을 구하세요.
# 2) 분산을 구하세요.
# 3) X에서 평균보다 큰 값이 나올 확률은 얼마인가요?
# 4) X의 확률질량 함수를 Bar 그래프로 그려보세요.
# 5) 5개의 표본을 무작위로 추출한 값의 산술평균을 계산해보세요. (표본평균)
# 6) 4번에서 그린 그래프에, 확률변수의 평균값을 빨간 세로선
# 으로 표시하고, 5번에서 계산한 표본평균을 파란 세로선으로 표시
# 하세요.(파란선은 코드를 돌릴때마다 위치가 바뀌어야 함.)
# 7) 5개의 표본으로 계산한 표본평균 300개 발생 -> 히스토그램
# 8) 20개의 표본으로 계산한 표본평균 300개 발생 -> 히스토그램

# 문제 2: 
# X1: 0과 1을 가지는 확률변수 (1이 나올 확률 0.3)
# X2: 0과 1을 가지는 확률변수 (1이 나올 확률 0.3)
# 확률변수 Y=X1+X2 일때, Y의 확률분포표를 작성하세요. 단, X1, X2는 서로
# 영향이 없음
import numpy as np
def X(s=1):
    rng=np.random.default_rng()
    u=rng.uniform(0, 1, s)
    conditions = [u <= 0.3, u <= 1]
    choices = [1, 0]
    result = np.select(conditions, choices, default=np.nan)
    return result

Y = X(1) + X(1) + X(1)
Y

