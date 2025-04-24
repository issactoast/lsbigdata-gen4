import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),
                          np.arange(12, 16)))
print("행렬:\n", matrix)
matrix.shape

# 0으로 채워진 길이가 10인 벡터는?
# np.empty(10)
# np.repeat(0, 10)
np.zeros(10)
np.zeros((2, 2))
y = np.arange(1, 5).reshape(2, 2)

# 1에서 20까지 채워진 4행 5열 행렬을 만들려면?
np.arange(1, 21).reshape(4, 5)
np.arange(1, 21).reshape(6, 3)  # 부족해도 에러
np.arange(1, 21).reshape(5, 5)  # 남아도 에러

# np.arange(1, 21) 20개 숫자 벡터지만, 5 by 5 행렬
# 우겨넣고 싶은 경우, 앞 숫자들 재활용을 통해
# 채워넣으려면?
vec1 = np.arange(1, 21)
vec1 = np.resize(vec1, 25).reshape(5, 5)


np.arange(1, 21).reshape(4, 5, order="C")
np.arange(1, 21).reshape(4, 5, order="F")

x=np.arange(1, 11).reshape(5, 2) * 2
x[1, 1]
x[3, 1]
x[2, 0]
x[:, 0]
x[2, :]
x[2]

y=np.arange(1, 21).reshape(4, 5)
y[2:4, 3]
y[2, 1:4]
y[1:4, 2:4]
y[1:3, [1, 3, 4]]

y[0:2, 1]
y[2:4, 3:5]
np.column_stack((y[0:2, 1], y[2:4, 3:5]))

# 1에서 20까지 숫자 중 랜덤하게 20개 숫자를 발생후
# 4행 5열 행렬 만드는 코드
np.random.seed(2025)
z=np.random.randint(1, 21, 20).reshape(4, 5)
z
z[:, 0] > 7
z[z[:, 0] > 7, :]

z[2, :] > 10
z[:, z[2, :] > 10]

# 열별 합계를 구하고 싶어요!
z.sum(axis=0)
z.sum(axis=1)

# 평균 점수 1등 학생의 점수 벡터는?
# 행: 학생
# 열: 1~5월 모의고사 점수
mean_vec=z.mean(axis=1)
mean_vec.max()
mean_vec == mean_vec.max()
z[mean_vec == mean_vec.max(),:]

# 모의고사 평균 점수가 10점 이상인
# 학생들 데이터 필터링
z[z.mean(axis=1) > 10, :]

# 모의고사 평균 점수가 10점 이상인 학생들
# 3월 이후 모의고사 점수 데이터 필터링
z[z.mean(axis=1) > 10, 2:]

z


a=np.arange(10)
a[2:6]

b=np.array([0, 2, 3, 6]) 
b > [4, 2, 1, 6]

# 1~5월 모의고사 점수
# 기존 1월~4월 모의고사 점수 평균보다 
# 5월 모의고사를 잘 본 학생 데이터 필터링
np.random.seed(20250224)
np.random.randint(1, 13)


# 1~5월 모의고사 점수
# 기존 1월~4월 모의고사 점수 평균보다 
# 5월 모의고사 점수를 비교했을때,
# 가장 점수가 많이 향상된 학생,
# 가장 점수가 떨어진 학생의 
# 평균 점수, 5월 모의고사 점수를 구하시오.
# 결과 예: 학생 1번, 평균 점수: 00점, 
#          5월 모의고사 00점, 차이 00점
# np.argmax()
# np.argmin()

# np.random.seed(1358)
# np.random.randint(1, 4)  # 발표자
# np.random.randint(1, 13) # 발표조
diff = z[:, 4] - z[:, :4].mean(axis=1)
diff.max() # 최대 향상치
diff.min() # 최대 하락치

## 벡터의 최대값의 위치 구하는코드
max_ind=np.where(diff == diff.max())[0]
min_ind=np.where(diff == diff.min())[0]

z[max_ind[0],:][4]
z[max_ind,:][0][4]
z[min_ind,:]

## 벡터의 최대값의 위치 구하는코드 2
max_ind=np.argmax(diff)
min_ind=np.argmin(diff)

z[max_ind,:][4]
z[min_ind,:]

# 가상환경에 패키지 설치
# conda activate 본인환경
# conda install matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(30, 30)

img1 = img1[0:16, 0:16]

# 행렬을 이미지로 표시
plt.figure(figsize=(10, 5)) # (가로, 세로) 크기 설정
plt.imshow(img1, cmap='gray', interpolation='nearest');
plt.colorbar()
plt.show()


import urllib.request

img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

import imageio
import numpy as np

# 이미지 읽기
jelly = imageio.imread("jelly.png")
type(jelly)
jelly.shape

# 베지밀 앞면 잘라와서 0~1로 바꾸기
img2=jelly[:, :, 0] / 255
img2.shape

# 행렬을 이미지로 표시
plt.figure(figsize=(10, 5)) # (가로, 세로) 크기 설정
plt.imshow(img2, cmap='gray', interpolation='nearest');
plt.colorbar()
plt.show()
# R: G: B:



# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)
# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])

my_array.shape
my_array[0,:,:]
my_array[1,:,:]

my_array[1, 1, 1:]