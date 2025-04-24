import numpy as np
img_mat = np.loadtxt('img_mat.csv', 
                     delimiter=',',
                     skiprows=1)

img_mat.shape

img_mat[:3, :4]

img_mat.max()
img_mat.min()

import matplotlib.pyplot as plt

# 행렬을 이미지로 표시
plt.figure(figsize=(10, 5)) # (가로, 세로) 크기 설정
plt.imshow(img_mat, cmap='gray', interpolation='nearest');
plt.colorbar()
plt.show()

# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
x.transpose()

x=np.array([1, 2, 3, 4]).reshape((2, 2))
y=np.array([3, 4, 1, 2]).reshape((2, 2))
x.dot(y)

# 성적데이터
np.random.seed(2025)
z=np.random.randint(1, 21, 20).reshape(4, 5)
z

w=np.array([0.1, 0.2, 0.3, 0.1, 0.3])
sum(np.array([19, 13, 4, 20, 13]) * w)

z.dot(w)

x=np.arange(1, 101).reshape((100, 1))
sum(x**2)
x.transpose().dot(x)

x=np.arange(1, 101)
x.shape
x.dot(x)

# np.matmul(x, x)

matC = np.random.rand(2, 3, 4)
matD = np.random.rand(2, 4, 5)
matC.shape
matD.shape
matmul_result = np.matmul(matC, matD)
matmul_result.shape


z = np.arange(10, 14).reshape((2, 2))
y = np.array([[1, 2], [3, 4]])
z
y
z * y


x=np.array([1, 2, 3, 4]).reshape((2, 2))
y=np.array([3, 4, 1, 2]).reshape((2, 2))
x
y
np.matmul(x, y)
np.matmul(y, x)

x
# x 행렬의 역행렬
inv_x=np.linalg.inv(x)
x
inv_x
np.matmul(x, inv_x)
np.matmul(inv_x, x)

w=np.array([1, 2, 2, 4]).reshape((2, 2))
w
np.linalg.inv(w)
np.linalg.det(w)

# 성적 데이터
# z^T z의 역행렬은 존재하나요?
# 성적데이터
np.random.seed(2025)
z=np.random.randint(1, 21, 20).reshape(4, 5)
z

np.linalg.det(np.matmul(z.transpose(),z)) 
zz=np.matmul(z.transpose(),z)
inv_zz=np.linalg.inv(np.matmul(z.transpose(),z))
np.matmul(zz, inv_zz)
np.matmul(inv_zz, zz)

# 역행렬은 정사각형 행렬만 존재한다.

a=np.array([1, 2, 3, 4]).reshape((2, 2))
b=np.array([5, 6, 7, 8]).reshape((2, 2))
c=np.array([9, 10, 11, 12]).reshape((2, 2))
matT=np.array([a, b])
matT.shape
np.matmul(matT, c)

np.matmul(a, b).dot(c)
np.matmul(a, np.matmul(b, c))

# 역행렬과 연립방정식
matA=np.array([3, 3, 2, 4]).reshape((2, 2))
matA
np.linalg.det(matA)
matA_inv=np.linalg.inv(matA)
b=np.array([1, 1])
matA_inv.dot(b)


array_2d = np.arange(1, 13).reshape((3, 4), order='F')
array_2d

np.apply_along_axis(max, axis=0, arr=array_2d)
np.apply_along_axis(np.mean, axis=1, arr=array_2d)
array_2d.mean(axis=1)
array_2d.sum(axis=0)
array_2d.sum(axis=1)

def my_sum(input):
    result = sum(input + 1)
    return result

a=np.array([1, 2, 3])
my_sum(a)

np.apply_along_axis(my_sum, axis=1, arr=array_2d)

# 입력 벡터의 각 원소들을 제곱한 후, 2를 곱하고,
# 그 값을 모두 더한 함수를 my_f라고 정의 한 후,
# my_f 를 행렬 array_2d의 각 행에 적용해주세요!
def my_f(input):
    result = 2*sum(input**2)
    return result

np.apply_along_axis(my_f, axis=1, arr=array_2d)

array_3d = np.arange(1, 25).reshape(2, 4, 3).transpose(0, 2, 1)
array_3d

np.apply_along_axis(sum, axis=0, arr=array_3d)
np.apply_along_axis(sum, axis=1, arr=array_3d)
np.apply_along_axis(sum, axis=2, arr=array_3d)

# np.random.seed(338225)
# np.random.randint(1, 13) # 발표조
# np.random.randint(1, 4)  # 발표자


