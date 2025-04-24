# 메서드: 객체에서 사용가능한 함수
# 어트리뷰트(속성): 객체의 특성정보를 담은 변수

# a가 리스트인 경우
a=[1, 2, 3]
a
b=a
c=a[:]
d=a.copy()

a[0]=20; b; c; d

# a가 넘파이 배열인 경우
import numpy as np
a=np.arange(1, 4)
a
b=a
c=a[:] # View (뷰) 개념 복사
d=a.copy()

a[0]=20; b; c; d

# 뷰인지 확인하는 방법 (넘파이)
print(a.base)
print(b.base)
print(c.base)
print(d.base)

c.base is a # c의 공유데이터가 a와 같은것은 확인