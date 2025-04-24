import numpy as np

def print_numbers():
    for i in range(1, 11):
        print(i)

print_numbers()

# 위의 함수를 1에서부터 내가 원하는 숫자(n)까지
# 정수를 출력하는 함수로 바꾸려면?
def print_numbers(n):
    for i in range(1, n+1):
        print(i)

print_numbers(10)

def outer_function(x):
    def inner_function(y):
        return y + 2
    result = inner_function(x)
    return result

outer_function(5)

start = 4
def find_even(start):
    while True:
        if start % 2 == 0:
            break
        start = start + 1
    return start


find_even(4)
find_even(5)
find_even(6)

# 처음으로 나오는 3의 배수
start = 1
def find_even(start):
    while start % 2 == 1: # 
        start += 1
        if start % 2 == 0:
            break
    print(start)

find_even(3)
find_even(4)


# 처음으로 나오는 3의 배수
start = 1
def find_even(start):
    while True:  
        start += 1
        if start % 3 == 0:
            break
    print(start)

# Q. 두번째로 나오는 짝수를 반환하는 함수!
# 예:
# 입력 3 -> 출력 6
# 입력 4 -> 출력 6
start = 3
def find_second_even(start):
    cnt = 0
    while True:
        if start % 2 == 0:
            cnt += 1
        if cnt == 2:
            break  
        start += 1
    print(start)

find_second_even(3)
find_second_even(4)
find_second_even(5)

start = 3
def find_second_even(start):
    cnt = 0
    while cnt < 2:
        if start % 2 == 0:
            cnt += 1
        start += 1
    start -= 1
    print(start)

find_second_even(3)
find_second_even(4)
find_second_even(5)

start=3
def find_second_even(start):
    arr1=[]
    while len(arr1) != 2:
        if start % 2 == 0:
            arr1.append(start)
        start += 1
    return arr1[-1]

find_second_even(3)
find_second_even(4)
find_second_even(5)

# 100번째, 1000번째, 3425번째 짝수!
