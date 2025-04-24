print('hello')

a=3

# Ctrl + Shift + P
# 명령어 팔레트 단축키

# 34의 6 거듭제곱한 후, 256으로 나눈 나머지의 2배는?

a = a + 10
a += 10
a

x = 10
x /= 3
print(x, type(x))

a = 100
a += 10
a -= 20
a //= 2
a

# True == 1
# False == 0
# and 는 곱셈하고 똑같아!
False * True

True or False
True + False
False or True
False + True
False or False
False + False
True or True
True + True
min(3, 5)
min(3, 1)
min(True + True, 1)
min(True + False, 1)
min(False + False, 1)


# 문자열 변수 할당
str1 = "Hello, "
str2 = "world!"

str1 + str2
str1 * 3
str1 + 3

# alt shift 아래화살표
# ctrl / 주석달기

# 리스트에서 멤버십 연산자 사용
my_list = [1, 2, 3, 4]
print(3 in my_list)
3 in my_list
10 in my_list


my_string = "Python programming"
"Python" in my_string
"python" in my_string

# 리스트 비교
x = [1, 2, 3]
y = [1, 2, 3]
z = x

x is z
y is z

x == y
x is y

x=15.2
type(x)


# 여러 줄 문자열
ml_str = """This is
a multi-line     \ ' "hello"
string"""
ml_str = "hi \t world!"
print(ml_str)


# 리스트 생성 예제
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]
print("Fruits:", fruits)

# 튜플 생성 예제
a = (10, 20, 30) # a = 10, 20, 30 과 동일
b = (42)
b = (42,)
print("좌표:", a)

a[1]
fruits[0]
numbers[2:5]

numbers = [10, 2, 23, 4, 5, 8, 9]
# 두번째 인덱스 이상, 5번째 인덱스 미만
numbers[2:5]
numbers[2:]

a
numbers[2] = 14
numbers
a[2] =10


# 딕셔너리 생성 예제
person = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}
print(person)
person['city']
person.get("city")
person.keys()
person.keys

fruits = {'apple', 'banana', 'cherry', 'apple'}
fruits[0]
fruits[1]
fruits.add("grape")
fruits.remove("banana")
fruits.discard("banana")

numbers = [1, 2, 3, 4, 5]
range_list = list(range(3, 5))
range_list

fruits = ["apple", "banana", "cherry",
          "apple", "banana"]
first_fruit = fruits[0]
last_fruit = fruits[-3]

fruits[1:-2]

squares = [x**2 for x in range(10)]
squares
# 1에서 100까지 제곱한 값들을 리스트로!
[x**2 for x in range(1, 101)]

# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2
combined_list * 3


# 리스트 각 원소별 반복
numbers = [5, 2, 3]
[x for x in numbers for _ in range(3)]

fruits = ["apple", "banana", "cherry"]

"banana" in fruits
[x == "banana" for x in fruits]


fruits = ["apple", "banana", "cherry"]
fruits.append("date")
fruits.append(2)
fruits.append([2])
fruits.extend([2])
fruits.extend([1, [2]])
fruits.extend("date")
fruits

# insert() 메서드 사용 예제
fruits = ["apple", "banana", "cherry"]
fruits.insert(1, "blueberry")
fruits


# remove() 메서드 사용 예제
fruits = ["apple", "banana", "cherry", "banana"]
fruits.remove("banana")
fruits.remove("banana")
fruits


# 리스트 생성
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# 제거할 인덱스 리스트
indices_to_remove = [1, 2, 0] # banana와 cherry를 제거

# 제거할 인덱스 리스트를 내림차순으로 정렬
indices_to_remove.sort()

# 각 인덱스에 대해 pop() 호출
popped_fruits = [fruits.pop(x) for x in indices_to_remove]
print("pop() 후 리스트:", fruits)


# reverse() 메서드 사용 예제
numbers = [3, 1, 4, 1, 5, 9]
numbers.sort()
numbers
numbers.reverse()


# 리스트 합계, 최소값, 최대값
numbers = [1, 2, 3, 4, 5]
sum(numbers)
min(numbers)
max(numbers)

# 멀티커서 단축키: CTRL + ALT + 아래화살표

# 리스트 중복 제거
numbers = [1, 2, 2, 3, 4, 4, 5]
a=set(numbers)
type(a)
unique_numbers = list(set(numbers))
unique_numbers


# 리스트를 문자열로 변환
fruits = ["apple", "banana", "cherry"]
", ".join(fruits)

fruits_str = "apple, banana, cherry"
fruits_list = fruits_str.split(", ")

matrix = [[1, 2, 3], [4, 5, 6], [7, 8]]
matrix[1][0]


data = [10, 20, 30]
print("메모리 주소:", id(data))

data = ["apple", "banana", "cherry"]
copy_data = data # 동일한 객체 참조

x = [1,2,3]
y = [1,2,3]
z = x

x is z

id(x)
id(y)
id(z)

x[1] = 10
x
z

original = ["apple", "banana", "cherry"]
copy_data = original[:] # 새 객체로 복사
shallow_copy = original.copy()
id(original)
id(copy_data)
id(shallow_copy)
shallow_copy


original = [["apple", "banana"], ["cherry", "grape"]]
shallow_copy = original.copy()
shallow_copy

id(original)
id(shallow_copy)

original[0][0] = "watermelon"
original

shallow_copy

original[0] = 3
original

shallow_copy

import copy
deep_copy=copy.deepcopy(original)
deep_copy

original[0][0] = "watermelon"
original

deep_copy