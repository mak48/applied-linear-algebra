# Задание
Описать **класс SLAE**
**docstring** """My SLAE Class"""
класс произвольных СЛАУ
**переменные** класса:
***обязательные***
*  name типа str - имя СЛАУ
*  a - матрица левой части, тип np.array
*  b - вектор правой части, тип np.array
***необязательные***
*  var_num число переменных СЛАУ, по умолчанию 0
*  eq_num число уравнений СЛАУ, по умолчанию 0
оба целые числа
**атрибуты**
*  get_a
*  get_b
*  get_var_num
*  get_eq_num
*  get_dim
**методы**
*  set_b
*  set_b_zero
*  x

**Дочерний класс SLAEhomogeneous**
**переменные** класса:
b - вектор правой части, тип np.array со значением по умолчанию None, не должен выводиться на экран при выведении на экран экземпляра класса (хотя в родительском классе уже есть переменная b, здесь ее нужно переопределить, чтобы она стала переменной со значением по умолчанию)
**атрибуты**
*  get_b
**методы**
*  set_b
*  x

**Дочерний класс SLAEsquare**
**переменные** класса:
singular: bool = None
square: bool = None
a_inv - обратная матрица для матрицы левой части, тип np.array
**атрибуты**
is_square  - возвращает True, если матрица квадратная
is_singular - возвращает True, если матрица вырождена
get_inv - - возвращает обратную матрицу, если она существует
**метод**
x

**класс SLAE**
**docstring** """My SLAE Class"""
**атрибуты**
*  get_a возвращает матрицу A
*  get_b  возвращает b
*  get_var_num сравнивает var_num с нулем, если 0, то вычисляет число столбцов A и перезаписывает var_num, возвращает var_num
*  get_eq_num аналогично get_var_num дествует с числом уравнений eq_num
*  get_dim возвращает кортеж tulpe из числа уравнений и числа переменных СЛАУ, использует get_eq_num и get_var_num
**методы**
*  set_b проверяет, что передаваемый в качестве аргумента вектор b соответствует матрице a по размерности, после чего перезаписывает значение переменной b экземпляра класса SLAE
*  set_b_zero у этого метода  нет дополнительных аргументов, метод вычисляет, какой размерности должен быть вектор b, составляет вектор из нулей и вызывает set_b
*  x возвращает кортеж, первый элемент True (если решение есть и единственно)  или False (иначе), второй - решение (если есть и единственно) или пустой np.array (np.array([]))

**Дочерний класс SLAEhomogeneous**
**переменные** класса:
b - вектор правой части, тип np.array со значением по умолчанию None, не должен выводиться на экран при выведении на экран экземпляра класса (хотя в родительском классе уже есть переменная b, здесь ее нужно переопределить, чтобы она стала переменной со значением по умолчанию)
**атрибуты**
*  get_b этот атрибут нужно переопределить, чтобы он возвращал вектор из нулей, размерности, соответствуюшей матрице A
**методы**
*  set_b этот метод  нужно переопределить, чтобы он ничего не возвращал и не делал, а только выводил сообщение 'b = 0 in homogeneous SLAE, use get_b instead'
*  x метод проверяет, что матрица квадратная и невырожденная, возвращает в этом случае кортеж с True и нулевым вектором соответствующей размерности, не использует np.linalg.solve,
иначе False и пустой np.array

**Дочерний класс SLAEsquare**
**переменные** класса:
singular: True если вырожденная матрица, иначе False
square: True если квадратная матрица, иначе False
a_inv обратная матрица, если существует, иначе None
**атрибуты**
is_square возвращает square, при необходимости предварительно вычисляя и заполняя эту переменную
is_singular - возвращает singular, при необходимости предварительно вычисляя и заполняя эту переменную
get_inv - - возвращает обратную матрицу, если она существует
**метод**
x проверяет невырожденность матрицы методами этого же класса, для невырожденной матрицы возвращает True и решение или False и  пустой np.array

Считать матрицу и вектор из файла 'ab1.xlsx'. Создать экземпляр класса SLAE с именем 'SLAE_1_1' и матрицей A1, вектором b1, вывести экземпляр на экран.
Вывести на экран атрибуты SLAE_1:
a, get_a, b, get_b

заменить b на вектор из нулей, вывести на экран текущее значение b
заменить b на вектор [4, 5, 6, 7], вывести на экран текущее значение b
Попытаться заменить b на вектор [-1, 2, -3, 4, 5, 6, 7], вывести на экран текущее значение b и решение СЛАУ
Создать экземпляр класса SLAEhomogeneous с именем 'SLAE_homo_1' и матрицей A1, вывести экземпляр на экран
Вывести на экран атрибуты SLAE_homo_1: a, get_a
вывести на экран текущее значение b
попытаться заменить b на вектор [1, 2, 3, 4, 5, 6, 7], вывести на экран текущее значение b
Создать экземпляр класса SLAEsquare с именем 'SLAE_sq_1' и матрицей A1, вектором b1 вывести экземпляр на экран
Вывести на экран атрибуты SLAE_sq_1: a, get_a
вывести решение x
заменить b на нулевой вектор
вывести на экран текущее значение b и решение x
попытаться заменить b на вектор [1, 2, 3, 4, 5], вывести на экран текущее значение b
Считать матрицу и вектор из файла 'ab2.xlsx'
Создать экземпляр класса SLAEsquare с именем 'SLAE_sq_2' и матрицей A2, вектором b2 вывести экземпляр на экран
Вывести на экран атрибуты SLAE_sq_2: a, get_a
вывести решение x
заменить b на нулевой вектор, вывести на экран текущее значение b и решение x
попытаться заменить b на вектор [1, 2, 3, 4, 5, 6], вывести на экран текущее значение b
