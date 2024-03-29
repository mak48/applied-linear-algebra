
## Задание 1.
Вычислить матричные нормы $||A_1||_1$, $||A_1||_2$, $||A_1||_\infty$, $||A_1||_{nuc}$ и $||A_1||_F$  для матрицы $A_1$
из файла  sem_9_task_1.xlsx, используя numpy, scipy и sympy (только нормы 1 и sympy.oo).

Вывести на экран на отдельных строках имя модуля (numpy, scipy и sympy) и нормы в виде numpy: ||A||1 = 91.0 ....

## Задание 2
Пусть $A_2$ - матрица, состоящая из первых десяти строк и первых десяти столбцов матрицы Задания 1.
С помощью разложения SVD (scipy.linalg.svdvals) получить сингулярные числа матрицы $A_2$. Изобразить на комплексной плоскости сингулярные числа, спектр матрицы $A_2$ и окружность с радиусом, равным спектральному радиусу матрицы $A_2$.

Вывести на экран спектр, спектральный радиус и сингулярные числа.


## Задание 3
Пусть $A_3$ - матрица, составленная из четных строк и нечетных столбцов матрицы Задания 1 (считаем, что нумерация с единицы!).
Проверить утверждение $||A||_F = \sqrt{\sigma_1^2 + \sigma_1^2 + ... + \sigma_1^n}$ на матрицах $A_1$, $A_2$ и $A_3$.

## Задание 4.
Вычислить матричные нормы 1, 2, $\infty$, ядерную и фробениусову единичной матрицы 3-го порядка, вывести на экран их значения, затем вывести ord тех норм, которые сохраняют единицу. Использовать списочные выражения.

## Задание 5.
На примере вектора $x$ из единиц и матрицы $A_1$ Задания 1 показать согласованность манхэттенской нормы с $||x||_1$, $||x||_2$, $||x||_\infty$.

Выписать в виде формулы определение согласованности матричной и векторной нормы, вывести на экран результат подстановки $A_1$ и $x$ в это неравенство, проверить, что оно выполняется.

## Задание 6.
На примере матрицы $A_2$ Задания 2 показать для норм 1, 2, $\infty$, ядерной и фробениусовой выполнение неравенства $||A^{-1}||\ge||A||^{-1}$. Вывести на экран значения левой и правой части и True или False в зависимости от того, выполняется ли оно.

Аналогично ($||A^{+}||\ge||A||^{-1}$?) сделать для псевдообратной матрицы к матрице, полученной из первых 10 строк и 8 столбцов матрицы $A_1$ Задания 1 и для матрицы $3\times 5$ из нулей.

Использовать вложенный цикл и zip, внешний цикл по матрицам (zip соединяет матрицу и обратную или псевдообратную), внутренний по ord норм.

## Задание 7*.
Из матрицы $A_1$ Задания 1 выделить всевозможные матрицы $n \times m$, состоящие из первых $n$ строк и $m$ столбцов. В файл sem_9_task_7.xlsx на отдельные листы с именами $n$ записать в отдельные столбцы нормы матриц $n \times m$, $m = 1, ..., 15$, использовать $m$ в качестве индекса в DataFrame, а ord норм как имена столбцов.

