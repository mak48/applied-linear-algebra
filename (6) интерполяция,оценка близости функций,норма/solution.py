import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy
from sympy import S, latex
from scipy.interpolate import lagrange, interp1d, CubicSpline
from scipy.linalg import norm
from numpy.polynomial.polynomial import Polynomial
from google.colab import files
from IPython.display import Math, Latex
import bezier
import scipy


#1
print("#1\n")
uploaded = files.upload()

for fn in uploaded.keys():
  print(f'User uploaded file "{fn}"')
df = pd.read_excel(fn, index_col=0, skiprows=[0], header=0)
print(f"""index: {df.index},\ncolumns: {df.columns},\n
values:\n{df.values},\nshape: {df.shape}""")

data = df.to_numpy()
Y1 = data[:, 1]
display(Y1)
X1 = df.index
poly1 = lagrange(X1, Y1)
X1ls = np.linspace(X1[0], X1[-1], 1000)
plt.plot(X1, Y1, 'ro', X1ls, poly1(X1ls), 'g-')

X1 = df.index
X1ls = np.linspace(X1[0], X1[-1], 1000)
for label, t_col_name, colors in zip(('день', 'ночь'),
                                     ('Давление', 'Давление.1'),
                                     (('r', 'g'), ('c', 'm'))):
    color1, color2 = colors
    Y1 = df[t_col_name].to_numpy()
    poly1 = lagrange(X1, Y1)
    plt.plot(X1, Y1, color1 + 'o', X1ls, poly1(X1ls),  color2 + '-',
             label=label)
plt.legend()

#2
print("#2\n")
arr = Y1
n = len(arr)

X = np.column_stack([np.ones(n), np.arange(1, n+1), np.arange(1, n+1)**2, np.arange(1, n+1)**3])
X_pseudo_inv = np.linalg.pinv(X)

coeffs = X_pseudo_inv.dot(arr)
poly = np.poly1d(coeffs[::-1])
poly_values = np.array([poly(i) for i in range(1, n+1)])

norm_diff = norm(arr - poly_values)
print("Норма разности:", norm_diff)

X1ls = np.linspace(1, 10, 1000)
poly_values = np.array([poly(i) for i in X1ls])

plt.figure()
plt.scatter(range(1, n+1), arr, color='red', label='Данные')
plt.plot(X1ls, poly_values, color='blue', label='Полином 3 степени')
plt.legend()
plt.grid(True)
plt.show()

#3
print("#3\n")
def f(x):
    return np.log(x + 1)

x_nodes = np.arange(0, 2.1, 0.5)
y_nodes = f(x_nodes)

cs = interp1d(x_nodes, y_nodes, kind='quadratic')
x_grid = np.arange(0, 2.1, 0.2)
y_true = f(x_grid)
y_spline = cs(x_grid)
norm_diff = np.linalg.norm(y_true - y_spline, ord=np.inf)

print("Значения функции f(x):\n", y_true)
print("Значения сплайна в узлах:\n", y_spline)
print("Норма разности:", round(norm_diff, 4))

#4
print("#4\n")
def f(x):
    return np.sin(2*x**2)
x = np.arange(0, np.pi+0.2, 0.2)
y = f(x)

cs = interp1d(x, y, kind='cubic')
x_new = np.arange(0, np.pi+0.1, 0.1)
y_new = f(x_new)
cs_new = cs(x_new)

for o in [1,2,3,np.inf]:
  norm1 = norm(y_new - cs_new, ord=o)
  print("Норма разности (ord=",o,"):", round(norm1, 4))

#5
print("#5\n")
def Y(X, p):
    Y1 = ((1+p)**p - abs(X-p) ** p) ** (1 / p)
    return (-Y1+2-p, Y1+2-p)

plt.axis('equal')
for p, color in zip((1, 3/2, 4, 5),('r--', 'g--', 'b:', 'c--')):
    X = np.linspace(-1, 1+2*p)
    Y1, Y2 = Y(X, p)
    plt.plot(X, Y1, color)
    plt.plot(X, Y2, color, label=f'$p={p}$')
plt.legend()

#6
print("#6\n")
ax = plt.gca()
# Устанавливаем оси координат
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
# Стрелки на концах осей координат
ax.plot(1, 0, '>k', transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, '^k', transform=ax.get_xaxis_transform(), clip_on=False)
ax.axis('equal')
# устанавливаем пределы по осям
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
# Тики (зарубки) на осях
# Позиции тиков (если не нужно изменять автоматические подписи к тикам)
ax.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4, 6])
ax.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4, 2])
# Установка подписей тиков
ax.set_xticklabels([r'$-4$', r'$-3$',r'$-2$',r'$-1$','   $0$', r'$+1$', r'$+2$',r'$+3$',r'$+4$','$x$'])
ax.set_yticklabels([r'$-4$', r'$-3$',r'$-2$',r'$-1$','   $0$', r'$+1$', r'$+2$',r'$+3$',r'$+4$', '$y$'])

def Y(X, p, a, b):
    Y1 = ((1/p)**p - abs(X-a) ** p) ** (1 / p)
    return (-Y1+b, Y1+b)

plt.axis('equal')
for p, color in zip((1, 2, 4),("red", "green", "cyan")):
  for a,b in zip((0,0,p,-p),(p,-p,0,0)):
    X = np.linspace(a-1/p, a+1/p)
    Y1, Y2 = Y(X, p,a,b)
    plt.plot(X, Y1, color[0]+'--')
    if (a==0 and b==p):
      plt.plot(X, Y2, color[0]+'--', label=f'$p={p}$')
    else:
      plt.plot(X, Y2, color[0]+'--')
    ax.scatter([a], [b], color=color)
plt.legend()

#7
print("#7\n")
def f4(x):
  return sympy.ln(x)/sympy.sqrt(x)

x = sympy.Symbol('x')
delta_x = 1
X4 = [1,2,3,4]

df4 = f4(x).diff(x)
Y4 = [f4(X4[0]), f4(X4[0])+df4.subs(x, X4[0]) * delta_x,
      f4(X4[-1])-df4.subs(x, X4[-1]) * delta_x, f4(X4[-1])]
nodes = np.asfarray([X4, Y4])
curve4 = bezier.Curve(nodes, degree=3)

def f(x):
  return np.log(x)/np.sqrt(x)
x_values = np.arange(1.000001, 4.1, 0.1)
y_values_f = f(x_values)
x_values = np.linspace(0.000001, 1, 31)
bezier_points = curve4.evaluate_multi(x_values)
norms = norm(y_values_f - bezier_points[1])
print(f'Норма разности: {norms}')

X4plt = np.linspace(*np.asfarray([X4[0], X4[-1]]), 1000)
npf4 = sympy.lambdify(x, f4(x))
fig, ax = plt.subplots()
ax.plot(X4, Y4, 'r--o', X4plt, npf4(X4plt), 'k:', lw=3)
curve4.plot(100, color='green', ax=ax)
