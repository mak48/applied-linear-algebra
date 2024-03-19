import numpy as np
import pandas as pd
import scipy.linalg
import sympy
import matplotlib.pyplot as plt
from sympy import latex, Matrix
from IPython.display import Latex
from google.colab import files

#1
print("#1\n")
A = np.array([[1, -5],
              [-2, 10]])
b = np.array([[4], [-7]])
X = [-0.5, 0.5]
Y1, Y2 = [[(b[i] - A[i, 0] * x) / A[i, 1] for x in X] for i in (0, 1)]
plt.plot(X, Y1)
plt.plot(X, Y2, 'g--')
plt.grid(True)
plt.show()

print(A[0, 0] / A[1, 0] == A[0, 1] / A[1, 1])

sol = np.linalg.pinv(A) @ b
display(Latex(f'псевдорешение\ {latex(Matrix(sol))}'))

plt.plot(X, Y1, 'm:', X, Y2, 'g--')
plt.scatter(*sol)
plt.grid(True)
plt.show()

def residual(A, b, sol):
    return np.linalg.norm(A @ sol - b)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-0.5, 0.5, 0.1)
Y = np.arange(-1.5, 0, 0.1)
Z = np.array([[residual(A, b, np.array([[x], [y]])) for x in X] for y in Y])
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z, vmin=Z.min() * 2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-0.5, 0.5, 0.1)
Y = np.arange(-1.5, 0, 0.1)
Z = np.array([[residual(A, b, np.array([[x], [y]])) for x in X] for y in Y])
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z, vmin=Z.min() * 2)
ax.scatter(*sol, residual(A, b, sol), c='k', s=150, marker=(6, 2))

min_res = min([min(zz) for zz in Z])
print(f'минимум невязки {min_res}, невязка псевдорешения {residual(A, b, sol)}, модуль разности {abs(residual(A, b, sol) - min_res)}')

#2
print("#2\n")

A = np.array([[2, 3,-1],[3,-2,1],[5,1,0]])
b = np.array([[5],[2],[0]])
Ab = np.array([[2, 3,-1,5],[3,-2,1,2],[5,1,0,0]])
if np.linalg.matrix_rank(A)==np.linalg.matrix_rank(Ab):
  print("система совместна")
else:
  print("система несовместна")

sol = np.linalg.pinv(A) @ b
display(Latex(f'Псевдорешение\ {latex(Matrix(sol))}'))

def residual(A, b, sol):
    return np.linalg.norm(A @ sol - b)
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
Z = np.arange(-1, 1, 0.1)
D = np.array([[[residual(A, b, np.array([[x],[y],[z]])) for x in X] for y in Y]for z in Z])
min_res=10**6
for i in range(len(D)):
  res = min([min(dd) for dd in D[i]])
  if res<min_res:
    min_res=res
print(f'Минимум невязки {min_res}, невязка псевдорешения {residual(A, b, sol)}, модуль разности {abs(residual(A, b, sol) - min_res)}')

#3
print("#3\n")
files.upload()
data = pd.read_excel("sem2_corona2020.xlsx", header=None, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 40, 49, 57, 72, 80, 91, 103])
arr = data.to_numpy()
n = len(arr[:, 1])

# a)
Q1 = arr[:, 1].reshape(n, 1).astype(float)
X2 = arr[:, 2].reshape(n, 1).astype(float)
K1 = np.hstack((X2, np.ones((n, 1))))
result_a = np.linalg.pinv(K1)@Q1
k2, b = result_a[:, 0]

# b)
X3 = arr[:, 3].reshape(n, 1).astype(float)
X4 = arr[:, 4].reshape(n, 1).astype(float)
K2 = np. hstack((X3, X4))
result_b = np. linalg.pinv(K2)@Q1
k3, k4 = result_b[:, 0]

# c)
X5 = arr[:, 5].reshape(n, 1).astype(float)
Q4 = arr[:, 4].reshape(n, 1).astype(float)
K3 = np.hstack((X2, X5, np.ones((n, 1))))
result_c = np.linalg.pinv(K3)@Q4
k2_1, k5, t = result_c[:, 0]

#график а)
X = np.array([min(X2), max(X2)])
Y = k2*X + b
plt.title("линейная регрессия Q1 = k2+X2 + b")
plt.plot(X, Y, 'g--', label = "линейная pегрессия")
plt.plot(X2, Q1, ' ', color='b', marker=(5,2), label = 'данные')
plt.legend()

#a)
print("a) коэффициенты для Q1 = k2*X2 + b:")
print("k2=", k2.round(3))
print("b=", b.round(3))
#b)
print("\nb) коэффициенты для Q1 = k3+X3 + k4*X4:")
print("k3=", k3.round(3))
print("k4=", k4.round(3))
#с)
print("\nc) коэффициенты для Q4 = k2*X2 + k5*X5 + t:")
print("k2=", k2_1.round(3))
print("k5=", k5.round(3))
print("t=", t)

#4
print("#4\n")
files.upload()
data = pd.read_excel("sem2_corona2020.xlsx", header=None, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 40, 49, 57, 72, 80, 91, 103])
arr = data.to_numpy()
n = len(arr[:, 1])

Q1 = arr[:, 1].reshape(n, 1).astype(float)
X2 = arr[:, 2].reshape(n, 1).astype(float)
K1 = np.hstack((X2, np.ones((n, 1))))
result_a = np.linalg.pinv(K1)@Q1
k2, b = result_a[:, 0]

X = np.array([min(X2), max(X2)])
Y = k2*X + b
plt.title("новая линейная регрессия")
plt.plot(X, Y, 'r--', label = "старая линейная pегрессия")
plt.plot(X2, Q1, ' ', color='r', marker=(5,2), label = 'старые данные')
plt.legend()

Q1_ = arr[:, 1].reshape(n, 1).astype(float)
X2_ = arr[:, 2].reshape(n, 1).astype(float)
d = []
for i in range(n):
  if Q1_[i]>1000 or X2_[i]>1000:
    d.append(i)
n-=len(d)
for i in range(len(d)):
    Q1_ = np.delete(Q1_,d[i]-i)
    X2_ = np.delete(X2_,d[i]-i)
X2_ = X2_.reshape(n, 1).astype(float)
Q1_ = Q1_.reshape(n, 1).astype(float)
K1_ = np.hstack((X2_, np.ones((n, 1))))
result_a_ = np.linalg.pinv(K1_)@Q1_
k2_, b_ = result_a_[:, 0]

X_ = np.array([min(X2), max(X2)])
Y_ = k2_*X_ + b_
plt.plot(X_, Y_, 'g--', label = "новая линейная pегрессия")
plt.plot(X2_, Q1_, ' ', color='g', marker=(5,2), label = 'новые данные')
plt.legend()