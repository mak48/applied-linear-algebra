import numpy as np
import scipy.linalg
import sympy
from sympy import S, latex, Eq, Matrix
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splev, CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline, BPoly, lagrange
from scipy.special import comb
from numpy.polynomial.polynomial import Polynomial
from IPython.display import Math, Latex
import bezier
from google.colab import files

#1
print("#1\n")
x = S('x')
X = (-np.pi/4, 0, np.pi/4)
Y = (2**0.5/2, 1, 2**0.5/2)
Lagrange1 = 0
for i in range(3):
    Li = Y[i]
    for j in range(3):
        if i != j:
            Li *= ((x - X[j]) / (X[i] - X[j]))
    Lagrange1 += Li
display(Latex(f'Полином\ Лагранжа\ {latex(Lagrange1)},\\\\\
упрощенный\ {latex(sympy.simplify(sympy.expand(Lagrange1)))}'))

display(Latex(f'Полином\ Лагранжа\ {latex(Lagrange1.evalf(2))},\\\\\
упрощенный\ {latex(sympy.simplify(sympy.expand(Lagrange1.evalf(2))))}'))

L = 0
n = len(X)
for i, Li in enumerate(Y):
    for j in range(n):
        if i != j:
            Li *= (x - X[j]) / (X[i] - X[j])
    L += Li
L = sympy.simplify(sympy.expand(L))
X_ls = np.linspace(X[0], X[-1],101)
Y_regr = [L.subs(x, item) for item in X_ls]
Y_cos = [np.cos(item) for item in X_ls]
plt.plot(X_ls, Y_regr, label=f"${sympy.latex(sympy.Eq(S('y'), L))}$")
plt.plot(X_ls, Y_cos, label=f"y = cos(x)")
plt.scatter(X, Y, color='red', label='data')
plt.grid()
plt.legend()

mse = np.sqrt(np.mean((np.array(Y_cos).astype(float) - np.array(Y_regr).astype(float))**2))
print("Среднеквадратическое отклонение:", mse)

#2
print("#2\n")
X2=[]
Y2=[]
for i in range(9):
  X2.append(-np.pi/4+i*(np.pi/16))
  Y2.append([np.cos(X2[i])])
A2 = np.array([[1, item, item ** 2] for item in X2])
res2 = np.linalg.pinv(A2) @ Y2
a2, b2, c2 = [round(item, 2) for item in  res2[:, 0]]
display(Latex(f'A = {latex(Matrix(A2))},\ \
Y = {Y},\ a = {a2},\ b = {b2},\ c = {c2}'))

X_ls2 = np.linspace(X2[0], X2[-1])
Y_cos2 = [np.cos(item) for item in X_ls2]
plt.plot(X_ls2, Y_cos2,'--g', label=f"y = cos(x)")
plt.plot(X_ls2, a2 + b2 * X_ls2 + c2 * X_ls2 ** 2, color = 'red',
         label='$y = {a} + {b}x + {c}x^2$'.format(a=round(a2),
                                                  b=round(b2), c=round(c2))
         )
plt.scatter(X2, Y2, color='red', label='data')
plt.grid()
plt.legend()

#3
print("#3\n")
poly5 = lagrange(X, Y)
display(Latex(f'Polynomial(lagrange(X, Y)).coef:\ \
{latex(Polynomial(poly5).coef)}\\\\lagrange(X, Y): '),
poly5)

x3_ls = np.linspace(min(X), max(X), 100)
plt.plot(X_ls, Y_regr, label=f"${sympy.latex(sympy.Eq(S('y'), L))}$")
plt.plot(x3_ls, Polynomial(poly5.coef[::-1])(x3_ls), label='Lagrange')
plt.plot(X_ls2, Y_cos2,'--g', label=f"y = cos(x)")
plt.scatter(X, Y, color='red', label='data')
plt.grid()
plt.legend()

#4
print("#4\n")
X = [-0.75,-0.375,0,0.375,0.75]
Y3 = [np.tan(x**3) for x in X]
Y4 = [np.tan(x**4) for x in X]
Y5 = [np.tan(x**5) for x in X]
Y = [Y3,Y4,Y5]
for i in range(3):
  plt.clf()
  plt.scatter(X, Y[i], color='red')
  spl1 = interp1d(X, Y[i],kind='quadratic')
  spl2 = interp1d(X, Y[i], kind='cubic')
  xs = np.linspace(X[0], X[-1], 100)
  plt.plot(xs, np.tan(xs**(i+3)), 'k-', xs, spl1(xs), 'c:', xs, spl2(xs), 'y--', lw=3)
  plt.savefig(str(i+3)+'.png')
  files.download(str(i+3)+'.png')

#5
print("#5\n")
# 2 способ
P1 = (-np.pi/6, np.sin(np.power(-np.pi/6,5)))
P2 = (0, 0)
P3 = (np.pi/6, np.sin(np.power(np.pi/6,5)))
P4 = (np.pi/3,np.sin(np.power(np.pi/3,5)))
P5 = (np.pi/2,np.sin(np.power(np.pi/2,5)))
X12, Y12 = [np.array(item).reshape(5, 1) for item in zip(P1, P2, P3, P4, P5)]
x = [0, 1]
bpX = BPoly(X12, x)
bpY = BPoly(Y12, x)
t_linspace = np.linspace(0, 1)
plt.plot(bpX(t_linspace), bpY(t_linspace), 'c-', lw=3)
plt.scatter(X12, Y12)

# 1 способ
def bernstein_poly(i, n, t):
    return comb(n, i) * (1 - t)**(n - i) * t**i

points = [(-np.pi/6, np.sin(np.power(-np.pi/6,5))),(0, 0),(np.pi/6, np.sin(np.power(np.pi/6,5))),
 (np.pi/3,np.sin(np.power(np.pi/3,5))),(np.pi/2,np.sin(np.power(np.pi/2,5)))]

t_values = np.linspace(0, 1)
bezier_curve = np.zeros_like(t_values)
bezier_curvey = np.zeros_like(t_values)
for i in range(5):
    x, y = points[i]
    bezier_curve += x * np.array([bernstein_poly(i, 4, t) for t in t_values])
    bezier_curvey += y * np.array([bernstein_poly(i, 4, t) for t in t_values])

plt.plot(bezier_curve, bezier_curvey, label='Bezier Curve', color='r')
plt.scatter([point[0] for point in points], [point[1] for point in points], color='b', label='сontrol points')
plt.legend()


#6
print("#6\n")
P1 = (-np.pi/6, np.sin(np.power(-np.pi/6,5)))
P2 = (0, 0)
P3 = (np.pi/6, np.sin(np.power(np.pi/6,5)))
P4 = (np.pi/3,np.sin(np.power(np.pi/3,5)))
P5 = (np.pi/2,np.sin(np.power(np.pi/2,5)))
X = []
Y = []
for point in (P1, P2, P3, P4, P5):
    X.append(point[0])
    Y.append(point[1])
nodes = np.array([X, Y])
curve = bezier.Curve(nodes, degree=4)
s_vals = np.linspace(-np.pi/6,np.pi/2, 5)
curve.evaluate_multi(s_vals)
curve.plot(100)