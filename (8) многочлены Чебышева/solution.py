import numpy as np
import sympy
from sympy import S, latex
from sympy.functions.special.polynomials import chebyshevt, chebyshevu
from numpy.polynomial.chebyshev import chebinterpolate, Chebyshev
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Latex
#1
print("#1\n")
def T_new(n, x):
    return ((x + sympy.sqrt(x ** 2 - 1)) ** n + (x - sympy.sqrt(x ** 2 - 1)) ** n) / 2
x = S('x')
X = np.linspace(-1, 1, 100)

plt.plot(X, sympy.lambdify(x, chebyshevt(1, x))(X), 'r',
         label=sympy.latex(sympy.Eq(S('cheb(1, x)'), chebyshevt(1, x)),
                           mode='inline'))
plt.plot(X, sympy.lambdify(x, chebyshevt(2, x))(X), 'm',
         label=sympy.latex(sympy.Eq(S('cheb(2, x)'), chebyshevt(2, x)),
                           mode='inline'))
plt.plot(X, sympy.lambdify(x, chebyshevt(3, x))(X), 'g',
         label=sympy.latex(sympy.Eq(S('cheb(3, x)'), chebyshevt(3, x)),
                           mode='inline'))
plt.plot(X, sympy.lambdify(x,chebyshevt(4, x))(X), 'b',
         label=sympy.latex(sympy.Eq(S('cheb(4, x)'), chebyshevt(4, x)),
                           mode='inline'))
plt.plot(X, sympy.lambdify(x, chebyshevt(5, x))(X), 'y',
         label=sympy.latex(sympy.Eq(S('cheb(5, x)'), chebyshevt(5, x)),
                           mode='inline'))
plt.legend()

#2
print("#2\n")
from scipy.linalg import norm
def T_new(n, x):
    return ((x + sympy.sqrt(x ** 2 - 1)) ** n + (x - sympy.sqrt(x ** 2 - 1)) ** n) / 2
x = S('x')
X = np.linspace(1, 2, 100)
print('Нормы:\n')
for i in [9,11,13]:
  difference = sympy.lambdify(x, chebyshevt(i, x))(X) -sympy.lambdify(x, T_new(i, x))(X)
  print(*[f'{i}(1): {norm(difference, 1)}'], sep='\n')
  print(*[f'{i}(2): {norm(difference, 2)}'], sep='\n')
  plt.plot(X, sympy.lambdify(x, chebyshevt(i, x))(X), '--r',
         label=sympy.latex(sympy.Eq(S('cheb(1, x)'), chebyshevt(i, x)),
                           mode='inline'))
  plt.plot(X, sympy.lambdify(x, T_new(i, x))(X), '.b',
         label=sympy.latex(sympy.Eq(S('cheb(2, x)'), T_new(i, x)),
                           mode='inline'))
  plt.show()

#3
print("#3\n")
def U(n, x):
    return (chebyshevt(n + 1, x).diff(x)/(n + 1)).expand().simplify()

def U_new(n, x):
    return (((x + sympy.sqrt(x ** 2 - 1)) ** (n + 1)\
             - (x - sympy.sqrt(x ** 2 - 1)) ** (n + 1))\
             / (2 * sympy.sqrt(x ** 2 - 1))).expand().simplify()
display(Latex('U(10, x) = {},\\\\U_{{new}}(10, x) = {}'.format(*[latex(item) for item in (U(10, x),
                                                  U_new(10, x),)])))

display(Latex(f'chebyshevu(10, x) = {latex(chebyshevu(10, x))}'))

#4
print("#4\n")
def f5(x):
  return sympy.sin(sympy.pi*x)

f5_norm0 = sympy.calculus.util.maximum(f5(x), x, domain=sympy.Interval(-1, 1))
f5_norm1 = sympy.Abs(f5(x)).integrate((x, -1, 1))
display(*[Latex('|f5|_{2} = {0} = {1}\
'.format(latex(item),
         round(item, 3), i)) for i, item in enumerate((f5_norm0, f5_norm1))])

#5
print("#5\n")
def f6(x):
    return x**6-x**2+2
def dot_prod_cheb(f, g, x):
    return (f * g / sympy.sqrt(1 - x ** 2)).integrate((x, -1, 1))
def f_cheb(f, n, x):
    res = 0
    for k in range(n + 1):
        cheb_k = chebyshevt(k, x)
        coef = dot_prod_cheb(cheb_k, f, x) / dot_prod_cheb(cheb_k, cheb_k, x)
        res += coef * cheb_k
    return res

res7 = chebinterpolate(f6, 4)
res7poly = sum([res7[k] * chebyshevt(k, x) for k in range(len(res7))])
cheb7 = Chebyshev(res7)

res6 = f_cheb(f6(x), 4, x)
X = np.linspace(-1, 1, 100)
plt.plot(X, f6(X), 'k-', X, cheb7(X), 'b--',
         X, sympy.lambdify(x, res6)(X), 'c:')
display(sympy.Eq(f6(x), res6))

f6_norm0 = sympy.calculus.util.maximum(f6(x)-f_cheb(f6(x), 4, x), x, domain=sympy.Interval(-1, 1))
f6_norm1 = sympy.Abs(f6(x)-f_cheb(f6(x), 4, x)).integrate((x, -1, 1))
display(*[Latex('|f6|_{1} = {0}\
'.format(round(item, 4), i)) for i, item in enumerate((f6_norm0, f6_norm1))])

#6
print("#6\n")
def f7(x):
    return x**7-x**2+x-4
res7 = chebinterpolate(f7, 6)
res7poly = sum([res7[k] * chebyshevt(k, x) for k in range(len(res7))])

Y1=sympy.calculus.util.maximum((f7(x)-res7poly), x, domain=sympy.Interval(-1, 1))
sympy.plot(((f7(x)-res7poly),(x, -1, 1)),(Y1, (x, -1, 1)), (-Y1, (x, -1, 1)))
res = (f7(x)-res7poly)
points1 = np.polynomial.chebyshev.chebpts1(6)
points2 = np.polynomial.chebyshev.chebpts2(7)

display(sympy.Eq(f7(x), res7poly.evalf(2)))
sympy.plot((res7poly, (x, -1, 1)), (f7(x), (x, -1, 1)))

#7
print("#7\n")
from google.colab import files
def f7(x):
    return np.exp(1/(1+x**2))
plt.plot(X, f7(X),'.k')
x = np.linspace(-1, 1, 100)
f4=np.empty(100)
f6=np.empty(100)
f10=np.empty(100)
for i,color,arr in zip([4,6,10],('--c','--b','r:'),(f4,f6,f10)):
  res7 = chebinterpolate(f7, i)
  cheb7 = Chebyshev(res7)
  plt.plot(X, cheb7(X),color)
  arr = cheb7(X)
  norm = np.max(np.abs(cheb7(x)- f7(x)))
  print(f'Chebyshev norm for k={i}: {norm}')
df = pd.DataFrame({'Col_A': f7(X),
                  'Col_B': f4,
                  'Col_C': f6,
                  'Col_D': f10})
with pd.ExcelWriter("task_7.xlsx") as writer:
  df.to_excel(writer,header=False,index=False)
  files.download("task_7.xlsx")