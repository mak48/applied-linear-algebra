import numpy as np
import sympy
import scipy.linalg
import numpy.linalg
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Matrix, latex
from google.colab import files
from IPython.display import Latex

#1
print("#1\n")
uploaded = files.upload()
M_ = pd.read_excel("sem_9_task_1.xlsx", header = None)
M = M_.to_numpy()
for package in (np, scipy):
  print(package.__name__, *[f'||M||{item} = {round(package.linalg.norm(M, ord=item), 3)}' for item in (1, 2, np.inf, 'nuc', 'fro')], sep=', ')
Msym = sympy.Matrix(M)
print("sympy",*[f'||M||{item} = {round(Msym.norm(ord=item), 3)}' for item in (1, 2, sympy.oo, 'fro')], sep=', ')

#2
print("#2\n")
M2 = M[:10, :10]

M2_spectr = numpy.linalg.eigvals(M2)
M2_spectr_abs = abs(M2_spectr)
M2_spectr_abs.sort()
radiusA3 = M2_spectr_abs[-1]
sing_val_M2 = scipy.linalg.svdvals(M2)
ax = plt.gca()
radius2 = radiusA3 ** 2
X = np.linspace(-radiusA3, radiusA3, 256)
Y = np.sqrt(radius2 - X ** 2)
ax.axis('equal')
ax.plot(sing_val_M2, np.zeros(len(sing_val_M2)), 'ro', np.real(M2_spectr), np.imag(M2_spectr), 'go', X, Y, 'c--', X, -Y, 'c--')
ax.set_xlabel('Re')
ax.set_ylabel('Im')
print("Спектр",M2_spectr)
print("Радиус",radiusA3)
print(f'Сингулярные числа {[round(item, 1) for item in sing_val_M2]}')

#3
print("#3\n")
M3 = M[1::2, ::2]
sing_val_M1 = scipy.linalg.svdvals(M)
sing_val_M3 = scipy.linalg.svdvals(M3)
for X, sing_val_X in zip([M,M2,M3],[sing_val_M1,sing_val_M2,sing_val_M3]):
  print(numpy.linalg.norm(sing_val_X, ord=2), numpy.linalg.norm(X, ord='fro'))

#4
print("#4\n")
A4 = np.array([[1,0,0],[0,1,0],[0,0,1]])
print(*[f'||M||{item} = {round(package.linalg.norm(A4, ord=item), 3)}' for item in (1, 2, np.inf, 'nuc', 'fro')], sep=', ')
ord_1 = [ord for ord in [1, 2, np.inf, 'fro'] if np.linalg.norm(A4, ord=ord) == 1]
print("Нормы, которые сохраняют единицу:", ord_1)

#5
print("#5\n")
x = np.ones(15)

for norms in [1,2,np.inf]:
  print(f'Норма {norms}: {np.linalg.norm(np.dot(M, x), ord=norms) <= np.linalg.norm(M, ord=1) * np.linalg.norm(x, ord=norms)}, {np.linalg.norm(np.dot(M, x), ord=norms)}<={np.linalg.norm(M, ord=1) * np.linalg.norm(x, ord=norms)}')


#6
print("#6\n")
submatrix = M[:10, :8]
submatrix = np.linalg.pinv(submatrix)
zero_matrix = np.zeros((3, 5))

for matrix in [M2, submatrix, zero_matrix]:
  for norm in [1, 2, np.inf, 'nuc', 'fro']:
    A = np.linalg.norm(matrix, ord=norm)
    A_inv = np.linalg.norm(np.linalg.pinv(matrix), ord=norm)
    print(f"\nNorm: {norm}\n||A^(-1)||: {A_inv}, ||A||^(-1): {A**(-1)}\nCondition: {A_inv >= A**(-1)}\n")

#7*
print("#7*\n")
with pd.ExcelWriter('sem_9_task_7.xlsx') as writer:
  for n in range(1, 16):
      norms = []
      for m in range(1, 16):
        matrix_nm = M[:n, :m]
        norms.append([np.linalg.norm(matrix_nm, ord=1),
                     np.linalg.norm(matrix_nm, ord=2),
                     np.linalg.norm(matrix_nm, ord=np.inf),
                     np.linalg.norm(matrix_nm, ord='nuc'),
                     np.linalg.norm(matrix_nm, ord='fro')])
      dfn = pd.DataFrame(norms,index=range(1, 16), columns=['1', '2', 'np.inf', 'nuc', 'fro'])
      dfn.to_excel(writer, sheet_name=str(n))
files.download('sem_9_task_7.xlsx')