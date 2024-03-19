import numpy as np
from google.colab import files
import pandas as pd
import sympy
import copy

#1
print("#1\n")
A = np.array([[2,-3,1,4],[3,-5,-1,5],[5,0,4,25]])
L = np.vstack((A[:,0],A[:,1],A[:,2])).T
P = np.vstack((A[:,3]))
X = np.linalg.solve(L,P)
print(X,'\n',np.allclose(P,L@X))

#2
print("#2\n")
uploaded = files.upload()
A2_ = pd.read_excel("SLAE_task_2.xlsx",sheet_name = "A2", header = None)
A2 = A2_.to_numpy()
print(A2)
pseudoa = np.linalg.pinv(A2)
print(pseudoa[:,0])
dfpseudo = pd.DataFrame(pseudoa)
with pd.ExcelWriter("SLAE_task_2_solution.xlsx") as writer:
  dfpseudo.to_excel(writer,sheet_name='A2',header=False,index=False)
  files.download("SLAE_task_2_solution.xlsx")

#3
print("#3\n")
A = sympy.Matrix(A2)
A_rref = A.rref()
cols = A_rref[1]
k = len(cols)
B = A[:, cols]
C = A_rref[0][:k, :]
A_pinv_my = C.T * (C * C.T) ** (-1) * (B.T * B) ** (-1) * B.T
print('A+=')
display(A_pinv_my.evalf(5))
print('B =')
display(B)
print('C =')
display(C)
A = np.array(A_pinv_my.evalf(5))
B = np.array(B)
C = np.array(C)
Adf = pd.DataFrame(A)
Bdf = pd.DataFrame(B)
Cdf = pd.DataFrame(C)
with pd.ExcelWriter("task_3_solution.xlsx") as writer:
    Bdf.to_excel(writer, sheet_name='B', header=False, index=False)
    Cdf.to_excel(writer, sheet_name='C', header=False, index=False)
    Adf.to_excel(writer, sheet_name='A3pinv', header=False, index=False)
files.download("task_3_solution.xlsx")

#4
print("#4\n")
uploaded = files.upload()
A3pinv = pd.read_excel("task_3_solution.xlsx", sheet_name='A3pinv', header=None)
A3pinv = A3pinv.to_numpy()
A4 = A3pinv@A3pinv.T
file_name_res = 'task_4.xlsx'
result=copy.deepcopy(A4)
with pd.ExcelWriter(file_name_res) as writer:
  Xdf = pd.DataFrame(result)
  Xdf.to_excel(writer, sheet_name="1", header=False, index=False)
  for i in range(2,10):
    result=result@A4
    Xdf = pd.DataFrame(result)
    Xdf.to_excel(writer, sheet_name=str(i), header=False, index=False)
files.download("task_4.xlsx" )

#5
print("#5\n")
def read_write_matrix(filename,*args):
  matrix=[]
  pseudo=[]
  for i in range(1,10):
    mdf = pd.read_excel(filename, sheet_name=str(i), header=None)
    m = mdf.to_numpy()
    p = np.linalg.pinv(m)
    matrix.append(m)
    pseudo.append(p)
  return matrix,pseudo

uploaded = files.upload()
args=[]
for i in range(1,10):
  args.append(str(i))
matrix,pseudo = read_write_matrix("task_4.xlsx",args)
for i in range(len(matrix)):
  print("diagonal: ")
  rows=[]
  for j in range(len(matrix[i][0])):
    rows.append(round(matrix[i][j][j],5))
  print(rows, sep=' ')
  print('\n',"last row of pseudo: ",pseudo[i][:][-1].T,'\n')

#6
print("#6\n")
def function(filename):
  mdf = pd.read_excel(filename, header=None)
  m = mdf.to_numpy()
  ans=[len(m),len(m[0])]
  mini=10**5
  maxi=-10**5
  sum=0
  pol=0
  for i in range(len(m)):
    for j in range(len(m[0])):
      sum+=m[i][j]
      if m[i][j]<mini:
        mini=m[i][j]
      if m[i][j]>maxi:
        maxi=m[i][j]
      if m[i][j]>0:
        pol+=1
  ans.append(sum)
  ans.append(pol)
  ans.append(mini)
  ans.append(maxi)
  return ans

uploaded = files.upload()
result=function("task_6.xlsx")
print("число строк матрицы: ",result[0], "\nчисло столбцов матрицы: ", result[1],
      "\nсумма элементов матриц: ",result[2],
      "\nчисло положительных элементов матрицы: ", result[3],
      "\nминимальный и максимальный элемент матрицы:",result[4],result[5])

#7
print("#7\n")
def change(matrix,args):
  new_matrix = copy.deepcopy(matrix)
  for i in args:
    for j in range(len(matrix[0])):
      if matrix[i-1][j]>0:
        new_matrix[i-1][j]=0
  return new_matrix
strings=[2,4,5]
uploaded = files.upload()
with pd.ExcelWriter("task_7_result.xlsx") as writer:
  for i in range(1,10):
    m = pd.read_excel("task_4.xlsx",sheet_name = str(i), header = None)
    m = m.to_numpy()
    new_m=change(m,strings)
    m_df = pd.DataFrame(new_m)
    m_df.to_excel(writer, sheet_name=str(i), header=False, index=False)
    mpseudo = np.linalg.pinv(m)
    new_m=change(mpseudo,strings)
    m_df = pd.DataFrame(new_m)
    m_df.to_excel(writer, sheet_name="pseudo"+str(i), header=False, index=False)
files.download("task_7_result.xlsx")