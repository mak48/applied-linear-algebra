import numpy as np
import sympy, scipy.linalg
#1
print("#1\n")
A = np.array([[2,-3,1],[3,-5,-1],[5,0,4]])
B = np.array([5,2,0])
C = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
for i in range(3):
  for j in range(3):
    C[i][j]=A[i][j]
  C[i][3]=B[i]
print(A,B,C,sep='\n\n')

#2
print("#2\n")
A = np.arange(3,18,1).reshape((5, 3))
print(A)
AT = A.T
print('\n',AT)
A2 = A*2
print('\n',A2)
A1 = np.concatenate((A,A2), axis=1)
print('\n',A1)
A3 = np.vstack((A[1],A[-1],A))
print('\n',A3)

#3
print("#3\n")
def print_row(A,n=0):
  print(A[n])

print_row(A)
for i in range(1,5):
  print_row(A,i)
print('\n')
print_row(A,-1)


#3*
print("#3*\n")
def concate(A,n=0):
  s = A[0:,n].reshape(5,1)
  A1=np.hstack((A,s))
  print('A=',A,'row=',s,'result=',A1,sep='\n\n')
concate(A,1)
print('\n')
concate(A)

#4
print("#4\n")
def m(A,*args):
  n = len(args)
  result = A
  for i in range(n):
    result = np.vstack((result,A[args[i]-1]))
  print(result,'\n')
A = np.array([[2,-3,1],[3,-5,-1],[5,0,4]])
m(A)
m(A,2)
m(A,1,3)


#5
print("#5\n")
A = np.array([[-2.1,-3,4.5],[3,5,-1.5],[7.2,1,0]])
B = np.array([10.23,20.65,9.34])
x = np.linalg.solve(A,B)
print(x, np.allclose(A@x,B), sep='\n')

#6
print("#6\n")
A = np.array([[1,2,5,9,1],[3,1,2,5,0]])
pseudo_1 = np.linalg.pinv(A)
print('numpy\n',pseudo_1,'\n')

A_0 = scipy.linalg.pinv(A)
print('scipy\n',A_0)

A_1 = sympy.Matrix(A)
pseudo_2 = A_1.pinv()
print('\nsympy')
display(pseudo_2.evalf(8))

#7
print("#7\n")
A = sympy.Matrix([[1,2,-5,9,1],[4,3,-3,4,1],[5,5,-8,13,2],[3,1,2,-5,0]])
A_rref = A.rref()
cols = A_rref[1]
k = len(cols)
B = A[:, cols]
print('B =')
display(B)
C = A_rref[0][:k, :]
print('C =')
display(C)
A_pinv_my = C.T * (C * C.T) ** (-1) * (B.T * B) ** (-1) * B.T
print('\n')
display('A+', A_pinv_my)
display(A_pinv_my.evalf(5))
A = np.array([[1,2,-5,9,1],[4,3,-3,4,1],[5,5,-8,13,2],[3,1,2,-5,0]])
A_pinv = np.linalg.pinv(A)
print('\nA+ (numpy)\n',A_pinv)

#8
print("#8\n")
A = sympy.Matrix([[1,2,5,9,1],[3,1,2,5,0]])
A_rref = A.rref()
cols = A_rref[1]
k = len(cols)
B = A[:, cols]
print('B =')
display(B)
C = A_rref[0][:k, :]
print('C =')
display(C)
A_pinv_my = C.T * (C * C.T) ** (-1) * (B.T * B) ** (-1) * B.T
print('\n')
display('A+', A_pinv_my)


#9
print("#9\n")
def function(n,k):
  n*=2
  A = np.zeros((n,n+k))
  for i in range(n):
    for j in range(n):
      if i==j and i%2==0:
        A[i][j]=1
      elif i==j and i%2==1:
        A[i][j]=3
      elif j-1>=0 and A[i][j-1]==1:
        A[i][j]=2
      elif i-1>=0 and A[i-1][j]==1:
        A[i][j]=4
  for m in range(n):
    for l in range(n,n+k):
      A[m][l]=1
  print(A)

function(2,1)