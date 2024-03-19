from google.colab import files


#1
print("#1\n")
A = sympy.Matrix([[1, 0, 0, -2],
                  [0, 1, 0, 1],
                  [0, 0, 3, 1]])
A_star_A = A.T * A
display(Latex(f'A^*A = {latex(A_star_A)}'))
A_star_A_sympy_ev = A_star_A.eigenvects()
display(Latex(f'Собственные\ векторы\ с\ \
собственными\ числами\ {latex(A_star_A_sympy_ev)}'))
A_star_A_sympy_eigenvalues = [num for num, mult, vectors in A_star_A_sympy_ev]
A_star_A_sympy_eigenvectors = [[vector.normalized() for vector in vectors]\
                               for num, mult, vectors in A_star_A_sympy_ev]
display(Latex(f'Собственные\ числа\ {latex(A_star_A_sympy_eigenvalues)}'),
        Latex(f'Нормализованные\ собственные\ \
        векторы\ {latex(A_star_A_sympy_eigenvectors)}'))
e0, e1, e5, e11 = A_star_A_sympy_eigenvectors
e0, = e0
e1, = e1
e5, = e5
e11, = e11
display(Latex('e_0 = {}, e_1 = {}, \
e_5 = {}, e_{{11}} = {}'.format(*map(latex, (e0, e1, e5, e11)))))
P = e11.row_join(e5).row_join(e1).row_join(e0)
sigma = (sympy.sqrt(11), sympy.sqrt(5), 1)
f1, f2, f3= [A * ei / sigma[i] for i, ei in enumerate((e11, e5, e1))]
Q = f1.row_join(f2).row_join(f3)
Sig = sympy.Matrix([[sympy.sqrt(11), 0, 0, 0], [0, sympy.sqrt(5), 0,0], [0, 0, 1, 0]])
display(Latex('Q = {}, Sig = {}, \
P = {}, Q  Sig  P^T =\
{}'.format(*map(latex, (Q, Sig, P, Q * Sig * P.T)))))
display(Latex('Q = {}, Sig = {}, P = {}\\\\Q  Sig  P^T = {}\
'.format(*[latex(item.evalf(3)) for item in (Q, Sig, P, Q * Sig * P.T)])))
A = np.array([[1, 0, 0, -2],
              [0, 1, 0, 1],
              [0, 0, 3, 1]])
A_star_A = A.T @ A
display(Latex(f'A^*A = {latex(A_star_A)}'),
        A_star_A)
A_star_A_eigen_vals, A_star_A_eigen_vects = np.linalg.eig(A_star_A)
print('Собственные числа \
{},\nсобственные\ векторы \n\
{}'.format(A_star_A_eigen_vals.round(2), A_star_A_eigen_vects.round(2)))
A_star_A_eigen_vals.sort()
A_star_A_eigen_vals_reversed = np.flip(A_star_A_eigen_vals)
display(A_star_A_eigen_vals.round(2), A_star_A_eigen_vals_reversed.round(2))
sigmas = [round(np.sqrt(item), 1) for item in A_star_A_eigen_vals_reversed if item >= 0]
Sigma = np.hstack((np.diag(sigmas), np.zeros((3, 1))))
e4, e0, e11, e12 = [item.reshape((4, 1)) for item in  A_star_A_eigen_vects.T]
A_star_A_eigen_vects_new = [e4, e0, e11, e12]
print(*[item.round(2).T for item in A_star_A_eigen_vects], sep='\n')
sigma = (sympy.sqrt(11), sympy.sqrt(5), 1)
f1, f2, f3 = [A @ ei / sigma[i] for i, ei in enumerate((e11, e5, e1))]
Q = np.hstack((f1, f2, f3))
Sig = np.hstack((np.diag(sigma), np.zeros((3, 1))))
display(*[item.evalf(3) for item in map(Matrix, (Q, Sig, P, Q @ Sig @ P.T))])



#2
print("#2\n")
Sigma_plus = np.vstack((np.diag([1 / item for item in sigma]), np.zeros((1, 3))))
A_pinv_my = P @ Sigma_plus @ Q.T
print("A_pinv_my")
display(A_pinv_my.evalf(3))
print("np.linalg.pinv(A)")
A_s = np.array(A).astype(np.float64)
A_lin = np.linalg.pinv(A_s)
display(A_lin)
print("матрицы A_pinv_my и np.linalg.pinv(A) равны")

#3
print("#3\n")
Q, sigma, P = np.linalg.svd(A_s, full_matrices=True)
Sig = np.hstack((np.diag(sigma), np.zeros((3, 1))))
Sigma_plus = np.vstack((np.diag([1 / item for item in sigma]), np.zeros((1, 3))))
display(Latex(f'P^T = {sympy.latex(P.round(2))}'),
Latex(f'\Sigma = {sympy.latex(Sig)}'),
Latex(f'Q = {sympy.latex(Q.round(2))}'),
Latex(f'Q\Sigma P^T = {sympy.latex((Q @ Sig @ P).round(2))}'))

print("\nA_pinv_my=")
A_pinv_my = P.T @ Sigma_plus @ Q.T
display(A_pinv_my)


#4
print("#4\n")
print("Загрузите matrix_task_4.xlsx")
uploaded = files.upload()
A = pd.read_excel("matrix_task_4.xlsx", header = None)
A = A.to_numpy()

print("Загрузите matrix_task_4_chol.xlsx")
uploaded = files.upload()
A_chol = pd.read_excel("matrix_task_4_chol.xlsx", header = None)
A_chol = A_chol.to_numpy()

#Холецкого
A_chol = sympy.Matrix(A_chol)
LA = A_chol.cholesky(hermitian=False)

A = sympy.Matrix(A)

#lu
LU_L, LU_U, LU_P = A.LUdecomposition()

#ldl
LDL_L, LDL_D = A_chol.LDLdecomposition(hermitian=False)

#qr
Q, R = A.QRdecomposition()


LA = np.array(LA).astype(np.float64)
LA = pd.DataFrame(LA)
LU_L = np.array(LU_L).astype(np.float64)
LU_L = pd.DataFrame(LU_L)
LU_U = np.array(LU_U).astype(np.float64)
LU_U = pd.DataFrame(LU_U)
LDL_L = np.array(LDL_L).astype(np.float64)
LDL_L = pd.DataFrame(LDL_L)
LDL_D = np.array(LDL_D).astype(np.float64)
LDL_D = pd.DataFrame(LDL_D)
Q = np.array(Q).astype(np.float64)
Q = pd.DataFrame(Q)
R = np.array(R).astype(np.float64)
R = pd.DataFrame(R)

with pd.ExcelWriter("matrix_task_4_ans.xlsx") as writer:
    LA.to_excel(writer, sheet_name='cholesky', header=False, index=False)
    LU_L.to_excel(writer, sheet_name='lu L matrix', header=False, index=False)
    LU_U.to_excel(writer, sheet_name='lu U matrix', header=False, index=False)
    LDL_L.to_excel(writer, sheet_name='ldl L matrix', header=False, index=False)
    LDL_D.to_excel(writer, sheet_name='ldl D matrix', header=False, index=False)
    Q.to_excel(writer, sheet_name='qr Q matrix', header=False, index=False)
    R.to_excel(writer, sheet_name='qr R matrix', header=False, index=False)
files.download("matrix_task_4_ans.xlsx")