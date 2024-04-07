#1
M = np.ones((10, 10))
for i in range(10):
  for j in range(10):
      M[i][j]= (-1)**(i*j)*(i-j)-(i+j)**2
for i in range(3,11):
  M_cur = M[:i,:i]
  print("i=",i)
  print( *[f'cond(M, {item})  = {round(np.linalg.cond(M_cur, p=item), 2)}'\
                    for item in (1, 2, np.inf, 'fro')], sep=', ')
print("sympy")
for i in range(3,6):
  M_cur = M[:i,:i]
  display(Latex(f'cond\_sympy = \
    {sympy.latex(sympy.Matrix(M_cur).condition_number().simplify())}'))

#2
uploaded = files.upload()
A2_ = pd.read_excel("sem_11_task_2.xlsx",sheet_name = "A2",index_col=0)
A2 = A2_.to_numpy()
b_ = pd.read_excel("sem_11_task_2.xlsx",sheet_name = "b2", header = None,index_col=0)
b = b_.to_numpy()
if (np.linalg.matrix_rank(A2)==20):
  print("матрица не вырождена")
else:
  print("матрица вырождена")
print("rank=",np.linalg.matrix_rank(A2))
if (np.linalg.matrix_rank(A2)==np.linalg.matrix_rank(np.hstack([A2,b]))):
  print("По теореме Кронекера Капелли совместна")
else:
  print("По теореме Кронекера Капелли несовместна")
print("число обусловленности =",np.linalg.cond(A2))
print("X=",np.linalg.solve(A2,b))

def computational_experiment(A, b, epsilon):
   b_epsilon = b + (epsilon * np.random.rand(*b.shape))
   X = np.linalg.solve(A, b_epsilon)
   X_ = np.linalg.solve(A, b)
   norm_X_diff = np.linalg.norm(X - X_)
   norm_b_diff = np.linalg.norm(b_epsilon - b)
   delta_b = norm_b_diff / np.linalg.norm(b)
   delta_X = norm_X_diff / np.linalg.norm(X)
   X_A =  np.linalg.norm(A) * np.linalg.norm(np.linalg.inv(A))
   res = delta_b/X_A <= delta_X <= X_A*delta_b
   return X, norm_X_diff, norm_b_diff, res

X, norm_X_diff, norm_b_diff, res = computational_experiment(A2, b, 0.001)
print("computational_experiment\n", "X =", X, "норма разности решений", norm_X_diff, "норма разности векторов", norm_b_diff, "неравенство", res)

#3
def calculate_error(A, A_inv, r):
    epsilon = np.random.uniform(-r, r, A.shape)
    A_distorted = A + epsilon
    A_inv_distorted = np.linalg.inv(A_distorted)
    delta_A_inv = np.linalg.norm(A_inv - A_inv_distorted) / np.linalg.norm(A_inv)
    delta_epsilon = np.linalg.norm(epsilon) / np.linalg.norm(A)
    return delta_A_inv, delta_epsilon

A = A2
A_inv = np.linalg.inv(A)
r_values = [0.001, 0.005, 0.01, 0.02, 0.05]

delta_A_inv_values = []
delta_epsilon_values = []

for r in r_values:
    delta_A_inv, delta_epsilon = calculate_error(A, A_inv, r)
    delta_A_inv_values.append(delta_A_inv)
    delta_epsilon_values.append(delta_epsilon)

chi_A = np.linalg.cond(A)
error_estimate = [chi_A * delta_epsilon / (1 - chi_A * delta_epsilon) for delta_epsilon in delta_epsilon_values]

plt.figure()
plt.scatter(delta_epsilon_values, delta_A_inv_values, color='green', label='δA^-1')
plt.scatter(delta_epsilon_values, error_estimate, color='purple', label='Оценка погрешности')
plt.xlabel('δε')
plt.ylabel('Error')
plt.legend()
plt.show()

#4
def check(A, b, num_experiments, Xn):
    count = 0
    result = []
    for i in range(num_experiments):
        delta_A = np.random.rand(*A.shape) * 0.02
        delta_b = np.random.rand(*b.shape) * 0.02
        A_ = A+delta_A
        b_ = b+delta_b
        X = np.linalg.solve(A, b)
        X_ = np.linalg.solve(A_, b_)
        Xn.append(X_)
        norm_X_diff = np.linalg.norm(X_ - X)
        norm_b_diff = np.linalg.norm(b_ - b)
        del_b = norm_b_diff / np.linalg.norm(b)
        delta_X = norm_X_diff / np.linalg.norm(X)
        X_A =  np.linalg.norm(A_) * np.linalg.norm(np.linalg.inv(A_))
        res = del_b/X_A <= delta_X <= X_A*del_b
        if res==True:
            count += 1
        euclidean_norm_Xn_X = np.linalg.norm(X_ - X)
        spectral_norm_A = np.linalg.norm(A_ - A)
        euclidean_norm_bn_b = np.linalg.norm(b_ - b)
        result.append([euclidean_norm_Xn_X, spectral_norm_A, euclidean_norm_bn_b, delta_X])
    return count / num_experiments, result

X = np.linalg.solve(A2, b)
Xn = [X]

count, result = check(A2, b, 100, Xn)
Xn = np.reshape(Xn,(101,20))

with pd.ExcelWriter('experiment4.xlsx') as writer:
    A_ext = np.column_stack((A2, b))
    df_A = pd.DataFrame(A_ext)
    df_A.to_excel(writer, sheet_name='SLAE', index=False)

    df_X = pd.DataFrame(Xn)
    df_X.to_excel(writer, sheet_name='Solutions', index=False)

    df = pd.DataFrame(result, columns=['||Xn-X*||', '||An-A||', '||bn-b||', 'delta_X'])
    df.to_excel(writer, sheet_name='Result', index=False)
files.download('experiment4.xlsx')


#5
uploaded = files.upload()
A1_ = pd.read_excel("sem_8_Example_1.xlsx",header = None)
A1 = A1_.to_numpy()
u1, sigma1, vh1 = scipy.linalg.svd(A1)
k = 130
A1r = u1[:, :k] @ np.diag(sigma1[:k]) @ vh1[:k, :]
np.set_printoptions(threshold=sys.maxsize)
display(np.round(A1r))
print(f'scipy.linalg.norm(A1 - A1r) = {scipy.linalg.norm(A1 - A1r, 2)}, \
ранг A1r = {np.linalg.matrix_rank(A1r)}')

#6
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))

img = io.imread(fn, as_gray=True)
u, s, vh = np.linalg.svd(img, full_matrices=True)
k = min(img.shape) // 2
img_half = u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]
plt.imshow(img_half, cmap='Greys_r')
k = min(img.shape)
for i in (10, 15, 25):
    t = k * i // 100
    imgplot = plt.imshow(u[:, :t] @ np.diag(s[:t]) @ vh[:t, :], cmap='Greys_r')
    print(f'{i}%')
    plt.show()

#7
uploaded = files.upload()
A3 = pd.read_excel('sem_11_task_7.xlsx', header=None)
A7 = A3.to_numpy()

U, S, Vt = np.linalg.svd(A7)

for rank in range(1,A7.shape[0]):
  Ar7 = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
  if np.linalg.norm(A7 - Ar7) <= 0.1 * np.linalg.norm(A7):
      break

print("Первые 10 наибольших сингулярных чисел:", S[:10])
print("Ранг матрицы Ar7:", rank)

with pd.ExcelWriter('sem_11_task_7_ans.xlsx') as writer:
    pd.DataFrame(U).to_excel(writer, sheet_name='U')
    pd.DataFrame(np.diag(S)).to_excel(writer, sheet_name='S')
    pd.DataFrame(Vt).to_excel(writer, sheet_name='Vt')
    pd.DataFrame(Ar7).to_excel(writer, sheet_name='Ar7')
files.download('sem_11_task_7_ans.xlsx')