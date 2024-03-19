import numpy as np
import numpy.linalg
import pandas as pd
from dataclasses import dataclass, field
from google.colab import files
class SLAE:
    """My SLAE Class"""
    def __init__(self, name, a, b, var_num=0, eq_num=0):
        self.name = name
        self.a = a
        self.b = b
        self.var_num = var_num
        self.eq_num = eq_num

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_var_num(self):
        if self.var_num == 0:
            self.var_num = self.a.shape[1]
        return self.var_num

    def get_eq_num(self):
        if self.eq_num == 0:
            self.eq_num = self.a.shape[0]
        return self.eq_num

    def get_dim(self):
        return (self.get_eq_num(), self.get_var_num())

    def set_b(self, new_b):
        if new_b.shape[0] == self.a.shape[0]:
            self.b = new_b
        else:
            print("Dimension mismatch between matrix A and vector b")

    def set_b_zero(self):
        self.b = np.zeros(self.a.shape[0])

    def x(self):
        A = np.hstack((self.a,self.b))
        if np.linalg.matrix_rank(self.a)==np.linalg.matrix_rank(A):
          return True, np.linalg.solve(self.a, self.b)
        else:
          return False, np.array([])

class SLAEhomogeneous(SLAE):
    def __init__(self, name, a, var_num=0, eq_num=0):
        super().__init__(name, a, None, var_num, eq_num)

    def get_b(self):
        return np.zeros(self.a.shape[0])

    def set_b(self, new_b):
        print("b = 0 in homogeneous SLAE, use get_b instead")

    def x(self):
        if np.linalg.matrix_rank(self.a) < self.a.shape[1]:
            return False, np.array([])
        else:
            return True, np.zeros(self.a.shape[1])

class SLAEsquare(SLAE):
    def __init__(self, name, a, b, var_num=0, eq_num=0):
        super().__init__(name, a, b, var_num, eq_num)
        self.singular = None
        self.square = None
        self.a_inv = None

    def is_square(self):
        if self.square is None:
            self.square = self.a.shape[0] == self.a.shape[1]
        return self.square

    def is_singular(self):
        if self.singular is None:
            self.singular = not np.linalg.matrix_rank(self.a) == self.a.shape[1]
        return self.singular

    def get_inv(self):
        if self.is_square() and not self.is_singular():
            self.a_inv = np.linalg.inv(self.a)
        return self.a_inv

    def x(self):
        if self.is_square() and not self.is_singular():
            return True, np.linalg.solve(self.a, self.b)
        else:
            return False, np.array([])
uploaded = files.upload()
A1_ = pd.read_excel("ab1.xlsx",sheet_name = "A", header = None)
A1 = A1_.to_numpy()
b1_ = pd.read_excel("ab1.xlsx",sheet_name = "b", header = None)
b1 = b1_.to_numpy()

SLAE_1_1 = SLAE('SLAE_1_1', A1, b1)
print(SLAE_1_1)
print(f"name = {SLAE_1_1.name}\na = {SLAE_1_1.a}\nb = {SLAE_1_1.b}")

print(f"a = {SLAE_1_1.a}\nget_a = {SLAE_1_1.get_a()}\nb = {SLAE_1_1.b}\nget_b = {SLAE_1_1.get_b()}")

SLAE_1_1.set_b(np.array([[0],[0],[0],[0]]))
print("b = ", SLAE_1_1.b)

SLAE_1_1.set_b(np.array([[4],[5],[6],[7]]))
print("b = ", SLAE_1_1.b)

SLAE_1_1.set_b(np.array([[-1],[2],[-3],[4],[5],[6],[7]]))
print("b = ", SLAE_1_1.b)
print("x = ", SLAE_1_1.x())

SLAE_homo_1 = SLAEhomogeneous('SLAE_homo_1',A1)
print(SLAE_homo_1)
print(SLAE_homo_1.name, "\n", SLAE_homo_1.a)

print(SLAE_homo_1.a, "\n", SLAE_homo_1.get_a())

print(SLAE_homo_1.b)
SLAE_homo_1.set_b(np.array([[1],[2],[3],[4],[5],[6],[7]]))
print(SLAE_homo_1.b)

SLAE_sq_1 = SLAEsquare('SLAE_sq_1',A1,b1)
print(SLAE_sq_1)
print("a = ",SLAE_sq_1.a,"\nget_a",SLAE_sq_1.get_a(),"\nx=",SLAE_sq_1.x())

SLAE_sq_1.set_b(np.array([[0],[0],[0],[0]]))
print("b = ",SLAE_sq_1.b,"\nx = ",SLAE_sq_1.x())
SLAE_sq_1.set_b(np.array([[1],[2],[3],[4],[5]]))
print("b = ",SLAE_sq_1.b)

uploaded = files.upload()
A2_ = pd.read_excel("ab2.xlsx",sheet_name = "A", header = None)
A2 = A2_.to_numpy()
b2_ = pd.read_excel("ab2.xlsx",sheet_name = "b", header = None)
b2 = b2_.to_numpy()

SLAE_sq_2 =SLAEsquare('SLAE_sq_2',A2,b2)
print("a = ",SLAE_sq_2.a,"\nget_a",SLAE_sq_2.get_a(),"\nx=",SLAE_sq_2.x())

SLAE_sq_2.set_b(np.array([[0],[0],[0],[0],[0],[0]]))
print("b = ",SLAE_sq_2.b,"\nx = ",SLAE_sq_2.x())
SLAE_sq_2.set_b(np.array([[1],[2],[3],[4],[5],[6]]))
print("b = ",SLAE_sq_2.b)