import numpy as np

# A = np.array([[0, 1, 0, 2, 0], 
#     [1, 0, 2, -1, 0],
#     [0, 0, -1, -2, 1]], dtype=float)
# B= np.array([4, 4, 6], dtype=float)
# C= np.array([2,-1,3,-10,1])
A = np.array([[30, 40, 20, 10], 
    [10, 50, 40, 30],
    [40, 10, 10, 20]], dtype=float)
B= np.array([7000, 6000, 4000], dtype=float)
C= np.array([-800,-900,-600,-700])

class Simplex():
  
  def __init__(self, A, B, C): 
        self.A=A
        self.B=B
        self.C=C
        self.m=A.shape[0]
        self.n=A.shape[1]
        self.primary_n=self.n
# Если на вход канон вид

  # # индексы базисных векторов
  # def get_basis_indices(self):
  #   identity_matrix = np.eye(min(self.n, self.m), dtype=int)
  #   self.basis_indices= []
  #   for j in range(self.m):
  #     identity_matrix_column=identity_matrix[:,j]
  #     for i in range(self.n):
  #       current_matrix_column=A[:,i]
  #       if (np.array_equal(identity_matrix_column, current_matrix_column)):
  #         self.basis_indices.append(i)
    

  # # C баз
  # def get_c_basis(self):
  #   self.c_basis=[]
  #   for k in self.basis_indices:
  #     c_basis_element=self.C[k]
  #     self.c_basis.append(c_basis_element)

  def add_basis(self):
    self.c_basis=[]
    self.basis_indices= []
    minside=min(self.n, self.m)
    identity_matrix = np.eye(minside)
    self.A=np.hstack((self.A, identity_matrix))
    self.C=np.hstack((self.C, np.zeros(minside)))
    for i in range(minside):
      self.n=self.n+1
      self.basis_indices.append(self.n-1)
      self.c_basis.append(0)
      
  # Оценки
  def get_evaluation(self):
    self.evaluation_list=[]
    evaluation_p0=((self.B*self.c_basis).sum()-0)
    self.evaluation_list.append(evaluation_p0)
    for j in range(self.n):
      # Cб *BPj - cj
      column_evaluation=(self.A[:,j]*self.c_basis).sum()-self.C[j]
      self.evaluation_list.append(column_evaluation)

  # тетта
  def get_theta(self, vector):
    theta=99999
    theta_index=0
    for i in range(vector.size):
      if ((vector[i] > 0) and (self.B[i] / vector[i]< theta)):
        theta=self.B[i] / vector[i]
        theta_index=i
    return theta_index

  # Выбор вектора выходящего из базиса, индексы элемента пересчета
  def get_new_element(self):
    max_evaluation_index = np.argmax(self.evaluation_list[1:])
    vector=self.A[:, max_evaluation_index]
    theta_index_row = self.get_theta(vector)
    return theta_index_row, max_evaluation_index
  # Пересчет таблицы
  def recalculating_table(self):
    ind_row , ind_col = self.get_new_element()
    
    # Замена индекса базиса и Cб
    self.basis_indices[ind_row] = ind_col
    self.c_basis[ind_row] = self.C[ind_col]
    # Пересчет ведущей стоки
    new_row=np.divide(self.A[ind_row , :], self.A[ind_row , ind_col])
    self.B[ind_row]=self.B[ind_row]/self.A[ind_row , ind_col]
    self.A[ind_row]=new_row
    # Пересчет остальных строк
    for i in range(self.m):
      if i !=ind_row :
        mnozh=(-self.A[i,ind_col])/self.A[ind_row , ind_col]
        new_b=self.B[ind_row]*mnozh+self.B[i]
        new_row=np.multiply(self.A[ind_row],mnozh)+self.A[i]
        self.A[i]=new_row
        self.B[i]=new_b

  def set_x(self):
    x=np.zeros(self.n)
    # for i in self.basis_indices:
    #   x[i]=self.B[self.basis_indices.index(i)]
    for i in range(self.m):
      x[self.basis_indices[i]]=self.B[i]
    return x[:self.primary_n]

  def forward(self):
    self.add_basis()
    self.get_evaluation()
    print(self.evaluation_list)
    while (max(self.evaluation_list[1:])>0):
      self.recalculating_table()
      self.get_evaluation()
    print(self.set_x(), "x")
    print(self.evaluation_list[0],"f(x)")
smpl= Simplex(A,B,C)
smpl.forward()
