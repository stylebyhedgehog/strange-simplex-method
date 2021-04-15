import numpy as np
import copy
import math

# A = np.array([[0, 1, 0, 2, 0], 
#     [1, 0, 2, -1, 0],
#     [0, 0, -1, -2, 1]], dtype=float)
# B= np.array([4, 4, 6], dtype=float)
# C= np.array([2,-1,3,-10,1])
# A = np.array([[12, 11, 13, 15], 
#     [2, 5, 4, 3],
#     [4, 13, 1, 2]], dtype=float)
# B= np.array([44, 14, 21], dtype=float)
# C= np.array([4, 5, 6, 7])
# A = np.array([[30, 40, 20, 10], 
#     [10, 50, 40, 30],
#     [40, 10, 10, 20]], dtype=float)
# B= np.array([7000, 6000, 4000], dtype=float)
# C= np.array([800,900,800,700])
A = np.array([[3, 8],
              [2, 1],
              [-1, 1]], dtype=float)
B = np.array([24, 8, 2], dtype=float)
C = np.array([4, 3])


class Simplex:

    def __init__(self, A, B, C, key_k=0):
        self.key_k=key_k
        self.A = A
        self.B = B
        self.primary_B = copy.copy(B)
        # постановка задачи на максимум а реализация на минимум
        self.C = (-1)*C
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.primary_n = self.n
        self.theta_first_iteration=1

    # Если на вход канон вид

    # индексы базисных векторов
    def get_basis_indices(self):
      identity_matrix = np.eye(min(self.n, self.m), dtype=int)
      self.basis_indices= []
      for j in range(self.m):
        identity_matrix_column=identity_matrix[:,j]
        for i in range(self.n):
          current_matrix_column=self.A[:,i]
          if (np.array_equal(identity_matrix_column, current_matrix_column)):
            self.basis_indices.append(i)

    # C баз
    def get_c_basis(self):
      self.c_basis=[]
      for k in self.basis_indices:
        c_basis_element=self.C[k]
        self.c_basis.append(c_basis_element)

    # Если на вход не канон вид
    def add_basis(self):
        self.c_basis = []
        self.basis_indices = []
        minside = self.m
        identity_matrix = np.eye(minside)
        self.A = np.hstack((self.A, identity_matrix))
        self.C = np.hstack((self.C, np.zeros(minside)))
        for i in range(minside):
            self.n = self.n + 1
            self.basis_indices.append(self.n - 1)
            self.c_basis.append(0)
        self.primary_basis_indices = copy.copy(self.basis_indices)
            # save

    # Оценки
    def get_evaluation(self):
        self.evaluation_list = []
        evaluation_p0 = ((self.B * self.c_basis).sum() - 0)
        self.evaluation_list.append(evaluation_p0)
        for j in range(self.n):
            # Cб *BPj - cj
            column_evaluation = (self.A[:, j] * self.c_basis).sum() - self.C[j]
            self.evaluation_list.append(column_evaluation)

    # тетта
    def get_theta(self, vector):
        theta = 99999
        theta_index = 0
        if(self.key_k !=0 and self.theta_first_iteration==1):
          self.theta_first_iteration=0
          for i in range(self.B.size):
            if self.B[i]<0:
              return i

        for i in range(vector.size):
            if ((vector[i] > 0) and (self.B[i] / vector[i] < theta)):
                theta = self.B[i] / vector[i]
                theta_index = i
        return theta_index

    # Выбор вектора выходящего из базиса, индексы элемента пересчета
    def get_new_element(self):
        max_evaluation_index = np.argmax(self.evaluation_list[1:])
        vector = self.A[:, max_evaluation_index]
        theta_index_row = self.get_theta(vector)
        return theta_index_row, max_evaluation_index

    # Пересчет таблицы
    def recalculating_table(self):
        ind_row, ind_col = self.get_new_element()

        # Замена индекса базиса и Cб
        self.basis_indices[ind_row] = ind_col
        self.c_basis[ind_row] = self.C[ind_col]
        # Пересчет ведущей стоки
        new_row = np.divide(self.A[ind_row, :], self.A[ind_row, ind_col])
        self.B[ind_row] = self.B[ind_row] / self.A[ind_row, ind_col]
        self.A[ind_row] = new_row
        # Пересчет остальных строк
        for i in range(self.m):
            if i != ind_row:
                mnozh = (-self.A[i, ind_col]) / self.A[ind_row, ind_col]
                new_b = self.B[ind_row] * mnozh + self.B[i]
                new_row = np.multiply(self.A[ind_row], mnozh) + self.A[i]
                self.A[i] = new_row
                self.B[i] = new_b

    def set_x(self):
        x = np.zeros(self.n)
        # for i in self.basis_indices:
        #   x[i]=self.B[self.basis_indices.index(i)]
        for i in range(self.m):
            x[self.basis_indices[i]] = self.B[i]

        if self.key_k==0:
          return x[:self.primary_n]
        else:
          return x[:(self.n-self.m)]

    def print_info(self, i):
        print("Итерация - ", i)
        print(self.A, " - A ")
        # постановка задачи на максимум а реализация на минимум
        print("x - ", self.set_x())
        print("Целевая функция - ", (-1)*self.evaluation_list[0])
        print("-"*33)

    def forward_simplex(self):
        self.add_basis()
        self.get_evaluation()
        i=0
        while max(self.evaluation_list[1:]) > 0:
            self.print_info(i)
            i+=1
            self.recalculating_table()
            self.get_evaluation()
        print("Результат: ")
        self.print_info(i)
        #Данные для анализа на чувствительность
        return self.primary_basis_indices, self.A, self.m
      
    def forward_only_result(self):
        if self.key_k==0:
          self.add_basis()
        else:
          self.get_basis_indices()
          self.get_c_basis()
        self.get_evaluation()
        while max(self.evaluation_list[1:]) > 0:
            self.recalculating_table()
            self.get_evaluation()
      
        return self.set_x(), (-1)*self.evaluation_list[0]
   
            
class SensAnalysis:
  def __init__(self, primary_basis_indices, res_A, m, primary_A,primary_B, primary_C):
        self.primary_basis_indices = primary_basis_indices
        self.res_A = res_A
        self.primary_A= primary_A
        self.primary_B = primary_B
        self.primary_C = primary_C
        self.m = m
        self.smpl= smpl

  # Поиск допустимых значений
  def sens_analysis_1(self):
      resalted_Basix = []
      for i in self.primary_basis_indices:
          resalted_Basix.append((self.res_A[:, i]).tolist())
      resalted_Basix = np.transpose(np.array(resalted_Basix))
      delta_b = []
      for j in range(self.m):
          delta_b.append((resalted_Basix[j] * self.primary_B).sum())
      delta_b = np.array(delta_b)
      left_confines=[]
      right_confines=[]
      for k in range(delta_b.size):
          left = resalted_Basix[:, k]
          right = delta_b
          self.normalization(left, right, k,left_confines,right_confines)
   
      return left_confines, right_confines , resalted_Basix

  def normalization(self, left, right, indx, left_confines, right_confines):
        right = (-1)*right
        mensh_ravno = []
        bolsh_ravno = []
        for i in range(left.size):
            if (left[i] < 0):
                mensh_ravno.append((right[i] / left[i]))
            elif (left[i] > 0):
                bolsh_ravno.append(right[i] / left[i])
            else:
                None

        if mensh_ravno:
            min_mensh_ravno=float('{:.3f}'.format(min(mensh_ravno)+self.primary_B[indx]))
        else:
            min_mensh_ravno="не ограничено"
        if bolsh_ravno:
            max_bolsh_ravno=float('{:.3f}'.format(max(bolsh_ravno)+self.primary_B[indx]))
        else:
            max_bolsh_ravno="не ограничено"

        if((type(max_bolsh_ravno) is str or type(min_mensh_ravno) is str) or max_bolsh_ravno<min_mensh_ravno):
            # print(max_bolsh_ravno, "<=delta<=",min_mensh_ravno)
            left_confines.append(max_bolsh_ravno)
            right_confines.append(min_mensh_ravno)
            print(max_bolsh_ravno, "<=b",indx+1,"<=",min_mensh_ravno)
        else:
            left_confines.append("не изменимо")
            right_confines.append("не изменимо")

  def sens_analysis_2(self, left_confines, right_confines):
      simplex_x=[]
      simplex_result=[]
      vector_b=[]
      for z in range(self.m):

        if (left_confines[z] != "не ограничено" and left_confines[z] !="не изменимо"
        and right_confines[z] != "не ограничено" and right_confines[z] !="не изменимо"):

          for d in range(math.ceil(left_confines[z]), math.floor(right_confines[z])):
            new_B=copy.copy(self.primary_B)
            new_B[z]=d
            primary_new_B=copy.copy(new_B)
            smpl=Simplex(self.primary_A, new_B, self.primary_C)
            x , y = smpl.forward_only_result()
            simplex_x.append(x)
            vector_b.append(primary_new_B.tolist())
            simplex_result.append(y)
      if simplex_result:
        max_value=max(simplex_result)
        max_index=simplex_result.index(max_value)
        b_with_max_value=vector_b[max_index]
        x_with_max_value=simplex_x[max_index]
        print(max_value, "f(x)")
        print(x_with_max_value, "x")
        print(b_with_max_value, "b")
      else:
        print("текущее решение не улучшить")
        

  def get_coefficient_b(self, new_B, resalted_Basix):
      test=[]
      for j in range(self.m):
        test.append((resalted_Basix[j] * new_B).sum())
  
      return np.array(test)

  def sens_analysis_3(self, left_confines, right_confines, resalted_Basix):
      simplex_x=[]
      simplex_result=[]
      vector_b=[]
     
      for z in range(self.m):
        if (left_confines[z] != "не ограничено" and left_confines[z] !="не изменимо" and right_confines[z] != "не ограничено" and right_confines[z] !="не изменимо"):
          new_B=copy.copy(self.primary_B)
          new_B[z]=round(left_confines[z])-1
          new_B=self.get_coefficient_b(new_B, resalted_Basix)
          smpl=Simplex(self.res_A, new_B, np.hstack((self.primary_C, np.zeros(self.m))),1)
          x , y = smpl.forward_only_result()
          simplex_x.append(x)
          vector_b.append(new_B.tolist())
          simplex_result.append(y)

          new_B1=copy.copy(self.primary_B)
          new_B1[z]=round(right_confines[z])+1
          new_B1=self.get_coefficient_b(new_B1, resalted_Basix)
          smpl=Simplex(self.res_A, new_B1, np.hstack((self.primary_C, np.zeros(self.m))),1)
          x , y = smpl.forward_only_result()
          simplex_x.append(x)
          vector_b.append(new_B1.tolist())
          simplex_result.append(y)
      
      if simplex_result:
        max_value=max(simplex_result)
        max_index=simplex_result.index(max_value)
        b_with_max_value=vector_b[max_index]
        x_with_max_value=simplex_x[max_index]
        print(max_value, "f(x)")
        print(x_with_max_value, "x")
        print(b_with_max_value, "coef")
      else:
        print("текущее решение не улучшить")

 


  def forward_sens_analysis(self):
      print("Анализ на чувствительность: ")
      print("(1) Интервал допустимых значений ограничений запасов удобрений: ")
      left_confines, right_confines, resalted_Basix = self.sens_analysis_1()
      print("(2) Максимальное значение целевой функции при изменении ограничений в допустимом диапазоне: ")
      self.sens_analysis_2(left_confines,right_confines)
      print("(3) Максимальное значение целевой функции при изменении ограничений в диапазоне за пределами допустимого: ")
      self.sens_analysis_3(left_confines, right_confines, resalted_Basix)
     
primary_A=copy.copy(A)
primary_B=copy.copy(B)
primary_C=copy.copy(C)
smpl = Simplex(A, B, C)


primary_basis_indices, res_A, m=smpl.forward_simplex()

sens_a = SensAnalysis(primary_basis_indices, res_A, m, primary_A, primary_B, primary_C)
sens_a.forward_sens_analysis()

