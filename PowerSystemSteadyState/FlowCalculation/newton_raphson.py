import numpy as np
import cmath
from scipy.linalg import lu_factor,lu_solve

from setup import FlowCalculationSetUp

########节点1是平衡节点######

np.seterr('raise')

class NetwonRaphsonFlowCalculation:
    def __init__(self, epsilon=0.001):
        self.node_number, self.balance_node, self.Y, self.power_dict = self.init_params(
        )
        self.node_voltage = self.init_node_voltage()  # 节点的电压
        self.PQ_node = self.init_PQ_node()  # PQ节点编号
        self.PQ_node_power_unbal = self.init_PQ_node_power_unbal(
        )  # PQ节点有有功无功功率
        self.jacobian_mat = np.zeros((len(self.PQ_node)*2, len(self.PQ_node)*2))
        self.solution = self.init_solution()
        self.iteration = 0  # 迭代次数
        self.epsilon = epsilon  # 修正阈值
        self.balance_node_power = complex(0, 0)  # 平衡节点的功率

    # 设置节点数量，平衡节点编号，节点导纳矩阵，节点功率
    def init_params(self):
        setup = FlowCalculationSetUp()
        return setup.node_number, setup.balance_node, setup.YB, setup.power

    # 初始化节点电压
    def init_node_voltage(self):
        node_voltage = np.zeros(self.node_number * 2)
        for node_index, node_info in self.power_dict.items():
            node_voltage[node_index * 2 - 1] = node_info['voltage'].real  # 实部
            node_voltage[node_index * 2 - 2] = node_info['voltage'].imag  # 虚部

        return node_voltage  # 包括有功和无功两部分

    # PQ节点的编号
    def init_PQ_node(self):
        PQ_node = [i for i in range(1, self.node_number + 1)]
        # 删除平衡节点
        PQ_node.pop(self.balance_node - 1)
        return PQ_node

    # 初始化PQ节点的有功无功功率
    def init_PQ_node_power_unbal(self):
        return np.zeros(len(self.PQ_node) * 2)  # 要包括有功和无功

    # 初始化PQ节点的电压修正量
    def init_solution(self):
        return np.zeros(len(self.PQ_node) * 2)  # 要包括有功和无功

    #计算节点编号为node_index的PQ节点的功率不平衡量
    def calc_unbal_PQ(self, node_index):
        sum_P, sum_Q = 0, 0
        # 编号为node_index节点的电压的实部和虚部
        e_node, f_node = self.node_voltage[node_index * 2 -
                                           1], self.node_voltage[node_index * 2
                                                                 - 2]
        # 根据公式 4-38a 4-38b page128
        for node_num in self.PQ_node:
            sum_P += (e_node * (
                self.Y[node_index - 1][node_num - 1].real *
                self.node_voltage[node_num * 2 - 1] -
                self.Y[node_index - 1][node_num - 1].imag *
                self.node_voltage[node_num * 2 - 2]) + f_node * (
                    self.Y[node_index - 1][node_num - 1].real *
                    self.node_voltage[node_num * 2 - 2] +
                    self.Y[node_index - 1][node_num - 1].imag *
                    self.node_voltage[node_num * 2 - 1]))

            sum_Q += (f_node * (
                self.Y[node_index - 1][node_num - 1].real *
                self.node_voltage[node_num * 2 - 1] -
                self.Y[node_index - 1][node_num - 1].imag *
                self.node_voltage[node_num * 2 - 2]) - e_node * (
                    self.Y[node_index - 1][node_num - 1].real *
                    self.node_voltage[node_num * 2 - 2] +
                    self.Y[node_index - 1][node_num - 1].imag *
                    self.node_voltage[node_num * 2 - 1]))

        # 无功不平衡量
        self.PQ_node_power_unbal[node_index * 2 -
                                 3] = self.power_dict[node_index]['power'].imag - sum_Q
        # 有功不平衡量
        self.PQ_node_power_unbal[node_index * 2 -
                                 4] = self.power_dict[node_index]['power'].real - sum_P

    # 计算雅可比矩阵的对角元
    def calc_jacobian_mat_diag(self, node_index, node_num):
        B_ii, e_i = self.Y[node_num -
                           1][node_num -
                              1].imag, self.node_voltage[node_num * 2 - 1]
        G_ii, f_i = self.Y[node_num -
                           1][node_num -
                              1].real, self.node_voltage[node_num * 2 - 2]
        sum_gf_plus_be = 0
        sum_ge_minus_bf = 0
        for node_pos in self.PQ_node:
            if node_pos != node_num:
                G_ij, f_j = self.Y[node_num -
                                   1][node_pos-1].real, self.node_voltage[
                                       node_num * 2 - 2]
                B_ij, e_j = self.Y[node_num -
                                   1][node_pos-1].imag, self.node_voltage[
                                       node_num * 2 - 1]
                sum_gf_plus_be += (G_ij * f_j + B_ij * e_j)
                sum_ge_minus_bf += (G_ij * e_j - B_ij * f_j)

        self.jacobian_mat[node_num * 2 - 4][
            node_num * 2 -
            4] = -B_ii * e_i + G_ii * f_i + sum_gf_plus_be  # H_ii
        self.jacobian_mat[node_num * 2 - 4][
            node_num * 2 -
            3] = G_ii * e_i + B_ii * f_i + sum_ge_minus_bf  # N_ii
        self.jacobian_mat[node_num * 2 - 3][
            node_num * 2 -
            4] = -G_ii * e_i - B_ii * f_i + sum_ge_minus_bf  # J_ii
        self.jacobian_mat[node_num * 2 - 3][
            node_num * 2 -
            3] = -B_ii * e_i + G_ii * f_i - sum_gf_plus_be  # L_ii

    # 计算雅可比矩阵的元素(非对角元)
    def calc_jacobian_mat_no_diag(self, node_index, node_num):
        B_ij, e_i = self.Y[node_index -
                           1][node_num -
                              1].imag, self.node_voltage[node_index * 2 - 1]
        G_ij, f_i = self.Y[node_index -
                           1][node_num -
                              1].real, self.node_voltage[node_index * 2 - 2]

        self.jacobian_mat[node_index * 2 -
                          4][node_num * 2 -
                             4] = -B_ij * e_i + G_ij * f_i  # H_ij
        self.jacobian_mat[node_index * 2 -
                          4][node_num * 2 -
                             3] = G_ij * e_i + B_ij * f_i  # N_ij
        self.jacobian_mat[node_index * 2 - 3][
            node_num * 2 - 4] = -self.jacobian_mat[node_index * 2 -
                                                   4][node_num * 2 - 3]  # J_ij
        self.jacobian_mat[node_index * 2 -
                          3][node_num * 2 -
                             3] = self.jacobian_mat[node_index * 2 -
                                                    4][node_num * 2 -
                                                       4]  # L_ij

    # 计算节点编号为node_index节点的雅可比矩阵的两行分量
    def calc_jacobian_mat(self, node_index):
        for node_num in self.PQ_node:
            if node_num == node_index:
                self.calc_jacobian_mat_diag(node_index, node_num)
            else:
                self.calc_jacobian_mat_no_diag(node_index, node_num)

    # 矩阵的LU分解(Crout分解)
    # def jacabian_lu_composition(self):
    #     # 下三角
    #     L = np.zeros_like(self.jacobian_mat)
    #     # 上三角
    #     U = np.zeros_like(self.jacobian_mat)

    #     # 若雅可比矩阵不是方阵
    #     if L.shape[0] != L.shape[1]:
    #         Exception('something wrong with jacobian matrix\'s rank')

    #     shape = L.shape[0]

    #     for k in range(shape):
    #         U[k,k] = 1
    #         for j in range(k,shape):
    #             sum_0 = sum(L[k,s]*U[s,j] for s in range(1,k-1))
    #             L[j,k] = self.jacobian_mat[k][k] - sum_0
    #         for j in range(k,shape):
    #             sum_1 = sum(L[k,s]*U[s,j] for s in range(1,k-1))
    #             U[k,j] = (self.jacobian_mat[k,j]-sum_1) / L[k,k]
        

        # # 计算l_i1
        # for i in range(shape):
        #     L[i][0] = self.jacobian_mat[i][0]
        #     U[i][i] = 1 # 上三角对角线元素全为1
        # # 计算u_1j
        # for j in range(1,shape):
        #     U[0][j] = self.jacobian_mat[0][j] / L[0][0]

        # for k in range(1, shape):
        #     #计算l_ik
        #     for i in range(k, shape):
        #         tmp = 0
        #         for r in range(k):
        #             tmp += (L[i][r] * U[r][k])
        #         L[i][k] = self.jacobian_mat[i][k] - tmp
        #     # 计算u_kj
        #     for j in range(k + 1, shape):
        #         tmp = 0
        #         for r in range(k):
        #             tmp += (L[k][r] * U[r][j])
        #         U[k][j] = (self.jacobian_mat[k][j] -
        #                            tmp) / L[k][k]

        # return L, U

    # 解修正方程式 Ax=b
    # A为雅可比矩阵 x为电压修正量 b为功率不平衡量
    def solve_modified_func(self):
        # 对雅可比矩阵进行LU分解
        lu,piv = lu_factor(self.jacobian_mat)
        self.solution = lu_solve((lu,piv),self.PQ_node_power_unbal)
        # _,L,U = scipy.linalg.lu(self.jacobian_mat)
        # L, U = self.jacabian_lu_composition()
        # L = np.nan_to_num(L)
        # U = np.nan_to_num(U)

        # 顺代求解 Ly=b
        # y = np.zeros_like(self.PQ_node_power_unbal)
        # shape = y.shape[0]
        # y[0] = self.PQ_node_power_unbal[0] / L[0][0]
        # for k in range(1, shape):
        #     tmp = 0
        #     for r in range(1, k):
        #         tmp += (L[k - 1][r - 1] * y[r - 1])
        #     y[k -
        #       1] = (self.PQ_node_power_unbal[k - 1] - tmp) / (L[k - 1][k - 1])

        # 回代求解 Ux=y
        # self.solution[-1] = y[-1]
        # for k in range(shape - 1, 0, -1):
        #     tmp = 0
        #     for r in range(k + 1, shape + 1):
        #         tmp += (U[k - 1][r - 1] * self.solution[r - 1])
        #     self.solution[k - 1] = y[k - 1] - tmp

    # 输出每个节点的幅值和相角
    def output_node_voltage(self):
        node_list = [i for i in range(1,self.node_number+1)]
        for node_index in node_list:
            voltage_complex = complex(self.node_voltage[node_index*2-1],self.node_voltage[node_index*2-2]) # 得到每个节点电压的直角坐标表示
            voltage_complex = list(cmath.polar(voltage_complex)) # 转换为极坐标 得到幅值和相角
            voltage_complex[1] *= (180/np.pi) # 将弧度转换为角度
            print('节点{}的电压幅值为{},相角为{}'.format(node_index,voltage_complex[0],voltage_complex[1]))

    def main_iter_loop(self):  # Newton-Raphson迭代的循环
        iteration = 0
        max_delta_e = float('inf')
        max_delta_f = float('inf')

        while (abs(max_delta_e) > self.epsilon) or (abs(max_delta_f) >
                                                    self.epsilon):
            iteration += 1
            for node_index in self.PQ_node:
                # 计算编号为node_index的节点的功率不平衡量
                self.calc_unbal_PQ(node_index)
                self.calc_jacobian_mat(node_index)

            # 解修正方程式
            #self.solve_modified_func()
            lu,piv = lu_factor(self.jacobian_mat)
            self.solution = lu_solve((lu,piv),self.PQ_node_power_unbal)
            
            # 判断迭代是否结束
            shape = self.solution.shape[0]
            delta_f_arr, delta_e_arr = np.zeros((int(shape / 2), 1)), np.zeros(
                (int(shape / 2), 1))
            for i in range(0, shape, 2):
                delta_f_arr[int(i / 2)] = self.solution[i]
                delta_e_arr[int(i / 2)] = self.solution[i + 1]

            # 更新最大的修正量
            tmp_delta_e = np.max(np.abs(delta_e_arr))
            tmp_delta_f = np.max(np.abs(delta_f_arr))
            
            if tmp_delta_e < max_delta_e:
                max_delta_e = tmp_delta_e
            if tmp_delta_f < max_delta_f:
                max_delta_f = tmp_delta_f
            
            # 更新节点电压 
            for node_index in self.PQ_node:
                self.node_voltage[node_index * 2 -
                                1] += self.solution[node_index * 2 - 3]  # 电压实部
                self.node_voltage[node_index * 2 -
                                2] += self.solution[node_index * 2 - 4]  # 电压虚部
            print('test')

        self.iteration = iteration
        print('迭代次数：{}'.format(self.iteration))

        # # 迭代结束 计算各PQ节点的电压
        # for node_index in self.PQ_node:
        #     self.node_voltage[node_index * 2 -
        #                       1] += self.solution[node_index * 2 - 3]  # 电压实部
        #     self.node_voltage[node_index * 2 -
        #                       2] += self.solution[node_index * 2 - 4]  # 电压虚部
        
        # 输出每个节点的电压幅值和相角
        self.output_node_voltage()

        # 计算平衡节点功率
        tmp = 0
        node_list = [i for i in range(1, self.node_number + 1)]
        for node_index in node_list:
            tmp += (self.Y[0][node_index - 1].conjugate() *
                    complex(self.node_voltage[node_index * 2 - 1],
                            self.node_voltage[node_index * 2 - 2]).conjugate())
        u_1 = complex(self.node_voltage[1], self.node_voltage[0])
        self.balance_node_power = u_1 * tmp


if __name__ == "__main__":
    obj = NetwonRaphsonFlowCalculation()
    obj.main_iter_loop()