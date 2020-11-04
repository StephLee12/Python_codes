import numpy as np
import cmath
from scipy.linalg import lu
from scipy.io import savemat

from setup import FlowCalculationSetUp

# 若出现运算异常 throw an exception
np.seterr('raise')

class NetwonRaphsonFlowCalculation:
    def __init__(self, epsilon=0.001):
        # 节点个数、平衡节点编号，导纳矩阵，功率，节点列表，线路数据
        self.node_number, self.balance_node, self.Y, self.power_dict, self.node_list,self.branches = self.init_params(
        )
        self.node_voltage = self.init_node_voltage()  # 节点的电压
        self.PQ_node = self.init_PQ_node()  # PQ节点编号
        # PQ节点有有功无功功率
        self.PQ_node_power_unbal = self.init_PQ_node_power_unbal()
        # 雅可比矩阵
        self.jacobian_mat = np.zeros(
            (len(self.PQ_node) * 2, len(self.PQ_node) * 2))
        self.solution = self.init_solution()  # 修正方程式的解
        self.iteration = 0  # 迭代次数
        self.epsilon = epsilon  # 迭代结束阈值
        self.balance_node_power = complex(0, 0)  # 平衡节点的功率
        self.branch_power = np.zeros((self.node_number,self.node_number),dtype=complex) # 线路损耗
        self.total_cost = complex(0,0) # 总网损

    # 设置节点数量，平衡节点编号，节点导纳矩阵，节点功率
    def init_params(self):
        setup = FlowCalculationSetUp()
        return setup.node_number, setup.balance_node, setup.YB, setup.power, setup.node_list,setup.branches

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
        sum_P_1, sum_P_2, sum_Q_1, sum_Q_2 = 0, 0, 0, 0
        # 编号为node_index节点的电压的实部和虚部
        e_node, f_node = self.node_voltage[node_index * 2 -
                                           1], self.node_voltage[node_index * 2
                                                                 - 2]
        # 根据公式 4-38a 4-38b page128
        for node_num in self.node_list:
            sum_P_1 += (self.Y[node_index - 1][node_num - 1].real *
                        self.node_voltage[node_num * 2 - 1] -
                        self.Y[node_index - 1][node_num - 1].imag *
                        self.node_voltage[node_num * 2 - 2])
            sum_P_2 += (self.Y[node_index - 1][node_num - 1].real *
                        self.node_voltage[node_num * 2 - 2] +
                        self.Y[node_index - 1][node_num - 1].imag *
                        self.node_voltage[node_num * 2 - 1])
            sum_Q_1 += (self.Y[node_index - 1][node_num - 1].real *
                        self.node_voltage[node_num * 2 - 1] -
                        self.Y[node_index - 1][node_num - 1].imag *
                        self.node_voltage[node_num * 2 - 2])
            sum_Q_2 += (self.Y[node_index - 1][node_num - 1].real *
                        self.node_voltage[node_num * 2 - 2] +
                        self.Y[node_index - 1][node_num - 1].imag *
                        self.node_voltage[node_num * 2 - 1])

        # e_i*sum(G_ij*e_j-B_ij*f_j)+f_i*sum(G_ij*f_j+B_ij*e_j)
        sum_P = e_node * sum_P_1 + f_node * sum_P_2
        # f_i*sum(G_ij*e_j-B_ij*f_j) + e_i*sum(G_ij*f_j+B_ij*e_j)
        sum_Q = f_node * sum_Q_1 - e_node * sum_Q_2
        # 无功不平衡量
        self.PQ_node_power_unbal[
            node_index * 2 -
            3] = self.power_dict[node_index]['power'].imag - sum_Q
        # 有功不平衡量
        self.PQ_node_power_unbal[
            node_index * 2 -
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
        for node_pos in self.node_list:
            if node_pos != node_num:
                G_ij, f_j = self.Y[node_num -
                                   1][node_pos -
                                      1].real, self.node_voltage[node_pos * 2 -
                                                                 2]
                B_ij, e_j = self.Y[node_num -
                                   1][node_pos -
                                      1].imag, self.node_voltage[node_pos * 2 -
                                                                 1]
                sum_gf_plus_be += (G_ij * f_j + B_ij * e_j)
                sum_ge_minus_bf += (G_ij * e_j - B_ij * f_j)

        self.jacobian_mat[node_num * 2 -
                          4][node_num * 2 -
                             4] = 2 * G_ii * f_i + sum_gf_plus_be  # H_ii
        self.jacobian_mat[node_num * 2 -
                          4][node_num * 2 -
                             3] = 2 * G_ii * e_i + sum_ge_minus_bf  # N_ii
        self.jacobian_mat[node_num * 2 -
                          3][node_num * 2 -
                             4] = -2 * B_ii * f_i + sum_ge_minus_bf  # J_ii
        self.jacobian_mat[node_num * 2 -
                          3][node_num * 2 -
                             3] = -2 * B_ii * e_i - sum_gf_plus_be  # L_ii

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

    # 输出每个节点的幅值和相角
    def output_node_voltage(self):
        print('##########每个节点的电压##########')
        for node_index in self.node_list:
            voltage_complex = complex(self.node_voltage[node_index * 2 - 1],
                                      self.node_voltage[node_index * 2 -
                                                        2])  # 得到每个节点电压的直角坐标表示
            voltage_complex = list(
                cmath.polar(voltage_complex))  # 转换为极坐标 得到幅值和相角
            voltage_complex[1] *= (180 / np.pi)  # 将弧度转换为角度
            print('Node{}: Voltage Magnitude is {} Phase Angle is {}°'.format(
                node_index, round(voltage_complex[0], 4),
                round(voltage_complex[1], 4)))

    # 计算平衡节点的功率和线路功率
    def calc_balance_node_power(self):
        print('##########平衡节点功率及总网损##########')
        
        # 计算平衡节点功率
        sum_y_1j_u_j = 0

        for node_index in self.node_list:
            y_1j = self.Y[0][node_index-1]
            u_j = complex(self.node_voltage[node_index*2-1],self.node_voltage[node_index*2-2])
            sum_y_1j_u_j += (y_1j.conjugate()*u_j.conjugate())
        
        u_1 = complex(self.node_voltage[1], self.node_voltage[0])
        self.balance_node_power = u_1 * sum_y_1j_u_j
        print('The power of balance node is {}+{}j'.format(round(self.balance_node_power.real,4),round(self.balance_node_power.imag,4)))

        # 计算总网损
        sum_power_PQ = complex(0,0)
        for node_index in self.PQ_node:
            sum_power_PQ += self.power_dict[node_index]['power']
        
        self.total_cost = self.balance_node_power + sum_power_PQ
        print('The total cost of the net is {}+{}j'.format(round(self.total_cost.real,4),round(self.total_cost.imag,4)))

    # 计算线路损耗
    def calc_branch_power(self):
        print('##########每条线路的功率##########')
        for branch in self.branches:
            # 从source到end的线路的导纳数据
            source,end,conductance,half_y = branch[0],branch[1],branch[2],branch[3]
            U_source = complex(self.node_voltage[source*2-1],self.node_voltage[source*2-2])
            U_end = complex(self.node_voltage[end*2-1],self.node_voltage[end*2-2])
            half_y_con,conductance_con = half_y.conjugate(),conductance.conjugate()
            # S_source_end = U_source[U_source*y_half_y+(U_source-U_end)*conductance]
            self.branch_power[source-1][end-1] = U_source*(
                U_source.conjugate()*half_y_con+
                (U_source.conjugate()-U_end.conjugate())*conductance_con
            )
            # S_end_source = S_end[U_end*y_half+(U_end-U_source)*conductance]
            self.branch_power[end-1][source-1] = U_end*(
                U_end.conjugate()*half_y_con+
                (U_end.conjugate()-U_source.conjugate())*conductance_con
            )
            print('Branch power from Node{} to Node{} is {}+{}j'.format(source,end,round(self.branch_power[source-1][end-1].real,4),round(self.branch_power[source-1][end-1].imag,4)))
            print('Branch power from Node{} to Node{} is {}+{}j'.format(end,source,round(self.branch_power[end-1][source-1].real,4),round(self.branch_power[end-1][source-1].imag,4)))

    # Newton-Raphson迭代 main function
    def run(self): 
        iteration = 0
        max_delta_e = float('inf')
        max_delta_f = float('inf')

        while (abs(max_delta_e) > self.epsilon) or (abs(max_delta_f) >
                                                    self.epsilon):
            iteration += 1
            # 计算编号为node_index的节点的功率不平衡量和雅可比矩阵分量
            for node_index in self.PQ_node:
                self.calc_unbal_PQ(node_index)
                self.calc_jacobian_mat(node_index)

            # 解修正方程式 LU分解
            _, L, U = lu(self.jacobian_mat)
            # savemat('jacobian.mat', {'jacobian': self.jacobian_mat})
            # savemat('b.mat', {'b': self.PQ_node_power_unbal})
            L_inv, U_inv = np.linalg.inv(L), np.linalg.inv(U)
            y = L_inv @ self.PQ_node_power_unbal
            self.solution = U_inv @ y

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
                                  1] += self.solution[node_index * 2 -
                                                      3]  # 电压实部
                self.node_voltage[node_index * 2 -
                                  2] += self.solution[node_index * 2 -
                                                      4]  # 电压虚部

        self.iteration = iteration
        print('##########迭代次数##########')
        print('Newton-Raphson\'s Iteration Times：{}'.format(self.iteration))
        # 输出每个节点的电压幅值和相角
        self.output_node_voltage()
        # 计算平衡节点功率和总网损
        self.calc_balance_node_power()
        # 计算线路功率
        self.calc_branch_power()


if __name__ == "__main__":
    obj = NetwonRaphsonFlowCalculation()
    obj.run()