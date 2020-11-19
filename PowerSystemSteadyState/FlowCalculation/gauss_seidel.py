import numpy as np
import cmath

from setup import FlowCalculationSetUp

np.seterr('raise')


class GaussSeidelFlowCalculation:
    def __init__(self, epsilon=0.001):
        # 节点个数 导纳矩阵 功率 节点列表 线路数据
        self.node_number, self.balance_node, self.Y, self.power_dict, self.node_list, self.branches = self.init_params(
        )
        self.node_voltage = self.init_node_voltage()  # 节点电压
        self.PQ_node = self.init_PQ_node()  # PQ节点编号
        self.iteration = 0  # 迭代次数
        self.epsilon = epsilon  # 迭代结束阈值
        self.balance_node_power = complex(0, 0)  # 平衡节点的功率
        self.branch_power = np.zeros((self.node_number, self.node_number),
                                     dtype=complex)  # 线路损耗
        self.total_cost = complex(0, 0)  # 总网损

    # 初始化参数
    def init_params(self):
        setup = FlowCalculationSetUp()
        return setup.node_number, setup.balance_node, setup.YB, setup.power, setup.node_list, setup.branches

    # 初始化节点电压
    def init_node_voltage(self):
        node_voltage = np.zeros(self.node_number, dtype=complex)
        for node_index, node_info in self.power_dict.items():
            node_voltage[node_index - 1] = node_info['voltage']
        return node_voltage

    # PQ节点的编号
    def init_PQ_node(self):
        PQ_node = [i for i in range(1, self.node_number + 1)]
        # 删除平衡节点
        PQ_node.pop(self.balance_node - 1)
        return PQ_node

    # 输出每个节点的幅值和相角
    def output_node_voltage(self):
        print('##########每个节点的电压##########')
        for node_index in self.node_list:
            voltage_complex = list(
                cmath.polar(self.node_voltage[node_index-1]))  # 转换为极坐标 得到幅值和相角
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
            y_1j = self.Y[0][node_index - 1]
            u_j = self.node_voltage[node_index-1]
            sum_y_1j_u_j += (y_1j.conjugate() * u_j.conjugate())

        u_1 = self.node_voltage[0]
        self.balance_node_power = u_1 * sum_y_1j_u_j
        print('The power of balance node is {}+({})j'.format(
            round(self.balance_node_power.real, 4),
            round(self.balance_node_power.imag, 4)))

        # 计算总网损
        sum_power_PQ = complex(0, 0)
        for node_index in self.PQ_node:
            sum_power_PQ += self.power_dict[node_index]['power']

        self.total_cost = self.balance_node_power + sum_power_PQ
        print('The total cost of the net is {}+({})j'.format(
            round(self.total_cost.real, 4), round(self.total_cost.imag, 4)))

    # 计算线路损耗
    def calc_branch_power(self):
        print('##########每条线路的功率##########')
        for branch in self.branches:
            # 从source到end的线路的导纳数据
            source, end, conductance, half_y = branch[0], branch[1], branch[
                2], branch[3]
            U_source = self.node_voltage[source-1]
            U_end = self.node_voltage[end-1]
            half_y_con, conductance_con = half_y.conjugate(
            ), conductance.conjugate()
            # S_source_end = U_source[U_source*y_half_y+(U_source-U_end)*conductance]
            self.branch_power[source - 1][end - 1] = U_source * (
                U_source.conjugate() * half_y_con +
                (U_source.conjugate() - U_end.conjugate()) * conductance_con)
            # S_end_source = S_end[U_end*y_half+(U_end-U_source)*conductance]
            self.branch_power[end - 1][source - 1] = U_end * (
                U_end.conjugate() * half_y_con +
                (U_end.conjugate() - U_source.conjugate()) * conductance_con)
            print('Branch power from Node{} to Node{} is {}+({})j'.format(
                source, end,
                round(self.branch_power[source - 1][end - 1].real, 4),
                round(self.branch_power[source - 1][end - 1].imag, 4)))
            print('Branch power from Node{} to Node{} is {}+({})j'.format(
                end, source,
                round(self.branch_power[end - 1][source - 1].real, 4),
                round(self.branch_power[end - 1][source - 1].imag, 4)))

    # Gauss-Seidel迭代 main function
    def run(self):
        iteration = 0
        max_delta = float('inf')

        while max_delta > self.epsilon:
            iteration += 1
            new_node_voltage = np.zeros_like(self.node_voltage,dtype=complex)
            new_node_voltage[0] = self.node_voltage[0]
            for node_index in self.PQ_node:
                Y_ii = self.Y[node_index-1][node_index-1]
                power_con = self.power_dict[node_index]['power'].conjugate()
                U_i_con = self.node_voltage[node_index-1].conjugate()
                sum_y_ij_u_j = 0
                
                for node_pos in self.node_list:
                    Y_ij = self.Y[node_index-1][node_pos-1]
                    if node_pos < node_index:
                       U_j = new_node_voltage[node_pos-1]
                       sum_y_ij_u_j += (Y_ij*U_j)
                       continue
                    if node_pos > node_index:
                        U_j = self.node_voltage[node_pos-1]
                        sum_y_ij_u_j += (Y_ij*U_j)
                        continue
                        
                new_node_voltage[node_index-1] = ((power_con/U_i_con)-sum_y_ij_u_j)/Y_ii
                    
            # 判断迭代是否结束
            diff_arr = abs(new_node_voltage - self.node_voltage)
            self.node_voltage = new_node_voltage
            tmp_delta = np.max(diff_arr)
            if tmp_delta < max_delta:
                max_delta = tmp_delta


        self.iteration = iteration
        print('##########迭代次数##########')
        print('Gauss-Seidel\'s Iteration Times：{}'.format(self.iteration))
        # 输出每个节点的电压幅值和相角
        self.output_node_voltage()
        # 计算平衡节点功率和总网损
        self.calc_balance_node_power()
        # 计算线路功率
        self.calc_branch_power()
        

if __name__ == "__main__":
    obj = GaussSeidelFlowCalculation()
    obj.run()