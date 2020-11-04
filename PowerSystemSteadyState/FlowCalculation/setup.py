import numpy as np


# 读取数据 一些准备工作
class FlowCalculationSetUp:
    def __init__(self):
        self.node_number = 0  # 节点数量
        self.node_list = None  # 节点编号列表
        self.balance_node = 0  # 平衡节点编号
        self.power = self.get_power_dict()  # 网络功率
        self.branches = None # 线路数据
        self.YB = self.get_YB()  # 导纳矩阵
        

    # 计算导纳矩阵
    def get_YB(self):
        mat = np.zeros((self.node_number, self.node_number), dtype=complex)
        branches = []

        with open('PowerSystemSteadyState/FlowCalculation/branch_data.txt',
                  'r') as f:
            # 按行读取branch_data中的数据
            for branch_data in f.readlines():
                branch_data = branch_data.strip('\n')
                # 匹配对应的数据
                source, end, r, x, y, k = branch_data.split()
                # convert to suitable data type
                source, end, r, x, y, k = int(source), int(end), float(
                    r), float(x), float(y), float(k)
                # get resistance
                resistance = complex(r, x)
                # get branch data
                branches.append([source,end,1/resistance,0.5*complex(0,y)])

                # 根据导纳矩阵的运算规则 计算矩阵中的元素
                mat[source -
                    1][end -
                       1], mat[end -
                               1][source -
                                  1] = 1 / (-resistance), 1 / (-resistance)
                mat[source - 1][source - 1] += (1 / resistance +
                                                0.5 * complex(0, y))
                mat[end - 1][end - 1] += (1 / resistance + 0.5 * complex(0, y))

        f.close()
        self.branches = branches
        return mat

    # @staticmethod
    # def calc_transformer_resistance(mat, resistance, k, source, end):
    #     # 变压器转换为 pi型等效电路 用阻抗表示
    #     resistance_tri = [
    #         resistance / (1 - k), resistance / k,
    #         resistance / (np.square(k) - k)
    #     ]
    #     # from source to end 的导纳 (非对角元)
    #     mat[source -
    #         1][end -
    #            1], mat[end -
    #                    1][source -
    #                       1] = -1 / resistance_tri[1], -1 / (resistance_tri[1])
    #     mat[source - 1][source - 1] += (1 / resistance_tri[1] +
    #                                     1 / resistance_tri[0])
    #     mat[end - 1][end - 1] += (1 / resistance_tri[1] +
    #                               1 / resistance_tri[2])

    # 将每个节点的功率 电压 节点类型用字典存储
    def get_power_dict(self):
        power_dict = {}
        node_number = 0

        with open('PowerSystemSteadyState/FlowCalculation/power_data.txt',
                  'r') as f:
            for power_data in f.readlines():
                power_data = power_data.strip('\n')
                node_index, voltage_amp, voltage_ang, power_real, power_imag, node_type = power_data.split(
                )
                node_index, voltage_amp, voltage_ang, power_real, power_imag, node_type = int(
                    node_index), float(voltage_amp), float(voltage_ang), float(
                        power_real), float(power_imag), int(node_type)
                # 获取平衡节点的编号
                if node_type == 3:
                    self.balance_node = node_index
                # 以每个节点的编号为key value为一个子字典 存储该节点的电压 功率 类型
                power_dict[node_index] = {
                    'voltage': complex(voltage_amp, voltage_ang),
                    'power': complex(power_real, power_imag),
                    'type': node_type
                }
                node_number += 1

        f.close()
        self.node_number = node_number
        self.node_list = [i for i in range(1, self.node_number + 1)]
        return power_dict


# if __name__ == "__main__":
#     obj = FlowCalculationSetUp()