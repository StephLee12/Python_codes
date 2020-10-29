import numpy as np


class FlowCalculationSetUp:
    def __init__(self):
        self.node_number = 0  # 节点数量
        self.balance_node = 0  # 平衡节点编号
        self.power = self.get_power_dict()  # 网络功率
        self.YB = self.get_YB()  # 导纳矩阵

    def get_YB(self):  #导纳矩阵
        mat = np.zeros((self.node_number, self.node_number), dtype=complex)

        with open('PowerSystemSteadyState/FlowCalculation/branch_data.txt',
                  'r') as f:
            for branch_data in f.readlines():
                branch_data = branch_data.strip('\n')
                source, end, r, x, y, k = branch_data.split()

                source, end, r, x, y, k = int(source), int(end), float(
                    r), float(x), float(y), float(k)
                resistance = complex(r, x)

                mat[source -
                    1][end -
                       1], mat[end -
                               1][source -
                                  1] = 1 / (-resistance), 1 / (-resistance)
                mat[source - 1][source - 1] += (1 / resistance +
                                                0.5 * complex(0, y))
                mat[end - 1][end - 1] += (1 / resistance + 0.5 * complex(0, y))

        f.close()
        return mat

    def get_power_dict(self):  # 网络功率(字典)
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
                if node_type == 3:
                    self.balance_node = node_index
                power_dict[node_index] = {
                    'voltage':complex(voltage_amp, voltage_ang),
                    'power':complex(power_real, power_imag),
                    'type':node_type
                }

                node_number += 1

        f.close()
        self.node_number = node_number
        return power_dict


# if __name__ == "__main__":
#     obj = FlowCalculationSetUp()
#     print(1)