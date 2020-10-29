# Calculate Flow in IEEE 9 Nodes Model

## Data Input

- ``branch_data.txt``每一列的数据分别为
  - 首端母线(节点)
  - 末端母线(节点)
  - 电阻标幺值
  - 电抗标幺值
  - 对地容纳标幺值
  - 变压器非标准变比
- ``power_data.txt``每一列数据分别为
  - 母线名(节点名)
  - 电压幅值
  - 电压相角
  - 有功功率
  - 无功功率
  - 节点类型
- **注意 节点2和节点3为发电机 属于PQ节点**

## Calculation

- ``gauss_seidel.py``为高斯-塞德尔迭代
- ``newton_raphson.py``为牛顿-拉斐森迭代
- ``setup.py``为数据读取