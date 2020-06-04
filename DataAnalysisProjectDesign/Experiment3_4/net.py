import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.special import comb
from copy import deepcopy

from netnode import NetNode


class Net:
    def __init__(self, path, beta=0.5, gamma=1,simulation_epochs=2000):
        self.path = path  # 网络图路径

        # node_list节点列表 其中的每个元素都是一个NetNode类
        # adj_dict邻接表 是个字典
        self.node_list, self.adj_dict = self.get_adj_dict()
        self.node_num = self.get_node_num()  #获得网络中节点数目的多少

        self.beta = beta  # 感染概率
        self.gamma = gamma  # 痊愈概率

        self.max_k_shell = 0  #最大k_shell值

        self.degree_res = None  #度排序结果
        self.degree_relation = None # 相关性
        self.ent_res = None  #信息熵排序结果
        self.ent_relation = None # 相关性
        self.k_shell_degree_res = None  #k_shell排序和度排序结合 结果
        self.k_shell_degree_relation = None #相关性
        self.k_shell_ent_res = None  #k_Shell排序和信息熵排序结合 结果
        self.k_shell_ent_relation = None #相关性
        self.prob_res = None  #概率模型排序结果
        self.prob_relation = None #相关性

        self.simulation_epochs = simulation_epochs
        self.sir_res = None  #sir传播排序结果

    # 获得网络中节点的个数
    def get_node_num(self):
        return len(self.node_list)

    # 获得邻接表
    def get_adj_dict(self):

        # 按行读取 网络数据
        # 每行数据为的第一个值为节点 其他为其邻接节点 用 ","分隔
        file_name = self.path
        node_list = []
        net_dict = {}
        with open(file_name) as f:
            for data in f.readlines():
                data = data.strip('\n')
                data = data.split(',')
                v = data[0]
                adj_v = data[1:]  # 获取邻接节点切片
                net_dict[int(v)] = adj_v  # 作为键值对 加入adj_dict
                node_list.append(NetNode(int(v), len(adj_v)))  #加入node_list

        return node_list, net_dict

    # k-shell算法
    def k_shell(self):
        adj_dict = deepcopy(self.adj_dict)

        ks_count = 1
        key_store = []

        while len(adj_dict) != 0:  #终止条件 网络中所有节点都有ks值

            # 第一次扫描 删除初始度为ks_count的节点
            for key in adj_dict.copy():
                val = adj_dict[key]
                if len(val) == ks_count:
                    self.node_list[key - 1].ks = ks_count  #将该节点的k-shell值标记
                    # 在网络中删除该节点
                    del adj_dict[key]  # 删除以该节点为键的键值对
                    if key in key_store:
                        idx = key_store.index(key)
                        key_store.pop(idx)
                    for key_1 in adj_dict:  # 在其他键对应的值中删除该节点
                        if str(key) in adj_dict[key_1]:
                            idx = adj_dict[key_1].index(str(key))
                            adj_dict[key_1].pop(idx)
                            #若删除该节点 某键的值的长度也变为 ks_count 说明该键 也要被删除
                            if len(adj_dict[key_1]) <= ks_count:
                                key_store.append(key_1)

            #去重复值
            key_store = list(set(key_store))
            # 再次扫描 删除因节点删除度为ks_count的节点
            while key_store != []:
                pop_list = []
                for key in key_store:
                    if adj_dict.__contains__(key):
                        self.node_list[key - 1].ks = ks_count  #将该节点的k-shell值标记
                        # 在网络中删除该节点
                        del adj_dict[key]  # 删除以该节点为键的键值对
                        pop_list.append(key)  #加入应该弹出key_store的列表pop_list
                        for key_1 in adj_dict:
                            if str(key) in adj_dict[key_1]:
                                idx = adj_dict[key_1].index(str(key))
                                adj_dict[key_1].pop(idx)
                                if len(adj_dict[key_1]) <= ks_count:
                                    if key_1 not in key_store:
                                        key_store.append(key_1)

                for pop_key in pop_list:
                    idx = key_store.index(pop_key)
                    key_store.pop(idx)

            ks_count += 1

        self.max_k_shell = ks_count

    # 针对概率模型的bfs
    def prob_bfs(self, node):
        # 返回一个列表 包含三个字典 分别为node的一跳、二跳、三跳邻居
        # 字典中的键为n跳节点的索引 值为它的前驱顶点索引
        node.visited = 1
        neighbors = [{} for i in range(3)]
        # 先处理一跳邻居
        for elem in self.adj_dict[node.idx]:
            neighbors[0][int(elem)] = [node.idx]
            self.node_list[int(elem) - 1].visited = 1
        #再处理二跳三跳邻居
        for i in range(1, 3):
            for key in neighbors[i - 1]:  #找key下一跳
                for elem in self.adj_dict[key]:  #elem即为key的下一跳
                    # 若被访问过 继续循环
                    if self.node_list[int(elem) - 1].visited == 1:
                        continue
                    # 若该节点 已经有前驱节点 执行append
                    if neighbors[i].__contains__(int(elem)):
                        neighbors[i][int(elem)].append(key)
                        self.node_list[int(elem) - 1].visited = 1

                    neighbors[i][int(elem)] = [key]
                    self.node_list[int(elem) - 1].visited = 1

        return neighbors

    # 每对节点进行一次bfs后 要将经过的所有节点的visited属性重置
    # 且对每个节点 计算rank后 要重置uninf_s和score
    def clear_prob_memory(self, node, neighbors):
        node.visited = 0
        for elem in neighbors:
            for key in elem:
                self.node_list[key - 1].visited = 0
                self.node_list[key - 1].uninf_s = 0
                self.node_list[key - 1].score = 0

    # 概率模型
    def prob_model(self):
        for node in self.node_list:
            rank = 0
            #获得node的一跳、二跳、三跳邻居及其前驱节点
            neighbors = self.prob_bfs(node)
            # 先处理一跳邻居
            # 一跳邻居的score均为beta unif_s均为1-beta
            neighbor_1 = neighbors[0]
            for key in neighbor_1:
                self.node_list[key - 1].uninf_s = 1 - self.beta
                self.node_list[key - 1].score = self.beta
                rank += self.node_list[key - 1].score

            # 处理二跳、三跳邻居
            for i in range(1, 3):
                neighbor_i = neighbors[i]
                for key in neighbor_i:
                    prev_nodes = neighbor_i[key]
                    uninf_s = 1
                    for prev_node in prev_nodes:
                        uninf_s *= self.node_list[prev_node - 1].uninf_s
                    score = 1 - uninf_s
                    self.node_list[key - 1].uninf_s = uninf_s
                    self.node_list[key - 1].score = score
                    rank += self.node_list[key - 1].score

            node.rank = rank
            # 重置遍历过的节点的score uninf_s visited值
            self.clear_prob_memory(node, neighbors)

    def clear_sir_memory(self,travel_nodes):
        for idx in travel_nodes:
            self.node_list[idx-1].visited = 0
            self.node_list[idx-1].flag_s = 1
            self.node_list[idx-1].flag_i = 0
            self.node_list[idx-1].flag_r =0
    
    # sir传播仿真
    def sir_simulation(self):
        # 仿真simulation_epochs次
        for _ in range(self.simulation_epochs):
            for node in self.node_list: # 遍历所有节点
                node.visited = 1
                neighbor_dict = {}
                travel_nodes = [node.idx] #记录所有访问过的节点
                
                for val in self.adj_dict[node.idx]:
                    neighbor_dict[int(val)] = [node.idx] #节点和前驱感染节点
                    self.node_list[int(val)-1].visited = 1 # 将邻居节点标记为访问过
                    travel_nodes.append(int(val))

                infected_sum = 0
                infected_record = [1]
                removed_record  = [0]
                suspect_record = [self.node_num-1]
                last_infected = [node.idx] #上一个步长被感染的节点
                
                timestep = 0
                while infected_record[timestep] != 0: #结束条件 没有被感染的节点
                    timestep += 1  # 步长+1            
                    # 在下一时刻 上一时刻被感染的节点 康复
                    removed_record.append(removed_record[timestep-1]+infected_record[timestep-1])
                    while len(last_infected) != 0:
                        idx = last_infected.pop()
                        self.node_list[idx-1].flag_i = 0
                        self.node_list[idx-1].flag_r = 1
                        self.node_list[idx-1].flag_s = 0
                
                    for key in neighbor_dict: #判断邻居节点是否被感染
                        rnd = np.random.uniform()
                        if rnd < 1-math.pow(1-self.beta,len(neighbor_dict[key])): #节点被感染
                            self.node_list[int(key)-1].flag_i = 1
                            self.node_list[int(key)-1].flag_s = 0
                            infected_sum += 1
                            last_infected.append(int(key))
                    
                    neighbor_dict.clear() #清空

                    # 本步长感染人数 和疑似人数的变化
                    infected_record.append(len(last_infected))
                    suspect_record.append(self.node_num-infected_record[timestep]-removed_record[timestep])

                    # 遍历新感染节点的邻居
                    for elem in last_infected:
                        for neighbor in self.adj_dict[elem]:
                            # 判断邻居是否被访问过 
                            if self.node_list[int(neighbor)-1].visited == 0: 
                                self.node_list[int(neighbor)-1].visited = 1
                                travel_nodes.append(int(neighbor))
                            # 判断邻居是否为S类别 如果是 创建键值对 添加到neighbor_dict
                            if self.node_list[int(neighbor)-1].flag_s == 1:
                                if neighbor_dict.__contains__(int(neighbor)):
                                    neighbor_dict[int(neighbor)].append(elem)
                                neighbor_dict[int(neighbor)] = [elem]    
                # node的传播结束 将数据存储到节点
                node.infected_record.append(infected_record)
                node.removed_record.append(removed_record)
                node.suspect_record.append(suspect_record)
                node.sir_val.append(infected_sum)
                # 重置访问过的节点的属性值
                self.clear_sir_memory(travel_nodes)

    # 度排序
    def degree_sort(self):
        sort_dict = {}
        # 获取每一个节点的dc值
        for node in self.node_list:
            dc_node = node.degree / (self.node_num - 1)
            sort_dict[node.idx] = dc_node

        sort_res = []
        # 对dc值排序
        sort_tuple = sorted(sort_dict.items(),
                            key=lambda x: x[1],
                            reverse=True)
        for elem in sort_tuple:
            sort_res.append(elem[0])

        self.degree_res = sort_res
        print('degree sort')
        print(sort_res)
        
        relation = self.calc_relation(sort_res)
        self.degree_relation = relation
        print('degree sort relation is {}'.format(relation))

        return sort_res

    # 信息熵排序
    def entropy_sort(self):
        sort_dict = {}
        # 获得所有节点的信息熵
        for node in self.node_list:
            ent = 0
            neighbor_degree = []  #node的邻居节点的度的列表
            neighbor_degree_sum = 0  #node的邻居节点的度的综合
            # 遍历node的邻居
            for neighbor in self.adj_dict[node.idx]:
                neighbor = int(neighbor)
                neighbor_degree.append(len(self.adj_dict[neighbor]))
                neighbor_degree_sum += len(self.adj_dict[neighbor])
            # 计算熵
            for degree in neighbor_degree:
                prob = degree / neighbor_degree_sum
                ent += prob * np.log(prob)

            sort_dict[node.idx] = -ent
        # 对信息熵排序
        sort_res = []
        sort_tuple = sorted(sort_dict.items(),
                            key=lambda x: x[1],
                            reverse=True)
        for elem in sort_tuple:
            sort_res.append(elem[0])
        self.ent_res = sort_res
        print('entropy sort')
        print(sort_res)

        relation = self.calc_relation(sort_res)
        self.ent_relation = relation
        print('entropy sort relation is {}'.format(relation))

        return sort_res

    # k_shell importance sort use degree sort when k_shell values are the same
    def k_shell_degree_sort(self):
        # 执行k_shell
        self.k_shell()

        sort_buckets = [{} for i in range(self.max_k_shell)]

        # 遍历所有节点 先将不同k_shell值的节点分开
        for node in self.node_list:
            dc_node = node.degree / (self.node_num - 1)  # 计算度
            # 将节点的度和索引作为键值对 放入对应的桶中
            sort_buckets[node.ks - 1][node.idx] = dc_node

        # 对于k_shell值相同的节点 用度进行排序
        sort_res = []
        for bucket in reversed(sort_buckets):
            res = []
            res_dict = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
            for elem in res_dict:
                res.append(elem[0])
            sort_res += res

        self.k_shell_degree_res = sort_res
        print('k-shell and degree sort')
        print(sort_res)

        relation = self.calc_relation(sort_res)
        self.k_shell_degree_relation = relation
        print('k-shell and degree sort relation is {}'.format(relation))
        return sort_res

    # k_shell importance sort use entropy sort when k_shell values are the same
    def k_shell_entropy_sort(self):
        #执行k_shell
        self.k_shell()

        sort_buckets = [{} for i in range(self.max_k_shell)]

        # 遍历所有节点 先将不同k_shell值的节点分开
        for node in self.node_list:
            ent = 0  #计算熵
            neighbor_degree = []  #node的邻居节点的度的列表
            neightbor_degree_sum = 0  #node的邻居节点的度的综合
            #遍历node的邻居
            for neighbor in self.adj_dict[node.idx]:
                neighbor = int(neighbor)
                neighbor_degree.append(len(self.adj_dict[neighbor]))
                neightbor_degree_sum += len(self.adj_dict[neighbor])

            # 计算熵
            for degree in neighbor_degree:
                prob = degree / neightbor_degree_sum
                ent += prob * np.log(prob)
            # 将节点的度和索引作为键值对 放入对应的桶中
            sort_buckets[node.ks - 1][node.idx] = -ent

        # 对于k_shell值相同的节点 用熵进行排序
        sort_res = []
        for bucket in reversed(sort_buckets):
            res = []
            res_dict = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
            for elem in res_dict:
                res.append(elem[0])
            sort_res += res

        self.k_shell_ent_res = sort_res
        print('k-shell and entropy sort')
        print(sort_res)

        relation = self.calc_relation(sort_res)
        self.k_shell_ent_relation = relation
        print('k-shell and entropy sort relation is {}'.format(relation))
        
        return sort_res

    # 概率模型排序
    def prob_sort(self):
        #执行概率模型
        self.prob_model()
        # 获得所有节点的rank
        ori_dict = {}
        for node in self.node_list:
            ori_dict[node.idx] = node.rank

        # 对rank排序
        sort_res = []
        res_dict = sorted(ori_dict.items(), key=lambda x: x[1], reverse=True)
        for elem in res_dict:
            sort_res.append(elem[0])

        self.prob_res = sort_res
        print('prob model sort')
        print(sort_res)

        relation = self.calc_relation(sort_res)
        self.prob_relation = relation
        print('prob model sort relation is {}'.format(relation))

        return sort_res

    # 仿真排序
    def sir_sort(self):
        # 进行仿真
        self.sir_simulation()      
        # 因为进行了1000次 仿真 先计算sir_val的均值
        for node in self.node_list:
            sir_arr = np.array(node.sir_val)
            node.sir_val = np.mean(sir_arr)
        
        ori_dict = {}
        for node in self.node_list:
            ori_dict[node.idx] = node.sir_val
        
        # 对rank排序
        sort_res = []
        res_dict = sorted(ori_dict.items(), key=lambda x: x[1], reverse=True)
        for elem in res_dict:
            sort_res.append(elem[0])

        self.sir_res = sort_res
        print('SIR simulation sort')
        print(sort_res)

        self.sir_plot()
        return sort_res
    # 绘制sir图像
    def sir_plot(self):
        for node in self.node_list:
            # 获得suspect inspect remove 1000次仿真中的最大步长
            max_len = [0,0,0] 
            for i in range(self.simulation_epochs):
                if len(node.suspect_record[i]) > max_len[0]:
                    max_len[0] = len(node.suspect_record[i])
                if len(node.infected_record[i]) > max_len[1]:
                    max_len[1] = len(node.infected_record[i])
                if len(node.removed_record[i]) > max_len[2]:
                    max_len[2] = len(node.removed_record[i])
            # 将不满足最大步长的仿真结果 填充值
            records = [node.suspect_record,node.infected_record,node.removed_record]
            for i in range(len(records)):
                record = records[i]
                for j in range(self.simulation_epochs):
                    record_j_len = len(record[j])
                    diff = max_len[i] - record_j_len
                    for _ in range(diff):
                        record[j].append(record[j][record_j_len-1])
            
            # 进行平均
            node.suspect_record = np.sum(np.array(node.suspect_record),axis=0) / self.simulation_epochs
            node.infected_record = np.sum(np.array(node.infected_record),axis=0) / self.simulation_epochs
            node.removed_record = np.sum(np.array(node.removed_record),axis=0) / self.simulation_epochs

            plt.figure()
            plt.title('Node {} SIR Simulation Result'.format(node.idx))
            plt.plot(node.suspect_record,color='y',label='suspected')
            plt.plot(node.infected_record,color='r',label='infected')
            plt.plot(node.removed_record,color='b',label='removed')
            plt.legend()
            plt.xlabel('Timestep')
            plt.ylabel('Number')
            plt.savefig('DataAnalysisProjectDesign/Experiment3_4/node{}.png'.format(node.idx))

    # 计算相似度
    def calc_relation(self,sort_res):
        # 初始化参数
        n_0,n_1,n_2,n_plus,n_minus = comb(self.node_num,2),0,0,0,0
        relation = 0
        # x表示sir仿真结果 y表示模型计算结果
        for i in range(self.node_num-1):
            x_i,y_i = self.sir_res[i],sort_res[i]
            for j in range(i+1,self.node_num):
                x_j,y_j = self.sir_res[j],sort_res[j]
                if x_i == x_j:
                    n_1 += 1
                if y_i == y_j:
                    n_2 += 1
                if (x_i < x_j and y_i < y_j) or (x_i > x_j and y_i > y_j):
                    n_plus += 1
                if (x_i < x_j and y_i > y_j) or (x_i > x_j and y_i < y_j):
                    n_minus += 1
        relation = (n_plus - n_minus) / np.sqrt((n_0-n_1) * (n_0-n_2))
        return relation

        
if __name__ == "__main__":
    file_name = 'DataAnalysisProjectDesign/Experiment3_4/net5.txt'
    net = Net(file_name)
    net.sir_sort()
    net.degree_sort()
    net.k_shell_degree_sort()
    net.entropy_sort()
    net.k_shell_entropy_sort()
    net.prob_sort()

