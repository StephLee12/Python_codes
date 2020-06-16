import pandas as pd
import numpy as np
import os
import math
import random

# subclass
import preprocess

from kmeans import KMeans

class DBSCAN:
    def __init__(self,info_path,data_path,epsilon,min_pts):
        #super(DBSCAN,self).__init__(info_path,data_path)
        self.info_path = info_path  #股票信息数据的路径
        self.data_path = data_path  #股票数据的路径
        self.info_dict = self.get_info_dict()
        self.stock_dict = self.get_stock_dict()

        self.data_df = self.transform_dict_to_df()  #存储数据的dataframe

        self.epsilon = epsilon
        self.min_pts = min_pts
        self.cluster_labels, self.cluster_num = self.get_cluster_info(
        )  #簇的labels 要聚类簇的数量
        self.actual_cluster_num = None #实际聚类的簇数目
        self.clusters = {} # 聚类后的结果
 
        self.purity = None
        self.nmi_mat = None

    
    # 获得股票信息词典
    def get_info_dict(self):
        # 若有已经保存的字典直接读取
        if os.path.exists(
                'DataAnalysisProjectDesign/Experiment5/stock_info.pkl'):
            return preprocess.read_dict(
                'DataAnalysisProjectDesign/Experiment5/stock_info.pkl')
        else:
            return preprocess.get_stock_field(self.info_path)

    # 获得股票数据词典
    def get_stock_dict(self):
        # 若有已经保存的字典直接读取
        if os.path.exists(
                'DataAnalysisProjectDesign/Experiment5/stock_data.pkl'):
            return preprocess.read_dict(
                'DataAnalysisProjectDesign/Experiment5/stock_data.pkl')
        else:
            return preprocess.get_stocks(self.data_path)

    # 将每个公司的数据合并为一个dataframe
    def transform_dict_to_df(self):
        # 将数据用dataframe存储
        df = pd.DataFrame(self.stock_dict)
        df = df.T  # 获得df的转置
        # 在最后插入一个新的列  标明每个公司对应的领域
        df.insert(df.shape[1], 'Field', '')
        for row in df.iterrows():
            idx = row[0]  # 获得公司名
            # 在字典信息中查询公司对应的领域
            field = self.info_dict[idx][0]
            # 填充数据
            df.at[idx, 'Field'] = field

        # 在最后插入新的一列，用于存储聚类结果
        df.insert(df.shape[1], 'Cluster', '')
        # 加入一列 存储该数据是否为核心对象
        df.insert(df.shape[1],'Core',0)
        # 加入一列 若该数据为核心对象 判断哪些点在它的epsilon邻域
        df.insert(df.shape[1],'Epsilon','')
        # 返回df的转置
        return df

    # 获取要聚类的簇数目
    def get_cluster_info(self):
        all_fields = list(self.data_df['Field'].unique())
        nan_idx = 0
        for elem in all_fields:
            if type(elem) == float:
                nan_idx = all_fields.index(elem)
        all_fields.pop(nan_idx)

        return all_fields, len(all_fields)

    # 判断每条数据是否为核心对象
    def is_core(self):
        df = self.data_df
        for elem_1 in df.iterrows():
            idx_1,series_1 = elem_1[0],elem_1[1].iloc[:-4]
            # 计算epsilon邻域内的点的个数
            in_epsilon = []
            p_count = 0
            for elem_2 in df.iterrows():
                idx_2,series_2 = elem_2[0],elem_2[1].iloc[:-4]
                if idx_1 != idx_2:
                    # 计算距离
                    dis = np.sqrt(np.sum(np.square(series_1-series_2)))
                    if dis <= self.epsilon:
                        in_epsilon.append(idx_2)
                        p_count += 1
            # 根据p_count判断是否为核心对象
            if p_count >= self.min_pts:
                df.loc[idx_1,'Core'] = 1
                df.at[idx_1,'Epsilon'] = in_epsilon

    # 簇扩展
    def expand_cluster(self,rand_choice,in_epsilon,non_selected,cluster_count):
        df = self.data_df

        i = 0
        pop_list = []
        # 直到邻域为空
        while in_epsilon != []:
            
            # 若点未被处理 将其标记为已处理 并标记簇号
            if in_epsilon[i] in non_selected:
                idx = non_selected.index(in_epsilon[i])
                non_selected.pop(idx)
                df.loc[in_epsilon[i],'Cluster'] = cluster_count
            
            # 判断邻域内的点 是否有核心对象
            # 若有 将该点的邻域也加入in_eplison
            if df.loc[in_epsilon[i],'Core'] == 1:
                p_in_epsilon = df.loc[in_epsilon[i],'Epsilon']
                # 
                for item in p_in_epsilon:
                    if item in in_epsilon or item == rand_choice or item in pop_list:
                        continue
                    else:
                        in_epsilon.append(item)
            
            # 将该点 即p弹出
            idx = in_epsilon.index(in_epsilon[i])
            pop_list.append(in_epsilon.pop(idx))   
        
        return non_selected
    
    # 对于每个簇 以该簇的众数作为该簇的类别
    def specify_cluster(self):
        
        df = self.data_df
        missing_cluster = [] #记录缺失的簇

        for i in range(self.cluster_num):
            # 获得每个簇的数据切片
            data = df[df['Cluster']==i].iloc[:,[-4,-3]]
            # 获得该簇的类别
            mode = None
            mode_list = list(data['Field'].mode())
            count = 0
            flag = 0
            while True:
                if self.clusters.__contains__(mode_list[count]):
                    count += 1
                    if count == len(mode_list):
                        missing_cluster.append(data)
                        flag = 1
                        break
                else:
                    mode = mode_list[count]
            
            if flag == 0:
                mode_num = data[data['Field']==mode].shape[0]
                self.clusters[mode] = [mode_num,data]
        
        # 处理缺失值
        labels_set = set(self.cluster_labels)
        cluster_set = set(list(self.clusters.keys()))
        missing_labels = list(labels_set - cluster_set)
        for i in range(len(missing_labels)):
            self.clusters[missing_labels[i]] = [2,missing_cluster[i]]

    # 聚类的主函数
    def cluster_main(self):
        df = self.data_df
        # 初始化未处理的点的列表
        non_selected = list(self.info_dict.keys())

        # 扫描一遍数据库 判断每个节点 是否为核心节点
        self.is_core()
        cluster_count = 0 #记录簇号
        
        # 直至non_selected为空 即所有点都被处理 循环结束
        while non_selected != []:
            # 随机抽取一个点 并判断该点是否为核心对象
            rand_choice = random.choice(non_selected)
            core_flag = df.loc[rand_choice,'Core']
            if core_flag == 0: #不是核心对象
                idx = non_selected.index(rand_choice)
                non_selected.pop(idx)
            else: # 若为核心对象
                df.loc[rand_choice,'Cluster'] = cluster_count #标记簇号
                # 该点标记为已处理
                idx = non_selected.index(rand_choice)
                non_selected.pop(idx)
                # 获得邻域内的点 进行簇扩展
                in_epsilon = df.loc[rand_choice,'Epsilon']
                non_selected = self.expand_cluster(rand_choice,in_epsilon,non_selected,cluster_count)
                cluster_count += 1

        self.actual_cluster_num= cluster_count
        if self.actual_cluster_num == self.cluster_num:
            self.specify_cluster()
        

    # 计算纯度
    def calc_purity(self):
        purity = 0

        for label in self.clusters:
            # 获得该簇中最多类别的数据的数量
            num = self.clusters[label][0]
            purity += num
        purity /= self.data_df.shape[0]    

        return purity

    # 辅助NMI计算 计算互信息
    def calc_information(self, label_i, label_j):
        info = 0

        # 获得聚类簇类别为label_i的数据
        label_i_data = self.clusters[label_i]
        label_i_df = label_i_data[1]
        # 获得该数据的size
        label_i_num = label_i_data[0]
        # 获得标签簇的数据的size
        label_j_num = self.data_df[self.data_df['Field']==label_j].shape[0]
        # 获得两者交集的size
        interset_num = label_i_df[label_i_df['Field']==label_j].shape[0]
        # 获得整个文档树的size
        total_length = self.data_df.shape[0]

        info = (interset_num / total_length) * math.log((total_length * interset_num) / (label_i_num * label_j_num) + 1e-3)
        

        return info

    # 辅助NMI计算 计算信息熵
    def calc_entropy(self, label):
        ent = 0

        label_num = self.clusters[label][0]  
        total_length = self.data_df.shape[0]

        ent = - (label_num / total_length) * math.log((label_num / total_length) + 1e-4)

        return ent

    # 计算NMI
    def calc_nmi(self):
        mat_shape = len(self.cluster_labels)
        # 行索引i作为聚类簇 列索引j作为标签簇
        nmi_mat = np.zeros([mat_shape, mat_shape])

        for i in range(mat_shape):
            label_i = self.cluster_labels[i]
            for j in range(mat_shape):
                label_j = self.cluster_labels[j]
                info_ij = self.calc_information(label_i, label_j)
                ent_i, ent_j = self.calc_entropy(label_i), self.calc_entropy(
                    label_j)
                nmi_mat[i][j] = 2 * info_ij / (ent_i + ent_j)
        
        self.nmi_mat = nmi_mat
        return nmi_mat


if __name__ == "__main__":
    info_path = 'DataAnalysisProjectDesign/Experiment5/stock_info.csv'
    data_path = 'DataAnalysisProjectDesign/Experiment5/data'
    arg_list = [
        (0.2,20),(0.2,25),(0,2,30),
        (0.25,20),(0.25,25),(0.25,30),
        (0.3,20),(0.3,25),(0.3,30),
        (0.35,20),(0.35,25),(0.35,30),
        (0.4,20),(0.4,25),(0.4,30),
        (0.45,20),(0.45,25),(0.4,30),
        (0.5,20),(0.5,25),(0.5,30)
    ]
    epsilon,min_pts = arg_list[9][0],arg_list[9][1]
    dbscan_obj = DBSCAN(info_path, data_path,epsilon,min_pts)
    dbscan_obj.cluster_main()
    purity = dbscan_obj.calc_purity()
    nmi_mat = dbscan_obj.calc_nmi()
    print('Purity is {}'.format(purity))
    print(nmi_mat)