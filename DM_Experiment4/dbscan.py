import numpy as np 
import pandas as pd
import random
import matplotlib.pyplot as plt

from preprocessing import Data

class DBSCAN:
    def __init__(self,data,epsilon=0.9,min_pts=6):
        self.data = data
        self.epsilon = epsilon
        self.min_pts = min_pts

    # 获取数据向量
    def get_data_vector(self,row):
        vec = []
        for col_name in self.data.fea_column:
            vec.append(self.data.df.loc[row,col_name])
        
        return np.array(vec)

    # 判断每条数据是否为核心对象
    def is_core(self):
        df = self.data.df
        for row in range(df.shape[0]):
            vec = self.get_data_vector(row)
            #计算epsilon邻域内的点数
            in_epsilon = [] #记录在邻域中的点索引
            p_count = 0 #统计在邻域中的数目
            for row_1 in range(df.shape[0]):
                if row_1 != row:
                    vec_1 = self.get_data_vector(row_1)
                    dis = np.sqrt(np.sum(np.square(vec-vec_1))) #计算欧氏距离
                    if dis <= self.epsilon:
                        in_epsilon.append(row_1)
                        p_count += 1
            # p_count与minpst比较
            if p_count >= self.min_pts:
                df.loc[row,self.data.core_column] = 1 #该点为核心对象
                df.at[row,self.data.in_epsilon_column] = in_epsilon #记录邻域内的结点                

    def expand_cluster(self,rand_choice,in_epsilon,non_selected,cluster_count):
        df = self.data.df
        
        i = 0
        pop_list = []
        while in_epsilon != []:

            if in_epsilon[i] in non_selected: # p在未被处理的列表里 将其标记为处理过 即从not_selected中弹出
                idx = non_selected.index(in_epsilon[i])
                non_selected.pop(idx)
                df.loc[in_epsilon[i],self.data.cluster_column] = cluster_count #标记簇号
            
            # 判断p是否为核心对象 如果是 则将其邻域中的点加入in_epsilon
            if df.loc[in_epsilon[i],self.data.core_column] == 1:
                p_in_epsilon = df.loc[in_epsilon[i],self.data.in_epsilon_column]
                for item in p_in_epsilon:
                    if item in in_epsilon or item == rand_choice or item in pop_list:
                        continue
                    else:
                        in_epsilon.append(item)
            #将p从in_epsilon中弹出 因为要加入新的元素
            idx = in_epsilon.index(in_epsilon[i])
            pop_list.append(in_epsilon.pop(idx))
        return non_selected

    def dbscan_main(self):
        df = self.data.df
        # 未被处理的点的列表
        non_selected = list(np.linspace(0,self.data.df.shape[0]-1,self.data.df.shape[0],dtype=int))
        
        self.is_core() # 遍历一次 数据集 判断每个点是否为核心结点
        cluster_count = 0 #记录簇号
        # 遍历non_selected
        while non_selected != []:
            rand_choice = random.choice(non_selected) #在non_selected中随机抽取一个点
            core_flag = df.loc[rand_choice,self.data.core_column]
            if core_flag == 0: #不是核心对象
                idx = non_selected.index(rand_choice)
                non_selected.pop(idx)
            else: #若为核心对象
                df.loc[rand_choice,self.data.cluster_column] = cluster_count #标记簇号
                # 将该结点标记为已处理 即从non_selected中弹出
                idx = non_selected.index(rand_choice)
                non_selected.pop(idx)
                in_epsilon = df.loc[rand_choice,self.data.in_epsilon_column] #获得邻域内的点
                non_selected=self.expand_cluster(rand_choice,in_epsilon,non_selected,cluster_count)
                cluster_count += 1
    
    def show_res(self):
        
        cluster_size = []
        # 画图
        ax = None
        color = ['r','g','b']
        cluster_num = self.data.df[self.data.cluster_column].unique().shape[0]
        for idx in range(cluster_num):
            df = self.data.df[self.data.df[self.data.cluster_column] == idx]
            df_size = str(df.shape[0])
            # 获取类别
            label = str(df[self.data.class_column].mode().get(0),encoding='utf-8')
            cluster_size.append('cluster '+str(idx)+ ' '+label + '\'s size:'+df_size)
            ax = df.plot.scatter(
                x=self.data.fea_column[0],
                y=self.data.fea_column[1],
                color = color[idx],
                label = label,
                ax = ax
            )
        for elem in cluster_size:
            print(elem)
        plt.title('DBSCAN Clustering Result')
        plt.show()

