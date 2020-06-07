import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import random

from preprocessing import Data

inf = float('inf')

class KMeans:
    
    def __init__(self,data,k,iteration=500):
        self.data = data
        self.cluster_num = k
        self.iteration = iteration
        self.cluster_avg = None

    # 获得数据的最小值和最大值 以便分配初始值
    def get_data_min_max(self):
        min_max = [] #每个元素都是一个list 存储min max

        for col_name in self.data.fea_column:
            min_val = self.data.df[col_name].min()
            max_val = self.data.df[col_name].max()
            min_max.append([min_val,max_val])
        
        return min_max
    
    # 初始化k个值 
    def k_initialize(self):
        init_val = [] #每个元素都是一个list

        min_max = self.get_data_min_max()
        for _ in range(self.cluster_num):#获取k个初始值
            tmp = [0] * len(min_max)
            for j in range(len(min_max)):
                # 将随机树保留4位小数
                rand_num = round(random.uniform(min_max[j][0],min_max[j][1]),2)
                tmp[j] = rand_num
            init_val.append(tmp)
        
        return init_val
    
    def kmeans_main(self):
        #初始化簇的平均值  每一个元素都是一个元组
        cluster_avg = self.k_initialize()
        # 初始化准则函数
        func_e = 0
        flag = 0 #用于判断func_e是否变化 flag==0变化 flag==1不变化
        iteration = 0
        while iteration <= self.iteration and flag == 0:
            #遍历数据集 将每个数据分配到最近的簇
            for row in range(self.data.df.shape[0]):
                # 获得一条数据
                vec = [] #储存一条数据的向量
                for col_name in self.data.fea_column:
                    vec.append(self.data.df.loc[row,col_name])
                vec = np.array(vec) # list to array
                #计算最近的距离
                cluster_idx = -1
                dis = inf
                for idx in range(self.cluster_num):
                    avg = np.array(cluster_avg[idx]) #list to array
                    tmp_dis = np.sqrt(np.sum(np.square(vec-avg))) #计算欧式距离
                    if tmp_dis < dis:
                        dis = tmp_dis
                        cluster_idx = idx
                # cluster_idx为该条数据属于的簇
                self.data.df.loc[row,self.data.cluster_column] = cluster_idx
            
            # 更新簇的平均值
            for idx in range(self.cluster_num):
                #找到cluster_column列中值为idx的所有数据
                data = self.data.df[self.data.df[self.data.cluster_column] == idx]
                tmp_count = 0
                for col_name in self.data.fea_column:
                    col_mean = round(data[col_name].mean(),4) #获取均值
                    cluster_avg[idx][tmp_count] = col_mean #更新
                    tmp_count += 1
            
            tmp_func_e = 0
            # 计算准则函数E
            for idx in range(self.cluster_num):
                #找到cluster_column列中值为idx的所有数据
                data = self.data.df[self.data.df[self.data.cluster_column] == idx]
                # 计算绝对值平方和
                for row in range(data.shape[0]):
                    # 获得一条数据
                    vec = [] #储存一条数据的向量
                    for col_name in self.data.fea_column:
                        vec.append(self.data.df.loc[row,col_name])
                    vec = np.array(vec) # list to array
                    avg = np.array(cluster_avg[idx]) #簇的平均值
                    res = np.sum(np.square(vec-avg)) #计算结果
                    tmp_func_e += res
            
            if round(tmp_func_e,4) == round(func_e,4): #若e值不再变化
                flag = 1
            else:
                func_e = round(tmp_func_e,4)
            
            iteration += 1
        
        self.cluster_avg = cluster_avg
    
    def show_res(self):
        
        cluster_size = []
        # 画图
        ax = None
        color = ['r','g','b']
        for idx in range(self.cluster_num):
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
        plt.title('K-Means Clustering Result')
        plt.show()