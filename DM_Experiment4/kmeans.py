import numpy as np 
import pandas as pd
import random

from preprocessing import Data

class KMeans:
    
    def __init__(self,data,k):
        self.data = data
        self.cluster_num = k

    # 获得数据的最小值和最大值 以便分配初始值
    def get_data_min_max(self):
        min_max = [] #每个元素都是一个tuple 存储min max

        for col_name in self.data.fea_column:
            min_val = self.data.df[col_name].min()
            max_val = self.data.df[col_name].max()
            min_max.append((min_val,max_val))
        
        return min_max
    
    # 初始化k个值 
    def k_initialize(self):
        init_val = [] #每个元素都是一个tuple

        min_max = self.get_data_min_max()
        for _ in range(self.cluster_num):#获取k个初始值
            tmp = [0] * len(min_max)
            for j in range(len(min_max)):
                # 将随机树保留4位小数
                rand_num = round(random.uniform(min_max[j][0],min_max[j][1]),4)
                tmp[j] = rand_num
            init_val.append(tuple(tmp))
        
        return init_val