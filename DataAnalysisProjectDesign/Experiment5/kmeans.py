import pandas as pd
import numpy as np
import os
import math

# subclass
import preprocess


class KMeans:
    def __init__(self, info_path, data_path, iterations=500):
        self.info_path = info_path  #股票信息数据的路径
        self.data_path = data_path  #股票数据的路径
        self.info_dict = self.get_info_dict()
        self.stock_dict = self.get_stock_dict()

        self.data_df = self.transform_dict_to_df()  #存储数据的dataframe

        self.cluster_avg = {}  #每个簇的质心
        self.cluster_labels, self.cluster_num = self.get_cluster_info(
        )  #簇的labels 聚类簇的数量
        self.iterations = iterations  # 最大迭代次数
        self.actual_iterations = None  # 实际迭代次数
        self.clusters ={} # 聚类后的结果

        self.purity = None  #聚类的纯度
        self.nmi_mat = None # 聚类的NMI矩阵

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

    # 初始化 选取k个质心
    def cluster_initialize(self):

        for elem in self.cluster_labels:
            idx = self.cluster_labels.index(elem)
            # 在数据集中找任意一个 field为elem的数据
            series = self.data_df[self.data_df['Field'] == elem]
            # 切分整个serires得到股票数据
            series = series.iloc[0, :-2].values
            # 将field和series作为键值对存储
            self.cluster_avg[idx] = series

    # 计算相似度
    def calc_similarity(self, series_1, series_2):
        pho = 0

        std_1, std_2 = np.std(series_1), np.std(series_2)
        mean_1, mean_2 = np.mean(series_1), np.mean(series_2)

        for i in range(len(series_1)):
            pho += (series_1[i] - mean_1) * (series_2[i] - mean_2)

        pho /= (std_1 * std_2 * len(series_1))
        return pho

    # 对于每个簇 以该簇的众数作为该簇的类别
    def specify_cluster(self):
        
        missing_cluster = [] #记录缺失的簇
        for i in range(self.cluster_num):
            # 获得每个簇的数据切片
            data = self.data_df[self.data_df['Cluster']==i].iloc[:,[-2,-1]]
            # 获得该簇的类别
            mode = None # 该簇的类别
            mode_list = list(data['Field'].mode()) #多数类的列表 因为可能会有相同数量的label
            count = 0 # 循环计数
            flag = 0 # 如果有缺失的簇 置1
            while True:
                # 若字典中已经有此类 则计数加1
                if self.clusters.__contains__(mode_list[count]):
                    count += 1
                    # 属于缺失簇的情况 flag置1 将数据保存
                    if count == len(mode_list):
                        missing_cluster.append(data)
                        flag = 1
                        break
                else:
                    # 字典中没有此类 获得该类
                    mode = mode_list[count]
                    break
                # 簇中类别为mode的个数
            if flag == 0:
                mode_num = data[data['Field']==mode].shape[0]
                self.clusters[mode] = [mode_num,data]

        # 处理缺失的簇
        labels_set = set(self.cluster_labels) #
        cluster_set = set(list(self.clusters.keys()))
        missing_labels = list(labels_set - cluster_set)
        for i in range(len(missing_labels)):
            self.clusters[missing_labels[i]] = [2,missing_cluster[i]]

    # 进行聚类的主函数
    def cluster_main(self):

        self.cluster_initialize()  #初始化质心
        func_e = 0  #初始化准则函数
        flag = 0  #用于判断func_e是否改变 flag若置1 说明迭代结束
        iteration = 0  #记录迭代次数

        while iteration <= self.iterations and flag == 0:
            # 遍历数据集 将每条数据分配到最近的簇
            for elem in self.data_df.iterrows():
                idx = elem[0]
                # 获得数据的时间序列
                series = elem[1].iloc[:-2].values
                # 分配到最近的簇
                sim = 0
                belong_cluster = None
                # 遍历所有的簇中心点
                for key in self.cluster_avg:
                    avg = self.cluster_avg[key]
                    calc_sim = self.calc_similarity(series, avg)
                    #calc_sim = round(calc_sim,4)
                    if abs(calc_sim) > sim:
                        sim = abs(calc_sim)
                        belong_cluster = key
                self.data_df.at[idx, 'Cluster'] = belong_cluster

            # 更新簇的均值
            for key in self.cluster_avg:
                old_avg = self.cluster_avg[key]
                # 获得该簇的所有数据
                data = self.data_df[self.data_df['Cluster'] ==
                                    key].iloc[:, :-2]
                new_avg = np.empty_like(old_avg)
                # 按列遍历data 获得每列的均值
                for elem in data.iteritems():
                    idx, series = elem[0], elem[1]
                    new_avg[idx] = series.mean()
                # 更新均值
                self.cluster_avg[key] = new_avg

            # 计算准则函数
            tmp_func_e = 0
            # 遍历每个簇
            for key in self.cluster_avg:
                avg = self.cluster_avg[key]
                data = self.data_df[self.data_df['Cluster'] == key]
                #按行遍历 avg 计算绝对值平方和
                for elem in data.iterrows():
                    idx, series = elem[0], elem[1]
                    # 获得每条序列的数据
                    series = series.iloc[:-2].values
                    res = np.sum(np.square(avg - series))
                    tmp_func_e += res

            # 判断准则函数是否变化
            if round(tmp_func_e, 4) == round(func_e, 4):
                flag = 1  #func_e不再变化 循环结束
            else:
                func_e = round(tmp_func_e, 4)

            iteration += 1

        self.actual_iterations = iteration
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
    kmeans_obj = KMeans(info_path, data_path)
    kmeans_obj.cluster_main()
    purity = kmeans_obj.calc_purity()
    nmi_mat = kmeans_obj.calc_nmi()
    print('Purity is {}'.format(purity))
    print(nmi_mat)