import numpy as np
import pandas as pd
from scipy.io import arff
import heapq
import math


# 加载数据
def load():
    #读取arff
    dataset = arff.loadarff('DM_Experiment1/labor.arff')
    #获取dataframe
    df = pd.DataFrame(dataset[0])
    #获取列名
    columns = list(df)
    nominal_column = []
    #获取标称属性的列名及对应的下标
    nominal_index = []
    count = 0
    for column in columns:
        if df[column].dtypes == 'object':
            nominal_column.append(column)
            nominal_index.append(count)

        count += 1
    return nominal_column, nominal_index, columns, df


# 数值属性用均值填充  标称属性用众数填充
def version_1(nominal_column, columns, df):
    #进行缺失值修改
    for column in columns:
        if column not in nominal_column:
            #如果是numeric类型，采用均值进行替换
            series = df[column]
            mean = series.mean()
            df[column].fillna(mean, inplace=True)
        else:
            #是标称型数据
            series = df[column]
            #获得没有？的series
            new_series = series[series != b'?']
            #求新series的众数 .mode()返回的还是series，根据下标得到mode
            mode = new_series.mode().get(0)
            series[series == b'?'] = mode


# 计算相似度 用5个最相似的均值和众数处理
def version_2(nominal_column, columns, sim_list, df):
    #进行缺失值修改
    column_count = 0
    for column in columns:
        if column not in nominal_column:  #如果是numeric类型
            for i in range(df.shape[0]):
                #处理缺失值
                if math.isnan(df.iat[
                        i, column_count]):  #df.iat[i,column_count].isnan():
                    sim = sim_list[i]  #得到缺失值位置的相似度向量
                    max_index = get_simlist_max(sim, len(sim))  #得到相似度从高到低排序的下标
                    first_index = max_index[0:5]  #每五个不断向后
                    mean = 0
                    counter_1 = 0  #如果相似度高对应位置也是nan 加1
                    for j in first_index:
                        if math.isnan(df.iat[j,
                                             column_count]):  #若出现对应的值是nan的情况
                            counter_1 += 1
                            continue
                        mean += df.iat[j, column_count]
                    if mean == 0:  #最初的五个相似度最高的均为nan
                        count = 1  #迭代下标数组
                        while mean == 0 and count < 53:
                            counter_2 = 0  #若出现对应的值是nan 加1
                            tmp_index = max_index[0 + count:5 + count]
                            for j in tmp_index:
                                if math.isnan(
                                        df.iat[j,
                                               column_count]):  #若出现对应的值是nan的情况
                                    counter_2 += 1
                                    continue
                                mean += df.iat[j, column_count]
                            if counter_2 == 5:  #避免除0的情况出现
                                count += 1
                                continue
                            else:
                                count += 1
                                df.iloc[i,
                                        column_count] = mean / (5 - counter_2)
                    else:
                        df.iloc[i, column_count] = mean / (5 - counter_1)
            column_count += 1
        else:  #是标称属性
            for ii in range(df.shape[0]):
                if df.iat[ii, column_count] == b'?':  #对于出现'?'的情况
                    sim = sim_list[ii]  #获得相似度向量 下面的步骤与数值属性基本相同
                    max_index = get_simlist_max(sim, len(sim))
                    first_index = max_index[0:5]
                    new_series = df.iloc[first_index, column_count]
                    mode = new_series.mode().get(0)
                    if mode == b'?':
                        count = 1
                        while mode == b'?' and count < 53:  #若出现mode是'?'
                            tmp_index = max_index[0 + count:5 + count]
                            new_series = df.iloc[tmp_index, column_count]
                            mode = new_series.mode().get(0)
                            count = count + 1
                        df.iloc[ii, column_count] = mode
                        count += 1
                    else:
                        df.iloc[ii, column_count] = mode
            column_count += 1


#对数值变量进行归一化
def normlization_part(attribute):
    index = np.argwhere(np.isnan(attribute))
    is_nan_index = []  #数据为nan的下标
    for i in index:
        is_nan_index.append(i[0])
    tmp = []  # 重新获取attribute_list中值不为nan的
    for i in range(len(attribute)):
        if i not in is_nan_index:
            tmp.append(attribute[i])
    tmp = np.array(tmp)  # list to array
    a_min, a_max = tmp.min(), tmp.max()
    new_a = []
    #归一化处理
    for i in range(len(attribute)):
        if i not in is_nan_index:
            new_elem = (attribute[i] - a_min) / (a_max - a_min)
            new_a.append(new_elem)
    return new_a, is_nan_index


def normlization(mat, nominal_index):
    for i in range(mat.shape[1]):
        if i not in nominal_index:  #数值属性列
            column = mat[:, i]
            column = list(map(float, column))  #将字符串转换为float
            column = np.array(column)
            column, is_nan_index = normlization_part(column)  #数据为nan的下标
            count = 0
            for j in range(mat.shape[0]):
                if j not in is_nan_index:
                    mat[j, i] = column[count]  #替换
                    count += 1
    return mat


#计算相似度 数值属性用欧氏距离
def cal_sim(mat, nominal_index):
    # shape[0]是矩阵行数 shape[1]是矩阵列数
    sim_list = [[0 for i in range(mat.shape[0])] for j in range(mat.shape[1])]
    for i in range(len(sim_list)):
        for j in range(len(sim_list)):
            if i == j:
                #对角线元素均为0
                sim_list[i][j] = 0
                continue

            obj_1, obj_2 = mat[i], mat[j]  #两个数据对象
            count_nominal = 0  #计算相同的标称变量
            #两个数值向量 用于计算欧式距离
            vec_i = []
            vec_j = []

            for k in range(mat.shape[1]):
                if k not in nominal_index:
                    #若是数值属性
                    if ((obj_1[:, k] == b'nan')
                            or (obj_2[:, k] == b'nan')):  #若某个对象的k属性不存在
                        continue
                    else:
                        num_i, num_j = float(obj_1[:, k]), float(obj_2[:, k])
                        vec_i.append(num_i)
                        vec_j.append(num_j)
                else:
                    #若是标称属性
                    if obj_1[:, k] == b'?' or obj_2[:, k] == b'?':
                        continue
                    elif obj_1[:, k] == obj_2[:, k]:
                        count_nominal += 1

            #计算欧式距离即数值变量的相似度
            vec_i, vec_j = np.array(vec_i), np.array(vec_j)
            sim_num = np.sqrt(np.sum(np.square(vec_j - vec_j)))

            #计算标称属性的相似度
            sim_nom = count_nominal / len(nominal_index)

            sim = sim_nom + sim_num
            sim_list[i][j] = sim
    return sim_list


#获得最大k元素的下标
def get_simlist_max(simlist, k):
    index = sorted(range(len(simlist)), key=lambda sub: simlist[sub])[-k:]
    index.reverse()
    return index


if __name__ == "__main__":
    # 加载数据
    nominal_column, nominal_index, columns, df = load()
    #version_1 数值属性用均值填充  标称属性用众数填充
    version_1(nominal_column, columns, df)
    #将dataframe转换为矩阵
    df_list = df.values.tolist()
    df_mat = np.mat(df_list)
    #矩阵归一化
    df_mat = normlization(df_mat, nominal_index)
    #计算相似度
    sim_list = cal_sim(df_mat, nominal_index)
    #version_2 用相似度最高的5个的均值和众数填充
    version_2(nominal_column, columns, sim_list, df)
