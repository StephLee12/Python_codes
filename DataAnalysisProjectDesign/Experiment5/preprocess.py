import pandas as pd
import numpy as np 
import math
import os
import pickle



def get_stock_field(path):
    # 函数返回值为一个字典
    # 键为公司名 值为公司对应领域

    # 股票信息的路径
    #path = 'DataAnalysisProjectDesign/Experiment5/stock_info.csv'
    # 读取股票对应的领域
    stock_df = pd.read_csv(path,
                           usecols=[0, 2],
                           header=None,
                           names=['Company', 'Field'])
    # dataframe转换为dict key为公司名  value为对应的领域
    stock_dict = stock_df.set_index('Company').T.to_dict('list')

    # 保存字典
    f = open('DataAnalysisProjectDesign/Experiment5/stock_info.pkl', 'wb')
    pickle.dump(stock_dict, f)
    f.close()

    return stock_dict


def get_stocks(path):
    # 返回一个字典
    # 键为公司名 值为对应的数据

    # 数据集的根路径
    #path = 'DataAnalysisProjectDesign/Experiment5/data'
    file_list = os.listdir(path)

    data_dict = {}

    # 依次读取每个文件
    for each_file in file_list:
        tmp_list = []
        company_str = each_file.split('.')[0]
        with open(path + '/' + each_file, 'r') as f:
            # 获取每一行的数据
            line = f.readlines()
            # 对行数据进行处理 获得股票价格差值
            for elem in line:
                elem = elem.strip()
                elem = elem.split()
                # 对数据归一化  R_i = ln P_(i+1) - ln P_i 
                p,adv_p = float(elem[3]),float(elem[4])
                tmp_list.append(math.log(adv_p)-math.log(p))
        data_dict[company_str] = tmp_list

    #保存字典 方便再一次读取
    f = open('DataAnalysisProjectDesign/Experiment5/stock_data.pkl', 'wb')
    pickle.dump(data_dict, f)
    f.close()

    return data_dict


def read_dict(path):
    f = open(path, 'rb')
    data_dict = pickle.load(f)

    return data_dict


if __name__ == "__main__":
    get_stock_field('DataAnalysisProjectDesign/Experiment5/stock_info.csv')
    get_stocks('DataAnalysisProjectDesign/Experiment5/data')
