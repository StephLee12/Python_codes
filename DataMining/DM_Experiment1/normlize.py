from scipy.io import arff
import pandas as pd
import numpy as np

#获取单个属性值的array
def single_attribute(data_tuple,data_tuple_len):   
    #先用list存储
    a_1,a_2,a_3,a_4 = [],[],[],[]

    
    for i in data_tuple:
        a_1.append(i[0])
        a_2.append(i[1])
        a_3.append(i[2])
        a_4.append(i[3])
        
    #list转换为array
    a_1 =  np.array(a_1)
    a_2 =  np.array(a_2)
    a_3 =  np.array(a_3)
    a_4 =  np.array(a_4)

    return a_1,a_2,a_3,a_4
#最大最小归一化
def normlize_v1(attribute):
    a_min,a_max = attribute.min(), attribute.max()
    new_a = []
    for i in attribute:
        new_elem = (i - a_min) / (a_max-a_min)
        new_a.append(new_elem)
    
    new_a = np.array(new_a)
    return new_a
#使用标准差计算ZSCORE
def normlize_v2(attribute):
    a_mean = attribute.mean()
    a_std = attribute.std()
    new_a  =[]
    for i in attribute:
        new_elem = (i - a_mean) / a_std
        new_a.append(new_elem)
    
    new_a = np.array(new_a)
    return new_a
#使用平均绝对偏差计算ZSCORE
def normlize_v3(attribute):
    a_mean = attribute.mean()
    a_len = len(attribute)
    a_sf = 0
    for i in attribute:
        a_sf = a_sf + abs(i - a_mean)
    
    a_sf = a_sf / a_len
    new_a  = []
    for i in attribute:
        new_elem = (i - a_mean) / a_sf
        new_a.append(new_elem)
    
    new_a =np.array(new_a)
    return new_a
#计算数字特征
def compute(attribute):
    
    a_statistic = []
    a_statistic.append(attribute.min()) #计算最小值
    a_statistic.append(attribute.max()) #计算最大值
    a_statistic.append(attribute.mean()) #计算均值
    a_statistic.append(attribute.std()) #计算标准差

    return a_statistic
#输出数字特征
def print_statistic(a_1,a_2,a_3,a_4):
    print("Sepallength:")
    print("Mininum is " + str(a_1[0]) + " Maximum is " + str(a_1[1]))
    print("Mean is " + str(a_1[2]) + " Std is " + str(a_1[3]))
    print("Sepalwidth:")
    print("Mininum is " + str(a_2[0]) + " Maximum is " + str(a_2[1]))
    print("Mean is " + str(a_2[2]) + " Std is " + str(a_2[3]))
    print("Petallength:")
    print("Mininum is " + str(a_3[0]) + " Maximum is " + str(a_3[1]))
    print("Mean is " + str(a_3[2]) + " Std is " + str(a_3[3]))
    print("Petalwidth:")
    print("Mininum is " + str(a_4[0]) + " Maximum is " + str(a_4[1]))
    print("Mean is " + str(a_4[2]) + " Std is " + str(a_4[3]))
#归一化version1主函数
def version_1(a_1,a_2,a_3,a_4):
    a_1_v1,a_2_v1,a_3_v1,a_4_v1 = normlize_v1(a_1),normlize_v1(a_2),normlize_v1(a_3),normlize_v1(a_4)
    matrix = [a_1_v1,a_2_v1,a_3_v1,a_4_v1]
    matrix = np.mat(matrix)
    print(matrix)
    #计算数字特征
    a_1_v1,a_2_v1,a_3_v1,a_4_v1 = compute(a_1_v1),compute(a_2_v1),compute(a_3_v1),compute(a_4_v1)
    #输出
    print("---------Output results used normlization-v1------------")
    print_statistic(a_1_v1,a_2_v1,a_3_v1,a_4_v1)
#归一化version2主函数
def version_2(a_1,a_2,a_3,a_4):
    a_1_v2,a_2_v2,a_3_v2,a_4_v2 = normlize_v2(a_1),normlize_v2(a_2),normlize_v2(a_3),normlize_v2(a_4)
    #计算数字特征
    a_1_v2,a_2_v2,a_3_v2,a_4_v2 = compute(a_1_v2),compute(a_2_v2),compute(a_3_v2),compute(a_4_v2)
    #输出
    print("---------Output results used normlization-v2------------")
    print_statistic(a_1_v2,a_2_v2,a_3_v2,a_4_v2)
#归一化version3主函数
def version_3(a_1,a_2,a_3,a_4):
    a_1_v3,a_2_v3,a_3_v3,a_4_v3 = normlize_v3(a_1),normlize_v3(a_2),normlize_v3(a_3),normlize_v3(a_4)
    #计算数字特征
    a_1_v3,a_2_v3,a_3_v3,a_4_v3 = compute(a_1_v3),compute(a_2_v3),compute(a_3_v3),compute(a_4_v3)
    #输出
    print("---------Output results used normlization-v3------------")
    print_statistic(a_1_v3,a_2_v3,a_3_v3,a_4_v3)

if __name__ == "__main__":
    # 读取arff文件
    dataset = arff.loadarff('DM_Experiment1/iris.arff')
    # 获取有效数据的元组
    data_tuple = dataset[0]
    df = pd.DataFrame(data_tuple)
    #print(df.values)
    #获取元组的长度
    data_tuple_len = len(data_tuple)
    #分别获取attribute
    a_1,a_2,a_3,a_4 = single_attribute(data_tuple,data_tuple_len)
    #进行归一化version1
    version_1(a_1,a_2,a_3,a_4)
    #进行归一化version2
    version_2(a_1,a_2,a_3,a_4)
    #进行归一化version3
    version_3(a_1,a_2,a_3,a_4)



