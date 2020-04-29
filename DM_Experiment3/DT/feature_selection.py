import numpy as np 
from scipy.io import arff
import pandas as pd

# 加载数据
def load():
    dataset = arff.loadarff('DM_Experiment1/iris.arff')
    df = pd.DataFrame(dataset[0])
    columns = list(df)
    last_column = columns.pop()

    return df,columns,last_column
#计算信息熵
def entropy(df,att_name):
    
    new_data_att = df[att_name].unique()#去除重复值
    ent = 0
    for i in new_data_att:
        pr = sum(df[att_name] == i) / df.shape[0] #计算概率
        ent += pr * np.log(pr) #计算熵
    
    return -ent
#计算条件信息熵
def case_entropy(df,x_att,y_att):

    new_x_att = df[x_att].unique()

    pr_x = df[x_att].value_counts() / df.shape[0]
    c_ent = 0

    for i in new_x_att:
        c_ent += pr_x[i] * entropy(df[df[x_att] == i], y_att)
    return c_ent
#计算信息增益  
def IG(df,columns,last_column):
    ig = []
    ent = entropy(df,last_column)
    for column in columns:
        c_ent = case_entropy(df,column,last_column)
        ig.append(ent - c_ent)
    return ig

#计算信息增益率
def split_info(df,att):
    new_att = df[att].unique()
    pr_x = df[att].value_counts() / df.shape[0]
    splitInfo = 0

    for i in new_att:
        splitInfo += pr_x[i] * np.log2(pr_x[i])
    
    return -splitInfo

def gain_ratio(ig,splitInfo):
    return ig / splitInfo

if __name__ == "__main__":
    df,columns,last_column= load()
    ig = IG(df,columns,last_column)
    #print(ig)
    print(np.log2())