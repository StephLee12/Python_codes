import numpy as np 
import pandas as pd

class IGAttribute:
    def __init__(self,value,att=None,ig_value=None):
        self.name = att
        self.value = value
        self.ig_value = ig_value

def entropy(df,att_name):
    
    new_data_att = df[att_name].unique()#去除重复值
    ent = 0
    for i in new_data_att:
        pr = sum(df[att_name] == i) / df.shape[0] #计算概率
        ent += pr * np.log(pr) #计算熵
    
    return -ent
#计算条件信息熵 标称型数据
def case_entropy_discrete(df,x_att,y_att):

    new_x_att = df[x_att].unique()

    pr_x = df[x_att].value_counts() / df.shape[0]
    c_ent = 0

    for i in new_x_att:
        c_ent += pr_x[i] * entropy(df[df[x_att] == i], y_att)
    return c_ent
#计算条件信息熵 数值型数据
# 产生划分点集合
def gen_candidates(df,column):
    column_list = list(df[column])
    column_list.sort()
    new_column_list = []
    for i in range(len(column_list) -1 ):
        elem = (column_list[i] + column_list[i+1]) /2
        new_column_list.append(elem)
    return new_column_list

def case_entropy_cont(df,x_att,y_att,ent):
    candidates = gen_candidates(df,x_att)
    best_ig = 0
    best_cand = 0
    for cand in candidates:
        pr_le = df[df[x_att] <= cand].shape[0] /df.shape[0]
        pr_gt = 1 - pr_le
        c_ent = 0
        c_ent += pr_le * entropy(df[df[x_att] <=cand],y_att)
        c_ent += pr_gt * entropy(df[df[x_att] > cand],y_att)
        ig = ent-c_ent
        if ig > best_ig:
            best_ig = ig
            best_cand = cand
    
    return IGAttribute(best_cand,x_att,best_ig)
#计算信息增益  
def IG(df,columns,last_column,nom_columns,num_columns):
    ig = []
    ent = entropy(df,last_column)
    for column in columns:
        if column in num_columns:
            ig.append(case_entropy_cont(df,column,last_column,ent))
        else:
            c_ent = case_entropy_discrete(df,column,last_column)
            ig.append(IGAttribute(None,column,ent-c_ent))
    return ig
