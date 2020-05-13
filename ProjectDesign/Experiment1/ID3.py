import numpy as np
import pandas as pd

from Node import IGNode
import AttSelect as ase


# using id3 algorithm
def ig_gen_DT(train_data, class_data, att_list, class_column, nom_columns,
              num_columns, belong_att):

    node = IGNode(class_data, None, belong_att)  # create a new node

    # class_data 中都在同一个类
    uni_class_data = class_data.unique()
    if (uni_class_data.shape[0] == 1):
        node.split_att = belong_att
        node.belong_att = uni_class_data[0]
        return node

    # 多数表决
    if (att_list == []):
        node.split_att = belong_att
        mode = class_data.mode().get(0)
        node.belong_att = mode
        return node
    # 选取分裂属性
    att_list, [split_att,value_selected] = ase.ig_att_selection(train_data, att_list,
                                               class_column, nom_columns,
                                               num_columns)
    node.split_att = split_att

    if split_att in num_columns: # 分裂属性是数值属性
        new_train_data = [
            train_data[train_data[split_att] <= value_selected],
            train_data[train_data[split_att] > value_selected]
        ]
        count = 0
        for i in new_train_data:
            if i.empty:
                mode = class_data.mode().get(0)
                node.children.append(IGNode(None,None,mode))
                count += 1
            else:
                new_class_data = i[class_column]
                if count == 0:
                    node.children.append(ig_gen_DT(i,new_class_data,att_list,class_column,nom_columns,num_columns,bytes('<=','utf-8')+bytes(str(value_selected),'utf-8'))) 
                else:
                    node.children.append(ig_gen_DT(i,new_class_data,att_list,class_column,nom_columns,num_columns,bytes('>','utf-8')+bytes(str(value_selected),'utf-8')))
        return node
    else:# 分裂属性是标称属性
        # 将此结点train_data分组
        split_att_list = train_data[split_att].unique()
        for att in split_att_list:
            new_train_data = train_data[train_data[split_att] == att]
            if new_train_data.empty:  #若为空 则加一个树叶到该结点,标记为D的多数类
                mode = class_data.mode().get(0)
                node.children.append(IGNode(None, None, mode))
            else:
                new_class_data = new_train_data[class_column]
                node.children.append(
                    ig_gen_DT(new_train_data, new_class_data, att_list,
                            class_column, nom_columns, num_columns, att))

    return node