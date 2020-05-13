import numpy as np
import pandas as pd

from Node import CARTNode
import AttSelect as ase


# using cart algorithm
def cart_gen_DT(train_data, class_data, att_list, class_column, nom_columns,
                num_columns, belong_att, left, right):

    node = CARTNode(class_data, None, belong_att, left, right)

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
    # attribute selection
    att_list, gini_att = ase.cart_att_selection(train_data, att_list,
                                                class_column, nom_columns,
                                                num_columns)
    node.split_att = gini_att.name

    if gini_att.name in num_columns:
        new_df = [
            train_data[train_data[node.split_att] <= gini_att.value],
            train_data[train_data[node.split_att] > gini_att.value]
        ]
        
        for i in range(2):
            if i == 0:
                if new_df[0].empty:
                    mode = class_data.mode().get(0)
                    node.left = CARTNode(None, None, mode, None, None)
                else:
                    new_class_data = new_df[0][class_column]
                    node.left = cart_gen_DT(new_df[0], new_class_data, att_list,
                                            class_column, nom_columns, num_columns,
                                            bytes('<=','utf-8')+bytes(str(gini_att.value),'utf-8'), None, None)
            else:  # i==1
                if new_df[1].empty:
                    mode = class_data.mode().get(0)
                    node.right = CARTNode(None, None, mode, None, None)
                else:
                    new_class_data = new_df[0][class_column]
                    node.right = cart_gen_DT(
                        new_df[0], new_class_data, att_list, class_column,
                        nom_columns, num_columns,
                        bytes('>','utf-8')+bytes(str(gini_att.value),'utf-8'), None, None)
        
        return node
    else:
        new_df = [
            train_data[train_data[node.split_att] == gini_att.value],
            train_data[train_data[node.split_att] != gini_att.value]
        ]

        for i in range(2):
            if i == 0:
                if new_df[0].empty:
                    mode = class_data.mode().get(0)
                    node.left = CARTNode(None, gini_att.value, mode, None, None)
                else:
                    new_class_data = new_df[0][class_column]
                    node.left = cart_gen_DT(new_df[0], new_class_data, att_list,
                                            class_column, nom_columns, num_columns,
                                            gini_att.value, None, None)
            else:  # i==1
                if new_df[1].empty:
                    mode = class_data.mode().get(0)
                    node.right = CARTNode(None, gini_att.value, mode, None, None)
                else:
                    new_class_data = new_df[0][class_column]
                    node.right = cart_gen_DT(
                        new_df[0], new_class_data, att_list, class_column,
                        nom_columns, num_columns,
                        bytes("not ", 'utf-8') + gini_att.value, None, None)

        return node
