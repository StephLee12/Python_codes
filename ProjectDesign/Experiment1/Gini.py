import numpy as np
import pandas as pd

import InformationGain as InfoG


class GiNiAttribute:
    def __init__(self, name=None, value=None, gini_value=None):
        self.name = name
        self.value = value
        self.gini = gini_value


def GINI(df, att_list, class_column, nom_columns, num_columns):

    gini_list = []  # gini attribute list
    for att in att_list:
        if att in num_columns:  #如果是数值属性
            ent = InfoG.entropy(df, class_column)
            ig_att = InfoG.case_entropy_cont(df, att, class_column, ent)
            value = ig_att.value
            gini_value = calc_gini_cont(df, att, value, class_column)
        else:  #如果是标称属性
            uni_value = df[att].unique()  # 去除重复值
            value = None  # record value
            gini_value = 1  # record gini value
            for i in uni_value:  # every value of this attribute
                if calc_gini_discreate(
                        df, att, i, class_column
                ) < gini_value:  # search minimal gini value
                    gini_value = calc_gini_discreate(df, att, i, class_column)
                    value = i  # change value

        gini_list.append(GiNiAttribute(att, value, gini_value))

    # from gini list select best attribute
    best_gini = 1
    idx = 0
    for k in range(len(gini_list)):
        if gini_list[k].gini < best_gini:
            best_gini = gini_list[k].gini
            idx = k

    att_list.pop(idx)  # update attribute list

    return att_list, gini_list[
        idx]  # return new attribute list and best gini attribute instance


def calc_gini_discreate(df, att, value, class_column):
    # split dataset via value in this attribute
    split_data = [df[df[att] == value], df[df[att] != value]]
    gini_value = 0  # initialize
    for i in split_data:
        tmp_gini = 0
        prob = i[class_column].value_counts() / i.shape[
            0]  # calculate probability
        tmp_gini = 1 - sum(np.square(np.array(prob)))
        tmp_gini *= i.shape[0] / df.shape[0]
        gini_value += tmp_gini

    return gini_value


def calc_gini_cont(df, att, value, class_column):
    split_data = [df[df[att] <= value], df[df[att] > value]]
    gini_value = 0
    for i in split_data:
        tmp_gini = 0
        prob = i[class_column].value_counts() / i.shape[
            0]  # calculate probability
        tmp_gini = 1 - sum(np.square(np.array(prob)))
        tmp_gini *= i.shape[0] / df.shape[0]
        gini_value += tmp_gini
    return gini_value