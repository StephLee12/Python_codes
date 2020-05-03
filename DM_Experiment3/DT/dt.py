import numpy as np
import pandas as pd
from scipy.io import arff
import feature_selection as dm


class DTNode:
    def __init__(self, data=None, split_att=None, belong_att=None):
        self.data = data
        self.split_att = split_att
        self.belong_att = belong_att
        self.children = []


# load data and normlize
def load(url):
    # load data
    dataset = arff.loadarff(url)
    df = pd.DataFrame(dataset[0])
    att_list = list(df)
    class_column = att_list.pop()

    return df, att_list, class_column


# C4.5 get max gain_ratio attribute
def att_selection(df, att_list, class_column):
    split_info = []
    # get information gain
    ig = dm.IG(df, att_list, class_column)
    # get split_info
    for att in att_list:
        split_info.append(dm.split_info(df, att))
    # list to array
    ig_arr, split_info_arr = np.array(ig), np.array(split_info)
    # calculate gain ratio
    gain_ratio = ig_arr / split_info_arr
    # select max gain ratio
    max_index = np.argmax(gain_ratio)
    # return selected att and att_list after deletion
    att_selected = att_list[max_index]
    att_list.pop(max_index)
    return att_list, att_selected


def gen_DT(train_data, class_data, att_list, class_column,belong_att):

    node = DTNode(class_data, None, belong_att)  # create a new node

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
    att_list, split_att = att_selection(train_data, att_list, class_column)
    node.split_att = split_att

    # 将此结点train_data分组
    split_att_list = train_data[split_att].unique()
    for att in split_att_list:
        new_train_data = train_data[train_data[split_att] == att]
        if new_train_data.empty:  #若为空 则加一个树叶到该结点,标记为D的多数类
            mode = class_data.mode().get(0)
            node.children.append(DTNode(None, att, mode))
        else:
            new_class_data = new_train_data[class_column]
            node.children.append(
                gen_DT(new_train_data, new_class_data, att_list, class_column,att))

    return node


if __name__ == "__main__":
    url = [
        'DM_Experiment3/DT/weather_nominal_train.arff',
        'DM_Experiment3/DT/weather_nominal_test.arff'
    ]
    train_data, att_list, class_column = load(url[0])
    class_data = train_data[class_column]
    test_data, _, __ = load(url[1])
    DTTree = gen_DT(train_data, class_data, att_list, class_column,None)
