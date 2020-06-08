import numpy as np
import pandas as pd
import re
import random

from collections import Counter

from id3_2 import IG3Node
from preprocessing import Data


# inherit from cartnode
# change fea_selection() method
class RFNode(IG3Node):
    def __init__(self, data, split_fea, val, num_flag, split_path, belong_fea,
                 leaf_flag, purity_flag):
        super(RFNode, self).__init__(data=data,
                                     split_fea=split_fea,
                                     val=val,
                                     num_flag=num_flag,
                                     split_path=split_path,
                                     belong_fea=belong_fea,
                                     leaf_flag=leaf_flag,
                                     purity_flag=purity_flag)

    # override
    def fea_selection(self):
        # get gini ,each elem is a tuple
        ig = self.calc_ig()
        # get gini value
        ig_val = []
        for i in range(len(ig)):
            ig_val.append(ig[i][0])

        # difference between carttree and random forest
        fea_list_length = len(self.data.fea_column)
        m = np.floor(np.sqrt(fea_list_length))  #取整
        rand_m_idx = random.sample(range(fea_list_length), int(m))
        # 获得被选中的特征的gini value
        choose_fea_ig_val = []
        for idx in rand_m_idx:
            choose_fea_ig_val.append(ig_val[idx])
        choose_fea_ig_val = np.array(choose_fea_ig_val)
        max_idx = np.argmax(choose_fea_ig_val)

        max_idx = rand_m_idx[max_idx]  #这里用了两次索引
        selected_fea = self.data.fea_column[max_idx]
        self.data.fea_column.pop(max_idx)

        # flag==0 means the feature is numeric
        # flag==1 means is nominal
        flag = 0
        divide_point = 0

        if selected_fea in self.data.nom_columns:
            flag = 1
        else:
            flag = 0
            divide_point = ig[max_idx][1]

        return (selected_fea, flag, divide_point)


class RandomForest:
    def __init__(self, data, dt_num=20):
        self.data = data  #全部数据
        self.dt_list = []  #基决策树
        self.root_list = []  #基决策树的根结点
        self.bagging_data = []  # bagging data
        self.bagging_idx = []
        self.oob_data = []  # oob data
        self.oob_idx = []
        self.dt_num = dt_num  #基决策树的数目

    # bagging产生数据
    def bagging(self):
        df = self.data.df
        #获得数据集的大下
        data_size = df.shape[0]
        #先获取idx 并对每个数据打上tag
        for i in range(self.dt_num):
            # 随机抽取data_size个样本
            rand_idx = list(
                np.random.randint(0, high=data_size, size=data_size))
            rand_idx_set = set(rand_idx)
            rand_idx_uni = list(rand_idx_set)
            #产生0,1,2,3,...,datasize-1的float
            total_idx_set = set(
                np.linspace(0, data_size - 1, num=data_size, dtype=int))
            #获得oob的index
            oob_idx = list(total_idx_set.difference(rand_idx_set))
            for j in rand_idx_uni:
                df.at[j, self.data.tag_column].append(i)
            #df.loc[rand_idx,self.data.tag_column].append(i)

            self.bagging_idx.append(rand_idx)
            self.oob_idx.append(oob_idx)

        # 产生bagging数据和oob数据
        for i in range(self.dt_num):
            bagging_df = df.iloc[self.bagging_idx[i], :]
            bagging_df.index = range(bagging_df.shape[0])
            oob_df = df.iloc[self.oob_idx[i], :]
            oob_df.index = range(oob_df.shape[0])
            self.bagging_data.append(bagging_df)
            self.oob_data.append(oob_df)

    def train_dt(self, train_data, split_path, val, num_flag):

        df = train_data.df  # get dataframe

        # create rfnode
        node = RFNode(data=train_data,
                      split_fea=None,
                      val=val,
                      num_flag=num_flag,
                      split_path=split_path,
                      belong_fea=None,
                      leaf_flag=0,
                      purity_flag=0)

        uni_class_data = node.data.class_data.unique()
        # 若该类别数为1
        if uni_class_data.shape[0] == 1:
            node.leaf_flag = 1  # 标记为叶子结点
            node.purity_flag = 1  #标记为纯结点
            node.belong_fea = uni_class_data[0]
            return node

        if len(node.data.fea_column) == 0:  # 特征都用完了 多数表决
            node.leaf_flag = 1
            node.purity_flag = 0
            mode = node.data.class_data.mode().get(0)
            node.belong_fea = mode
            return node

        selected_fea, flag, divide_point = node.fea_selection()
        node.split_fea = selected_fea

        if flag == 0:  #数值属性作为分裂属性
            #根据分裂属性 将数据分裂
            split_df = [
                df[df[selected_fea] <= divide_point],
                df[df[selected_fea] > divide_point]
            ]
            tmp_count = 0
            for data in split_df:
                #如果split_data中有一个为空 以train_data中多数表决 该结点为叶结点
                if data.empty:
                    mode = node.data.class_data.mode().get(0)
                    #创建新的data instance
                    data_obj = Data(path=None,
                                    dataset=None,
                                    df=df,
                                    fea_column=node.data.fea_column,
                                    nom_columns=node.data.nom_columns,
                                    num_columns=node.data.num_columns,
                                    class_column=node.data.class_column,
                                    class_data=node.data.class_data)
                    #作为该结点的子代
                    node.children.append(
                        RFNode(data=data_obj,
                               split_fea=None,
                               val=None,
                               num_flag=-1,
                               split_path=split_path +
                               bytes(selected_fea, encoding='utf-8') +
                               bytes(' not exist', encoding='utf-8'),
                               belong_fea=mode,
                               leaf_flag=1,
                               purity_flag=0))
                    tmp_count += 1
                else:
                    # tmp_count == 0 对于数值型为 <=
                    # tmp_count == 1 对于数值型为 >
                    if tmp_count == 0:
                        data_obj = Data(
                            path=None,
                            dataset=None,
                            df=data,
                            fea_column=node.data.fea_column,
                            nom_columns=node.data.nom_columns,
                            num_columns=node.data.num_columns,
                            class_column=node.data.class_column,
                            class_data=data[node.data.class_column])
                        node.children.append(
                            self.train_dt(
                                train_data=data_obj,
                                split_path=split_path +
                                bytes(selected_fea, encoding='utf-8') +
                                bytes('<=', encoding='utf-8') +
                                bytes(str(divide_point), encoding='utf-8') +
                                bytes(' ', encoding='utf-8'),
                                val=bytes('<=', encoding='utf-8') +
                                bytes(str(divide_point), encoding='utf-8'),
                                num_flag=0))
                        tmp_count += 1
                    else:
                        data_obj = Data(
                            path=None,
                            dataset=None,
                            df=data,
                            fea_column=node.data.fea_column,
                            nom_columns=node.data.nom_columns,
                            num_columns=node.data.num_columns,
                            class_column=node.data.class_column,
                            class_data=data[node.data.class_column])
                        node.children.append(
                            self.train_dt(
                                train_data=data_obj,
                                split_path=split_path +
                                bytes(selected_fea, encoding='utf-8') +
                                bytes('>', encoding='utf-8') +
                                bytes(str(divide_point), encoding='utf-8') +
                                bytes(' ', encoding='utf-8'),
                                val=bytes('>', encoding='utf-8') +
                                bytes(str(divide_point), encoding='utf-8'),
                                num_flag=1))
        else:  # flag == 1 标称型数据
            # 获取分裂属性列的所有不重复元素
            selected_fea_value = df[selected_fea].unique()
            for val in selected_fea_value:
                split_df = df[df[selected_fea] == val]
                if split_df.empty:
                    mode = node.data.class_data.mode().get(0)
                    data_obj = Data(path=None,
                                    dataset=None,
                                    df=df,
                                    fea_column=node.data.fea_column,
                                    nom_columns=node.data.nom_columns,
                                    num_columns=node.data.num_columns,
                                    class_column=node.data.class_column,
                                    class_data=node.data.class_data)
                    node.children.append(
                        RFNode(data=data_obj,
                                split_fea=None,
                                val=None,
                                num_flag=-1,
                                split_path=split_path +
                                bytes(selected_fea, encoding='utf-8') +
                                bytes(' not exist', encoding='utf-8'),
                                belong_fea=mode,
                                leaf_flag=1,
                                purity_flag=0))
                else:
                    data_obj = Data(
                        path=None,
                        dataset=None,
                        df=split_df,
                        fea_column=node.data.fea_column,
                        nom_columns=node.data.nom_columns,
                        num_columns=node.data.num_columns,
                        class_column=node.data.class_column,
                        class_data=split_df[node.data.class_column])
                    node.children.append(
                        self.train_dt(train_data=data_obj,
                                        split_path=split_path +
                                        bytes(selected_fea, encoding='utf-8') +
                                        bytes(' ', encoding='utf-8'),
                                        val=val,
                                        num_flag=-1))

        return node

    @staticmethod
    def get_train_params(df):

        # get class_column name and feature column name
        fea_list = list(df)
        tag_idx, classify_idx = fea_list.index('tag'), fea_list.index(
            'classify')
        fea_list.pop(tag_idx)  #弹出tags
        fea_list.pop(classify_idx)  #弹出classify
        class_column = fea_list.pop()
        fea_column = fea_list

        nom_columns = []
        num_columns = []
        # get nominal column name and numeric column name
        # transform bytes to str and float respectively
        for col_name in fea_column:
            if df[col_name].dtypes == 'object':
                nom_columns.append(col_name)
            else:
                num_columns.append(col_name)

        class_data = df[class_column]
        #(fea_column,nom_columns,num_columns,class_column,class_data)
        return {
            'fea_column': fea_column,
            'nom_columns': nom_columns,
            'num_columns': num_columns,
            'class_column': class_column,
            'class_data': class_data
        }

    def train_rf(self):
        for data_df in self.bagging_data:
            params_dict = self.get_train_params(data_df)
            data = Data(path=None,
                        dataset=None,
                        df=data_df,
                        fea_column=params_dict.get('fea_column'),
                        nom_columns=params_dict.get('nom_columns'),
                        num_columns=params_dict.get('num_columns'),
                        class_column=params_dict.get('class_column'),
                        class_data=params_dict.get('class_data'))
            tree = self.train_dt(train_data=data,
                                 split_path=bytes(' ', encoding='utf-8'),
                                 val=None,
                                 num_flag=-1)
            self.dt_list.append(tree)
            self.root_list.append(tree)

    # 对于数值型数据 得到分裂点
    @staticmethod
    def match_node_val(s):
        s = str(s, encoding='utf-8')
        res = re.findall(r"\d+\.?\d*", s)
        val = float(res[0])
        return val

    # 测试单个oob
    def test_dt(self, oob, k):  #k代表第k个oob
        df = oob  # get dataframe

        node = self.root_list[k]  # get root
        # 在树中搜索 若结点是叶子结点 则循环结束
        while node.leaf_flag != 1:
            split_fea = node.split_fea  # 分裂的属性
            trans_val = df.loc[split_fea]  # 第row行 'split_fea'列的数据
            if split_fea in self.data.num_columns: # 若分裂属性是数值属性
                val = None #分裂值 即divide point
                flags = [0,0] # flag为1说明不为叶结点
                num_flags = [-1,-1] # -1表示无 0表示<= 1表示>
                for i in range(len(node.children)): #遍历node的子代
                    if node.children[i].val != None:
                        flags[i] = 1
                        num_flags[i] = node.children[i].num_flag
                        val = self.match_node_val(node.children[i].val)
                if val == None: #若两边都为叶结点
                    node = node.children[0]
                else:
                    if trans_val <= val: #若该条数据的值小于divide point的值
                        idx = -1
                        for i in range(len(num_flags)):
                            if num_flags[i] == 0: #找到 <= 的子结点
                                idx = i
                                break
                        if idx == -1:
                            node = node.children[0]
                        else:
                            node = node.children[idx]
                    else: 
                        idx = -1
                        for i in range(len(num_flags)):
                            if num_flags[i] == 1: #找到 >的子结点
                                idx = i
                                break
                        if idx == -1:
                            node = node.children[0]
                        else:
                            node = node.children[idx]
            else: # 若分裂属性是标称属性
                val = None
                idx = -1
                for i in range(len(node.children)): #遍历node的子代                       
                    if node.children[i].val == None:
                        continue
                    if trans_val == node.children[i].val: #找到对应值相等的子结点
                        val = node.children[i].val
                        idx = i
                        break
                if val == None:
                    node = node.children[0]
                else:    
                    node = node.children[idx]
        #内层循环结束 得到叶结点
        # 打上标签
        belong_fea = node.belong_fea
        df.loc[self.data.classify_column].append(belong_fea)

    def test_rf(self):
        correct_rate = 0
        # 遍历oob数据 每个决策树投票
        for oob in self.oob_data:  #外层循环 遍历数据
            for i in range(oob.shape[0]):  #遍历每个oob
                data = oob.iloc[i, :]
                for j in range(self.dt_num):  #每个决策树投票
                    if j in data.loc[self.data.tag_column]:
                        continue
                    else:
                        self.test_dt(data, j)
        
        # 初始化混淆矩阵
        labels = list(self.data.class_data.unique())
        mat_shape = len(labels)
        conf_mat = np.zeros([mat_shape,mat_shape])

        #遍历所有的oob 每一条数据进行投票
        oob_data_len = 0
        correct_rate = 0
        for oob in self.oob_data:
            for i in range(oob.shape[0]):
                # 获得该条数据的投票结果
                classify_list = oob.loc[i, self.data.classify_column]
                mode = Counter(classify_list).most_common(1)[0][0]
                # 将其填入混淆矩阵
                true_label = oob.loc[i,self.data.class_column]
                pred_label = mode
                true_label_idx = labels.index(true_label)
                pred_label_idx = labels.index(pred_label)
                conf_mat[true_label_idx][pred_label_idx] += 1
                # 如果分类正确 对应计数加1
                if mode == oob.loc[i, self.data.class_column]:
                    correct_rate += 1
            oob_data_len += oob.shape[0]

        correct_rate = correct_rate / oob_data_len
        return correct_rate,conf_mat


# if __name__ == "__main__":
#     rf = RandomForest()
