import numpy as np
import pandas as pd
import re

from id3_2 import IG3Node
from preprocessing import Data


# inherit from ig3node
class CARTNode(IG3Node):
    def __init__(self, data, split_fea, val, num_flag, split_path, belong_fea,
                 leaf_flag, purity_flag):
        super(CARTNode, self).__init__(data=data,
                                       split_fea=split_fea,
                                       val=val,
                                       num_flag=num_flag,
                                       split_path=split_path,
                                       belong_fea=belong_fea,
                                       leaf_flag=leaf_flag,
                                       purity_flag=purity_flag)

        self.left_child = None
        self.right_child = None

    # 计算熵
    def calc_entropy(self, df, fea_name):
        # 去除一列数据中的重复值
        uni_fea_data = df[fea_name].unique()

        ent = 0
        for val in uni_fea_data:
            pr = sum(df[fea_name] == val) / df.shape[0]
            # 概率计算也可用 self.data.df[fea_name].value_counts / self.data.df.shape[0]
            ent += pr * np.log(pr)

        return -ent  # 注意返回的是 -ent

    # 对于数值型数据 计算条件信息熵时
    # 先产生划分点集合
    def gen_divide_points(self, fea_name):
        fea_elem = list(self.data.df[fea_name])
        fea_elem.sort()

        divide_points = []
        for i in range(len(fea_elem) - 1):
            p = (fea_elem[i] + fea_elem[i + 1]) / 2
            divide_points.append(p)

        return divide_points

    # 计算条件信息熵
    def calc_case_ent_cont(self, fea_name):
        divide_points = self.gen_divide_points(fea_name)

        self.data.class_column_ent = self.calc_entropy(self.data.df,
                                                       self.data.class_column)
        # 计算每个划分点的信息增益
        best_ig = 0
        best_divide_p = 0
        for p in divide_points:
            # <= p 的概率
            pr_le = self.data.df[
                self.data.df[fea_name] <= p].shape[0] / self.data.df.shape[0]
            # > p 的概率
            pr_gt = 1 - pr_le

            # 计算信息增益
            case_ent = 0
            case_ent += pr_le * self.calc_entropy(
                self.data.df[self.data.df[fea_name] <= p],
                self.data.class_column)
            case_ent += pr_gt * self.calc_entropy(
                self.data.df[self.data.df[fea_name] > p],
                self.data.class_column)
            ig = self.data.class_column_ent - case_ent

            if ig > best_ig:
                best_ig = ig
                best_divide_p = p

        return best_divide_p, best_ig

    # 计算标称数据的gini
    def calc_gini_nom(self, fea_name, fea_val):
        df = self.data.df
        # split dataframe
        split_df = [df[df[fea_name] == fea_val], df[df[fea_name] != fea_val]]

        gini_val = 0
        for data in split_df:
            tmp_gini = 0
            pr = data[self.data.class_column].value_counts() / data.shape[0]
            tmp_gini = 1 - sum(np.square(np.array(pr)))
            tmp_gini *= data.shape[0] / df.shape[0]
            gini_val += tmp_gini

        return gini_val

    # 计算数值数据的gini
    def calc_gini_num(self, fea_name, divide_point):
        df = self.data.df
        split_df = [
            df[df[fea_name] <= divide_point], df[df[fea_name] > divide_point]
        ]

        gini_val = 0
        for data in split_df:
            tmp_gini = 0
            pr = data[self.data.class_column].value_counts() / data.shape[0]
            tmp_gini = 1 - sum(np.square(np.array(pr)))
            tmp_gini *= data.shape[0] / df.shape[0]
            gini_val += tmp_gini

        return gini_val

    # 计算每一列的gini
    def calc_gini(self):
        gini = []
        for col_name in self.data.fea_column:
            # 若是数值属性
            if col_name in self.data.num_columns:
                best_divide_p, _ = self.calc_case_ent_cont(col_name)
                gini_val = self.calc_gini_num(col_name, best_divide_p)
                gini.append((gini_val, best_divide_p))
            else:  # 若是标称属性
                fea_val = self.data.df[col_name].unique()
                gini_val = 1
                best_fea_val = None
                for val in fea_val:
                    tmp = self.calc_gini_nom(col_name, val)
                    if tmp < gini_val:
                        gini_val = tmp
                        best_fea_val = val
                gini.append((gini_val, best_fea_val))

        return gini

    # 特征筛选
    def fea_selection(self):

        # get gini,each elem is a tuple
        gini = self.calc_gini()
        # get gini value
        gini_val = []
        for i in range(len(gini)):
            gini_val.append(gini[i][0])

        # find min gini value
        gini_val = np.array(gini_val)
        max_idx = np.argmin(gini_val)

        selected_fea = self.data.fea_column[max_idx]
        self.data.fea_column.pop(max_idx)

        # flag==0 means the feature is numeric
        # flag==1 means is nominal
        flag = 0
        divide_point = 0
        best_nom_fea_val = None

        if selected_fea in self.data.nom_columns:
            flag = 1
            best_nom_fea_val = gini[max_idx][1]
        else:
            flag = 0
            divide_point = gini[max_idx][1]

        return (selected_fea, flag, divide_point, best_nom_fea_val)


class CARTTree:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.tree = self.build_tree(train_data=self.train_data,
                                    split_path=bytes(' ', encoding='utf-8'),
                                    val=None,
                                    num_flag=-1)
        self.root = self.get_root()

    def get_root(self):
        return self.tree

    def build_tree(self, train_data, split_path, val, num_flag):

        df = train_data.df  # get dataframe

        # create a node
        node = CARTNode(data=train_data,
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

        selected_fea, flag, divide_point, best_nom_fea_val = node.fea_selection(
        )
        node.split_fea = selected_fea

        if flag == 0:  # 数值属性
            split_df = [
                df[df[selected_fea] <= divide_point],
                df[df[selected_fea] > divide_point]
            ]
            # 是二叉树
            for i in range(2):
                if i == 0:  # 左子代
                    if split_df[i].empty:
                        mode = node.data.class_data.mode().get(0)
                        data_obj = Data(path=None,
                                        dataset=None,
                                        df=df,
                                        fea_column=node.data.fea_column,
                                        nom_columns=node.data.nom_columns,
                                        num_columns=node.data.num_columns,
                                        class_column=node.data.class_column,
                                        class_data=node.data.class_data)
                        node.left_child = CARTNode(
                            data=data_obj,
                            split_fea=None,
                            val=None,
                            num_flag=-1,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes(' not exist', encoding='utf-8'),
                            belong_fea=mode,
                            leaf_flag=1,
                            purity_flag=0)
                    else:
                        data_obj = Data(
                            path=None,
                            dataset=None,
                            df=split_df[i],
                            fea_column=node.data.fea_column,
                            nom_columns=node.data.nom_columns,
                            num_columns=node.data.num_columns,
                            class_column=node.data.class_column,
                            class_data=split_df[i][node.data.class_column])
                        node.left_child = self.build_tree(
                            train_data=data_obj,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes('<=', encoding='utf-8') +
                            bytes(str(divide_point), encoding='utf-8') +
                            bytes(' ', encoding='utf-8'),
                            val=bytes('<=', encoding='utf-8') +
                            bytes(str(divide_point), 'utf-8'),
                            num_flag=0)
                else:  #右子代
                    if split_df[i].empty:
                        mode = node.data.class_data.mode().get(0)
                        data_obj = Data(path=None,
                                        dataset=None,
                                        df=df,
                                        fea_column=node.data.fea_column,
                                        nom_columns=node.data.nom_columns,
                                        num_columns=node.data.num_columns,
                                        class_column=node.data.class_column,
                                        class_data=node.data.class_data)
                        node.right_child = CARTNode(
                            data=data_obj,
                            split_fea=None,
                            val=None,
                            num_flag=-1,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes(' not exist', encoding='utf-8'),
                            belong_fea=mode,
                            leaf_flag=1,
                            purity_flag=0)
                    else:
                        data_obj = Data(
                            path=None,
                            dataset=None,
                            df=split_df[i],
                            fea_column=node.data.fea_column,
                            nom_columns=node.data.nom_columns,
                            num_columns=node.data.num_columns,
                            class_column=node.data.class_column,
                            class_data=split_df[i][node.data.class_column])
                        node.right_child = self.build_tree(
                            train_data=data_obj,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes('>', encoding='utf-8') +
                            bytes(str(divide_point), encoding='utf-8') +
                            bytes(' ', encoding='utf-8'),
                            val=bytes('>', encoding='utf-8') +
                            bytes(str(divide_point), encoding='utf-8'),
                            num_flag=1)
        else:  # 数据是标称型
            split_df = [
                df[df[selected_fea] == best_nom_fea_val],
                df[df[selected_fea] != best_nom_fea_val]
            ]

            for i in range(2):
                if i == 0:  #左子代
                    if split_df[i].empty:
                        mode = node.data.class_data.mode().get(0)
                        data_obj = Data(path=None,
                                        dataset=None,
                                        df=df,
                                        fea_column=node.data.fea_column,
                                        nom_columns=node.data.nom_columns,
                                        num_columns=node.data.num_columns,
                                        class_column=node.data.class_column,
                                        class_data=node.data.class_data)
                        node.left_child = CARTNode(
                            data=data_obj,
                            split_fea=None,
                            val=None,
                            num_flag=-1,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes(' not exist', encoding='utf-8'),
                            belong_fea=mode,
                            leaf_flag=1,
                            purity_flag=0)
                    else:
                        data_obj = Data(
                            path=None,
                            dataset=None,
                            df=split_df[i],
                            fea_column=node.data.fea_column,
                            nom_columns=node.data.nom_columns,
                            num_columns=node.data.num_columns,
                            class_column=node.data.class_column,
                            class_data=split_df[i][node.data.class_column])
                        node.left_child = self.build_tree(
                            train_data=data_obj,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes(' not ', encoding='utf-8') +
                            best_nom_fea_val + bytes(' ', encoding='utf-8'),
                            val=bytes(' not ', encoding='utf-8') +
                            best_nom_fea_val,
                            num_flag=-1)
                else:  #右子代
                    if split_df[i].empty:
                        mode = node.data.class_data.mode().get(0)
                        data_obj = Data(path=None,
                                        dataset=None,
                                        df=df,
                                        fea_column=node.data.fea_column,
                                        nom_columns=node.data.nom_columns,
                                        num_columns=node.data.num_columns,
                                        class_column=node.data.class_column,
                                        class_data=node.data.class_data)
                        node.right_child = CARTNode(
                            data=data_obj,
                            split_fea=None,
                            val=None,
                            num_flag=-1,
                            split_path=split_path +
                            bytes(' not exist', encoding='utf-8'),
                            belong_fea=mode,
                            leaf_flag=1,
                            purity_flag=0)
                    else:
                        data_obj = Data(
                            path=None,
                            dataset=None,
                            df=split_df[i],
                            fea_column=node.data.fea_column,
                            nom_columns=node.data.nom_columns,
                            num_columns=node.data.num_columns,
                            class_column=node.data.class_column,
                            class_data=split_df[i][node.data.class_column])
                        node.right_child = self.build_tree(
                            train_data=data_obj,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes('\'s', encoding='utf-8') + best_nom_fea_val +
                            bytes(' ', encoding='utf-8'),
                            val=best_nom_fea_val,
                            num_flag=-1)
        return node

    # 对于数值型数据 得到分裂点
    @staticmethod
    def match_node_val(s):
        s = str(s, encoding='utf-8')
        res = re.findall(r"\d+\.?\d*", s)
        val = float(res[0])
        return val
    #在测试集上检验
    def test(self):
        df = self.test_data.df

        node = self.root
        for row in range(df.shape[0]): #最外层 遍历测试集
            # 在树中搜索 若结点是叶子结点 则循环结束
            while node.leaf_flag != 1:
                split_fea = node.split_fea# 分裂的属性
                trans_val = df.at[row,split_fea]# 第row行 'split_fea'列的数据
                if split_fea in self.test_data.num_columns:# 若分裂属性是数值属性
                    val = None #分裂值 即divide point
                    flags = [0,0] # flag为1说明不为叶结点
                    num_flags = [-1,-1] # -1表示无 0表示<= 1表示>
                    if node.left_child.val != None: #判断左结点
                        flags[0] = 1
                        num_flags[0] = node.left_child.num_flag
                        val = self.match_node_val(node.left_child.val)
                    if node.right_child.val != None: #判断右结点
                        flags[1] = 1
                        num_flags[1] = node.right_child.num_flag
                        val = self.match_node_val(node.right_child.val)
                    if val == None: #两边都为叶结点
                        node = node.left_child
                    else:
                        if trans_val <= val: # 小于divide point
                            idx = -1
                            for i in range(2):
                                if num_flags[i] == 0:
                                    idx = i
                                    break
                            if idx == 1:
                                node = node.right_child
                            else:
                                node = node.left_child
                        else: #大于divide point
                            idx = -1
                            for i in range(2):
                                if num_flags[i] == 1:
                                    idx = i
                                    break
                            if idx == 1:
                                node = node.right_child
                            else:
                                node = node.left_child
                else: #分裂属性是标称属性
                    val = None
                    idx = -1
                    if trans_val == node.right_child.val:
                        val = node.right_child.val
                        node = node.right_child
                    else:
                        node = node.left_child
            #内层循环结束 得到叶结点
            # 打上标签
            belong_fea = node.belong_fea
            df.at[row,node.data.classify_column] = belong_fea
            #重置node
            node = self.root

    def get_conf_mat(self):
        # 获取dataframe
        df = self.test_data.df
        labels = list(self.test_data.class_data.unique())  # 获取class 的label
        mat_shape = len(labels)  #矩阵的秩
        conf_mat = np.zeros([mat_shape, mat_shape])  #初始化混淆矩阵
        judge = {'accuracy': 0, 'er': 0, 'precision': 0, 'recall': 0, 'f': 0}

        for row in range(df.shape[0]):  #填充混淆矩阵
            true_label = df.at[row, self.test_data.class_column]
            pred_label = df.at[row, self.test_data.classify_column]

            true_label_idx = labels.index(true_label)
            pred_label_idx = labels.index(pred_label)

            conf_mat[true_label_idx][pred_label_idx] += 1

        return conf_mat, judge
