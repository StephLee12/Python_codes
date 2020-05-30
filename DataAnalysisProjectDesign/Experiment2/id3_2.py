import numpy as np
import pandas as pd
import re

from preprocessing import Data


class IG3Node:
    def __init__(self, data, split_fea, val,num_flag,split_path, belong_fea, leaf_flag,
                 purity_flag):
        self.data = data
        self.children = []
        self.split_fea = split_fea  #该结点由哪个feature分裂
        self.val = val #某feature的值
        self.num_flag = num_flag #0表示<= 1表示> -1表示无
        self.split_path = split_path  # 该结点之前的分裂路径
        self.leaf_flag = 0  # 是否是叶子结点 0为不是 1为是
        self.belong_fea = belong_fea  # 得到分类的标签
        #若为叶子结点 且该结点类别唯一  说明该结点纯 置1 否则置0
        self.purity_flag = 0

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

    # 对于标称型数据 计算条件信息熵
    def calc_case_ent_discrete(self, fea_name):

        # 去除重复值
        uni_fea_data = self.data.df[fea_name].unique()
        fea_pr = self.data.df[fea_name].value_counts() / self.data.df.shape[0]

        case_ent = 0
        for val in uni_fea_data:
            case_ent += fea_pr[val] * self.calc_entropy(
                self.data.df[self.data.df[fea_name] == val],
                self.data.class_column)

        return case_ent

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

    # 计算信息增益
    def calc_ig(self):
        ig = []
        class_column_ent = self.calc_entropy(self.data.df,
                                             self.data.class_column)

        for col_name in self.data.fea_column:
            if col_name in self.data.num_columns:
                divide_point, ig_val = self.calc_case_ent_cont(col_name)
                ig.append((ig_val, divide_point))
                #  append 一个元组 因为在建树时 要用的divide_point
            else:
                case_ent = self.calc_case_ent_discrete(col_name)
                ig_val = class_column_ent - case_ent
                ig.append((ig_val, 0))
                # 对于标称数据 divide_point设0即可

        return ig

    def fea_selection(self):

        # get ig,each elem in ig is a tuple
        ig = self.calc_ig()
        # get ig value
        ig_val = []
        for i in range(len(ig)):
            ig_val.append(ig[i][0])

        ig_val = np.array(ig_val)  # to array
        max_idx = np.argmax(ig_val)  # find max ig

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


class ID3Tree:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.tree = self.build_tree(train_data=self.train_data,
                                    split_path=bytes(' ', encoding='utf-8'),
                                    val=None,
                                    num_flag=-1)
        self.root = self.get_root()

    # 获得root结点
    def get_root(self):
        return self.tree

    # 建树
    def build_tree(self, train_data, split_path,val,num_flag):

        df = train_data.df  # get dataframe

        node = IG3Node(data=train_data,
                       split_fea=None,
                       val=val,
                       num_flag=num_flag,
                       split_path=split_path,
                       belong_fea=None,
                       leaf_flag=0,
                       purity_flag=0)

        uni_class_data = node.data.class_data.unique()
        if uni_class_data.shape[0] == 1:  # 该结点类别数为1
            node.leaf_flag = 1  #标记为叶子结点
            node.purity_flag = 1  #标记为纯结点
            node.belong_fea = uni_class_data[0]
            return node

        if len(node.data.fea_column) == 0:  # 特征都已用完 采用多数表决
            node.leaf_flag = 1  # 标记为叶子结点
            node.purity_flag = 0  #标记为不纯结点
            mode = node.data.class_data.mode().get(0)  # 表决
            node.belong_fea = mode
            return node

        # 获得分裂属性
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
                        IG3Node(data=data_obj,
                                split_fea=None,
                                val=None,
                                num_flag=-1,
                                split_path=split_path +
                                bytes(selected_fea,encoding='utf-8')+
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
                            self.build_tree(
                                train_data=data_obj,
                                split_path=split_path + 
                                bytes(selected_fea,encoding='utf-8')+
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
                            self.build_tree(
                                train_data=data_obj,
                                split_path=split_path +
                                bytes(selected_fea,encoding='utf-8') +
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
                        IG3Node(data=data_obj,
                                split_fea=None,
                                val=None,
                                num_flag=-1,
                                split_path=split_path +
                                bytes(selected_fea,encoding='utf-8')+
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
                        self.build_tree(train_data=data_obj,
                                        split_path=split_path +
                                        bytes(selected_fea, encoding='utf-8')+
                                        bytes(' ',encoding='utf-8'),
                                        val=val,
                                        num_flag=-1))

        return node
    
    # 对于数值型数据 得到分裂点
    @staticmethod
    def match_node_val(s):
        s = str(s,encoding='utf-8')
        res = re.findall(r"\d+\.?\d*",s)
        val = float(res[0])
        return val
        
    # 在测试集上检验
    def test(self):
        df = self.test_data.df
        
        node = self.root
        for row in range(df.shape[0]): #最外层 遍历测试集
            while node.leaf_flag != 1: # 在树中搜索 若结点是叶子结点 则循环结束
                split_fea = node.split_fea # 分裂的属性
                trans_val = df.at[row,split_fea] # 第row行 'split_fea'列的数据
                if split_fea in self.test_data.num_columns: # 若分裂属性是数值属性
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
            # 内层循环结束 到达叶子结点
            # 将数据打上标签
            belong_fea = node.belong_fea
            df.at[row,node.data.classify_column] = belong_fea
            # 重置node
            node = self.root

    
    # 计算混淆矩阵
    def get_conf_mat(self):
        # 获取dataframe
        df = self.test_data.df
        labels  = list(self.test_data.class_data.unique()) # 获取class 的label
        mat_shape = len(labels) #矩阵的秩
        conf_mat = np.zeros([mat_shape,mat_shape]) #初始化混淆矩阵
        judge = {
            'accuracy':0,
            'er':0,
            'precision':0,
            'recall':0,
            'f':0
        }

        for row in range(df.shape[0]): #填充混淆矩阵
            true_label = df.at[row,self.test_data.class_column]
            pred_label = df.at[row,self.test_data.classify_column]
            
            true_label_idx = labels.index(true_label)
            pred_label_idx = labels.index(pred_label)

            conf_mat[true_label_idx][pred_label_idx] += 1
        
        
        return conf_mat,judge
        
        