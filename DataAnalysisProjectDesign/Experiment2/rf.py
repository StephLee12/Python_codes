import numpy as np
import pandas as pd
import re

from collections import Counter

from cart_2 import CARTNode
from preprocessing import Data


# inherit from cartnode
# change fea_selection() method
class RFNode(CARTNode):
    def __init__(self,data,split_fea,val,num_flag,split_path,belong_fea,leaf_flag,purity_flag):
        super(RFNode,self).__init__(
            data=data,
            split_fea=split_fea,
            val=val,
            num_flag=num_flag,
            split_path=split_path,
            belong_fea=belong_fea,
            leaf_flag=leaf_flag,
            purity_flag=purity_flag
        )
    # override
    def fea_selection(self):
        # get gini ,each elem is a tuple
        gini = self.calc_gini()
        # get gini value
        gini_val = []
        for i in range(len(gini)):
            gini_val.append(gini[i][0])
        
        # difference between carttree and random forest
        fea_list_length = len(self.data.fea_column)
        m = np.floor(np.sqrt(fea_list_length)) #取整
        rand_m_idx = list(np.random.randint(0,fea_list_length,size=int(m)))
        # 获得被选中的特征的gini value
        choose_fea_gini_val = []
        for idx in rand_m_idx:
            choose_fea_gini_val.append(gini_val[idx])
        choose_fea_gini_val = np.array(choose_fea_gini_val)
        min_idx = np.argmin(choose_fea_gini_val)
        
        min_idx = rand_m_idx[min_idx] #这里用了两次索引
        selected_fea = self.data.fea_column[min_idx]
        self.data.fea_column.pop(min_idx)

        # flag==0 means the feature is numeric
        # flag==1 means is nominal
        flag = 0
        divide_point = 0
        best_nom_fea_val = None

        if selected_fea in self.data.nom_columns:
            flag = 1
            best_nom_fea_val = gini[min_idx][1]
        else:
            flag = 0
            divide_point = gini[min_idx][1]

        return (selected_fea, flag, divide_point, best_nom_fea_val)

class RandomForest:
    def __init__(self,data,dt_num=20):
        self.data = data #全部数据
        self.dt_list = [] #基决策树
        self.root_list = [] #基决策树的根结点
        self.bagging_data = [] # bagging data
        self.bagging_idx = []
        self.oob_data = [] # oob data
        self.oob_idx = []
        self.dt_num = dt_num #基决策树的数目
    
    # bagging产生数据
    def bagging(self):
        df = self.data.df
        #获得数据集的大下
        data_size = df.shape[0]
        #先获取idx 并对每个数据打上tag
        for i in range(self.dt_num):
            # 随机抽取data_size个样本
            rand_idx = list(np.random.randint(0,high=data_size,size=data_size))
            rand_idx_set = set(rand_idx)
            rand_idx_uni = list(rand_idx_set)
            #产生0,1,2,3,...,datasize-1的float
            total_idx_set = set(np.linspace(0,data_size-1,num=data_size,dtype=int))
            #获得oob的index
            oob_idx = list(total_idx_set.difference(rand_idx_set))
            for j in rand_idx_uni:
                df.at[j,self.data.tag_column].append(i)
            #df.loc[rand_idx,self.data.tag_column].append(i)

            self.bagging_idx.append(rand_idx)
            self.oob_idx.append(oob_idx)
        
        # 产生bagging数据和oob数据
        for i in range(self.dt_num):
            self.bagging_data.append(df.iloc[self.bagging_idx[i],:])
            self.oob_data.append(df.iloc[self.oob_idx[i],:])


    def train_dt(self,train_data,split_path,val,num_flag):
        
        df = train_data.df # get dataframe

        # create rfnode
        node = RFNode(
            data=train_data,
            split_fea=None,
            val=val,
            num_flag=num_flag,
            split_path=split_path,
            belong_fea=None,
            leaf_flag=0,
            purity_flag=0
        )

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
                        node.left_child = RFNode(
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
                        node.left_child = self.train_dt(
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
                        node.right_child = RFNode(
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
                        node.right_child = self.train_dt(
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
                        node.left_child = RFNode(
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
                        node.left_child = self.train_dt(
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
                        node.right_child = RFNode(
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
                        node.right_child = self.train_dt(
                            train_data=data_obj,
                            split_path=split_path +
                            bytes(selected_fea, encoding='utf-8') +
                            bytes('\'s', encoding='utf-8') + best_nom_fea_val +
                            bytes(' ', encoding='utf-8'),
                            val=best_nom_fea_val,
                            num_flag=-1)
        return node

    @staticmethod
    def get_train_params(df):

        # get class_column name and feature column name
        fea_list = list(df)
        fea_list.pop() #弹出tags
        fea_list.pop() #弹出classify
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

        return (fea_column,nom_columns,num_columns,class_column,class_data)

    def train_rf(self):
        for data_df in self.bagging_data:
            params_list = self.get_train_params(data_df)
            data = Data(
                path=None,
                dataset=None,
                df=data_df,
                fea_column=params_list[0],
                nom_columns=params_list[1],
                num_columns=params_list[2],
                class_column=params_list[3],
                class_data=params_list[4]
            )
            tree = self.train_dt(
                train_data=data,
                split_path=bytes(' ',encoding='utf-8'),
                val=None,
                num_flag=-1
            )
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
    def test_dt(self,oob,k): #k代表第k个oob
        df = oob # get dataframe

        node = self.root_list[k] # get root
        #correct_num = 0
        for row in range(df.shape[0]): #最外层 遍历测试集
            # 在树中搜索 若结点是叶子结点 则循环结束
            while node.leaf_flag != 1:
                split_fea = node.split_fea# 分裂的属性
                split_fea_idx = self.data.fea_column.index(split_fea)
                trans_val = df.iloc[row,split_fea_idx] # 第row行 'split_fea'列的数据
                if split_fea in self.data.num_columns:# 若分裂属性是数值属性
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
            df.iloc[row,-2].append(belong_fea)
            #重置node
            node = self.root_list[k]
        
        
        
    def test_rf(self):
        correct_rate = 0
        for oob in self.oob_data: #外层循环 遍历数据
            #tmp_correct_rate = 0
            for i in range(self.dt_num):
                self.test_dt(oob,i)

        #遍历所有的oob 每一条数据进行投票
        oob_data_len = 0
        correct_rate = 0
        for oob in self.oob_data:
            for i in range(oob.shape[0]):
                classify_list = oob.iloc[i,-2]
                mode = Counter(classify_list).most_common(1)[0][0]
                if mode == oob.iloc[i,-3]:
                    correct_rate += 1
            oob_data_len += oob.shape[0]
        
        correct_rate = correct_rate / oob_data_len
        return correct_rate

    
# if __name__ == "__main__":
#     rf = RandomForest()
