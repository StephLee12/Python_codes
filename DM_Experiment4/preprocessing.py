import pandas as pd
from scipy.io import arff


class Data:
    def __init__(self,
                 path=None,
                 dataset=None,
                 df=None,
                 fea_column=[],
                 nom_columns=[],
                 num_columns=[],
                 class_column=[],
                 class_data=None):
        self.path = path  # dataset path
        self.dataset = dataset  # load arff
        self.df = df  # convert dataset to df
        self.fea_column = fea_column  # feature's column name
        self.nom_columns = nom_columns  # nominal column name
        self.num_columns = num_columns  # numeric column name
        self.class_column = class_column  # label column name
        self.class_data = class_data  # label data
        self.cluster_column = 'cluster' # 聚类后的簇号
        # 用于DBSCAN 是否为核心对象 该列数据为0或1 0表示不为核心对象 1表示为核心对象
        self.core_column = 'core' 
        # 用于DBSCAN 若该数据为核心对象 哪些点在它的epsilon邻域
        self.in_epsilon_column = 'in_epsilon'
        
        

    # clear last instance memory
    def clear_memory(self):
        self.dataset = None
        self.df = None
        self.fea_column = []
        self.nom_columns = []
        self.num_columns = []
        self.class_column = []
        self.class_data = None

    def load_data(self):
        # get dataset and dataframe
        self.dataset = arff.loadarff(self.path)  # load dataset
        self.df = pd.DataFrame(self.dataset[0])  # convert to df

        # get class_column name and feature column name
        fea_list = list(self.df)
        self.class_column = fea_list.pop()
        self.fea_column = fea_list

        # get nominal column name and numeric column name
        # transform bytes to str and float respectively
        for col_name in self.fea_column:
            if self.df[col_name].dtypes == 'object':
                self.nom_columns.append(col_name)
            else:
                self.num_columns.append(col_name)

        self.class_data = self.df[self.class_column]
        self.df.insert(self.df.shape[1], self.cluster_column,None)
        self.df.insert(self.df.shape[1],self.core_column,0)
        self.df.insert(self.df.shape[1],self.in_epsilon_column,None)
        

    # fill missing data
    def fill_missing_data(self):
        # fill missing date in nominal columns
        for col_name in self.nom_columns:
            col_slice = self.df[col_name]
            # calculate mode
            mode = col_slice.mode().get(0)
            col_slice[col_slice == b'?'] = mode

        # fill missing data in numeric columns
        # here just calculate mean
        for col_name in self.num_columns:
            col_slice = self.df[col_name]
            # calculate mean
            mean = col_slice.mean()
            col_slice.fillna(mean, inplace=True)

    # def transform_bytes(self):
    #     # for numeric data——transform bytes to float
    #     for col_name in self.nom_columns:
    #         self.df[col_name].astype(str)


# if __name__ == "__main__":
#     path = 'DataAnalysisProjectDesign/Experiment1/adult_train.arff'
#     data = Data(path)
#     data.load_data()
#     data.fill_missing_data()
#     print(data.df)