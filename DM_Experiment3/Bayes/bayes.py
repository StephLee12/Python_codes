import numpy as np
import pandas as pd
from scipy.io import arff

class BayesDict:
    def __init__(self,name=None,dic=None,pro=None):
        self.name = name
        self.dic = dic
        self.pro = pro

def load(url):
    # load data
    dataset = arff.loadarff(url)
    df = pd.DataFrame(dataset[0])
    columns = list(df)

    return df,columns

def get_series_pro(df,column_name):
    # get a unique series probability
    pr_class = df[column_name].value_counts() / df.shape[0]
    pr_class = pr_class.to_dict()
    return pr_class

def train_bayes_main(df,columns):
    
    pr_class_dict = get_series_pro(df,columns[-1]) # get last column probability

    bayes_dict_bucket = [] # initialize dict bucket
    for i in pr_class_dict:
        bayesDict = BayesDict(i,None,pr_class_dict[i]) # create a new instance
        new_series = df[df[columns[-1]] == i]
        sum_dict = {}
        for column in columns:
           if column != columns[-1]:
               new_dict = get_series_pro(new_series,column)
               sum_dict.update(new_dict)
           else:
               continue
        bayesDict.dic = sum_dict  
        bayes_dict_bucket.append(bayesDict)

    return bayes_dict_bucket

def classify_bayes_main(test_df,columns,bayes_bucket):
    
    columns.append('play_classify')
    test_df.insert(len(columns)-1,'play_classify','')

    for i in range(test_df.shape[0]):
        pro_dict = {} # initialzie 
        for bucket_elem in bayes_bucket: # begin calculate probability
            pro = bucket_elem.pro # initialize pro to iterate
            pro_class_name = bucket_elem.name # key 
            dict_class = bucket_elem.dic
            for column in columns:
                if (column != columns[-2]) and (column != columns[-1]):
                    att = test_df.loc[i,column]
                    if att in dict_class: # 如果存在key
                        pro *= dict_class[att]
                    else: #如果不存在key
                        pro = 0
                        break
                else:
                    continue
            pro_dict[pro_class_name] = pro
        # sort get max probability
        sorted_pro = sorted(pro_dict.items(),key=lambda x: x[1],reverse=True)
        # fill classify result
        test_df.loc[i,columns[-1]] = sorted_pro[0][0] 
    
    return test_df
            
        
if __name__ == "__main__":
    url = ['DM_Experiment3/Bayes/weather_nominal_train.arff','DM_Experiment3/Bayes/weather_nominal_test.arff']
    train_data , columns = load(url[0]) # get training data
    test_data, _ = load(url[1]) # get test data
    bayes_bucket = train_bayes_main(train_data,columns)
    res = classify_bayes_main(test_data,columns,bayes_bucket)
    print(res)