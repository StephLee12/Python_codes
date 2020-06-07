import numpy as np
import pandas as pd
from scipy.io import arff

# load data and normlize
def load_normlize_data(url):
    # load data
    dataset = arff.loadarff(url)
    df = pd.DataFrame(dataset[0])
    columns = list(df)

    column_sum = []
    for i in range(len(columns)-1):
        column_sum.append(df[columns[i]].sum())

    # normlize data
    for i in range(len(columns) -1):
        for j in range(df.shape[0]):
            df.loc[j,columns[i]] = df.loc[j,columns[i]] / column_sum[i]

    return df,columns

# vote
def knn_vote(best_k,train_df,columns):
      
    vote_list = train_df.loc[best_k,columns[-2]]
    mode = vote_list.mode().get(0)
    return mode
# calculate distance
def cal_dis(vec_train,vec_test):
    
    vec_train,vec_test = np.array(vec_train), np.array(vec_test)
    dis = np.sqrt(np.sum(np.square(vec_train - vec_test)))
    return dis

def cal_correct_rate(test_df,columns):
    
    # 构建混淆矩阵
    labels = list(test_df[columns[-2]].unique())
    mat_shape = len(labels)
    conf_mat = np.zeros([mat_shape,mat_shape])
    
    correct_sum = 0
    for i in range(test_df.shape[0]):
        true_label_idx = labels.index(test_df.at[i,columns[-2]])
        pred_label_idx = labels.index(test_df.at[i,columns[-1]])
        conf_mat[true_label_idx][pred_label_idx] += 1
        if test_df.at[i,columns[-2]] == test_df.at[i,columns[-1]]:
            correct_sum += 1
    return conf_mat,correct_sum / test_df.shape[0]

# knn main function
def knn_classify_main(train_df,test_df,columns,k=3):
    
    # add a new column to record classification result
    columns.append('class_classify')
    test_df.insert(len(columns)-1,'class_classify','')
    
    for i in range(test_df.shape[0]):
        # get test_data vector
        vec_test = []
        for ii in range(len(columns) -2):
            vec_test.append(test_df.iat[i,ii])
        
        best_k , best_k_dis = [], []
        for j in range(train_df.shape[0]):
            # get train_data vector
            vec_train = []
            for jj in range(len(columns) -2):
                vec_train.append(train_df.iat[j,jj])
            # calculate distance
            dis = cal_dis(vec_train,vec_test)
            if j < k: # add k initial elems
                best_k.append(j)
                best_k_dis.append(dis)
            else: # substitute elem
                for kk in range(len(best_k)):
                    if dis < best_k_dis[kk]:
                        best_k[kk] = j
                        best_k_dis[kk] = dis
                        break   
        
        vote_res = knn_vote(best_k,train_df,columns)
        test_df.at[i,columns[-1]] = vote_res
    
    
    conf_mat,correct_rate = cal_correct_rate(test_df,columns)
    return test_df,conf_mat,correct_rate



if __name__ == "__main__":
    url = ['DM_Experiment3/KNN/iris.2D.train.arff', 'DM_Experiment3/KNN/iris.2D.test.arff']
    train_data,columns = load_normlize_data(url[0])
    test_data,_ = load_normlize_data(url[1])
    test_df,conf_mat,correct_rate = knn_classify_main(train_data,test_data,columns,k=3)
    print(conf_mat)
    print('knn classification correct rate is:' + str(correct_rate))
    
    
