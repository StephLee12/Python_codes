import pandas as pd 
from scipy.io import arff

import MissingData as Md

def load(url):
    
    dataset = arff.loadarff(url) # read data
    df = pd.DataFrame(dataset[0]) # turn to dataframe
    att_list = list(df) # get columns name
    class_column = att_list.pop() # get last column name and att_list remove last element

    nom_columns,num_columns = [],[] # nominal columns and numeric columns


    for att in att_list:
        if df[att].dtypes == 'object':
            nom_columns.append(att)
        else:
            num_columns.append(att)

    # preprocess dataset e.g. fill missing data
    df = Md.fill_nom_data(df,nom_columns) # fill nominal column missing value
    df = Md.bytes_to_float(df,num_columns) # turn bytes into float
    return df,att_list,class_column,nom_columns,num_columns