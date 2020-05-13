import numpy as np  
import pandas as pd 
#import math

#import Load

# 将bytes类型转换为float
def bytes_to_float(df,num_columns):
    for column in num_columns:
        df[column].astype(float)

    return df

# ```
# # normlize numeric data
# def norm_num_data(df,num_columns):
#     df = bytes_to_float(df,num_columns)
#     num_df = df[num_columns]
#     print(num_df)
#     for column in num_columns:
#         min,max = df[column].min(),df[column].max()
#         for i in range(df.shape[0]):
#             pass
#     return num_df


# use mode to fill the missing value
def fill_nom_data(df,nom_columns):
    for column in nom_columns:
        column_slice = df[column]
        mode = column_slice.mode().get(0)
        column_slice[column_slice==b'?'] = mode
    return df

# if __name__ == "__main__":
#     url = 'ProjectDesign/Experiment1/adult_train.arff'
#     df,att_list,class_column,nom_columns,num_columns=Load.load(url)
#     fill_nom_data(df,nom_columns)
#     #num_df = norm_num_data(df,num_columns)