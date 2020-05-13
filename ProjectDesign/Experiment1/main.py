import numpy as np
import pandas as pd
from scipy.io import arff

import Gini
import InformationGain as InfoG
import Load
from ID3 import ig_gen_DT
from CART import cart_gen_DT

# select iris or adult
def select_dataset(choice, iris_url, adult_url):
    if choice == '1':
        return iris_url
    else:
        return adult_url
        
# generate DT main function (router)
def genDT(train_data, class_data, att_list, class_column,nom_columns,num_columns,belong_att,
          tree_choice):
    if tree_choice == '1':  # id3
        return ig_gen_DT(train_data, class_data, att_list, class_column,nom_columns,num_columns,
                         belong_att)

    else:  # cart
        return cart_gen_DT(train_data, class_data, att_list, class_column,nom_columns,num_columns,
                           belong_att, None, None)

# entry
if __name__ == "__main__":
    iris_url = [
        'ProjectDesign/Experiment1/iris_train.arff',
        'ProjectDesign/Experiment1/iris_test.arff'
    ]
    adult_url = [
        'ProjectDesign/Experiment1/adult_train.arff',
        'ProjectDesign/Experiment1/adult_test.arff'
    ]

    data_choice = input('Enter 1 for iris DT; Enter 2 for adult DT:')
    tree_choice = input('Enter 1 for ID3; Enter 2 for CART:')

    url = select_dataset(data_choice, iris_url, adult_url)
    train_data, att_list, class_column,nom_columns,num_columns = Load.load(url[0])
    class_data = train_data[class_column]
    test_data, _, __,___,____ = Load.load(url[1])
    DTTree = genDT(train_data, class_data, att_list, class_column,nom_columns,num_columns, None,
                   tree_choice)
