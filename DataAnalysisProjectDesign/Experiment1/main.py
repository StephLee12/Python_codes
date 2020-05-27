import numpy as np
import pandas as pd
from scipy.io import arff

from preprocessing import Data
from id3_2 import ID3Tree
from cart_2 import CARTTree


# select iris or adult
def select_dataset(choice, iris_path, adult_path):
    if choice == '1':
        return iris_path
    else:
        return adult_path


# build different type of dt
def dt_router(train_data_obj, test_data_obj, choice):
    if choice == '1':  # id3
        return ID3Tree(train_data=train_data_obj, test_data=test_data_obj)
    else:  # cart
        return CARTTree(train_data=train_data_obj, test_data=test_data_obj)


def main():

    # set path
    iris_path = [
        'DataAnalysisProjectDesign/Experiment1/iris_train.arff',
        'DataAnalysisProjectDesign/Experiment1/iris_test.arff'
    ]
    adult_path = [
        'DataAnalysisProjectDesign/Experiment1/adult_train.arff',
        'DataAnalysisProjectDesign/Experiment1/adult_test.arff'
    ]

    # get choice
    data_choice = input('Enter 1 for iris DT; Enter 2 for adult DT:')
    tree_choice = input('Enter 1 for ID3; Enter 2 for CART:')

    path = select_dataset(data_choice, iris_path, adult_path)

    # create train data instance
    train_data_obj = Data(path[0])
    train_data_obj.load_data()
    train_data_obj.fill_missing_data()
    # create test data instance
    test_data_obj = Data(path[1])
    test_data_obj.clear_memory()
    test_data_obj.load_data()
    test_data_obj.fill_missing_data()

    tree = dt_router(train_data_obj, test_data_obj, tree_choice)

    tree.test()

    conf_mat, judge = tree.get_conf_mat()

    return tree, conf_mat, judge


# main
if __name__ == "__main__":

    tree, conf_mat, judge = main()
    print(conf_mat)
