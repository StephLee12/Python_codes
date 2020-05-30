import numpy as np
import pandas as pd
from scipy.io import arff

from preprocessing import Data
from cart_2 import CARTTree
from rf import RandomForest


# select iris or adult
def select_dataset(choice, iris_path, adult_path):
    if choice == '1':
        return iris_path
    else:
        return adult_path

def main():

    # set path
    iris_path = 'DataAnalysisProjectDesign/Experiment2/iris_train.arff'
    adult_path = 'DataAnalysisProjectDesign/Experiment2/adult_train.arff'

    print('Random forest is composed with many CART decision trees')
    # get choice
    data_choice = input('Enter 1 for iris ; Enter 2 for adult :')

    path = select_dataset(data_choice, iris_path, adult_path)

    # create data instance
    data_obj = Data(path)
    data_obj.load_data()
    data_obj.fill_missing_data()
    
    # create random forest
    rf = RandomForest(
        data=data_obj,
        dt_num=20
    )

    rf.bagging()
    rf.train_rf()
    correct_rate = rf.test_rf()

    return correct_rate



# main
if __name__ == "__main__":

    correct_rate = main()
    print(correct_rate)
