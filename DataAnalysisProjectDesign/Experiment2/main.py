import numpy as np
import pandas as pd
from scipy.io import arff

from preprocessing import Data
from rf import RandomForest
from id3_2 import ID3Tree,IG3Node

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

    # get choice
    data_choice = input('Enter 1 for iris; Enter 2 for adult:')

    dt_num = int(input('Enter your expected tree number:'))

    path = select_dataset(data_choice,iris_path,adult_path)

    # create data instance
    data_obj = Data(path)
    data_obj.load_data()
    data_obj.fill_missing_data()

    # create random forest
    rf = RandomForest(
        data=data_obj,
        dt_num=dt_num
    )

    rf.bagging()
    rf.train_rf()
    correct_rate,conf_mat = rf.test_rf()

    return dt_num,correct_rate,conf_mat

if __name__ == "__main__":
    dt_num,correct_rate,conf_mat = main()
    print('Random Forest is Composed of {} ID3 Decision Trees'.format(dt_num))
    print('Random Forest Classification Accuracy:{}'.format(correct_rate))
    print('Random Forest Classificaton Confusion Matrix Shows Below')
    print(conf_mat)