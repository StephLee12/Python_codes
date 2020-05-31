import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff

from preprocessing import Data
from kmeans import KMeans
from dbscan import DBSCAN


def algorithm_router(choice, data):
    if choice == '1':  #kmeans
        kmeans_obj = KMeans(data=data, k=3, iteration=500)
        kmeans_obj.kmeans_main()
        print(kmeans_obj.cluster_avg)
        kmeans_obj.show_res()
    else:  #dbscan
        dbscan_obj = DBSCAN(data=data,epsilon=0.9,min_pts=6)
        dbscan_obj.dbscan_main()
        dbscan_obj.show_res()


def main():

    path = 'DM_Experiment4/iris.arff'

    choice = input('Use KMeans Enter 1;Use DBSCAN Enter 2:')

    # load data
    data_obj = Data(path=path)
    data_obj.load_data()

    algorithm_router(choice, data_obj)


if __name__ == "__main__":
    main()