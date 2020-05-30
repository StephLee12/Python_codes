import numpy as np 
import pandas as pd 
from scipy.io import arff

from preprocessing import Data
from kmeans import KMeans
from dbscan import DBSCAN

def main():

    path = 'DM_Experiment4/iris.arff'

    # load data
    data_obj = Data(path=path)
    data_obj.load_data()

    kmeans_obj = KMeans(data_obj,k=3)
    #kmeans_obj.k_initialize()
    #min_max = kmeans_obj.get_data_min_max()

    return kmeans_obj

if __name__ == "__main__":
    obj = main()