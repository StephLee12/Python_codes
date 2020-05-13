import numpy as np
import pandas as pd

import Gini
import InformationGain as InfoG


# get best split attribute via id3
def ig_att_selection(df, att_list, class_column, nom_columns, num_columns):
    # get information gain
    ig = InfoG.IG(df, att_list, class_column, nom_columns, num_columns)
    att_selected = None
    value_selected = None
    max_ig = 0
    idx = None
    for i in range(len(ig)):
        if ig[i].ig_value > max_ig:
            max_ig = ig[i].ig_value
            att_selected = ig[i].name
            value_selected = ig[i].value
            idx = i
    att_list.pop(idx)

    return att_list, [att_selected, value_selected]


# get best split attribute via gini
def cart_att_selection(df, att_list, class_column, nom_columns, num_columns):

    new_att_list, gini_att = Gini.GINI(df, att_list, class_column, nom_columns,
                                       num_columns)

    return new_att_list, gini_att