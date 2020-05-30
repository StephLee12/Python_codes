from scipy.io import arff
import pandas as pd
import numpy as np
from itertools import chain,combinations


def load():
    # load data
    dataset = arff.loadarff('DM_Experiment2/weather_nominal.arff')
    db = dataset[0]
    df = pd.DataFrame(dataset[0])
    columns = list(df)

    return db, df, columns

def gen_database(db):

    db = db.tolist()  # array to list
    db_list = []  # every elem in list is tuple convert that to list

    for tu in db:
        list_conv = list(tu)
        list_conv_length = len(list_conv)
        for i in range(list_conv_length):
            list_conv[i] = str(list_conv[i], 'utf-8')
        db_list.append(list_conv)

    return db_list


def get_support(elem, db_list):

    sup = 0

    for trans in db_list:
        if (elem in trans):
            sup += 1

    return sup / len(db_list)


def gen_l1(params, df, columns, db_list):

    l1_set = []
    min_sup = params[0]

    for column in columns:
        series = df[column].unique()  # delete duplicate element
        tmp = series.tolist()  #array to list
        for i in tmp:  #split every series into single element
            i = str(i, 'utf-8')
            sup = get_support(i, db_list)
            if (sup >= min_sup):
                l1_set.append(
                    ([i], sup))  # append a tuple, last element is support

    return l1_set


def get_max_subset(ori_set):

    max_subset = []

    max_subset.append(ori_set[1:])
    for i in range(1, len(ori_set) - 1):
        tmp = ori_set[:i] + ori_set[i + 1:]
        max_subset.append(tmp)
    max_subset.append(ori_set[:-1])

    return max_subset


def has_infre_subset(max_subset, lk):

    for i in max_subset:
        flag = 0
        for j in lk:
            if (set(i) == set(j)):
                flag = 1
                break
        if (flag == 0):
            return True

    return False


def gen_apriori(k, lk):  # k here represents k-1

    item_set = []

    lk_nosup = []
    for ii in lk:
        lk_nosup.append(ii[0])

    lk_nosup_length = len(lk_nosup)

    for i in range(lk_nosup_length - 1):
        for j in range(i + 1, lk_nosup_length):
            com_ij = lk_nosup[i] + lk_nosup[j]  #combine i and j
            uni_ij = list(set(com_ij))  # delete duplicate element
            if (len(uni_ij) == k + 1):  # consider it as a candidate
                max_subset = get_max_subset(
                    uni_ij)  # get uni_ij's k-item subset
                if (not has_infre_subset(
                        max_subset,
                        lk_nosup)):  # uni_ij doesn't have infrequent subset
                    item_set.append(uni_ij)

    return item_set


def is_loop_continue(fre_set):

    if (fre_set == []):
        return False

    return True


def run_apriori(params, l1_set, db_list):

    min_sup = params[0]
    db_len = len(db_list)
    fre_set = [[] for i in range(len(l1_set))
               ]  #create a bucket to store all frequent itemsets
    fre_set[0] = l1_set  # l1_set puts into the first bucket

    k = 1
    while (is_loop_continue(fre_set[k - 1])):
        ck = gen_apriori(k, fre_set[k - 1])
        for i in ck:
            sup_count = 0
            for j in db_list:
                if (set(i).issubset(set(j))):
                    sup_count += 1
            sup = sup_count / db_len
            if (sup >= min_sup):
                flag = 0
                for tmp in fre_set[k]:
                    if (i,sup) == tmp:
                        flag = 1
                        break
                if flag == 0:
                    fre_set[k].append((i, sup))
                
        k += 1

    count = 1
    for i in fre_set:
        if len(i) != 0:
            print("L%d itemset size is:%d" % (count, len(i)))
        count += 1
    return fre_set

def get_all_subsets(item):
    
    all_subsets = list(chain.from_iterable(combinations(item,r) for r in range(1,len(item)+1)))
    all_subsets.pop()

    for i in range(len(all_subsets)):
        all_subsets[i] = set(all_subsets[i])

    return all_subsets

def get_fre_set_len(fre_set):
    len = 0

    tmp = fre_set[0]
    while tmp != []:
        len += 1
        tmp = fre_set[len]
    
    return len


def dig_rules(fre_set,min_conf):
    strong_rules = [] #强规则

    fre_set_len = get_fre_set_len(fre_set)

    for i in range(1,fre_set_len):
        fre_list = fre_set[i]
        for j in fre_list:
            item = j[0] # 元组中的trans
            sup_item = j[1] # 支持度
            item_subset = get_all_subsets(item)
            item = set(item)
            for subset in item_subset:
                diff_set = item.difference(subset) #获得全集的补集
                subset_len = len(subset)
                tmp_list = fre_set[subset_len-1]
                for fre_elem in tmp_list: #遍历fre_set找到subset的支持度
                    tmp_set = fre_elem[0]
                    tmp_sup = fre_elem[1]
                    if set(tmp_set) == subset: #找到对应的subset
                        conf = sup_item / tmp_sup #计算置信度
                        if conf > min_conf:
                            strong_rules.append(((subset,diff_set),conf)) #用元组保存数据
                            break
                    else:
                        continue

    return strong_rules


if __name__ == "__main__":

    params = [0.2, 0.75]  # min_sup, min_conf

    db, df, columns = load()
    db_list = gen_database(db)
    l1_set = gen_l1(params, df, columns, db_list)
    fre_set = run_apriori(params, l1_set, db_list)
    strong_rules = dig_rules(fre_set,params[1])
    print(strong_rules)
    
