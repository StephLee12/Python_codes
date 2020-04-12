import numpy as np

def normlize_v1(attribute):
    a_min,a_max = attribute.min(), attribute.max()
    new_a = []
    for i in attribute:
        new_elem = (i - a_min) / (a_max-a_min)
        new_a.append(new_elem)
    
    new_a = np.array(new_a)
    return new_a

if __name__ == "__main__":
    
    mylist = [90,80,85]
    arr = np.array(mylist)
    new_arr = normlize_v1(arr)
    print(new_arr)