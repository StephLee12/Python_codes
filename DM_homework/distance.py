import numpy as np 

if __name__ == "__main__":
    i_list,j_list,k_list = [0.667,0.500,1.000],[1.000,1.000,0],[0,0,0.500]
    vec_i,vec_j,vec_k  = np.array(i_list),np.array(j_list),np.array(k_list)
    sim_num = []
    sim_num.append(np.sqrt(np.sum(np.square(vec_i - vec_j))))
    sim_num.append(np.sqrt(np.sum(np.square(vec_i - vec_k))))
    sim_num.append(np.sqrt(np.sum(np.square(vec_j - vec_k))))
    print(sim_num)