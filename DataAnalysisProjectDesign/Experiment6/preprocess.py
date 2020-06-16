import numpy as np 

movie_num = 193609 #电影总数
user_num = 610 #用户数

def load_mat(path):
    
    #初始化评分矩阵
    mat = np.zeros([user_num,movie_num]) 
    
    # 读取txt
    with open(path) as f:
        for data in f.readlines():
            data = data.strip('\n')
            data = data.split(',')
            user_id,movie_id,score = int(data[0]),int(data[1]),float(data[2])
            mat[user_id-1,movie_id-1] = score
    
    np.savetxt('DataAnalysisProjectDesign/Experiment6/mat.txt',mat,fmt='%.2f')


# if __name__ == "__main__":
    # rating_path = 'DataAnalysisProjectDesign/Experiment6/ratings.txt'
    # load_mat(path)