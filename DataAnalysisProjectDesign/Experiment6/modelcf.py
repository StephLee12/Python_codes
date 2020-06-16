import numpy as np 

user_like = 3 #3分之上用户喜欢
user_diss = 2 #2分以下用户不喜欢

class ModelCF:
    def __init__(self,mat_path,recommend_k=50):
        self.mat_path = mat_path #读取评分矩阵的路径
        self.mat = self.load_mat() #评分矩阵

        self.max_iterations = 1500 # SVD最大迭代次数
        self.lr = 1e-3 # 学习率 SVD的超参
        self.Lambda = 1e-3 # SVD的超参 正则项
        self.K = 5 # SVD中的潜在因子
        self.svd_loss = 0 # SVD最后的loss
        self.svd_mat = None # SVD分解得到的矩阵
        self.zero_idx = [] #存储每一个user的所有缺失值

        self.recommend_k = recommend_k #推荐列表容量
        self.conf_mat = None #推荐的混淆矩阵
        self.precision = 0 # 准确率
        self.recall = 0 #召回率
    
    # 加载打分矩阵
    def load_mat(self):
        return np.loadtxt(self.mat_path)
    
    # SVD分解
    def svd(self):
        # 初始化 U V
        U = np.random.normal(size=(self.mat.shape[0],self.K))
        V = np.random.normal(size=(self.K,self.mat.shape[1]))
        svd_mat = None
        loss = 0

        # 获取打分矩阵中所有非零元素下标
        # non_zeros是一个元组 其中的每个元素为一个array 存储行列下标
        non_zeros = np.where(self.mat != 0)

        # 开始SVD迭代
        for _ in range(self.max_iterations):
            dU,dV = np.zeros_like(U),np.zeros_like(V)
            for row,column in non_zeros:
                for k in range(self.K):
                    dU[row,k] += (-2.0) * loss * V[k,column]
                    dV[k,column] += (-2.0) * loss * U[row,k]
            
            dU += 2.0 * self.Lambda * U
            dV += 2.0 * self.Lambda * V
            U = U - self.lr * dU
            V = V - self.lr * dV
            svd_mat = np.dot(U,V)
            loss = (np.square(self.mat - svd_mat) * (self.mat != 0)).sum()
        
        self.svd_mat = svd_mat
        self.svd_loss = loss


    # 通过SVD矩阵 将打分矩阵 填充
    def pred_score(self):
        self.svd()

        for i in range(self.mat.shape[0]):
            # 获得缺失位置
            series = self.mat[i]
            zero_idx = np.where(series == 0)

            self.zero_idx.append(zero_idx)

            # 根据SVD矩阵填充打分矩阵
            for idx in zero_idx:
                self.mat[i][idx] = self.svd_mat[i][idx]
    
    # 进行推荐
    def recommend(self):
        self.pred_score()
        # 初始化混淆矩阵
        conf_mat = np.zeros([2,2])
        # 遍历每个用户
        for i in range(self.mat.shape[0]):
            series = self.mat[i]

            # 获得从大到小排序的索引
            sort_series_idx = list(reversed(np.argsort(series)))

            # 获得推荐列表
            recom_list = []
            for k in range(self.recommend_k):
                if sort_series_idx[k] in self.zero_idx[i]:
                    recom_list.append(sort_series_idx[k])
            
            # 判断推荐的产品 用户是否喜欢 不推荐的是否喜欢
            for j in self.zero_idx[i]:
                # 若某项为推荐商品
                if j in recom_list:
                    if self.mat[i][j] >= user_like: #推荐的用户喜欢
                        conf_mat[0][0] += 1
                    elif self.mat[i][j] <= user_diss: # 推荐的用户不喜欢
                        conf_mat[1][0] += 1
                else:
                    if self.mat[i][j] >= user_like: # 不推荐的用户喜欢
                        conf_mat[0][1] += 1
                    elif self.mat[i][j] <= user_diss: #不推荐的用户不喜欢
                        conf_mat[1][1] += 1
        
        self.conf_mat = conf_mat
        
        # 计算召回率和准确率
        precision = conf_mat[0][0] / np.sum(conf_mat[:][0])
        recall = conf_mat[0][0] / np.sum(conf_mat[0])

        self.recall = recall
        self.precision = precision

        return conf_mat,precision,recall

if __name__ == "__main__":
    path =  'DataAnalysisProjectDesign/Experiment6/mat.txt'
    model_cf = ModelCF(path)
    conf_mat,precision,recall = model_cf.recommend()
    print('Precision is {}'.format(precision))
    print('Recall is {}'.format(recall))
    print('Confusion Matrix shows below')
    print(conf_mat)