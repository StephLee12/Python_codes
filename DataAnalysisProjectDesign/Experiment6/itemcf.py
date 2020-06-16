import numpy as np  

user_like = 3 #3分之上用户喜欢
user_diss = 2 #2分以下用户不喜欢

class ItemCF:
    def __init__(self,mat_path,k=3,recommend_k=50):
        self.mat_path = mat_path #读取评分矩阵的路径
        self.mat = self.load_mat() #评分矩阵

        self.sim_mat = self.calc_sim_mat() #项的相似度矩阵
        self.k = k #最相似的k个项
        self.recommend_k = recommend_k #推荐列表容量
        self.zero_idx = [] #存储每一个user的所有缺失值
        
        self.conf_mat = None #推荐的混淆矩阵
        self.precision = 0 # 准确率
        self.recall = 0 #召回率

    # 加载打分矩阵
    def load_mat(self):
        return np.loadtxt(self.mat_path)

    # 计算项的相似度矩阵 用皮尔逊相关系数
    def calc_sim_mat(self):
        # 初始化
        sim_mat = np.zeros([self.mat.shape[1],self.mat.shape[1]])
        for i in range(sim_mat.shape[0]):
            for j in range(sim_mat.shape[0]):
                if i == j:
                    sim_mat[i,j] = 1
                else:
                    # 计算使用皮尔逊相关系数
                    item_i,item_j = self.mat[:][i],self.mat[:][j]
                    mean_i,mean_j = item_i.mean(),item_j.mean()
                    std_i,std_j = item_i.std(),item_j.std()
                    sim = 0
                    for k in range(self.mat.shape[1]):
                        sim += (item_i[k]-mean_i) * (item_j[k]-mean_j)
                    sim /= (std_i * std_j)
                    sim_mat[i][j] = sim
        
        return sim_mat

    
    # 进行打分
    def pred_score(self):
        for i in range(self.mat.shape[0]):
            # 获得缺失位置
            series = self.mat[i]
            zero_idx = np.where(series == 0)

            self.zero_idx.append(list(zero_idx))

            # 获得相似度列表
            sim_list = list(self.sim_mat[i])
            sim_list.pop(i)

            # 对相似度从高到低排序
            sort_sim_list =  sorted(sim_list,reverse=True)

            # 选取最相似的k个项
            most_k_sim = sort_sim_list[:self.k]
            most_k_idx = []
            for elem in most_k_sim:
                idx = sim_list.index(elem)
                most_k_idx.append(idx)
            
            # 打分
            for idx in zero_idx:
                # 获得相似度最高的k个项的打分
                score = []
                for k_idx in most_k_idx:
                    score.append(self.mat[i][k_idx])
                
                # 计算得分
                final_score = (np.array(score) * np.array(most_k_sim)) / sum(most_k_sim)

                # 预测值填充进打分矩阵
                self.mat[i][idx] = final_score
    
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
    item_cf = ItemCF(path)
    conf_mat,precision,recall = item_cf.recommend()
    print('Precision is {}'.format(precision))
    print('Recall is {}'.format(recall))
    print('Confusion Matrix shows below')
    print(conf_mat)
