import numpy as np 

user_like = 3 #3分之上用户喜欢
user_diss = 2 #2分以下用户不喜欢

class DiffusionBGRecom:
    def __init__(self,mat_path,recommend_k=50):
        self.mat_path = mat_path #存储打分矩阵的路径
        self.mat = self.load_mat() #打分矩阵

        #获得二部图 用字典表示
        self.user_dict,self.item_dict = self.get_dict()
        self.zero_idx = [] #存储每一个user的所有缺失值
        self.diffusion_mat = None #扩散矩阵

        self.recommend_k = recommend_k

    # 加载打分矩阵
    def load_mat(self):
        return np.loadtxt(self.mat_path)
    
    # 获得二部图
    def get_dict(self):
        # user_dict usrid为key userid对应行所有不为0的索引为value
        # item_dict itemid为key itemid对应列所有不为0的索引为value
        user_dict,item_dict = {},{}

        for i in range(self.mat.shape[0]):
            series = self.mat[i]
            non_zero_idx = list(np.where(series != 0))

            self.zero_idx.append(list(np.where(series == 0)))

            user_dict[i] = non_zero_idx
        
        for j in range(self.mat.shape[1]):
            series = self.mat[:][j]
            non_zero_idx = list(np.where(series != 0))
            item_dict[j] = non_zero_idx
        
        return user_dict,item_dict
    
    # 进行扩散
    def diffusion(self):
        # 初始化扩散后的二部图
        first_diff_mat = np.zeros_like(self.mat)
        second_diff_mat = np.zeros_like(self.mat)

        # 由item向user扩散
        for key in self.item_dict:
            user_list = self.item_dict[key] #item对应的所有有打分的user
            score_arr = self.mat[:][key][user_list] #获得所有的score
            sum_score = np.sum(score_arr)
            
            loop_count = 0
            for user in user_list:
                # 进行列归一
                weight = score_arr[loop_count] / sum_score
                first_diff_mat[user][key] = weight
                loop_count += 1
        
        # 由user向item扩散
        for key in self.user_dict:
            item_list = self.user_dict[key] #user对应的所有打分的item
            score_arr = self.mat[key][item_list] # 获得所有的score
            sum_score = np.sum(score_arr) #得分总和
            first_diff_series = first_diff_mat[key][item_list]

            loop_count = 0
            for item in item_list:
                # 每个item对应的权重
                weight = score_arr[loop_count] / sum_score
                second_diff_mat[key][item] = weight * first_diff_series[item]
                loop_count += 1
        
        self.diffusion_mat = second_diff_mat
    
    #进行推荐
    def recommend(self):
        self.diffusion()
        conf_mat = np.zeros([2,2])

        for i in range(self.mat.shape[0]):
            # 获得与扩散矩阵相乘的向量
            vector = np.zeros(self.mat.shape[1])
            for j in range(self.mat.shape[1]):
                if j not in self.zero_idx[i]:
                    vector[j] = 1
            
            # 矩阵相乘
            res_vec = np.transpose(np.dot(self.diffusion_mat,vector))

            # 获得从大到小的排序索引
            sort_res_idx = list(reversed(np.argsort(res_vec)))

            # 获得推荐列表
            recom_list = []
            for k in range(self.recommend_k):
                if sort_res_idx[k] in self.zero_idx[i]:
                    recom_list.append(sort_res_idx[k])
            
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
    diffusion_recom = DiffusionBGRecom(path)
    conf_mat,precision,recall = diffusion_recom.recommend()
    print('Precision is {}'.format(precision))
    print('Recall is {}'.format(recall))
    print('Confusion Matrix shows below')
    print(conf_mat)
