import numpy as np  
import pandas as pd 
import scipy.io as scio
from math import pow
import matplotlib.pyplot as plt 

# 参数
Pho = 7.8  #密度
Sigma = 4.03e6
C = 460 # 比热
T0 = 298 # 初始温度
V = 10*4.5*0.024 # 体积
dV = V/(256*320) 
dM = dV*Pho/(1000)
dS = 100*45*pow(10,-6)/(256*320)
S_CUT = np.sqrt(dS) * 0.24* pow(10,-3) #横截面积
frame_time = 1/383 #每一帧的时间
R0 = 0.73 * 0.1 / (45 * 0.24)

# 加载电流密度数据
current_density = scio.loadmat('Training_1/heat.mat')['J']
# 加载温度数据
T = scio.loadmat('Training_1/TData.mat')['TData']
# 计算产生的焦耳热
Q = np.zeros_like(T)
R = np.zeros_like(T)
for i in range(Q.shape[0]):
    for j in range(Q.shape[1]):
        for k in range(Q.shape[2]):
            if k == 0:
                R[i,j,k] = R0
                continue
            Q[i,j,k] = C*dM*(T[i,j,k]-T[i,j,k-1])
            R[i,j,k] = Q[i,j,k]/(S_CUT**2 * Sigma * Pho * pow(10,3) * C * frame_time)
np.save(file='Training_1/Q.npy',arr=Q)
np.save(file='Training_1/R.npy',arr=R)
# 计算阻抗

for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        for k in range(R.shape[2]):
            if k == 0:
                
                continue
            



# R = np.loadtxt('Traning_1/R.txt')
# # 取某一-帧画等高线 第39帧
# plt.style.use('seaborn-whitegrid')
# x = np.linspace(1,256,256)
# y = np.linspace(1,320,320)
# X,Y =  np.meshgrid(x,y)
# contour = plt.contour(X,Y,R[:,:,38].T,20,cmaps='viridis')
# plt.colorbar()
# plt.show()

