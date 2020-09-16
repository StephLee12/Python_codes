import numpy as np 
import matplotlib.pyplot as plt 

R = np.load(file='Training_1/R.npy')
Q = np.load(file='Training_1/Q.npy')
# 取某一-帧画等高线 第39帧
plt.style.use('seaborn-whitegrid')
x = np.linspace(1,256,256)
y = np.linspace(1,320,320)
X,Y =  np.meshgrid(x,y)
contour = plt.contour(X,Y,R[:,:,38].T,20,cmaps='viridis')
#plt.colorbar()
plt.show()