import scipy.io as scio
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors

# path =  'Training_1/TData.mat'
# data = scio.loadmat(path)['TData']

# test_data = data[200,200,:]
# plt.plot(test_data)
# plt.show()

# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         series = data

plt.style.use('seaborn-whitegrid')
data = scio.loadmat('Training_1/heat.mat')['J']
x = np.linspace(1,256,256)
y = np.linspace(1,320,320)

X,Y = np.meshgrid(x,y)
#colorlist = ['w','gray','aqua']
#cmaps = colors.LinearSegmentedColormap('mylist',colorlist,N=3000)
#cset = plt.contour(X,Y,data.T,cmap=cmaps)
#plt.quiver(X,Y,Y,-X)
contour = plt.contour(X,Y,data.T,100,cmap='viridis')
#plt.clabel(contour)
#plt.colorbar(cset)
plt.savefig('Training_1/contour.png')
plt.colorbar()
plt.show()


