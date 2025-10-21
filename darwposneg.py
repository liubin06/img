import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path1 = r'D:\python\results\self\0.1_128_acc.csv'
def draw(temperature,path):
    data_raw = np.array(pd.read_csv(path,header=0,sep=','))
    data = temperature * np.log(data_raw[:,4:])
    x = data_raw[:,0]
    # plt.plot(x,data_raw[:,1],label=r'acc')
    # plt.plot(x, data_raw[:, 3], label=r'loss')
    plt.plot(x, data[:, 0], label=r'posmean')
    plt.plot(x, data[:, 1], label=r'posvar')
    plt.plot(x, data[:, 2], label=r'negmean')
    plt.plot(x, data[:, 3], label=r'negvar')
    plt.legend()
    plt.show()
draw(0.1,path1)