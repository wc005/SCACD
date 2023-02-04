import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import dataclass
from itertools import product

import utils
from dataclass import exchange_rate
# from tools import read


# mae = []
# mse = []
# results = read("results/ETT_final.pkl")
# for key in results[200].keys():
#     mse.append(results[200][key]['mse'])
#     mae.append(results[200][key]['mae'])
#
# print(np.mean(np.array(mse)))
# print(np.mean(np.array(mae)))


# with open("results/traffic.pkl", "rb") as tf:
#     result = pickle.load(tf)
#     print(result)
df = pd.read_csv('../../data/weather/weather.csv', usecols=['OT'])
path_pic = 'dis_pic/{}_distribution.pdf'.format('weather')
# ETT, exchange_rate, national_illness, traffic, weather, electricity
window = 820
step = 800
x_data = np.array(df)
x_data = np.squeeze(x_data)
mu = []
std = []
for i in range(0, 9000 - window, step):
    array = np.array(x_data[i:i + window])
    mu1 = np.mean(array)
    std1 = np.std(array)
    mu.append(mu1)
    std.append(std1)
mu = np.array(mu).squeeze()
std = np.array(std).squeeze()
z = np.array(range(0, len(mu)))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter3D(mu, std, z, color="red", s=7)
ax1.plot3D(mu, std, z, color='blue', linewidth=1)
# , fontproperties='Calibri'
ax1.set_xlabel('$\mu$')
ax1.set_ylabel('$\sigma$')
ax1.set_zlabel('$t$')
plt.savefig(path_pic, bbox_inches='tight')
plt.show()
#
#
# mu1 = np.mean(head)
# std1 = np.std(head)
# print(mu1)
# print(std1)
# mu2 = np.mean(middle)
# std2 = np.std(middle)
# print(mu2)
# print(std2)
# mu3 = np.mean(tail)
# std3 = np.std(tail)
# print(mu3)
# print(std3)
# plt.subplot(3, 1, 1)
# plt.hist(head)
# plt.subplot(3, 1, 2)
# plt.hist(middle)
# plt.subplot(3, 1, 3)
# plt.hist(tail)
# plt.show()
