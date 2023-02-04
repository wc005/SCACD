import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn
import pandas as pd
datalist =['ETT', 'exchange_rate', 'national_illness', 'traffic', 'weather', 'electricity' ]
datadict ={'ETT':'ETT', 'exchange_rate':'Exchange_rate', 'national_illness':'ILI',
           'traffic':'Traffic', 'weather':'Weather', 'electricity':'Electricity'}
# datalist =[ 'weather' ]
for item in datalist:
    datapath = '../../data/{}/{}.csv'.format(item, item)
    df = pd.read_csv(datapath, usecols=['OT'])

    # x_data = np.array(df)[0:50]
    x_data = np.array(df)
    y = np.squeeze(x_data)
    x = np.arange(0, len(y))

    yf = abs(fft(y))                # 取绝对值
    # 处理直流分量
    yf[0] = yf[0]/len(y)
    # 归一化处理
    yf_nor = yf/len(x) * 2

    xf = np.arange(len(y))        # 频率

    # plt.subplot(111)
    # plt.plot(x, y, linewidth=0.5)
    # plt.title('{}'.format(item), fontsize=7)
    # plt.tick_params(axis='both', which='major', labelsize=7)
    # plt.xlabel('values', fontdict=[])
    plt.figure(dpi=170, figsize=(9, 6))
    plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')


    title = '{} - FFT(two sides)'.format(datadict[item])
    plt.title(title, fontsize=18, color='darkviolet')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'

    plt.subplot(1, 1, 1)
    plt.plot(xf, yf, 'b', linewidth=1)

    # plt.legend()
    path_pic = './FFTpic/{}.pdf'.format(item)
    plt.savefig(path_pic, bbox_inches='tight')
    # plt.show()