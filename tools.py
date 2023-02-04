import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch

def drawloss(path_pic, trainloss, validloss):

    train_loss = np.array(trainloss)
    valid_loss = np.array(validloss)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    plt.figure()
    plt.subplot(1, 1, 1)
    # plt.title("t={}".format('GroundTruth'))
    plt.plot(train_loss, label='train_loss', color="red", linewidth=1)
    plt.xlabel('epoch(×10)')
    plt.ylabel('loss')
    # plt.subplot(2, 1, 2)
    # plt.title("t={}".format('SCACD'))
    plt.plot(valid_loss, label='valid_loss', color="blue", linewidth=1)
    plt.legend()
    plt.savefig(path_pic, bbox_inches='tight')

class tools:
    def __init__(self, n, item, step, dataset):
        super(tools, self).__init__()
        self.item = item
        self.step = step
        self.num = n
        self.dataset = dataset

    def draw(self, Truth, pre):
        groundtrueth = np.mean(Truth.cpu().clone().detach().numpy(), 0)
        SCACD = np.mean(pre.cpu().clone().detach().numpy(), 0)
        data = torch.stack((Truth, pre), dim=1)
        # 路径
        path_name = './picture/{}/{}_{}_{}'.format(self.dataset, self.item, self.step, self.num)
        path_pic = './picture/{}/{}_{}_{}.pdf'.format(self.dataset, self.item, self.step, self.num)
        # 保存数据
        with open(path_name, "wb") as tf:
            pickle.dump(data, tf)

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'
        plt.figure()
        plt.subplot(1, 1, 1)
        # plt.title("t={}".format('GroundTruth'))
        plt.plot(groundtrueth, label='GroundTruth', color="red", linewidth=1)
        # plt.subplot(2, 1, 2)
        # plt.title("t={}".format('SCACD'))
        plt.plot(SCACD, label='SCACD', color="blue", linewidth=1)
        plt.legend()

        plt.savefig(path_pic, bbox_inches='tight')
        # plt.show()
        self.num = self.num + 1

    def read(path):
        with open(path, "rb") as tf:
            result = pickle.load(tf)
            print(result)
            return result


def draw_heatmap(data, i, window):

    ax = sns.heatmap(data, annot=True, annot_kws={'size': 10})
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Adaptive Covariance Matrix')  # 图标题
    # ax.set_xlabel('input features')  # x 轴标题
    # ax.set_ylabel('first hidden layer')
    # plt.legend()
    plt.savefig('./Cov/Cov_{}_{}.pdf'.format(window, i), bbox_inches='tight')
    # plt.show()