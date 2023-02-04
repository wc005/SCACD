import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch


def get_data(path, key, window, step, Scaler):
    df = pd.read_csv(path, usecols=[key], encoding='ISO-8859-1')
    x_data = np.array(df)
    x_data = np.squeeze(x_data)
    # 归一化
    x_data = Scaler.fit_transform(x_data.reshape(-1, 1))
    x_data = np.nan_to_num(x_data, nan=0)
    result = []
    for i in range(0, x_data.shape[0] - window, step):
        array = np.array(x_data[i:i + window])
        result.append(array)
    data = np.array(result).squeeze()
    data = np.nan_to_num(data, nan=np.nanmean(data))
    return data


def getset(data, RNN_len, batch_size):
    rand_index = np.random.randint(0, data.shape[0]-RNN_len, size=data.shape[0]-RNN_len)
    one_data = []
    for i in range(RNN_len):
        one_data.append(data[rand_index + i, :])
    # (batch_size,step, window)
    data = np.array(one_data).swapaxes(0, 1)
    x_data = data[:, 0:RNN_len - 1:, ]
    y_data = data[:, -1, ]
    # 数据 shuffle
    index = [i for i in range(len(x_data))]
    np.random.shuffle(index)
    x_data = x_data[index]
    y_data = y_data[index]
    train_count = int(len(x_data) / 10 * 8)
    # 训练数据
    trian_x = x_data[0:train_count, ]
    trian_y = y_data[0:train_count, ]
    train_dataset = Data.TensorDataset(torch.from_numpy(trian_x),
                                       torch.from_numpy(trian_y)
                                       )
    trainload = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    # 验证集
    val_count = int(len(x_data) / 10 * 8)
    # val_x = x_data[train_count:val_count, ]
    # val_y = y_data[train_count:val_count, ]
    # val_dataset = Data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    # valload = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    valload = 0
    # 测试集
    test_x = x_data[val_count:, ]
    test_y = y_data[val_count:, ]
    test_dataset = Data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    testload = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True)
    # testload = 0
    return trainload, valload, testload
