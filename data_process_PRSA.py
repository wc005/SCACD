import pandas as pd
import numpy as np


def get_data_train(path, window, step):
    df = pd.read_csv(path, usecols=["HULL"])
    x_data = np.array(df)

    x_data = np.squeeze(x_data)
    result = []
    train_window = x_data.shape[0] / 3 * 2

    for i in range(0, int(train_window - window), step):
        array = np.array(x_data[i:i + window])
        result.append(array)
    result = np.array(result)
    return result


def get_data_test(path, window, step):
    df = pd.read_csv(path, usecols=["PRES"])
    x_data = np.array(df)
    x_data = np.squeeze(x_data)
    result = []
    test_window = x_data.shape[0] / 3 * 2
    for i in range(int(test_window), int(x_data.shape[0] - window), step):
        array = np.array(x_data[i:i + window])
        result.append(array)
    result = np.array(result)
    return result


def get_batch(x, step, batch_size):
    rand_index = np.random.randint(0, x.shape[0] - (step+1), size=batch_size)
    one_data = []
    for i in range(step):
        one_data.append(x[rand_index + i, :])
    # (step, batch_size, window)
    data = np.array(one_data).swapaxes(0, 1)
    x_data = data[:, 0:step - 1:, ]
    y_data = data[:, -1:, ]
    return x_data, y_data

