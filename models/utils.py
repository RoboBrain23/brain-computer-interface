import numpy as np
from keras.utils import to_categorical
import os

def filter_data(data, low=9, high=15, fs=128, order=4,is_fb=False):
    """
    filter the data with the bandpass filter :

    if is_fb is True, the filter bank will be used, the data will be filtered
    by the bandpass filter in the frequency range of each harmonics of the low frequency and the high frequency.

    if is_fb is False, the data will be filtered by the bandpass filter
    in the frequency range of first harmonics of the low frequency and the high frequency.

    :param data: the data to be filtered

    :param low: the lowest frequency of the our target frequencies

    :param high: the highest frequency of the our target frequencies

    :param fs: the sampling frequency

    :param order: the order of the filter

    :return: the filtered data
    """
    harmonics = get_harmonics(low, high)
    if is_fb:
        filtered_data = []
        for freq_range in harmonics:
            wn1, wn2 = 2 * freq_range[0] / fs, 2 * freq_range[1] / fs
            channel_data_list = []
            for i in range(data.shape[1]):
                b, a = butter(order, [wn1, wn2], 'bandpass')
                filtedData = filtfilt(b, a, data[:, i])
                channel_data_list.append(filtedData)
            filtered_data.append(np.array(channel_data_list))
        return np.array(filtered_data)
    else:
        wn1 = 2*harmonics[0][0]/fs
        wn2 = 2*harmonics[-1][-1]/fs
        channel_data_list = []
        for i in range(data.shape[1]):
            b, a = butter(order, [wn1,wn2], 'bandpass')
            filtedData = filtfilt(b, a, data[:,i])
            channel_data_list.append(filtedData)
        channel_data_list = np.array([channel_data_list])

        return channel_data_list
    

def get_harmonics(start_freq, end_freq):
    harmonics = []
    # i = 1
    for i in range(1,4):
        harmonics.append([int(i * start_freq) - 2, int(i * end_freq) + 2])
        # i += 1
    return harmonics


def train_val_split(y_label):
    """
    split the training data into training set and validation set randomly with the ratio of 10:1

    :param y_label: the label of the training data

    :return: the training set and validation set
    """
    train_list = np.arange(y_label.shape[0])
    val_list = np.random.choice(train_list, int(y_label.shape[0] // 10))
    train_list = [train_list[i] for i in range(len(train_list)) if (i not in val_list)]
    return train_list, val_list


def train_data_generator(batch_size, data,  y_label, start_time, train_list,num_classes,input_shape,is_fb=False):
    """
    generate the training data

    :param batch_size:  the size of the batch

    :param data: the data of the training data

    :param y_label:  the label of the training data

    :param start_time:  the start time of the training data

    :param train_list:  the list of the training data

    :param num_classes:  the number of the classes

    :param input_shape:  the shape of the input data

    :return:  the training data
    """
    input_shape = (batch_size,)+input_shape
    if input_shape[-1] != 1:
        input_shape = input_shape[:-1] + (1,)
    x_train1 = np.zeros(input_shape, dtype=np.float32)
    if is_fb:
        x_train2 = np.zeros(input_shape, dtype=np.float32)
        x_train3 = np.zeros(input_shape, dtype=np.float32)
    # x_train4 = np.zeros((batch_size, self.channel, win_train, 1), dtype=np.float32)
    y_train = np.zeros((batch_size, num_classes), dtype=np.float32)
    while True:
        # get training samples of batch_size trials
        for i in range(batch_size):
            # randomly selecting the single-trial
            k = np.random.choice(train_list)
            # get the label of the single-trial
            y_data = y_label[k]
            # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time, 450 frames is the data range we used
            time_start = np.random.randint(0,int(600 - input_shape[2]))
            x1 = int(start_time[k]) + time_start
            x2 = int(start_time[k]) + time_start + input_shape[2]
            x_2 = np.reshape(data[0][:, x1:x2], input_shape[1:])
            x_train1[i] = x_2                    
            y_train[i] = to_categorical(y_data, num_classes=4, dtype='float32')
            x_train1[i] = np.reshape(data[0][:, x1:x2], input_shape[1:])
            if is_fb:
                x_train2[i] = np.reshape(data[1][:, x1:x2], input_shape[1:])
                x_train3[i] = np.reshape(data[2][:, x1:x2], input_shape[1:])
            # x_train4[i] = np.reshape(train_data4[:, x1:x2], (self.channel, win_train, 1))
            y_train[i] = to_categorical(y_data, num_classes=num_classes, dtype='float32')
        
        if is_fb:
            x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)
            yield x_train, y_train
        else:
            yield x_train1, y_train



def test_data_generator(batch_size, data, y_label, start_time,num_classes,input_shape,is_fb=False):
    input_shape = (batch_size,)+input_shape
    if input_shape[-1] != 1:
        input_shape = input_shape[:-1] + (1,)

    x_train1 = np.zeros(input_shape, dtype=np.float32)
    if is_fb:
        x_train2 = np.zeros(input_shape, dtype=np.float32)
        x_train3 = np.zeros(input_shape, dtype=np.float32)
    # x_train4 = np.zeros((batch_size, self.channel, win_train, 1), dtype=np.float32)
    y_train = np.zeros((batch_size, num_classes), dtype=np.float32)
    for i in range(batch_size):
        k = np.random.randint(0, y_label.shape[0])
        y_data = y_label[k]
        time_start = np.random.randint(0, int(1000 - input_shape[2])) - start_time[0]
        x1 = int(start_time[k]) + time_start
        x2 = int(start_time[k]) + time_start + input_shape[2]
        x_train1[i] = np.reshape(data[0][:, x1:x2], input_shape[1:])
        if is_fb:
            x_train2[i] = np.reshape(data[1][:, x1:x2], input_shape[1:])
            x_train3[i] = np.reshape(data[2][:, x1:x2], input_shape[1:])
        # x_train4[i] = np.reshape(train_data4[:, x1:x2], (self.channel, win_train, 1))
        y_train[i] = to_categorical(y_data, num_classes=num_classes, dtype='float32')
    
    if is_fb:
        x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)
        return x_train, y_train
    else:
        return x_train1, y_train


def save_model(model, model_name, save_path=''):
    """
    save the model to the save_path

    :param model: the model that need to be saved

    :param model_name: the name of the model

    :param save_path: the path of the model
    """
    model_name = os.path.join(save_path, model_name)
    model_name = model_name + '.h5'
    print(model_name)
    model.save(model_name)
