import numpy as np
from keras.utils import to_categorical
import os


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


def train_data_generator(batch_size, data, y_label, start_time, train_list, num_classes, input_shape):
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
    input_shape = (batch_size,) + input_shape
    x_train = np.zeros(input_shape, dtype=np.float32)
    y_train = np.zeros((batch_size, num_classes), dtype=np.float32)
    while True:
        labels_count = np.zeros(num_classes)
        i = 0
        while np.sum(labels_count) < batch_size:
            k = np.random.choice(train_list)
            y_data = y_label[k]
            if labels_count[y_data] < batch_size // num_classes:
                labels_count[y_data] += 1
            else:
                continue
            time_start = np.random.randint(0, int(630 - input_shape[2])) - start_time[0]
            x1 = int(start_time[k]) + time_start
            x2 = int(start_time[k]) + time_start + input_shape[2]
            x_train_data = data[x1:x2, :]
            # todo: add preprocessing functions here

            x_train[i] = np.reshape(x_train_data[0][:, x1:x2], input_shape[1:])
            y_train[i] = to_categorical(y_data, num_classes=num_classes, dtype='float32')
            i += 1
        yield x_train, y_train


def test_data_generator(batch_size, data, y_label, start_time, num_classes, input_shape):
    """
    generate the test data

    :param batch_size:  the size of the batch

    :param data: the data of the test data

    :param y_label:  the label of the test data

    :param start_time:  the start time of the test data

    :param num_classes:  the number of the classes

    :param input_shape:  the shape of the input data

    :return:  the test data
    """
    input_shape = (batch_size,) + input_shape
    x_train = np.zeros(input_shape, dtype=np.float32)
    y_train = np.zeros((batch_size, num_classes), dtype=np.float32)
    for i in range(batch_size):
        k = np.random.randint(0, y_label.shape[0])
        y_data = y_label[k]
        time_start = np.random.randint(0, int(630 - input_shape[2])) - start_time[0]
        x1 = int(start_time[k]) + time_start
        x2 = int(start_time[k]) + time_start + input_shape[2]
        x_train_data = data[x1:x2, :]
        # todo: add preprocessing functions here

        x_train[i] = np.reshape(x_train_data[0][:, x1:x2], input_shape[1:])
        y_train[i] = to_categorical(y_data, num_classes=num_classes, dtype='float32')
    return x_train, y_train


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
