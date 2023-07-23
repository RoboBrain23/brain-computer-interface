import multiprocessing
import sys

import numpy as np
import tensorflow as tf
from keras.models import load_model
from scipy.signal import butter, filtfilt

from online_processing.utils.data_streamer import get_epoc, empty_buffer
from online_processing.utils.stimulus import start_stimulus

from data_acquisition.modules.utils.Logger import Logger, app_logger

# logger = Logger(__name__)
logger = app_logger  # Log all in app.log


def reference(data):
    data = data - data.mean(axis=1)[:, np.newaxis]
    return data


def filter_data(data, low=9, high=15, fs=128, order=4, is_fb=False):
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
        wn1 = 2 * harmonics[0][0] / fs
        wn2 = 2 * harmonics[-1][-1] / fs
        channel_data_list = []
        for i in range(data.shape[1]):
            b, a = butter(order, [wn1, wn2], 'bandpass')
            filtedData = filtfilt(b, a, data[:, i])
            channel_data_list.append(filtedData)
        channel_data_list = np.array([channel_data_list])

        return channel_data_list


def get_harmonics(start_freq, end_freq):
    harmonics = []
    i = 1
    for i in range(1, 4):
        harmonics.append([int(i * start_freq) - 2, int(i * end_freq) + 2])
        i += 1
    return harmonics


def preprocess_trial(data, fs=128, order=4, low=9, high=15):
    print(data.shape)
    # data = reference(data)
    data = filter_data(data[:, 6:8], low, high, fs=fs, order=order)
    return data.T


def predict_trial(data, model_path, model_type='normal'):
    data = preprocess_trial(data)
    # print(data.shape)
    # data shape should be (1,2, window_size*fs, 1)
    # add new axis for batch size
    # data = data[np.newaxis, :, :, np.newaxis,np.newaxis]
    data = data.transpose(1, 0, 2)[np.newaxis, :, :, :]
    if model_type != 'lite':
        model = load_model(model_path, compile=False)
        # print(model.input_shape)
        prediction = model.predict(data, verbose=0)
    else:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
    # print(model.summary())
    y_pred = np.argmax(prediction, axis=1)[0]
    # print(prediction[0])
    if prediction[0][y_pred] > 0.3:
        return y_pred
    else:
        return 4


def get_direction(y_pred):
    labels_name = ['top', 'right', 'down', 'left', 'no movement']
    return labels_name[y_pred]


def data_streamer(q, w, fs, terminate_flag):
    empty_buffer()
    while not terminate_flag.value:
        data = get_epoc(w, fs)
        # logger.info("Date is received")
        q.put(data)
        # logger.info("Date is added to the Queue")


def stimulus(terminate_flag):
    start_stimulus()
    terminate_flag.value = True


def model(q, terminate_flag):
    while not terminate_flag.value:
        if not q.empty():
            # logger.info("Data found in the Queue")
            data = q.get()
            # logger.info("Date is get from the Queue")

            # print(f"OLD data: {data}")
            # np.save("data.npy", data)
            # TODO: Implement your model code here.
            # data = reference(data)

            # data = preprocess_trial(data=data)
            command = predict_trial(data=data, model_path="./models/TCNN-1.8s-1000_-_Copy.h5")

            final_command = get_direction(y_pred=command)
            print(f"Final Command: {final_command}")


if __name__ == '__main__':
    window_size = 1.8
    sampling_frequency = 128

    terminate_flag = multiprocessing.Value('b', False)
    data_queue = multiprocessing.Queue()

    try:
        # creating processes
        p1 = multiprocessing.Process(name="StimulusProcess", target=stimulus, args=(terminate_flag,))
        p2 = multiprocessing.Process(name="DataStreamerProcess", target=data_streamer,
                                     args=(data_queue, window_size, sampling_frequency, terminate_flag,))
        p3 = multiprocessing.Process(name="ModelProcess", target=model, args=(data_queue, terminate_flag,))

        p1.start()
        p2.start()
        p3.start()

        while True:
            if terminate_flag.value:
                p1.terminate()
                p2.terminate()
                p3.terminate()
                break

        p1.join()
        p2.join()
        p3.join()

        print("All processes finished!")

    except KeyboardInterrupt:
        print('Program has ended')