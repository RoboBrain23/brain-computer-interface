import multiprocessing
import sys

from online_processing.utils.data_streamer import get_epoc, empty_buffer
from online_processing.utils.stimulus import start_stimulus

from data_acquisition.modules.utils.Logger import Logger, app_logger

# logger = Logger(__name__)
logger = app_logger  # Log all in app.log

def data_streamer(q, w, fs, terminate_flag):
    empty_buffer()
    while not terminate_flag.value:
        data = get_epoc(w, fs)
        logger.info("Date is received")
        q.put(data)
        logger.info("Date is added to the Queue")


def stimulus(terminate_flag):
    start_stimulus()
    terminate_flag.value = True


def model(q, terminate_flag):
    while not terminate_flag.value:
        if not q.empty():
            logger.info("Data found in the Queue")
            data = q.get()
            logger.info("Date is get from the Queue")
            # TODO: Implement your model code here.
            print(f"Got data: {data}")


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
