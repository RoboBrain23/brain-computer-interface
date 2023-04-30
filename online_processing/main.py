import asyncio
import multiprocessing
import threading
from time import sleep

from online_processing.utils.data_streamer import *
from online_processing.utils.stimulus import start_stimulus


async def data_streamer(w, fs):
    return get_epoc(w, fs)


async def stimulus():
    start_stimulus()


def start_acquisition():
    window_size = 20
    sampling_frequency = 128

    try:
        stimulus_thread = threading.Thread(name="StimulusThread", target=asyncio.run, args=(stimulus(),))
        stimulus_thread.start()

        asyncio.run(data_streamer(window_size, sampling_frequency))

    except Exception as e:
        print("Treading ERROR")


def model():
    # TODO: Implement your model code here.
    i = 0
    while True:
        sleep(1)
        print(i)
        i += 1


def main():
    try:
        # creating processes
        p1 = multiprocessing.Process(target=start_acquisition)
        p2 = multiprocessing.Process(target=model)

        # starting process 1
        p1.start()
        # starting process 2
        p2.start()

        # wait until process 1 is finished
        p1.join()
        # wait until process 2 is finished
        p2.join()

        # both processes finished
        print("Done!")

    except KeyboardInterrupt:
        print('Program has ended')


if __name__ == '__main__':
    main()
