import os
import sys
import time
import queue

from data_acquisition.modules.utils.FileHandler import create_csv_file

dir_path = os.path.dirname(os.path.realpath(__file__))
packages_path = os.path.join(dir_path, "cyKit")
sys.path.append(packages_path)

import data_acquisition.modules.headset.cyKit.cyPyWinUSB as hid
from data_acquisition.modules.headset.cyKit.cyCrypto.Cipher import AES

from data_acquisition.modules.utils.Logger import Logger, app_logger

# logger = Logger(__name__)
logger = app_logger  # Log all in app.log

tasks = queue.Queue()


class EEG(object):

    def __init__(self):

        self._recording_state = False
        self.hid = None
        self.delimiter = ", "
        self.key = ""
        self.cipher = None

        devices_used = 0

        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devices_used += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                # EPOC+ in 16-bit Mode.
                k = ['\0'] * 16
                k = [self.serial_number[-1], self.serial_number[-2], self.serial_number[-2], self.serial_number[-3],
                     self.serial_number[-3], self.serial_number[-3], self.serial_number[-2], self.serial_number[-4],
                     self.serial_number[-1], self.serial_number[-4], self.serial_number[-2], self.serial_number[-2],
                     self.serial_number[-4], self.serial_number[-4], self.serial_number[-2], self.serial_number[-1]]

                self.key = str(''.join(k))
                self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)
                device.set_raw_data_handler(self._data_handler)
        if devices_used == 0:
            logger.error("Can't find EPOC+ usb")
            os._exit(0)
        # sn = self.serial_number

    def _data_handler(self, data):
        join_data = ''.join(map(chr, data[1:]))
        data = self.cipher.decrypt(bytes(join_data, 'latin-1')[0:32])
        if str(data[1]) == "32":  # No Gyro Data.
            return
        tasks.put(data)

    def _convert_epoc_plus(self, value_1, value_2):
        edk_value = "%.8f" % (
                ((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) - 128) * 32.82051289))
        return edk_value

    def get_data(self):

        data = tasks.get()
        # d = ""
        # for i in data:
        #     d += f"{i}, "
        # logger.warning(f"The BCI stream data : {d}")

        # print(str(data[0])) #COUNTER

        try:
            packet_data = ""
            for i in range(2, 16, 2):
                packet_data = packet_data + str(
                    self._convert_epoc_plus(str(data[i]), str(data[i + 1]))) + self.delimiter

            for i in range(18, len(data), 2):
                packet_data = packet_data + str(
                    self._convert_epoc_plus(str(data[i]), str(data[i + 1]))) + self.delimiter

            packet_data = packet_data[:-len(self.delimiter)]
            return str(packet_data)

        except Exception as exception2:
            print(str(exception2))

    def _is_tasks_empty(self):
        """

        :return: True, if a Queue instance has zero elements, otherwise it returns False
        """
        return tasks.empty()

    def clear_tasks(self):
        """
        Safely clear the contents of the queue.
        """
        with tasks.mutex:
            tasks.queue.clear()

    def stop_acquisition(self):
        """
        Change the recording state to break the recording while loop.
        """
        self._recording_state = False

    def log_headset_info(self):
        logger.info("vendor_name : " + self.hid.vendor_name)
        logger.info("vendor_id : " + str(self.hid.vendor_id))
        logger.info("product_name : " + self.hid.product_name)
        logger.info("product_id : " + str(self.hid.product_id))
        logger.info("version_number : " + str(self.hid.version_number))
        logger.info("serial_number : " + self.hid.serial_number)

    def _pause_recording(self, duration):
        if duration > 0:
            time.sleep(duration)

    def _open_file(self, file_path, header):
        try:
            file = open(file_path, "a+", newline='')
            file.write(header + "\n")
            file.flush()
            os.fsync(file.fileno())
            return file
        except Exception as e:
            print("Recording err:")
            print(e)

    def start_recording(self, csv_data_file: str, csv_meta_data_file: str, preparation_duration: int,
                        stimulation_duration: int, rest_duration: int, frequencies: dict, directions_order: list):
        """
        Start recording EEG data into .csv file

        :param csv_data_file: The path of the csv file which will be used for recording EEG data.

        :param csv_meta_data_file: The path of the csv file which will be used for recording the direction and its corresponding starting row.

        :param stimulation_duration: Duration of stimulation recording session (EPOC duration)

        :param preparation_duration: Delay before recording session for preparation period

        :param rest_duration: The rest duration, which at that time the stimulus is stopped.

        :param frequencies: Dictionary of the direction as a key and a frame as a value which freq=60/frame

        :param directions_order: List of the direction in order.

        :type csv_data_file: str

        :type csv_meta_data_file: str

        :type stimulation_duration: int

        :type preparation_duration: int

        :type rest_duration: int

        :type frequencies: dict

        :type directions_order: list
        """

        # Open the targeted csv file for the first time to start recording process.
        data_header = "AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4"
        meta_data_header = "starting_row,label"

        raw_data_file = self._open_file(csv_data_file, data_header)
        meta_data_file = self._open_file(csv_meta_data_file, meta_data_header)

        starting_time = 0
        current_row = 0

        self._recording_state = True

        # Append the raw data into CSV file.
        try:
            for directions in directions_order:
                for direction in directions:
                    if not self._recording_state:
                        break

                    current_direction = direction
                    current_frequency = frequencies[direction]

                    # Delay the recording process if needed.
                    logger.info(f"Start PREPARATION for {preparation_duration} seconds")
                    self._pause_recording(preparation_duration)
                    logger.info("End PREPARATION")

                    logger.info(
                        f"Recording stage started with direction {current_direction} and {current_frequency} HZ")
                    starting_time = time.time()
                    self._recording_state = True

                    meta_data = f"{current_row}, {current_direction}"
                    meta_data_file.write(meta_data + "\n")

                    self._clear_tasks()
                    logger.info(f"Is tasks queue empty: {self._is_tasks_empty()}")

                    logger.info(f"start STIMULATION RECORDING")
                    while self._recording_state:

                        if time.time() - starting_time >= stimulation_duration:
                            logger.info(f"End STIMULATION RECORDING")
                            break

                        if self._is_tasks_empty():
                            continue

                        current_row += 1
                        raw_data = self.get_data()
                        raw_data_file.write(raw_data + "\n")

                    if self._recording_state:
                        logger.info(f"Start REST")
                        self._pause_recording(rest_duration)
                        logger.info("End REST")

        except KeyboardInterrupt:
            print("You stop the recording process!")
        finally:
            raw_data_file.close()
            meta_data_file.close()
            logger.info(
                "The recording session is done after {} seconds and saved at :".format(time.time() - starting_time))
            logger.info("File Path>> " + csv_data_file)

    def epoc_streamer(self, w: float, fs: int):
        logger.info("START Streaming process")
        epoc = []
        num_of_rows = int(w * fs)

        for _ in range(num_of_rows):
            data_row = self.get_data()
            epoc.append([data_row])

        logger.info("FINISH Streaming process")
        return epoc


if __name__ == '__main__':
    headset = EEG()
    while 1:
        data = tasks.get()

        d = ''
        for i in range(len(data)):
            d += f"{data[i]}, "

        print(d)
