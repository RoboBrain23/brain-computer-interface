import os
import sys
import time
import queue

dir_path = os.path.dirname(os.path.realpath(__file__))
packages_path = os.path.join(dir_path, "cyKit")
sys.path.append(packages_path)

import data_acquisition.modules.headset.cyKit.cyPyWinUSB as hid
from data_acquisition.modules.headset.cyKit.cyCrypto.Cipher import AES

from data_acquisition.modules.utils.Logger import Logger

logger = Logger(__name__)

tasks = queue.Queue()


class EEG(object):

    def __init__(self):

        self._recording_state = False
        self.hid = None
        self.delimiter = ", "

        devices_used = 0

        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devices_used += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                device.set_raw_data_handler(self.data_handler)
        if devices_used == 0:
            os._exit(0)
        sn = self.serial_number

        # EPOC+ in 16-bit Mode.
        k = ['\0'] * 16
        k = [sn[-1], sn[-2], sn[-2], sn[-3], sn[-3], sn[-3], sn[-2], sn[-4], sn[-1], sn[-4], sn[-2], sn[-2], sn[-4],
             sn[-4], sn[-2], sn[-1]]

        self.key = str(''.join(k))
        self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)

    def data_handler(self, data):
        join_data = ''.join(map(chr, data[1:]))
        data = self.cipher.decrypt(bytes(join_data, 'latin-1')[0:32])
        if str(data[1]) == "32":  # No Gyro Data.
            return
        tasks.put(data)

    def convert_epoc_plus(self, value_1, value_2):
        edk_value = "%.8f" % (
                ((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) - 128) * 32.82051289))
        return edk_value

    def get_data(self):

        data = tasks.get()
        # print(str(data[0])) #COUNTER

        try:
            packet_data = ""
            for i in range(2, 16, 2):
                packet_data = packet_data + str(self.convert_epoc_plus(str(data[i]), str(data[i + 1]))) + self.delimiter

            for i in range(18, len(data), 2):
                packet_data = packet_data + str(self.convert_epoc_plus(str(data[i]), str(data[i + 1]))) + self.delimiter

            packet_data = packet_data[:-len(self.delimiter)]
            return str(packet_data)

        except Exception as exception2:
            print(str(exception2))

    def is_tasks_empty(self):
        """

        :return: True, if a Queue instance has zero elements, otherwise it returns False
        """
        return tasks.empty()

    def print_headset_info(self):

        print("vendor_name : " + self.hid.vendor_name)
        print("vendor_id : " + str(self.hid.vendor_id))
        print("product_name : " + self.hid.product_name)
        print("product_id : " + str(self.hid.product_id))
        print("version_number : " + str(self.hid.version_number))
        print("serial_number : " + self.hid.serial_number)

    def start_recording(self, csv_data_file: str, csv_meta_data_file: str, recording_duration: int,
                        delay_before_recording: int, rest_duration: int,
                        frequencies: dict):
        """
        Start recording EEG data into .csv file

        :param csv_data_file: The path of the csv file which will be used for recording EEG data.

        :param csv_meta_data_file: The path of the csv file which will be used for recording the direction and its corresponding starting row.

        :param recording_duration: Duration of recording session (EPOC duration)

        :param delay_before_recording: Delay before recording session for preparation period

        :param rest_duration: The rest duration, which at that time the stimulus is stopped.

        :param frequencies: Dictionary of the direction as a key and a frame as a value which freq=60/frame

        :type csv_data_file: str

        :type csv_meta_data_file: str

        :type recording_duration: int

        :type delay_before_recording: int

        :type rest_duration: int

        :type frequencies: dict
        """

        # Open the targeted csv file for the first time to start recording process.

        data_header = "F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4, label, frequency"
        meta_data_header = "starting_row, label"

        raw_data_file = self.open_file(csv_data_file, data_header)
        meta_data_file = self.open_file(csv_meta_data_file, meta_data_header)

        logger.info(f"The preparation time: {delay_before_recording} sec, "
                    f"recording time: {recording_duration} sec, "
                    f"rest time: {rest_duration} sec with a total recording time of {recording_duration + rest_duration} sec, "
                    f"a total time of {delay_before_recording + recording_duration + rest_duration} sec for the one EPOC")
        log_t = 0  # For logging
        starting_time = 0
        current_row = 1

        # Append the raw data into CSV file.
        try:
            for direction, frame in frequencies.items():
                logger.info(f"at t = {log_t} sec:")
                current_direction = direction
                current_frequency = 60 / frame

                # Delay the recording process if needed.
                logger.info("preparation stage started for {} seconds".format(delay_before_recording))
                self.preparation_delay(delay_before_recording)
                logger.info("preparation stage is finished!")

                log_t += delay_before_recording
                logger.info(f"at t = {log_t} sec:")
                logger.info(f"Recording stage started with direction {current_direction} and {current_frequency} HZ")

                starting_time = time.time()
                self._recording_state = True

                logger.info(f"Current row: {current_row}, Current direction: {current_direction}")
                meta_data = f"{current_row}, {current_direction}"
                meta_data_file.write(meta_data + "\n")

                while self._recording_state:

                    # Stop recording after the recording and rest duration.
                    if time.time() - starting_time > recording_duration + rest_duration:
                        log_t += rest_duration
                        logger.info(f"at t = {log_t} sec:")
                        logger.info(f"This epoc is ended after {time.time() - starting_time} seconds")
                        break

                    if self.is_tasks_empty():
                        continue

                    # Change the direction and frequency to the rest value after epoc duration
                    if recording_duration < time.time() - starting_time < recording_duration + rest_duration and current_direction != "rest":
                        log_t += recording_duration
                        logger.info(f"at t = {log_t} sec:")
                        logger.info(
                            "Changing the current_direction from {} ({})HZ to rest (0)HZ".format(current_direction,
                                                                                                 current_frequency))
                        current_direction = "rest"
                        current_frequency = 0
                        logger.info(f"Current row: {current_row}, Current direction: {current_direction}")
                        meta_data = f"{current_row}, {current_direction}"
                        meta_data_file.write(meta_data + "\n")
                        # logger.warning(f"Current row: {current_row}, Current direction: {current_direction}")

                    current_row += 1

                    raw_data = self.get_data() + ", {}, {}".format(current_direction, current_frequency)

                    raw_data_file.write(raw_data + "\n")

                    # print(self.get_data())    # print the recorded data.
        except KeyboardInterrupt:
            print("You stop the recording process!")
        finally:
            raw_data_file.close()
            meta_data_file.close()
            logger.info(
                "The recording session is done after {} seconds and saved at :".format(time.time() - starting_time))
            logger.info("File Path>> " + csv_data_file)

    def preparation_delay(self, delay_before_recording):
        if delay_before_recording > 0:
            time.sleep(delay_before_recording)

    def open_file(self, file_path, header):
        try:
            file = open(file_path, "a+", newline='')
            file.write(header + "\n")
            file.flush()
            os.fsync(file.fileno())

            return file
        except Exception as e:
            print("Recording err:")
            print(e)

    def stop_recording(self):
        self._recording_state = False
