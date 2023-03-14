import os
import sys
import time
import queue

dir_path = os.path.dirname(os.path.realpath(__file__))
packages_path = os.path.join(dir_path, "cyKit")
sys.path.append(packages_path)

import data_acquisition.modules.headset.cyKit.cyPyWinUSB as hid
from data_acquisition.modules.headset.cyKit.cyCrypto.Cipher import AES

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

    def start_recording(self, file_path: str, recording_duration: int, delay_before_recording: int):
        """
        Start recording EEG data into .csv file


        :param file_path: The path of the csv file which will be used for recording EEG data.

        :param recording_duration: Duration of recording session (EPOC duration)

        :param delay_before_recording: Delay before recording session for preparation period

        :type file_path: str

        :type recording_duration: int

        :type delay_before_recording: int
        """

        # Delay the recording process if needed.
        if delay_before_recording > 0:
            print("delay starting for {} seconds".format(delay_before_recording))
            time.sleep(delay_before_recording)

        print("delays is finished!")
        starting_time = time.time()

        self._recording_state = True
        try:
            record_csv_file = open(file_path, "a+", newline='')

            csv_header = "F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4"

            record_csv_file.write(csv_header + "\n")

            record_csv_file.flush()
            os.fsync(record_csv_file.fileno())

            # Append the raw data into CSV file.
            try:
                while self._recording_state:

                    # Stop recording after the given duration
                    if time.time() - starting_time > recording_duration:
                        print("Time is up after {} seconds".format(time.time() - starting_time))
                        break

                    if self.is_tasks_empty():
                        continue

                    record_csv_file.write(self.get_data() + "\n")
                    # print(self.get_data())    # print the recorded data.
            except KeyboardInterrupt:
                print("You stop the recording process!")
            finally:
                record_csv_file.close()
                print(
                    "The recording session is done after {} seconds and saved at :".format(time.time() - starting_time))
                print("File Path>> " + file_path)

        except Exception as e:
            print("Recording err:")
            print(e)
        pass

    def stop_recording(self):
        self._recording_state = False
