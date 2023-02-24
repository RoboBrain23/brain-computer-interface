import os
import time

import cyPyWinUSB as hid
import queue
from cyCrypto.Cipher import AES

tasks = queue.Queue()


class EEG(object):

    def __init__(self):
        self.hid = None
        self.delimiter = ", "

        devices_used = 0

        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devices_used += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                device.set_raw_data_handler(self.dataHandler)
        if devices_used == 0:
            os._exit(0)
        sn = self.serial_number

        # EPOC+ in 16-bit Mode.
        k = ['\0'] * 16
        k = [sn[-1], sn[-2], sn[-2], sn[-3], sn[-3], sn[-3], sn[-2], sn[-4], sn[-1], sn[-4], sn[-2], sn[-2], sn[-4],
             sn[-4], sn[-2], sn[-1]]

        self.key = str(''.join(k))
        self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)

    def dataHandler(self, data):
        join_data = ''.join(map(chr, data[1:]))
        data = self.cipher.decrypt(bytes(join_data, 'latin-1')[0:32])
        if str(data[1]) == "32":  # No Gyro Data.
            return
        tasks.put(data)

    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (
                ((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) - 128) * 32.82051289))
        return edk_value

    def get_data(self):

        data = tasks.get()
        # print(str(data[0])) #COUNTER

        try:
            packet_data = ""
            for i in range(2, 16, 2):
                packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1]))) + self.delimiter

            for i in range(18, len(data), 2):
                packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1]))) + self.delimiter

            packet_data = packet_data[:-len(self.delimiter)]
            return str(packet_data)

        except Exception as exception2:
            print(str(exception2))

    def is_tasks_empty(self):
        """

        :return: True, if a Queue instance has zero elements, otherwise it returns False
        """
        return tasks.empty()

    def get_headset_info(self):

        print("vendor_name : " + self.hid.vendor_name)
        print("vendor_id : " + str(self.hid.vendor_id))
        print("product_name : " + self.hid.product_name)
        print("product_id : " + str(self.hid.product_id))
        print("version_number : " + str(self.hid.version_number))
        print("serial_number : " + self.hid.serial_number)


    def create_csv_file(self, prefix="", path=os.path.realpath("")):
        """
        Creates a CSV file for recording EEG data with a unique file name based on the current date and time.

        :param prefix: The prefix that will be added at the beginning of the CSV file name.
        :param path: The path of the records folder.
        :return: The full path of the CSV file.
        """


        # Create a folder for records.
        if not os.path.exists(path + "/EEG-Records"):
            try:
                os.mkdir(path + "/EEG-Records")
            except Exception as msg:
                print("Failed to Create Directory: '" + path + "/EEG-Records/' \r\n Please Check Permissions. ")
                print(str(msg))
                return

        # Add timestamp to the name of recorded file.
        file_name = prefix + "_" if prefix != "" else ""
        file_name += str(time.strftime("%d.%m.%y_%H.%M.%S"))

        # Create a CSV record file.
        file_path = path + "/EEG-Records/" + file_name + '.csv'

        return file_path

    def record_csv(self):
        """
        Record the EPOC+ raw data in a CSV file
        """
        file_path = self.create_csv_file()

        try:
            record_file = open(file_path, "a+", newline='')

            csv_header = "F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4"

            record_file.write(csv_header + "\n")

            record_file.flush()
            os.fsync(record_file.fileno())

            # Append the raw data into CSV file.
            try:
                while 1:
                    while self.is_tasks_empty():
                        pass
                    record_file.write(self.get_data() + "\n")
                    print(self.get_data())
            except KeyboardInterrupt:
                record_file.close()
                print("File Path>> " + file_path)

        except Exception as e:
            print(e)
