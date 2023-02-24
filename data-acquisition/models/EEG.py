import os

import logging
import cyPyWinUSB as hid
import queue
from cyCrypto.Cipher import AES


tasks = queue.Queue()

class EEG(object):

    def __init__(self):
        self.hid = None
        self.delimiter = ", "

        devicesUsed = 0

        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devicesUsed += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                device.set_raw_data_handler(self.dataHandler)
        if devicesUsed == 0:
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

    def isTasksEmpty(self):
        return tasks.empty()

    def getDeviceInfo(self):
        print("vendor_name : " + self.hid.vendor_name)
        print("vendor_id : " + str(self.hid.vendor_id))
        print("product_name : " + self.hid.product_name)
        print("product_id : " + str(self.hid.product_id))
        print("version_number : " + str(self.hid.version_number))
        print("serial_number : " + self.hid.serial_number)




