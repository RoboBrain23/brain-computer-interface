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

sensor_bits = {
    'F3': [10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
    'FC5': [28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9],
    'AF3': [46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27],
    'F7': [48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45],
    'T7': [66, 67, 68, 69, 70, 71, 56, 57, 58, 59, 60, 61, 62, 63],
    'P7': [84, 85, 86, 87, 72, 73, 74, 75, 76, 77, 78, 79, 64, 65],
    'O1': [102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 80, 81, 82, 83],
    'O2': [140, 141, 142, 143, 128, 129, 130, 131, 132, 133, 134, 135, 120, 121],
    'P8': [158, 159, 144, 145, 146, 147, 148, 149, 150, 151, 136, 137, 138, 139],
    'T8': [160, 161, 162, 163, 164, 165, 166, 167, 152, 153, 154, 155, 156, 157],
    'F8': [178, 179, 180, 181, 182, 183, 168, 169, 170, 171, 172, 173, 174, 175],
    'AF4': [196, 197, 198, 199, 184, 185, 186, 187, 188, 189, 190, 191, 176, 177],
    'FC6': [214, 215, 200, 201, 202, 203, 204, 205, 206, 207, 192, 193, 194, 195],
    'F4': [216, 217, 218, 219, 220, 221, 222, 223, 208, 209, 210, 211, 212, 213]
}
quality_bits = [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

battery_values = {
    "255": 100,
    "254": 100,
    "253": 100,
    "252": 100,
    "251": 100,
    "250": 100,
    "249": 100,
    "248": 100,
    "247": 99,
    "246": 97,
    "245": 93,
    "244": 89,
    "243": 85,
    "242": 82,
    "241": 77,
    "240": 72,
    "239": 66,
    "238": 62,
    "237": 55,
    "236": 46,
    "235": 32,
    "234": 20,
    "233": 12,
    "232": 6,
    "231": 4,
    "230": 3,
    "229": 2,
    "228": 2,
    "227": 2,
    "226": 1,
    "225": 0,
    "224": 0,
}

g_battery = 0

def get_level(data, bits):
    """
    Returns sensor level value from data using sensor bit mask in micro volts (uV).
    """
    level = 0
    for i in range(13, -1, -1):
        level <<= 1
        b, o = (bits[i] / 8) + 1, bits[i] % 8
        level |= (ord(data[b]) >> o) & 1
    return level


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

    def _clear_tasks(self):
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

    def handle_quality(self, sensors):
        """
        Sets the quality value for the sensor from the quality bits in the packet data.
        Optionally will return the value.

        :param sensors - reference to sensors dict in Emotiv class.

        """
        if self.old_model:
            current_contact_quality = get_level(self.raw_data, quality_bits) / 540
        else:
            current_contact_quality = get_level(self.raw_data, quality_bits) / 1024
        sensor = ord(self.raw_data[0])
        if sensor == 0 or sensor == 64:
            sensors['F3']['quality'] = current_contact_quality
        elif sensor == 1 or sensor == 65:
            sensors['FC5']['quality'] = current_contact_quality
        elif sensor == 2 or sensor == 66:
            sensors['AF3']['quality'] = current_contact_quality
        elif sensor == 3 or sensor == 67:
            sensors['F7']['quality'] = current_contact_quality
        elif sensor == 4 or sensor == 68:
            sensors['T7']['quality'] = current_contact_quality
        elif sensor == 5 or sensor == 69:
            sensors['P7']['quality'] = current_contact_quality
        elif sensor == 6 or sensor == 70:
            sensors['O1']['quality'] = current_contact_quality
        elif sensor == 7 or sensor == 71:
            sensors['O2']['quality'] = current_contact_quality
        elif sensor == 8 or sensor == 72:
            sensors['P8']['quality'] = current_contact_quality
        elif sensor == 9 or sensor == 73:
            sensors['T8']['quality'] = current_contact_quality
        elif sensor == 10 or sensor == 74:
            sensors['F8']['quality'] = current_contact_quality
        elif sensor == 11 or sensor == 75:
            sensors['AF4']['quality'] = current_contact_quality
        elif sensor == 12 or sensor == 76 or sensor == 80:
            sensors['FC6']['quality'] = current_contact_quality
        elif sensor == 13 or sensor == 77:
            sensors['F4']['quality'] = current_contact_quality
        elif sensor == 14 or sensor == 78:
            sensors['F8']['quality'] = current_contact_quality
        elif sensor == 15 or sensor == 79:
            sensors['AF4']['quality'] = current_contact_quality
        else:
            sensors['Unknown']['quality'] = current_contact_quality
            sensors['Unknown']['value'] = sensor
        return current_contact_quality


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

                    logger.info(f"Recording stage started with direction {current_direction} and {current_frequency} HZ")
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

if __name__ == '__main__':
    headset = EEG()
    while 1:
        data = tasks.get()
        g_battery = 0

        counter = data[0]
        # counter = ord(c)
        if counter > 127:
            g_battery = battery_values[str(counter)]
            counter = 128

        d=''

        for i in range(len(data)):
            d += f"{data[i]}, "
        print(f"battery: {data[16]}")
        print(f"G_battery: {g_battery}")

        print(d)