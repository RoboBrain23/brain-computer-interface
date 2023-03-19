import asyncio
import threading
import time

from data_acquisition.config.config import FREQUENCIES_DICT, POSITIONS
from data_acquisition.modules.headset.EEG import EEG
from data_acquisition.modules.utils.stimulus.blankboard.BlankboardStimulus import BlankboardStimulus
# from data_acquisition.modules.utils.stimulus.checkerboard.CheckerboardStimulus import CheckerboardStimulus
from data_acquisition.modules.utils.FileHandler import create_csv_file

from data_acquisition.modules.utils.Logger import Logger, app_logger

# logger = Logger(__name__)
logger = app_logger  # Log all in app.log


class SSVEP:
    def __init__(self, preparation_duration: int, stimulation_duration: int, rest_duration: int, frequencies: dict,
                 full_screen_mode: bool, order: list):

        self._preparation_duration = preparation_duration
        self._stimulation_duration = stimulation_duration
        self._rest_duration = rest_duration
        self._full_stimulation_duration = self._preparation_duration + self._stimulation_duration

        self._frequencies = frequencies
        self._direction_order = order

        self._stimulus_full_screen_mode = False
        self._stimulus_screen_width = 1024
        self._stimulus_screen_height = 768
        self._full_screen_mode = full_screen_mode

        self._session_state = True
        self._epoc = None
        self._stimulus = None

    async def start_recording(self):
        """
        Start recording eeg data and store the data in csv file
        """

        prefix = ""
        base_file_name = time.strftime("%d.%m.%y_%H.%M.%S")
        suffix = "meta_data"
        logger.info("STARTING CREATING csv FILES")

        # Create two .csv file one for the raw data and the second for the data indexing.
        csv_data_file_path = create_csv_file(prefix, base_file_name)
        csv_meta_data_file = create_csv_file(prefix, base_file_name, suffix)

        self._epoc.start_recording(csv_data_file_path, csv_meta_data_file, self._preparation_duration,
                                   self._stimulation_duration, self._rest_duration, self._frequencies,
                                   self._direction_order)

    def stop_recording(self):
        self._epoc.stop_acquisition()

    def start_stimulation_gui(self):
        """
        Start the stimulation GUI
        """
        self._stimulus = BlankboardStimulus(self._frequencies, self._preparation_duration, self._stimulation_duration,
                                            self._rest_duration, self._full_screen_mode, self._direction_order)
        self._stimulus.run()

        # self._stimulus = CheckerboardStimulus(self._stimulus_full_screen_mode, self._stimulus_screen_width,
        #                                       self._stimulus_screen_height, self._frequencies,
        #                                       self._full_stimulation_duration, self._rest_duration)
        # self._stimulus.run(self._is_training_mode)

    def start(self):
        try:
            self._epoc = EEG()
            # Create a thread for recording data and start the GUI in the main thread.
            asyncio_thread = threading.Thread(target=asyncio.run, args=(self.start_recording(),))
            asyncio_thread.start()

            self.start_stimulation_gui()
            logger.info("GUI Closed!")
            self.stop_recording()
            logger.info("Recording Closed!\n")

        except Exception as e:
            logger.error('Error occurred while recording')
            self.stop_recording()

    def set_session_state(self, state: bool):
        self._session_state = state

    def get_session_state(self):
        return self._session_state


if __name__ == '__main__':
    preparation_duration = 1
    stimulation_duration = 1
    rest_duration = 1
    full_screen = False

    ssvep = SSVEP(preparation_duration, stimulation_duration, rest_duration, FREQUENCIES_DICT, full_screen, POSITIONS)
    ssvep.start()
