from data_acquisition.config.config import FREQUENCIES_DICT
from data_acquisition.modules.ssvep import SSVEP

preparation_duration = 2
stimulation_duration = 4
rest_duration = 4
full_screen = True

ssvep = SSVEP(preparation_duration, stimulation_duration, rest_duration, FREQUENCIES_DICT, full_screen)
ssvep.start()
