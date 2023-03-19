import random

from data_acquisition.config.config import *
from data_acquisition.modules.ssvep import SSVEP

preparation_duration = 2
stimulation_duration = 5
rest_duration = 4
full_screen = True

order = [TOP_POSITION, RIGHT_POSITION, DOWN_POSITION, LEFT_POSITION]
random.shuffle(order)

ssvep = SSVEP(preparation_duration, stimulation_duration, rest_duration, FREQUENCIES_DICT, full_screen, order)
ssvep.start()
