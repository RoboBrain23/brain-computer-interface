import random

from data_acquisition.config.config import *
from data_acquisition.modules.ssvep import SSVEP

subject_name = ""
no_of_sessions = 5
preparation_duration = 2
stimulation_duration = 5
rest_duration = 4
full_screen = True

base_directions = [TOP_POSITION, RIGHT_POSITION, DOWN_POSITION, LEFT_POSITION]
final_order = []
for i in range(no_of_sessions):
    final_order.append(random.shuffle(base_directions))


ssvep = SSVEP(subject_name, preparation_duration, stimulation_duration, rest_duration, FREQUENCIES_DICT, full_screen, final_order)
ssvep.start()
