from data_acquisition.modules.utils.stimulus.blankboard.BlankboardStimulus import BlankboardStimulus
from data_acquisition.config.config import FREQUENCIES_DICT

stimulus = BlankboardStimulus()


def start_stimulus():
    stimulus.stim(FREQUENCIES_DICT)
